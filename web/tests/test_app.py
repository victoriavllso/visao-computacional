import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from app import images, pipeline
from app.main import app

NUTRIENT_KEYS = set(pipeline.NUTRIENTS)


def jpeg(seed: int, size=(60, 80)) -> bytes:
    image = np.random.default_rng(seed).integers(0, 255, (*size, 3), np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("APP_MOCK_VERDICT", raising=False)
    return TestClient(app)


def upload(client, front=None, back=None, name="Biscoito"):
    files = {
        "front": ("frente.jpg", front or jpeg(1), "image/jpeg"),
        "back": ("verso.jpg", back or jpeg(2), "image/jpeg"),
    }
    return client.post("/api/analyses", files=files, data={"name": name})


def test_upload_returns_documented_result(client):
    response = upload(client)
    assert response.status_code == 201
    entry = response.json()
    assert entry["name"] == "Biscoito"
    result = entry["result"]
    assert result["verdict"] in pipeline.VERDICTS
    assert result["reasons"] and all(isinstance(r, str) for r in result["reasons"])
    assert set(result["nutrition"]) == {
        "basis",
        "added_sugars_g",
        "saturated_fat_g",
        "sodium_mg",
    }
    assert set(result["required_warnings"]) == NUTRIENT_KEYS
    assert set(result["found_warnings"]) == NUTRIENT_KEYS
    assert result["back_photo_type"] in {
        "table",
        "linear",
        "old_format",
        "no_table",
        "uncertain",
    }
    assert result["mock"] is True


def test_photos_are_served_downscaled(client):
    entry = upload(client, front=jpeg(3, size=(2000, 1500))).json()
    response = client.get(entry["photos"]["front"]["thumb"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    thumb = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(thumb.shape[:2]) == images.THUMB_SIZE


@pytest.mark.parametrize("verdict", pipeline.VERDICTS)
def test_forced_mock_verdict(client, monkeypatch, verdict):
    monkeypatch.setenv("APP_MOCK_VERDICT", verdict)
    assert upload(client).json()["verdict"] == verdict


def test_mock_is_deterministic(client):
    first = upload(client).json()["result"]
    second = upload(client).json()["result"]
    assert first == second


def test_history_list_detail_delete(client):
    first = upload(client, name="A").json()
    second = upload(client, front=jpeg(5), name="B").json()

    listed = client.get("/api/analyses").json()
    assert [entry["id"] for entry in listed] == [second["id"], first["id"]]
    assert "result" not in listed[0]

    detail = client.get(f"/api/analyses/{first['id']}").json()
    assert detail["result"] == first["result"]

    assert client.delete(f"/api/analyses/{first['id']}").status_code == 204
    assert client.get(f"/api/analyses/{first['id']}").status_code == 404
    assert client.get(first["photos"]["front"]["view"]).status_code == 404
    assert client.delete(f"/api/analyses/{first['id']}").status_code == 404
    assert [entry["id"] for entry in client.get("/api/analyses").json()] == [
        second["id"]
    ]


def test_json_export(client):
    entry = upload(client).json()
    response = client.get(f"/api/analyses/{entry['id']}/export")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    exported = response.json()
    assert exported["id"] == entry["id"]
    assert exported["result"]["nutrition"] == entry["result"]["nutrition"]


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "vazio"),
        (b"not an image at all", "Formato"),
        (b"\xff\xd8\xff" + b"\0" * 100, "corrompido"),
    ],
)
def test_bad_upload_is_rejected(client, data, message):
    files = {
        "front": ("f.jpg", jpeg(1), "image/jpeg"),
        "back": ("b.jpg", data, "image/jpeg"),
    }
    response = client.post("/api/analyses", files=files)
    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert "verso" in response.json()["detail"]
    assert client.get("/api/analyses").json() == []


def test_too_large_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(images, "MAX_UPLOAD_BYTES", 1000)
    response = upload(client, front=jpeg(1, size=(200, 200)))
    assert response.status_code == 422
    assert "grande" in response.json()["detail"]


def test_missing_photo_is_rejected(client):
    files = {"front": ("f.jpg", jpeg(1), "image/jpeg")}
    assert client.post("/api/analyses", files=files).status_code == 422


def test_unknown_or_malformed_id(client):
    assert client.get("/api/analyses/" + "0" * 32).status_code == 404
    assert client.get("/api/analyses/..%2F..%2Fetc").status_code in (404, 422)


def test_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Analisar produto" in response.text
    assert client.get("/static/app.js").status_code == 200


def readings(required, found, photo_type="table"):
    return pipeline.Readings({}, required, found, photo_type, {})


def test_summarize_missing_warning_is_a_violation():
    required = {"added_sugars": False, "saturated_fat": False, "sodium": True}
    found = {"added_sugars": False, "saturated_fat": False, "sodium": False}
    verdict, reasons = pipeline.summarize(readings(required, found))
    assert verdict == "non_compliant"
    assert reasons == ["Deveria ter ALTO EM SÓDIO e não tem."]


def test_summarize_violation_wins_over_unknown():
    required = {"added_sugars": True, "saturated_fat": None, "sodium": False}
    found = {"added_sugars": False, "saturated_fat": False, "sodium": False}
    assert pipeline.summarize(readings(required, found))[0] == "non_compliant"


def test_summarize_unknown_is_inconclusive():
    required = {"added_sugars": True, "saturated_fat": False, "sodium": False}
    found = {"added_sugars": True, "saturated_fat": None, "sodium": False}
    assert pipeline.summarize(readings(required, found))[0] == "inconclusive"


def test_summarize_unreadable_back_is_inconclusive():
    flags = dict.fromkeys(pipeline.NUTRIENTS, False)
    verdict, _ = pipeline.summarize(readings(flags, flags, "old_format"))
    assert verdict == "inconclusive"


def test_summarize_all_match_is_compliant():
    required = {"added_sugars": True, "saturated_fat": False, "sodium": True}
    assert pipeline.summarize(readings(required, dict(required)))[0] == "compliant"
