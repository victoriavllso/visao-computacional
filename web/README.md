# Web app

Send a photo of the **front** and of the **back** of a food package and see
whether its "ALTO EM" warnings follow ANVISA's front-of-pack rule
(RDC 429/2020, IN 75/2020):

- ✅ **De acordo com a ANVISA**: every required warning is printed, and no other.
- ⚠️ **Não foi possível decidir**: a value or a warning couldn't be read, or the
  back photo has no new-format table (old per-portion format, no table).
- ❌ **Viola as regras**: a required warning is missing (e.g. "Deveria ter ALTO
  EM SÓDIO e não tem"), or a warning is printed below the limit.

The result shows the reasons, the warnings required vs printed, the nutrition
table read from the back photo, the photos, and a JSON export. Past analyses
stay in the **Histórico**, where they can be reopened or deleted.

> **The readings are a mock for now** (see below): the interface works end to
> end, but the verdicts and values are made up until the pipelines are plugged in.

## Running

```bash
pixi run -e web app          # http://localhost:8000
pixi run -e web test-web     # API tests (FastAPI TestClient)
```

The `web` environment adds FastAPI, uvicorn and python-multipart to the
default one, which stays light. History and photos are kept in `data/app/`
(git-ignored; `APP_DATA_DIR` changes it): `history.sqlite3` and
`photos/<id>/`. Delete the folder to start over.

The server listens on 127.0.0.1 only. To open it on a phone on the same
network, run `pixi run -e web uvicorn app.main:app --app-dir web --host 0.0.0.0`
and open `http://<computer's IP>:8000`. Note that browsers only offer the
camera button on phones; on desktop both buttons open the file picker.

## Layout

| File | Role |
| :--- | :--- |
| `app/main.py` | FastAPI app: the page and the JSON API |
| `app/pipeline.py` | **Adapter to the vision pipelines** (mock for now) |
| `app/storage.py` | History in SQLite, photos on disk |
| `app/images.py` | Upload checks (JPEG/PNG/WebP by content, 20 MB), smaller copies |
| `static/` | The page: plain HTML/CSS/JS, no build step |
| `tests/` | API and verdict tests |

API: `POST /api/analyses` (multipart: `front`, `back`, optional `name`),
`GET /api/analyses`, `GET|DELETE /api/analyses/<id>`,
`GET /api/analyses/<id>/export` (JSON download),
`GET /api/analyses/<id>/photos/<front|back>/<view|thumb>`.

## Plugging in the pipelines

The backend calls only `app.pipeline.analyze(front_path, back_path)`, which
returns:

```python
{
  "verdict": "compliant" | "inconclusive" | "non_compliant",
  "reasons": ["Deveria ter ALTO EM SÓDIO e não tem.", ...],
  "nutrition": {"basis": "100g" | "100ml" | None,
                "added_sugars_g": float | None,
                "saturated_fat_g": float | None,
                "sodium_mg": float | None},
  "required_warnings": {"added_sugars": bool | None, "saturated_fat": ..., "sodium": ...},
  "found_warnings":    {"added_sugars": bool | None, "saturated_fat": ..., "sodium": ...},
  "back_photo_type": "table" | "linear" | "old_format" | "no_table" | "uncertain",
  "versions": {"back": str, "front": str, "rules": str},
  "mock": bool,
}
```

`None` means "couldn't tell". The verdict and the reasons come from
`summarize()`, which is real code; only `read_photos()` is a mock. It makes
up the readings from a hash of the two photos (the same photos always give
the same result, and different ones cover the three verdicts); set
`APP_MOCK_VERDICT=compliant|inconclusive|non_compliant` to force one.

To plug the real pipelines in, replace `read_photos()` so that it returns a
`Readings` built from:

- **back**: `vision.back.pipeline.read_back(cv2.imread(back_path))` →
  `result.reading.facts` (basis, added_sugars_g, saturated_fat_g, sodium_mg)
  and `result.photo_type.value`;
- **rules**: `rules.anvisa.assess(facts, Exemption.NONE)` → a nutrient in
  `required` is True, in `unknown` None, otherwise False (all None when the
  back photo isn't readable);
- **front** (`src/vision/front/`, in progress on another branch): True/False
  per warning printed, None when it can't tell.

The docstring of `app/pipeline.py` has the details (exemptions, imports).
Set `mock=False` and real versions, so the page stops showing the "resultado
simulado" notice.

## Known limitations

- HEIC photos (iPhone's format) are rejected; iOS Safari usually converts
  them to JPEG when uploading through the page, other apps may not.
- No accounts: everyone who can open the page sees the same history.
- The analysis runs inside the request; a slow pipeline makes the page wait
  (with a "Analisando…" notice) rather than queuing the job.
