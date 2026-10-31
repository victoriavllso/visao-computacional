import json
from types import SimpleNamespace

from labeling.labels import compact_labels, load_labels, truth_facts
from rules.anvisa import Basis


def product(code="789", sugars=10.0, fat=2.0, sodium_g=0.3, quantity="500 g"):
    return SimpleNamespace(
        code=code,
        quantity=quantity,
        off_added_sugars_100g=sugars,
        off_saturated_fat_100g=fat,
        off_sodium_100g=sodium_g,
    )


def test_without_hand_label_uses_off():
    facts, source = truth_facts(product(), {})

    assert source == "off"
    assert (facts.added_sugars_g, facts.sodium_mg, facts.basis) == (
        10.0,
        300.0,
        Basis.SOLID,
    )


def test_hand_label_wins_by_barcode():
    labels = {
        "789": {
            "status": "ok",
            "basis": "100ml",
            "added_sugars": 7.9,
            "saturated_fat": 0.0,
            "sodium": 35,
        }
    }
    facts, source = truth_facts(product(), labels)

    assert source == "hand"
    assert (facts.basis, facts.added_sugars_g, facts.sodium_mg) == (
        Basis.LIQUID,
        7.9,
        35,
    )


def test_empty_hand_field_is_unknown_not_off():
    labels = {
        "789": {
            "status": "partial",
            "added_sugars": None,
            "saturated_fat": 1.0,
            "sodium": 20,
        }
    }
    facts, _ = truth_facts(product(), labels)

    assert facts.added_sugars_g is None
    # No basis marked by hand: falls back to the package quantity.
    assert facts.basis is Basis.SOLID


def test_no_table_and_old_format_have_no_truth():
    assert truth_facts(product(), {"789": {"status": "no_table"}}) == (None, "no_table")
    # Old format: per portion only, values not comparable with per 100 g.
    old = {"789": {"status": "old_format", "sodium": 75}}
    assert truth_facts(product(), old) == (None, "old_format")


def test_most_recent_label_wins_across_people(tmp_path):
    def write(name, entries):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries))

    write(
        "ana",
        [{"code": "789", "status": "ok", "sodium": 1, "saved_at": "2026-10-10T10:00"}],
    )
    write(
        "bruno",
        [
            {
                "code": "789",
                "status": "ok",
                "sodium": 3,
                "saved_at": "2026-10-11T09:00",
            },
            {
                "code": "123",
                "status": "ok",
                "sodium": 2,
                "saved_at": "2026-10-09T09:00",
            },
        ],
    )

    labels = load_labels(tmp_path)

    assert labels["789"]["sodium"] == 3 and labels["789"]["labeler"] == "bruno"
    assert len(labels) == 2


def test_broken_lines_are_skipped(tmp_path):
    (tmp_path / "hand-labeled.jsonl").write_text(
        '{"code": "789", "status": "ok", "sodium": 1}\n'
        '{"code": "123", "sta\n'  # write cut short
        "<<<<<<< HEAD\n"  # merge conflict marker
        '{"status": "ok"}\n'  # no barcode
        '{"code": 456, "status": "ok", "sodium": 2}\n'
    )

    labels = load_labels(tmp_path)

    assert sorted(labels) == ["456", "789"]


def test_compact_keeps_latest_entry_per_barcode(tmp_path):
    (tmp_path / "hand-labeled.jsonl").write_text(
        '{"code": "789", "sodium": 1, "saved_at": "2026-10-10T10:00"}\n'
        '{"code": "123", "sodium": 5, "saved_at": "2026-10-10T11:00"}\n'
    )
    (tmp_path / "bruno.jsonl").write_text(
        '{"code": "789", "sodium": 3, "saved_at": "2026-10-11T09:00"}\n'
    )

    assert compact_labels(tmp_path) == (3, 2)

    assert [path.name for path in tmp_path.iterdir()] == ["hand-labeled.jsonl"]
    labels = load_labels(tmp_path)
    assert labels["789"]["sodium"] == 3 and labels["789"]["labeler"] == "bruno"
    assert labels["123"]["labeler"] == "hand-labeled"
