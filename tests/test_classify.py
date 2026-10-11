import pytest

from vision.back.classify import Features, PhotoType, classify


def make(**overrides):
    values = dict(
        ruled_rows=10,
        nutrition_words=7,
        has_100_header=False,
        has_added_sugars=False,
        has_per_100_text=False,
        has_new_format_text=False,
        has_old_format_text=False,
    )
    values.update(overrides)
    return Features(**values)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(nutrition_words=1), PhotoType.NO_TABLE),
        (dict(has_100_header=True), PhotoType.TABLE),
        (dict(has_new_format_text=True), PhotoType.TABLE),
        (dict(has_added_sugars=True, ruled_rows=0), PhotoType.TABLE),
        (dict(has_per_100_text=True, ruled_rows=0), PhotoType.LINEAR),
        (dict(has_old_format_text=True), PhotoType.OLD_FORMAT),
        # New evidence wins: a new table can mention 'porção'.
        (dict(has_old_format_text=True, has_100_header=True), PhotoType.TABLE),
        # Transitional label: added sugars row, but per portion only.
        (dict(has_old_format_text=True, has_added_sugars=True), PhotoType.OLD_FORMAT),
        # No evidence either way: not old just because OCR missed '100 g'.
        (dict(), PhotoType.UNCERTAIN),
    ],
)
def test_classify(overrides, expected):
    assert classify(make(**overrides)) is expected
