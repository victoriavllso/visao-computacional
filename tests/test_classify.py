import pytest

from vision.back.classify import Features, PhotoType, classify


def make(**overrides):
    values = dict(
        ruled_rows=10,
        nutrition_words=7,
        has_100_header=False,
        has_added_sugars=False,
        has_per_100_text=False,
        has_explicit_linear=False,
        has_new_format_text=False,
        has_old_format_text=False,
    )
    values.update(overrides)
    return Features(**values)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(nutrition_words=1), PhotoType.NO_TABLE),
        # A header or added sugars row is enough, even with little text read.
        (dict(nutrition_words=2, has_100_header=True), PhotoType.TABLE),
        # A lone '100 g' is usually the net weight on the package.
        (dict(nutrition_words=0, has_100_header=True), PhotoType.NO_TABLE),
        (dict(has_100_header=True), PhotoType.TABLE),
        (dict(has_new_format_text=True), PhotoType.TABLE),
        (dict(has_added_sugars=True, ruled_rows=0), PhotoType.TABLE),
        # Linear only with explicit linear wording and no ruled lines.
        (
            dict(has_per_100_text=True, has_explicit_linear=True, ruled_rows=0),
            PhotoType.LINEAR,
        ),
        (dict(has_per_100_text=True, has_explicit_linear=True), PhotoType.TABLE),
        (dict(has_per_100_text=True, ruled_rows=0), PhotoType.TABLE),
        (dict(has_old_format_text=True), PhotoType.OLD_FORMAT),
        # New evidence wins: a new table can mention 'porção'.
        (dict(has_old_format_text=True, has_100_header=True), PhotoType.TABLE),
        # Transitional label: added sugars row, but per portion only. Old
        # format by the labeling rule (no 100 g/ml column); PR #15 reads it
        # as a table: open team decision.
        pytest.param(
            dict(has_old_format_text=True, has_added_sugars=True),
            PhotoType.OLD_FORMAT,
            marks=pytest.mark.xfail(reason="transitional labels read as tables"),
        ),
        # No evidence either way: not old just because OCR missed '100 g'.
        (dict(), PhotoType.UNCERTAIN),
    ],
)
def test_classify(overrides, expected):
    assert classify(make(**overrides)) is expected
