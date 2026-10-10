"""Back of the package → nutrition facts per 100 g/ml.

photo ─► locate + straighten table ─► OCR ─► parse rows ─► NutritionFacts
"""

from dataclasses import dataclass

import cv2

from vision.back.locate import Table, locate_table
from dataclasses import replace

from rules.anvisa import Nutrient
from vision.back.ocr import TARGET_TEXT_HEIGHT, prepare, read_row, read_words
from vision.back.parse import (
    Reading,
    Word,
    is_unit,
    parse_number,
    parse_table,
    value_in_column,
)

# Re-read the value region of each found label with digits-only OCR:
# "fill" only missing values, "override" every value, None to skip.
REFINE = "fill"


@dataclass
class BackResult:
    reading: Reading
    table: Table | None


def read_back(image, locate=True, preprocess=True) -> BackResult:
    """`locate` and `preprocess` switch the classical steps off, to measure
    what each one adds."""
    if isinstance(image, str):
        path, image = image, cv2.imread(image)
        if image is None:
            raise FileNotFoundError(path)

    table = locate_table(image) if locate else None
    crop = table.image if table is not None else image

    clean = prepare(crop) if preprocess else crop
    words = read_words(clean, preprocess=False)
    reading = parse_table(words)
    if REFINE and preprocess:
        reading = refine(clean, reading)
    return BackResult(reading=reading, table=table)


def refine(clean, reading):
    """Digits-only re-reading of the row right of each label, in the
    preprocessed image. Local to the label, so curved tables don't matter,
    and the strip holds only that row's numbers."""
    values = {}
    # Text is normalized to TARGET_TEXT_HEIGHT; OCR word boxes are less
    # reliable (merged words come out several rows tall).
    half = 0.8 * TARGET_TEXT_HEIGHT
    for nutrient, (x, cy, _) in reading.value_spans.items():
        if REFINE == "fill" and reading.facts.amount(nutrient) is not None:
            continue

        words = read_row(clean, x, cy - half, clean.shape[1], cy + half)
        numbers = [
            w for w in words if not is_unit(w) and parse_number(w.text) is not None
        ]
        if not numbers:
            continue

        anchor = Word("", int(x) - 1, int(cy - half), 1, int(2 * half))
        value = value_in_column(numbers, anchor, reading.headers)
        values[nutrient] = value if value is not None else parse_number(numbers[0].text)

    if not values:
        return reading

    fields = {
        Nutrient.ADDED_SUGARS: "added_sugars_g",
        Nutrient.SATURATED_FAT: "saturated_fat_g",
        Nutrient.SODIUM: "sodium_mg",
    }
    facts = replace(reading.facts, **{fields[n]: v for n, v in values.items()})
    return replace(reading, facts=facts)
