"""Back of the package → nutrition facts per 100 g/ml.

photo ─► locate + straighten table ─► OCR ─► parse rows ─► NutritionFacts
"""

from dataclasses import dataclass, replace

import cv2

from rules.anvisa import Nutrient, NutritionFacts
from vision.back.locate import Table, locate_table
from vision.back.ocr import (
    TARGET_TEXT_HEIGHT,
    prepare,
    read_row,
    read_words,
    read_words_rapidocr,
)
from vision.back.parse import (
    Reading,
    Word,
    is_unit,
    parse_number,
    parse_table,
    value_in_column,
)

# Re-read the value region of each found label as a single text line:
# "fill" only missing values, "override" every value, None to skip.
REFINE = "fill"
# "tesseract": classical preprocessing + Tesseract; "rapidocr": learned OCR
# on the located and straightened color crop; "combined": RapidOCR's value
# for each nutrient, Tesseract's where RapidOCR has none. RapidOCR is more
# accurate when it reads a value, but its text detector misses isolated
# small cells such as a lone '0'. On 300 dev products, all three decisions
# right: Tesseract 41%, RapidOCR 23%, combined 51%.
ENGINE = "combined"


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

    if ENGINE == "rapidocr":
        return BackResult(reading=parse_table(read_words_rapidocr(crop)), table=table)

    clean = prepare(crop) if preprocess else crop
    words = read_words(clean, preprocess=False)
    reading = parse_table(words)
    if REFINE and preprocess:
        reading = refine(clean, reading)

    if ENGINE == "combined":
        reading = combine(parse_table(read_words_rapidocr(crop)), reading)
    return BackResult(reading=reading, table=table)


def combine(primary, fallback):
    """Each value from `primary`, from `fallback` where `primary` has none."""
    facts = NutritionFacts(
        basis=primary.facts.basis or fallback.facts.basis,
        **{
            field: (
                getattr(primary.facts, field)
                if getattr(primary.facts, field) is not None
                else getattr(fallback.facts, field)
            )
            for field in ("added_sugars_g", "saturated_fat_g", "sodium_mg")
        },
    )
    return replace(
        fallback,
        facts=facts,
        labels=primary.labels | fallback.labels,
        matches={**fallback.matches, **primary.matches},
        headers=primary.headers or fallback.headers,
    )


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
