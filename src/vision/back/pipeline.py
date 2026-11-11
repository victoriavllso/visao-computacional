"""Back of the package → nutrition facts per 100 g/ml.

photo ─► locate + straighten table ─► OCR ─► parse rows ─► NutritionFacts
"""

from dataclasses import dataclass, replace

import cv2

from rules.anvisa import Nutrient, NutritionFacts
from vision.back.classify import Features, PhotoType, classify, features
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
    plausible,
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
# Photos with fewer character-shaped blobs are rejected before any OCR.
# Conservative: barcodes and icons also look like characters.
MIN_TEXT_BLOBS = 40
# Skip the learned OCR when the Tesseract pass already says the photo can't
# be checked (old format, no table).
GATE_LEARNED_OCR = True
# Photos with nutrition text but no format evidence even after the learned
# OCR: TABLE reads them (risk: checking an old per-portion label), OLD_FORMAT
# abstains (risk: no answer for a readable label). On 255 dev hand labels the
# 10 uncertain photos were 6 old format, 3 tables, 1 no table, and none of
# the 3 tables was read right: abstain.
UNCERTAIN_AS = PhotoType.OLD_FORMAT


@dataclass
class BackResult:
    reading: Reading
    table: Table | None
    photo_type: PhotoType
    features: Features

    @property
    def readable(self):
        """Only readings of new-format tables or linear text can be checked
        against RDC 429/2020."""
        return self.photo_type.readable


def read_back(image, locate=True, preprocess=True, debug_dir=None) -> BackResult:
    """`locate` and `preprocess` switch the classical steps off, to measure
    what each one adds.  `debug_dir`, when set, saves intermediate images and
    a JSON summary to that directory."""
    if isinstance(image, str):
        path, image = image, cv2.imread(image)
        if image is None:
            raise FileNotFoundError(path)

    if debug_dir is not None:
        from pathlib import Path

        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        from vision.back.debug import save_original

        save_original(debug_dir, image)

    table = locate_table(image, debug_dir=debug_dir) if locate else None
    crop = table.image if table is not None else image

    # Stage 1, before OCR: only photos with almost no text are rejected.
    if table is not None and table.text_blobs < MIN_TEXT_BLOBS:
        empty = parse_table([])
        photo_features = features(table, [], empty)
        result = BackResult(empty, table, PhotoType.NO_TABLE, photo_features)
        if debug_dir is not None:
            from vision.back.debug import save_summary

            save_summary(debug_dir, result)
        return result

    # Stage 2, after the fast OCR: the format is in the text.
    if ENGINE == "rapidocr":
        words = read_words_rapidocr(crop)
        reading = parse_table(words)
        if debug_dir is not None:
            from vision.back.debug import save_ocr_rapidocr

            save_ocr_rapidocr(debug_dir, crop, words)
    else:
        clean = prepare(crop) if preprocess else crop
        words = read_words(clean, preprocess=False)
        reading = parse_table(words)
        if REFINE and preprocess:
            reading = refine(clean, reading)

        if debug_dir is not None:
            from vision.back.debug import save_binarized, save_ocr_tesseract

            save_binarized(debug_dir, clean)
            save_ocr_tesseract(debug_dir, clean, words)

    photo_features = features(table, words, reading)
    photo_type = classify(photo_features)

    # Stage 3: the slow learned OCR, skipped only on positive evidence of an
    # old label. The fast OCR reading little text isn't evidence of no table
    # (on dev, 20 of 116 tables were rejected that way).
    needs_learned = photo_type is not PhotoType.OLD_FORMAT
    if ENGINE == "combined" and (needs_learned or not GATE_LEARNED_OCR):
        learned = read_words_rapidocr(crop)
        reading = combine(parse_table(learned), reading)
        photo_features = features(table, words + learned, reading)
        photo_type = classify(photo_features)

        if debug_dir is not None:
            from vision.back.debug import save_ocr_rapidocr

            save_ocr_rapidocr(debug_dir, crop, learned)

    if photo_type is PhotoType.UNCERTAIN:
        # Still no format evidence with all the text read.
        photo_type = UNCERTAIN_AS

    result = BackResult(reading, table, photo_type, photo_features)

    if debug_dir is not None:
        from vision.back.debug import save_parsed_values, save_summary

        save_parsed_values(debug_dir, crop, reading)
        save_summary(debug_dir, result)

    return result


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
        value = value if value is not None else parse_number(numbers[0].text)
        if value is not None and plausible(nutrient, value):
            values[nutrient] = value

    if not values:
        return reading

    fields = {
        Nutrient.ADDED_SUGARS: "added_sugars_g",
        Nutrient.SATURATED_FAT: "saturated_fat_g",
        Nutrient.SODIUM: "sodium_mg",
    }
    facts = replace(reading.facts, **{fields[n]: v for n, v in values.items()})
    return replace(reading, facts=facts)
