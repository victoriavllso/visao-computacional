"""Back of the package → nutrition facts per 100 g/ml.

photo ─► locate + straighten table ─► OCR ─► parse rows ─► NutritionFacts
"""

from dataclasses import astuple, dataclass, field, replace

import cv2

from rules.anvisa import Nutrient, NutritionFacts
from vision.back.classify import Features, PhotoType, classify, features
from vision.back.locate import Table, locate_table
from vision.back.orient import osd_rotation, rotate, vertical_text
from vision.back.ocr import (
    TARGET_TEXT_HEIGHT,
    prepare,
    read_row,
    rapidocr_lines,
    read_words,
    words_from_lines,
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
# "fill" only missing values, "override" every value, None to skip. It helped
# Tesseract alone (+3 pts, README); next to RapidOCR its values were right
# 14% of the time on dev (14 values), so off, like the parser's nearby-row
# rules (parse.NEARBY_ROWS).
REFINE = None
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
# Turn photos taken on their side or upside down before reading them
# (vision/back/orient.py).
ORIENT = True
# When the located table leaves a value unread, or looks empty, RapidOCR
# reads the whole photo and fills only the missing values. The locator's box
# is the ruled lines plus one text height: tables whose lines are faint or
# broken under the label column lose the labels ('Sódio' is short and at the
# far left), and bent bottom rows fall outside. Tesseract keeps the crop (it
# needs the scaling and binarization); RapidOCR reads whole photos well.
FULL_PHOTO_FALLBACK = True
# RapidOCR's PP-OCRv6 model size: "small" or "medium" (vision/back/ocr.py).
RAPIDOCR_MODEL = "small"


@dataclass
class BackResult:
    reading: Reading
    table: Table | None
    photo_type: PhotoType
    features: Features
    # Clockwise degrees the photo was turned before reading it.
    rotation: int = 0
    # Lines read by the learned OCR, when it ran (orientation cue).
    lines: list = field(default_factory=list, repr=False)

    @property
    def readable(self):
        """Only readings of new-format tables or linear text can be checked
        against RDC 429/2020."""
        return self.photo_type.readable


def read_back(
    image, locate=True, preprocess=True, debug_dir=None, orient=ORIENT, cache=None
) -> BackResult:
    """`locate` and `preprocess` switch the classical steps off, to measure
    what each one adds.  `debug_dir`, when set, saves intermediate images and
    a JSON summary to that directory. `cache`, an OcrCache of this photo
    (vision/back/cache.py), returns stored OCR results instead of running
    the OCR again."""
    if isinstance(image, str):
        path, image = image, cv2.imread(image)
        if image is None:
            raise FileNotFoundError(path)

    result, rotation = _read(image, locate, preprocess, debug_dir, cache), 0
    if not orient:
        return result
    # OSD only suggests a turn: it also asked for 180 degrees on upright
    # photos the parser read right (dev, confidence up to 4.6). Each turn is
    # kept only if the photo reads better turned; ties keep the photo as is.
    suggested = _ocr(cache, "osd", image, lambda: osd_rotation(image))
    if suggested:
        candidates = [suggested]
    elif vertical_text(result.lines):
        # On its side and OSD wasn't sure which way: the reading tells.
        candidates = [90, 270]
    else:
        candidates = []
    for degrees in candidates:
        candidate = _read(
            rotate(image, degrees), locate, preprocess, None, cache, degrees
        )
        if reading_score(candidate) > reading_score(result):
            result, rotation = candidate, degrees
    return replace(result, rotation=rotation)


def reading_score(result):
    """How well a photo was read: the right way up gives a readable type, the
    '100 g' header and values; text turned the wrong way gives fewer."""
    facts = result.reading.facts
    values = sum(
        value is not None
        for value in (facts.added_sugars_g, facts.saturated_fat_g, facts.sodium_mg)
    )
    return (result.readable, values + bool(result.reading.headers))


def _ocr(cache, key, image, compute):
    """compute(), or its stored result when a cache is given."""
    return compute() if cache is None else cache.get(key, image, compute)


def _tesseract(cache, key, image):
    """Tesseract's words on an image already preprocessed."""
    if cache is None:
        return read_words(image, preprocess=False)
    rows = cache.get(
        key, image, lambda: [astuple(w) for w in read_words(image, preprocess=False)]
    )
    return [Word(*row) for row in rows]


def _rapidocr(cache, key, image):
    """RapidOCR's words (one per text line) on a color image."""
    key = f"{key}_{RAPIDOCR_MODEL}"
    return words_from_lines(
        _ocr(cache, key, image, lambda: rapidocr_lines(image, RAPIDOCR_MODEL))
    )


def _read(image, locate, preprocess, debug_dir, cache=None, rotation=0) -> BackResult:
    # Cache keys: rotation and classical steps, then the OCR call.
    prefix = f"r{rotation}/{'' if locate else 'nolocate/'}"
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
        words = _rapidocr(cache, prefix + "rapidocr", crop)
        reading = parse_table(words)
        if debug_dir is not None:
            from vision.back.debug import save_ocr_rapidocr

            save_ocr_rapidocr(debug_dir, crop, words)
    else:
        clean = prepare(crop) if preprocess else crop
        words = _tesseract(
            cache, prefix + ("tesseract" if preprocess else "tesseract_raw"), clean
        )
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
    learned = words if ENGINE == "rapidocr" else []
    if ENGINE == "combined" and (needs_learned or not GATE_LEARNED_OCR):
        learned = _rapidocr(cache, prefix + "rapidocr", crop)
        reading = combine(parse_table(learned), reading)
        photo_features = features(table, words + learned, reading)
        photo_type = classify(photo_features)

        if debug_dir is not None:
            from vision.back.debug import save_ocr_rapidocr

            save_ocr_rapidocr(debug_dir, crop, learned)

    # Stage 4: the table's box lost what was missing; the whole photo may
    # have it. Only when a box was cut (otherwise RapidOCR already read the
    # whole photo) and the crop left something to find.
    if (
        ENGINE == "combined"
        and FULL_PHOTO_FALLBACK
        and table is not None
        and table.box is not None
        and needs_full_photo(photo_type, reading)
    ):
        full = _rapidocr(cache, prefix + "rapidocr_full", image)
        reading = combine(reading, parse_table(full), names=(None, "rapidocr_full"))
        if not photo_type.readable:
            photo_features = features(table, words + learned + full, reading)
            photo_type = classify(photo_features)

    if photo_type is PhotoType.UNCERTAIN:
        # Still no format evidence with all the text read.
        photo_type = UNCERTAIN_AS

    result = BackResult(reading, table, photo_type, photo_features, lines=learned)

    if debug_dir is not None:
        from vision.back.debug import save_parsed_values, save_summary

        save_parsed_values(debug_dir, crop, reading)
        save_summary(debug_dir, result)

    return result


def needs_full_photo(photo_type, reading):
    """The crop looked empty or format-less, or a readable table has a
    value left unread."""
    if photo_type in (PhotoType.NO_TABLE, PhotoType.UNCERTAIN):
        return True
    return photo_type.readable and any(
        reading.facts.amount(nutrient) is None for nutrient in Nutrient
    )


def combine(primary, fallback, names=("rapidocr", "tesseract")):
    """Each value from `primary`, from `fallback` where `primary` has none.
    A name of None keeps that reading's sources as they are (already
    prefixed by an earlier combine)."""
    sources = {}
    for nutrient in Nutrient:
        for name, reading in zip(names, (primary, fallback)):
            if reading.facts.amount(nutrient) is not None:
                source = reading.sources.get(nutrient, "?")
                sources[nutrient] = f"{name}:{source}" if name else source
                break
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
        sources=sources,
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
    sources = {**reading.sources, **{n: "refine" for n in values}}
    return replace(reading, facts=facts, sources=sources)
