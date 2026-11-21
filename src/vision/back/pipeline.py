"""Back of the package → nutrition facts per 100 g/ml.

photo ─► locate + straighten table ─► OCR ─► parse rows ─► NutritionFacts
"""

import re
from dataclasses import astuple, dataclass, field, replace
from pathlib import Path

import cv2

from rules.anvisa import Nutrient, NutritionFacts
from vision.back.classify import Features, PhotoType, classify, features
from vision.back.locate import Table, locate_table
from vision.back.orient import osd_rotation, rotate, vertical_text
from vision.back.ocr import (
    NATIVE,
    RAPIDOCR_MAX_SIDE,
    TARGET_TEXT_HEIGHT,
    prepare,
    read_row,
    rapidocr_lines,
    read_words,
    words_from_lines,
)
from vision.back.parse import (
    NOT_SIGNIFICANT,
    Reading,
    Word,
    is_unit,
    normalize,
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
# Level RapidOCR's tilted text lines before grouping rows, on images that
# weren't straightened (no table located, the whole photo):
# ocr.words_from_lines. Measured on dev and off: whole-photo values right
# 68% -> 82%, but decisions unchanged (all three right -1 product: 4 fixed,
# 5 broken, the same at 1 and 2 degrees). Leveling already-straightened
# crops too: -3 products (8 fixed, 11 broken).
LEVEL_TEXT = False
# A second RapidOCR model read on the crop of a readable table only when
# values are still unread (after the small model and Tesseract), filling only
# those; None to skip. On dev (OCR cache): medium alone instead of small
# gives 90.2% vs 90.6% all three right (15 products fixed, 18 broken: they
# fail on different photos), as this fallback 91.9% (+9, none broken), run
# on 19% of the photos. Before Tesseract instead of after: 92.0% (+1), but
# run on 29% of the photos, unreadable ones too.
FALLBACK_MODEL = "medium"
# The larger model reads only the rows still missing a value: a strip of
# the crop around each label (the full width, ROW_STRIP label heights above
# and below its center) instead of the whole crop, as its cost grows with
# the text it reads. A missing nutrient whose label no engine found still
# gets the whole crop.
FALLBACK_ROWS = True
ROW_STRIP = 1.0
# A number whose box touches a strip's top or bottom edge (within this many
# pixels) was cut by it: a fragment of the row above or below, not the
# label's row (dev: '1,6' g of saturated fat read as '8' from the top of
# the next row's digits). Dropped; None keeps it. Words with letters stay:
# a label on slanted text can be as tall as its strip.
ROW_EDGE = 1


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


# The fallbacks for what the first reading left unread (stages 3b and 4:
# the larger model, the whole photo) run on the orientation chosen, not on
# each orientation tried: the turn is chosen on the first readings
# (Tesseract + small RapidOCR), and only orientations that tie there get
# their fallbacks to decide. False finishes every candidate before
# choosing, as before: up to three medium-model calls on a turned photo.
FALLBACKS_AFTER_TURN = True


@dataclass
class _Context:
    """What the fallbacks need to finish a reading (_read, then _finish)."""

    image: object  # the photo, turned
    crop: object  # what the OCR read: the located table, or the photo
    table: Table | None
    straight: bool  # the crop was straightened
    prefix: str  # cache key prefix (rotation, classical steps)
    words: list  # the fast OCR's words
    learned: list  # the learned OCR's words, when it ran
    # (reading, scale) of each engine, learned first: label positions and
    # '100 g' headers, in crop coordinates times the scale (Tesseract reads
    # a rescaled image).
    readings: list = field(default_factory=list)


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

    entries = [(*_read(image, locate, preprocess, debug_dir, cache), 0)]
    if orient:
        # OSD only suggests a turn: it also asked for 180 degrees on upright
        # photos the parser read right (dev, confidence up to 4.6). Each turn
        # is kept only if the photo reads better turned; ties keep the photo
        # as is.
        suggested = _ocr(cache, "osd", image, lambda: osd_rotation(image))
        if suggested:
            candidates = [suggested]
        elif vertical_text(entries[0][0].lines):
            # On its side and OSD wasn't sure which way: the reading tells.
            candidates = [90, 270]
        else:
            candidates = []
        for degrees in candidates:
            turned = _read(
                rotate(image, degrees), locate, preprocess, None, cache, degrees
            )
            entries.append((*turned, degrees))

    def finish(entry):
        result, context, degrees = entry
        return _finish(result, context, cache), context, degrees

    if not FALLBACKS_AFTER_TURN:
        best = _best([finish(entry) for entry in entries])
    else:
        best = _best(entries)
        # Orientations that read equally well before the fallbacks: theirs
        # decide (dev: 3 turned photos read only with them).
        best = _best([finish(entry) for entry in best])
    result, context, rotation = best[0]
    if debug_dir is not None and context is not None:
        from vision.back.debug import save_parsed_values, save_summary

        save_parsed_values(Path(debug_dir), context.crop, result.reading)
        save_summary(debug_dir, result)
    return replace(result, rotation=rotation)


def _best(entries):
    """The (result, context, rotation) entries that read best, in order: the
    photo as is first when it ties."""
    top = max(reading_score(result) for result, _, _ in entries)
    return [entry for entry in entries if reading_score(entry[0]) == top]


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


def _rapidocr(cache, key, image, level=False, model=None):
    """RapidOCR's words (one per text line) on a color image; `level` turns
    tilted text level (ocr.words_from_lines)."""
    model = model or RAPIDOCR_MODEL
    key = f"{key}_{model}"
    return words_from_lines(
        _ocr(cache, key, image, lambda: rapidocr_lines(image, model)),
        level=level and LEVEL_TEXT,
    )


def _read(image, locate, preprocess, debug_dir, cache=None, rotation=0):
    """Stages 1 to 3: locate, the fast OCR, the learned OCR. Returns the
    reading and what _finish needs to complete it (None: nothing left to
    do). The photo type may still be UNCERTAIN."""
    # Cache keys: rotation and classical steps, then the OCR call.
    prefix = f"r{rotation}/{'' if locate else 'nolocate/'}"
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        from vision.back.debug import save_original

        save_original(debug_dir, image)

    table = locate_table(image, debug_dir=debug_dir) if locate else None
    crop = table.image if table is not None else image
    # A located table is already straightened; the whole photo isn't.
    straight = table is not None and table.box is not None

    # Stage 1, before OCR: only photos with almost no text are rejected.
    if table is not None and table.text_blobs < MIN_TEXT_BLOBS:
        empty = parse_table([])
        photo_features = features(table, [], empty)
        result = BackResult(empty, table, PhotoType.NO_TABLE, photo_features)
        if debug_dir is not None:
            from vision.back.debug import save_summary

            save_summary(debug_dir, result)
        return result, None

    # Stage 2, after the fast OCR: the format is in the text.
    if ENGINE == "rapidocr":
        words = _rapidocr(cache, prefix + "rapidocr", crop, not straight)
        reading = parse_table(words)
        if debug_dir is not None:
            from vision.back.debug import save_ocr_rapidocr

            save_ocr_rapidocr(debug_dir, crop, words)
    else:
        clean = prepare(crop) if preprocess else crop
        scale = clean.shape[1] / crop.shape[1]
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
    readings = [(reading, 1.0 if ENGINE == "rapidocr" else scale)]
    if ENGINE == "combined" and (needs_learned or not GATE_LEARNED_OCR):
        learned = _rapidocr(cache, prefix + "rapidocr", crop, not straight)
        learned_reading = parse_table(learned)
        readings.insert(0, (learned_reading, 1.0))
        reading = combine(learned_reading, reading)
        photo_features = features(table, words + learned, reading)
        photo_type = classify(photo_features)

        if debug_dir is not None:
            from vision.back.debug import save_ocr_rapidocr

            save_ocr_rapidocr(debug_dir, crop, learned)

    result = BackResult(reading, table, photo_type, photo_features, lines=learned)
    context = _Context(image, crop, table, straight, prefix, words, learned, readings)
    return result, context


def _finish(result, context, cache=None) -> BackResult:
    """Stages 3b and 4, the fallbacks for what the first reading left
    unread, then the final type of photos still without format evidence."""
    if context is None:
        return result
    reading = result.reading
    photo_type, photo_features = result.photo_type, result.features

    # Stage 3b: the larger model on the same crop, for what is still unread.
    if (
        ENGINE == "combined"
        and FALLBACK_MODEL
        and FALLBACK_MODEL != RAPIDOCR_MODEL
        and photo_type.readable
        and _missing(reading)
    ):
        strips = _missing_rows(reading, context) if FALLBACK_ROWS else None
        if strips is None:
            larger = parse_table(
                _rapidocr(
                    cache,
                    context.prefix + "rapidocr",
                    context.crop,
                    not context.straight,
                    FALLBACK_MODEL,
                )
            )
        else:
            larger = _read_rows(cache, context, strips)
        reading = combine(reading, larger, names=(None, f"rapidocr_{FALLBACK_MODEL}"))

    # Stage 4: the table's box lost what was missing; the whole photo may
    # have it. Only when a box was cut (otherwise RapidOCR already read the
    # whole photo) and the crop left something to find.
    table = context.table
    if (
        ENGINE == "combined"
        and FULL_PHOTO_FALLBACK
        and table is not None
        and table.box is not None
        and needs_full_photo(photo_type, reading)
    ):
        full = _rapidocr(cache, context.prefix + "rapidocr_full", context.image, True)
        reading = combine(reading, parse_table(full), names=(None, "rapidocr_full"))
        if not photo_type.readable:
            text = context.words + context.learned + full
            photo_features = features(table, text, reading)
            photo_type = classify(photo_features)

    if photo_type is PhotoType.UNCERTAIN:
        # Still no format evidence with all the text read.
        photo_type = UNCERTAIN_AS

    return replace(
        result, reading=reading, photo_type=photo_type, features=photo_features
    )


def _missing_rows(reading, context):
    """Vertical spans (y0, y1) of the crop around the label of each nutrient
    still unread, merged where they overlap; None when the label of one of
    them wasn't found by any engine (the whole crop is read instead)."""
    # A missing value may be declared in the 'não contém quantidades
    # significativas' sentence, which a strip doesn't hold whole (dev: 2
    # values lost): the whole crop.
    text = normalize(" ".join(w.text for w in context.words + context.learned))
    if NOT_SIGNIFICANT in text:
        return None
    height = context.crop.shape[0]
    spans = []
    for nutrient in Nutrient:
        if reading.facts.amount(nutrient) is not None:
            continue
        found = next(
            (
                (engine.value_spans[nutrient], scale)
                for engine, scale in context.readings
                if nutrient in engine.value_spans
            ),
            None,
        )
        if found is None:
            return None
        (_, cy, h), scale = found
        cy, h = cy / scale, h / scale
        y0 = max(0, int(cy - ROW_STRIP * h))
        spans.append((y0, min(height, int(cy + ROW_STRIP * h) + 1)))
    merged = []
    for y0, y1 in sorted(spans):
        if merged and y0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], y1))
        else:
            merged.append((y0, y1))
    return merged


def _read_rows(cache, context, strips):
    """The larger model's reading of the strips, in crop coordinates, with
    the first reading's '100 g' headers (the strips don't hold them)."""
    # The scale RapidOCR reads the whole crop at, so the text has the same
    # size; the NATIVE detector doesn't enlarge it again.
    scale = min(1.0, RAPIDOCR_MAX_SIDE / max(context.crop.shape[:2]))
    words = []
    for y0, y1 in strips:
        strip = context.crop[y0:y1]
        if scale < 1:
            strip = cv2.resize(
                strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        key = f"{context.prefix}rapidocr_rows{y0}-{y1}"
        model = FALLBACK_MODEL + NATIVE
        words += [
            Word(
                word.text,
                int(word.x / scale),
                int(word.y / scale) + y0,
                int(word.w / scale),
                int(word.h / scale),
                word.conf,
            )
            for word in _rapidocr(cache, key, strip, not context.straight, model)
            if ROW_EDGE is None or not _cut(word, strip.shape[0])
        ]
    headers = next(
        (
            [
                replace(h, cx=h.cx / scale, cy=h.cy / scale, w=int(h.w / scale))
                for h in engine.headers
            ]
            for engine, scale in context.readings
            if engine.headers
        ),
        [],
    )
    return parse_table(words, headers)


def _cut(word, height):
    """A number (no letters) whose box touches the top or bottom edge of a
    strip this tall."""
    if re.search(r"[^\W\d_]", word.text):
        return False
    return word.y <= ROW_EDGE or word.y + word.h >= height - ROW_EDGE


def _missing(reading):
    return any(reading.facts.amount(nutrient) is None for nutrient in Nutrient)


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
