"""Front of the package → which "ALTO EM" warnings (RDC 429/2020) it shows.

The warning has a fixed design: a black panel with a magnifier, "ALTO EM"
and the nutrients, all in the same bold font. The learned OCR reads it as
separate lines ("ALTO EM", "AÇÚCAR", "ADICIONADO", ...), so the warning is
found in two steps:

    anchor  a line reading "ALTO EM"
    panel   lines near the anchor with about the same text height: the
            nutrients of that warning (ingredients and claims elsewhere on
            the package are in other sizes)

and each nutrient is the distinctive word of its phrase, matched with
tolerance for OCR errors.

Two methods, same result (read_front):

    "ocr"   RapidOCR on the whole photo, anchor + panel as above
    "yolo"  a detector (detector.py) boxes the panels; RapidOCR reads each
            crop, with a margin, at a size where the panel text is legible;
            the nutrients are matched in the crops' text
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from rules.anvisa import Nutrient
from vision.back.ocr import read_words_rapidocr
from vision.back.parse import Word, normalize, similarity

ANCHOR = "altoem"
# Imported products may carry the Spanish warning ("ALTO EN AZÚCARES /
# GRASAS SATURADAS / SODIO"), which isn't the RDC 429/2020 one. Rejecting
# "ALTO EN" as an anchor lost 2 real warnings on dev (the OCR reads "ALTO
# EM" as "ALTOEN"): a panel is Spanish by its Spanish words instead.
NOT_ANCHORS = ()
SPANISH = ("grasa", "azucares", "calorias")
# Words of each phrase: "açúcar adicionado", "gordura saturada", "sódio".
# Each word may be read on its own line, and inside the warning panel any
# one of them identifies the nutrient.
KEYS = {
    Nutrient.ADDED_SUGARS: ("adicionad", "acucar"),
    Nutrient.SATURATED_FAT: ("saturad", "gordura"),
    Nutrient.SODIUM: ("sodio",),
}
MIN_SIMILARITY = 0.8
# "panel": lines near the anchor (in anchor text heights) with about its
# text height; "anywhere": any line on a photo with an anchor.
SCOPE = "panel"
# Horizontal panels put up to three nutrients side by side; perspective
# makes the nutrients up to ~1.8x the anchor's height (dev).
PANEL_REACH = (14, 7)  # horizontal, vertical
PANEL_HEIGHT_RATIO = (0.5, 2.0)

# Crops of detected panels: margin around the box (fraction of its size,
# each side), for boxes cut a bit short, and the short side the crop is
# scaled up to, so small panels reach a legible text size.
CROP_MARGIN = 0.15
CROP_MIN_SIDE = 480
# "any": a nutrient read anywhere in a crop counts; "anchor": only in crops
# where "ALTO EM" is also read.
CROP_RULE = "any"
# Inside a detected panel the only words are "ALTO EM" and the three
# nutrients, so a looser match is safe there.
CROP_MIN_SIMILARITY = MIN_SIMILARITY


@dataclass
class FrontResult:
    warnings: dict  # Nutrient -> bool
    anchors: list[Word] = field(default_factory=list)
    panel: list[Word] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    boxes: list = field(default_factory=list)  # detector.Box, method "yolo"

    @property
    def has_warning(self):
        return any(self.warnings.values())


def contains(text, key, min_similarity=MIN_SIMILARITY):
    """Whether normalized `text` has a stretch similar to `key`."""
    if key in text:
        return True
    n = len(key)
    return any(
        similarity(text[i : i + n], key) >= min_similarity
        for i in range(max(len(text) - n + 1, 1))
    )


def is_anchor(text, not_anchors=NOT_ANCHORS):
    return contains(text, ANCHOR) and not any(k in text for k in not_anchors)


def is_spanish(words, spanish=SPANISH):
    return any(k in normalize(w.text) for w in words for k in spanish)


def near(word, anchor, reach=PANEL_REACH, height_ratio=PANEL_HEIGHT_RATIO):
    reach_x, reach_y = reach
    low, high = height_ratio
    return (
        abs(word.x + word.w / 2 - (anchor.x + anchor.w / 2)) <= reach_x * anchor.h
        and abs(word.cy - anchor.cy) <= reach_y * anchor.h
        and low * anchor.h <= word.h <= high * anchor.h
    )


def find_warnings(
    words,
    scope=SCOPE,
    keys=KEYS,
    reach=PANEL_REACH,
    height_ratio=PANEL_HEIGHT_RATIO,
    not_anchors=NOT_ANCHORS,
    spanish=SPANISH,
) -> FrontResult:
    """The other parameters exist to measure each rule (evaluation/front.py)."""
    anchors = [w for w in words if is_anchor(normalize(w.text), not_anchors)]
    anchors = [
        a
        for a in anchors
        if not is_spanish(
            [w for w in words if near(w, a, reach, height_ratio)], spanish
        )
    ]
    if scope == "panel":
        panel = [
            w for w in words if any(near(w, a, reach, height_ratio) for a in anchors)
        ]
    else:
        panel = list(words) if anchors else []

    texts = [normalize(w.text) for w in panel]
    warnings = {
        nutrient: any(contains(text, key) for text in texts for key in alternatives)
        for nutrient, alternatives in keys.items()
    }
    return FrontResult(warnings, anchors, panel, words)


def crop_panel(image, box, margin=CROP_MARGIN, min_side=CROP_MIN_SIDE):
    """Crop of the box with a margin, scaled so its short side is at least
    min_side. Returns the crop, its origin in the photo and the scale."""
    height, width = image.shape[:2]
    mx, my = margin * box.w, margin * box.h
    x0, y0 = max(int(box.x - mx), 0), max(int(box.y - my), 0)
    x1 = min(int(box.x + box.w + mx), width)
    y1 = min(int(box.y + box.h + my), height)
    crop = image[y0:y1, x0:x1]
    scale = max(min_side / max(min(crop.shape[:2]), 1), 1.0)
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, x0, y0, scale


def read_panel(image, box, margin=CROP_MARGIN, min_side=CROP_MIN_SIDE):
    """RapidOCR lines of one detected panel, in photo coordinates."""
    crop, x0, y0, scale = crop_panel(image, box, margin, min_side)
    return [
        Word(
            w.text,
            int(x0 + w.x / scale),
            int(y0 + w.y / scale),
            int(w.w / scale),
            int(w.h / scale),
            w.conf,
        )
        for w in read_words_rapidocr(crop)
    ]


def find_warnings_in_panels(
    panels,
    boxes=(),
    rule=CROP_RULE,
    keys=KEYS,
    spanish=SPANISH,
    min_similarity=CROP_MIN_SIMILARITY,
) -> FrontResult:
    """`panels`: the lines read in each detected panel. A Spanish panel
    isn't an RDC 429/2020 warning."""
    kept, anchors = [], []
    for words in panels:
        crop_anchors = [w for w in words if is_anchor(normalize(w.text))]
        if is_spanish(words, spanish) or (rule == "anchor" and not crop_anchors):
            continue
        kept += words
        anchors += crop_anchors

    texts = [normalize(w.text) for w in kept]
    warnings = {
        nutrient: any(
            contains(text, key, min_similarity)
            for text in texts
            for key in alternatives
        )
        for nutrient, alternatives in keys.items()
    }
    every = [w for words in panels for w in words]
    return FrontResult(warnings, anchors, kept, every, list(boxes))


def read_front(image, method="ocr", scope=SCOPE, model=None) -> FrontResult:
    """Which warnings the front photo shows. `image`: path or BGR array;
    `method`: "ocr" (whole photo) or "yolo" (detected panels)."""
    if isinstance(image, (str, Path)):
        path, image = image, cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError(path)
    if method == "ocr":
        return find_warnings(read_words_rapidocr(image), scope)
    if method == "yolo":
        from vision.front.detector import MODEL_PATH, detect_panels

        boxes = detect_panels(image, model or MODEL_PATH)
        panels = [read_panel(image, box) for box in boxes]
        return find_warnings_in_panels(panels, boxes)
    raise ValueError(f"unknown method {method!r}")
