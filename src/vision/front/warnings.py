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
"""

from dataclasses import dataclass, field

import cv2

from rules.anvisa import Nutrient
from vision.back.ocr import read_words_rapidocr
from vision.back.parse import Word, normalize, similarity

ANCHOR = "altoem"
# Imported products may carry the Spanish warning ("ALTO EN"), which isn't
# the RDC 429/2020 one; one letter apart, so rejected explicitly.
NOT_ANCHORS = ("altoen",)
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


@dataclass
class FrontResult:
    warnings: dict  # Nutrient -> bool
    anchors: list[Word] = field(default_factory=list)
    panel: list[Word] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)

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
) -> FrontResult:
    """The other parameters exist to measure each rule (evaluation/front.py)."""
    anchors = [w for w in words if is_anchor(normalize(w.text), not_anchors)]
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


def read_front(image, scope=SCOPE) -> FrontResult:
    if isinstance(image, str):
        path, image = image, cv2.imread(image)
        if image is None:
            raise FileNotFoundError(path)
    return find_warnings(read_words_rapidocr(image), scope)
