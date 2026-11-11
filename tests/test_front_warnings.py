import numpy as np
import pytest

from rules.anvisa import Nutrient
from vision.back.parse import Word
from vision.front.detector import Box, Detector, fold_of
from vision.front.warnings import (
    contains,
    crop_panel,
    find_warnings,
    find_warnings_in_panels,
)


def line(text, x, y, h=40):
    return Word(text, x, y, len(text) * h // 2, h)


def found(result):
    return {n for n, present in result.warnings.items() if present}


# A warning panel as the OCR reads it: one line per word.
PANEL = [
    line("ALTO EM", 100, 100),
    line("AÇÚCAR", 100, 150),
    line("ADICIONADO", 100, 195),
    line("SÓDIO", 300, 150),
]


def test_panel_words_on_separate_lines():
    assert found(find_warnings(PANEL)) == {Nutrient.ADDED_SUGARS, Nutrient.SODIUM}


def test_ocr_errors_are_tolerated():
    words = [
        line("ALT0 EM", 100, 100),
        line("GORDURA", 100, 150),
        line("SATURAOA", 100, 195),
    ]
    assert found(find_warnings(words)) == {Nutrient.SATURATED_FAT}


def test_no_anchor_no_warning():
    words = [line("Gordura saturada 0 g", 100, 100), line("Sódio 120 mg", 100, 150)]
    assert found(find_warnings(words)) == set()


def test_small_text_far_from_the_panel_is_ignored():
    # Ingredients: smaller text, elsewhere on the package.
    ingredients = line("sal (sódio), gordura saturada", 100, 900, h=15)
    result = find_warnings(PANEL + [ingredients])
    assert Nutrient.SATURATED_FAT not in found(result)
    # Scope "anywhere" takes it.
    assert Nutrient.SATURATED_FAT in found(
        find_warnings(PANEL + [ingredients], "anywhere")
    )


def test_contains():
    assert contains("acucaradicionado", "adicionad")
    assert contains("adicionaoo", "adicionad")
    assert not contains("adoçante", "adicionad")


# YOLO + OCR: the lines read in each detected panel.
def test_panel_crops():
    sugar = [line("ALTO EM", 0, 0), line("AÇÚCAR ADICIONADO", 0, 50)]
    fat = [line("GORDURA SATURADA", 0, 0)]  # "ALTO EM" not read
    result = find_warnings_in_panels([sugar, fat])
    assert found(result) == {Nutrient.ADDED_SUGARS, Nutrient.SATURATED_FAT}
    anchored = find_warnings_in_panels([sugar, fat], rule="anchor")
    assert found(anchored) == {Nutrient.ADDED_SUGARS}


def test_spanish_panel_is_not_a_warning():
    spanish = [line("ALTO EN", 0, 0), line("AZÚCARES", 0, 50), line("SODIO", 0, 100)]
    assert found(find_warnings_in_panels([spanish])) == set()


def test_crop_panel_margin_and_scale():
    image = np.zeros((1000, 2000, 3), np.uint8)
    crop, x0, y0, scale = crop_panel(image, Box(100, 200, 100, 50, 0.9), 0.2, 200)
    assert (x0, y0) == (80, 190)
    assert scale == 200 / 70
    assert crop.shape[0] == round(70 * scale)
    # At the border the margin is cut.
    _, x0, y0, _ = crop_panel(image, Box(0, 0, 100, 50, 0.9), 0.2, 0)
    assert (x0, y0) == (0, 0)


def test_detector_decodes_end_to_end_output():
    # Letterbox of a 2000 x 1000 photo into 1000 x 1000: scale 0.5, 250 px
    # padding on top.
    output = np.array([[100, 300, 200, 400, 0.9, 0], [0, 0, 10, 10, 0.01, 0]])
    boxes = Detector.decode(output, 0.5, 0, 250, (1000, 2000, 3), 0.1)
    assert len(boxes) == 1
    assert (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h) == (200, 100, 200, 200)


def test_detector_decodes_classic_output_with_nms():
    # (4 + 1 class, anchors): two overlapping boxes, one kept.
    output = np.array([[150, 152], [350, 352], [100, 100], [100, 100], [0.9, 0.8]])
    boxes = Detector.decode(output, 1.0, 0, 0, (1000, 1000, 3), 0.1)
    assert [(b.x, b.y, b.score) for b in boxes] == [(100, 300, pytest.approx(0.9))]


def test_fold_is_stable():
    assert fold_of("7891000502204") == fold_of("7891000502204")
    assert {fold_of(str(code)) for code in range(100)} == set(range(5))
