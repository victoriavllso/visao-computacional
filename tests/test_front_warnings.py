from rules.anvisa import Nutrient
from vision.back.parse import Word
from vision.front.warnings import contains, find_warnings


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
