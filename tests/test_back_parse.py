import pytest

from rules.anvisa import Basis
from vision.back.parse import Word, group_rows, parse_number, parse_table


def row(y, *cells, h=20):
    """Words of one table row; each cell is (x, text)."""
    return [Word(text, x, y, 10 * len(text), h) for x, text in cells]


# Nescau Cereal nutrition table (vertical layout).
NESCAU = [
    *row(0, (300, "100"), (340, "g"), (420, "30"), (450, "g"), (520, "%VD*")),
    *row(
        40,
        (0, "Valor"),
        (60, "energético"),
        (180, "(kcal)"),
        (300, "388"),
        (420, "116"),
        (520, "6"),
    ),
    *row(80, (20, "Açúcares"), (110, "totais"), (180, "(g)"), (300, "30"), (420, "9")),
    *row(
        120,
        (30, "Açúcares"),
        (120, "adicionados"),
        (240, "(g)"),
        (300, "29"),
        (420, "8,6"),
        (520, "17"),
    ),
    *row(
        160,
        (0, "Gorduras"),
        (90, "totais"),
        (160, "(g)"),
        (300, "4,8"),
        (420, "1,4"),
        (520, "2"),
    ),
    *row(
        200,
        (20, "Gorduras"),
        (110, "monoinsaturadas"),
        (260, "(g)"),
        (300, "2,1"),
        (420, "0,6"),
    ),
    *row(
        240,
        (20, "Gorduras"),
        (110, "saturadas"),
        (210, "(g)"),
        (300, "1,5"),
        (420, "0,5"),
        (520, "8"),
    ),
    *row(280, (0, "Sódio"), (60, "(mg)"), (300, "367"), (420, "110"), (520, "5")),
]


def test_vertical_table():
    facts = parse_table(NESCAU).facts

    assert facts.basis is Basis.SOLID
    assert facts.added_sugars_g == 29
    assert facts.saturated_fat_g == 1.5  # not the monoinsaturadas row
    assert facts.sodium_mg == 367


def test_horizontal_table_takes_value_next_to_label():
    # Side-by-side layout: two label/value groups on the same row.
    words = [
        *row(0, (200, "100"), (230, "ml"), (600, "100"), (630, "ml")),
        *row(
            40,
            (0, "Açúcares"),
            (90, "adicionados"),
            (200, "7,9"),
            (300, "3"),
            (400, "Gorduras"),
            (490, "saturadas"),
            (600, "0,2"),
            (700, "0"),
        ),
        *row(
            80,
            (0, "Proteínas"),
            (200, "1,1"),
            (400, "Sódio"),
            (460, "(mg)"),
            (600, "35"),
        ),
    ]
    facts = parse_table(words).facts

    assert facts.basis is Basis.LIQUID
    assert facts.added_sugars_g == 7.9
    assert facts.saturated_fat_g == 0.2
    assert facts.sodium_mg == 35


def test_tolerates_ocr_errors():
    words = [
        *row(0, (0, "Acucares"), (90, "adlcionados"), (200, "(g)"), (260, "O")),
        *row(40, (0, "Gorduras"), (90, "saturadas(g)"), (260, "3,O")),
        *row(80, (0, "S6dio(mg)"), (260, "120mg")),
    ]
    facts = parse_table(words).facts

    assert facts.added_sugars_g == 0
    assert facts.saturated_fat_g == 3.0
    assert facts.sodium_mg == 120


def test_missing_rows_are_none():
    facts = parse_table(row(0, (0, "Proteínas"), (200, "6,2"))).facts

    assert facts.added_sugars_g is None
    assert facts.saturated_fat_g is None
    assert facts.sodium_mg is None
    assert facts.basis is None


def test_rows_group_despite_small_vertical_offsets():
    words = [
        Word("Sódio", 0, 100, 50, 20),
        Word("367", 300, 106, 30, 20),
        Word("Fibras", 0, 140, 50, 20),
    ]

    assert [[w.text for w in r] for r in group_rows(words)] == [
        ["Sódio", "367"],
        ["Fibras"],
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("8,7", 8.7),
        ("29", 29),
        ("0,04", 0.04),
        ("367mg", 367),
        ("1.5g", 1.5),
        ("O", 0),
        ("(g)", None),
        ("%VD", None),
        ("<0,5", 0.5),
        ("", None),
        ("“29", 29),
        ("367.", 367),
        ("S", None),
        # Thousands grouped with a dot (Brazilian tables).
        ("1.180", 1180),
        ("1.047mg", 1047),
        ("12.000", 12000),
        ("0.500", 0.5),
        ("1.5", 1.5),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_unit_misread_as_digit_is_not_the_value():
    words = row(0, (0, "Açúcares"), (90, "adicionados"), (200, "(9)"), (260, "8,7"))

    assert parse_table(words).facts.added_sugars_g == 8.7


def test_value_drifted_out_of_the_row_is_still_found():
    # Curved package: the number sits lower than its label.
    words = [
        Word("Sódio", 0, 100, 50, 20),
        Word("(mg)", 60, 100, 40, 20),
        Word("367", 300, 116, 30, 20),
    ]

    assert parse_table(words).facts.sodium_mg == 367


@pytest.mark.parametrize(
    "header, basis",
    [
        (["1009"], Basis.SOLID),
        (["100", "9"], Basis.SOLID),
        (["100", "mi"], Basis.LIQUID),
    ],
)
def test_misread_headers(header, basis):
    words = [Word(t, 300 + 60 * i, 0, 50, 20) for i, t in enumerate(header)]

    assert parse_table(words).facts.basis is basis


def test_value_aligned_with_100g_header_beats_nearer_portion_value():
    # The 100 g value was not read; the nearest number is the portion value.
    words = [
        *row(0, (300, "100"), (340, "g"), (420, "60"), (450, "g")),
        *row(40, (0, "Sódio"), (60, "(mg)"), (300, "579"), (420, "347")),
        *row(80, (0, "Açúcares"), (90, "adicionados"), (420, "0")),
    ]
    facts = parse_table(words).facts

    assert facts.sodium_mg == 579
    # No number under '100 g' on this row: falls back to the row value.
    assert facts.added_sugars_g == 0


@pytest.mark.parametrize(
    "label", [["Aç", "adicionados"], ["Aç.", "adic."], ["Açúcares", "adicionados"]]
)
def test_abbreviated_added_sugars(label):
    words = row(
        0, *[(90 * i, t) for i, t in enumerate(label)], (300, "(g)"), (360, "23")
    )

    assert parse_table(words).facts.added_sugars_g == 23


def test_linear_format():
    # Small packages: values as running text instead of a table.
    words = (
        row(
            0,
            (0, "Por"),
            (40, "100"),
            (80, "g"),
            (100, "(40"),
            (140, "g,"),
            (180, "%VD*):"),
        )
        + row(
            40,
            (0, "Açúcares"),
            (90, "adicionados"),
            (200, "23"),
            (230, "g"),
            (260, "(9,4"),
            (310, "g,"),
            (340, "18%)"),
            (400, "Gorduras"),
            (490, "saturadas"),
            (590, "4,7"),
            (620, "g"),
            (650, "(1,9"),
        )
        + row(80, (0, "Sódio"), (60, "113"), (100, "mg"), (140, "(45"), (180, "mg,"))
    )
    facts = parse_table(words).facts

    assert facts.basis is Basis.SOLID
    assert (facts.added_sugars_g, facts.saturated_fat_g, facts.sodium_mg) == (
        23,
        4.7,
        113,
    )


def test_labels_inside_phrases_without_spaces():
    words = [
        *row(0, (0, "Acucaresadicionados (g)"), (300, "29")),
        *row(40, (0, "Gordurasmonoinsaturadas (g)"), (300, "2,1")),
        *row(80, (0, "Gordurassaturadas (g)"), (300, "1,5")),
        *row(120, (0, "Sodio (mg)"), (300, "367")),
    ]
    facts = parse_table(words).facts

    assert (facts.added_sugars_g, facts.saturated_fat_g, facts.sodium_mg) == (
        29,
        1.5,
        367,
    )


def test_tall_boxes_do_not_take_the_value_of_the_row_above():
    # Learned OCR boxes are about as tall as the row spacing.
    def line(y, text, *values):
        return [Word(text, 0, y, 300, 70)] + [
            Word(v, x, y + 5, 40, 60) for x, v in zip((660, 860), values)
        ]

    words = [
        Word("100g", 650, 170, 60, 57),
        *line(620, "Gorduras totais (g)", "24", "8,3"),
        *line(690, "Gorduras saturadas (g)", "4,1", "1,4"),
    ]

    assert parse_table(words).facts.saturated_fat_g == 4.1


def sentence(y, text):
    return row(y, *[(60 * i, t) for i, t in enumerate(text.split())])


def test_not_significant_amounts_count_as_zero():
    words = [
        *row(0, (300, "100"), (340, "g")),
        *row(40, (0, "Sódio"), (60, "(mg)"), (300, "120")),
        *sentence(80, "Não contém quantidades significativas de açúcares"),
        *sentence(110, "adicionados, gorduras saturadas e sódio."),
    ]
    facts = parse_table(words).facts

    assert facts.added_sugars_g == 0
    assert facts.saturated_fat_g == 0
    # A value read from the table wins over the sentence.
    assert facts.sodium_mg == 120


def test_not_significant_without_spaces_and_general_terms():
    # Learned OCR drops spaces; 'açúcares' and 'gorduras totais' cover added
    # sugars and saturated fat.
    words = [
        Word(
            "Naocontemquantidadessignificativasdeacucares,gordurastotais.",
            0,
            0,
            600,
            30,
        )
    ]
    facts = parse_table(words).facts

    assert (facts.added_sugars_g, facts.saturated_fat_g, facts.sodium_mg) == (
        0,
        0,
        None,
    )


def test_not_significant_list_stops_at_the_next_sentence():
    words = [
        *sentence(0, "Não contém quantidades significativas de gorduras trans."),
        *sentence(30, "*Percentual de valores diários fornecidos pela porção. Sódio"),
    ]

    assert parse_table(words).facts.sodium_mg is None


def test_impossible_amounts_are_not_values():
    # %VD and merged digits: 477 g of fat per 100 g can't be.
    words = [
        *row(0, (0, "Gorduras"), (90, "saturadas"), (200, "(g)"), (260, "477")),
        *row(40, (0, "Sódio"), (200, "(mg)"), (260, "411415")),
    ]
    facts = parse_table(words).facts
    assert facts.saturated_fat_g is None
    assert facts.sodium_mg is None


def test_a_plausible_row_wins_over_an_impossible_one():
    words = [
        *row(0, (0, "Gorduras"), (90, "saturadas"), (200, "(g)"), (260, "477")),
        *row(40, (0, "Gorduras"), (90, "saturadas"), (200, "(g)"), (260, "4,7")),
    ]
    assert parse_table(words).facts.saturated_fat_g == 4.7
