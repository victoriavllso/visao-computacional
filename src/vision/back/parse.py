"""Turn OCR words from a nutrition table into nutrition facts.

Pure logic on word positions, independent of the OCR engine. Relies on the
RDC 429/2020 table layout: one row per nutrient, label on the left and the
per-100 g/ml column first among the values.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from rules.anvisa import Basis, Nutrient, NutritionFacts


@dataclass(frozen=True)
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float = 100.0

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def right(self):
        return self.x + self.w


@dataclass
class Reading:
    facts: NutritionFacts
    # Row text and value read per nutrient, for debugging.
    matches: dict = field(default_factory=dict)
    # Nutrients whose label was found, with or without a value.
    labels: set = field(default_factory=set)
    # Where each found label's values start: (x, row center y, text height).
    value_spans: dict = field(default_factory=dict)
    headers: list = field(default_factory=list)
    # Which rule gave each value: "column" ('100 g' column on its row),
    # "first_right", "column_rows" (column, nearby rows), "near",
    # "not_significant", "refine"; after combining engines, prefixed with
    # the engine ("rapidocr:column").
    sources: dict = field(default_factory=dict)


# Distinctive words of each label (full and abbreviated: 'Aç. adic.',
# 'Gord. sat.'), and words that look similar but belong to other rows.
LABELS = {
    Nutrient.ADDED_SUGARS: (("adicionados", "adic"), ()),
    # 'Saturated': English tables printed next to the Brazilian one are per
    # serving (dev: 'Saturated fat' matched at 0.78, its row's number taken).
    Nutrient.SATURATED_FAT: (
        ("saturadas",),
        ("insaturadas", "monoinsaturadas", "saturated"),
    ),
    # 'Sodium': imported products with an English table next to the Portuguese.
    Nutrient.SODIUM: (("sodio", "sodium"), ()),
}
MIN_SIMILARITY = 0.75
# A unit in parentheses at the end of a label word: 'Sódio(mg)', '(g)' read
# as '(9)'. Only these are cut: RapidOCR reads the accent of 'Só' as a
# parenthesis ('S(odio', '(odi (mg)'), and cutting from any '(' left 's'
# (dev: 17 sodium rows lost, their values on the same row).
UNIT_SUFFIX = re.compile(
    r"\(\s*(?:m?g|ml|mi|kcal|kj|ng|mq|µg|mcg|9|6|o|a)?\s*\)?\s*$", re.IGNORECASE
)

# Letters OCR confuses with digits inside numbers.
DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"})
# Punctuation OCR attaches to numbers: quotes, bars, stars from dirt or lines.
NUMBER_JUNK = "()[]{}|:;*.,'\"“”‘’`´~^_-–—!¡"
NUMBER = re.compile(r"^[<≤]?(\d+(?:[.,]\d+)?)(mg|g|ml)?$")
# Brazilian tables write decimals with a comma and may group thousands with
# a dot: "1.180 mg" is 1180 (dev: two sodium values read as 1.18 and 1.047).
THOUSANDS = re.compile(r"^[1-9]\d{0,2}\.\d{3}$")

# Physical limits per 100 g/ml: no nutrient weighs more than the food, and
# pure salt has ~39,000 mg of sodium per 100 g. A larger reading is an OCR
# error (digits merged, the %VD column), not a value: on dev, 6 of the 21
# products wrongly called non-compliant came from readings like 477 g of
# saturated fat or 411,415 mg of sodium.
MAX_PER_100 = {
    Nutrient.ADDED_SUGARS: 100.0,
    Nutrient.SATURATED_FAT: 100.0,
    Nutrient.SODIUM: 40000.0,
}


# Rules beyond the label's own row: the '100 g' column on nearby rows and the
# nearest number anywhere. On dev their values were right 12-36% of the time
# (vs 90-95% for the label's own row) and gave 6 of the 13 products wrongly
# called non-compliant. Off: an unread value is unknown (inconclusive) rather
# than a guess; false accusations 13 -> 7, violations caught unchanged.
NEARBY_ROWS = False


def plausible(nutrient, value):
    return 0 <= value <= MAX_PER_100[nutrient]


def normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text)


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def parse_number(text):
    """'8,7' → 8.7, '367mg' → 367, 'O' → 0; None if not a number."""
    text = text.replace(" ", "").strip(NUMBER_JUNK)
    if not text:
        return None

    match = NUMBER.match(text) or NUMBER.match(text.translate(DIGIT_FIXES))
    if not match:
        return None
    number = match.group(1)
    if THOUSANDS.match(number):
        number = number.replace(".", "")
    return float(number.replace(",", "."))


# Tokens of an OCR text line: a parenthesized group, a run of letters, a
# number with what is glued after it, or other punctuation. Linear text is
# often read without spaces: 'saturadas4,4g(0,5g,3%)·Sódio 99mg' →
# 'saturadas', '4,4g', '(0,5g,3%)', 'Sódio', '99mg'.
TOKEN = re.compile(
    r"\([^\s()\[\]·•]*\)?"  # '(0,5g,3%)', '(mg)', '(odio' (accent read as '(')
    r"|\[[^\s()\[\]·•]*\]?"
    r"|\d[\d.,]*(?:mg|ml|kcal|kj|g|%)?"  # a number and its unit, not the next word
    r"|[^\W\d_]+"  # letters
    r"|[^\s\w(\[·•]+",  # other punctuation
    re.IGNORECASE,
)


def split_word(word):
    """The tokens of a word holding a whole text line, as Words.

    RapidOCR returns a text line as one box. In linear text the label and its
    value are then one word ('Sódio 155 mg (74 mg, 4%)'), so the value, which
    the parser looks for in the words after the label, was never read (dev:
    ~45 values); and neighboring cells merge ('120 6', value and %VD, parsed
    as 1206). Each token gets the share of the box width of its characters
    (RapidOCR gives no per-character positions). Words without spaces (all of
    Tesseract's) are kept as they are."""
    if not re.search(r"\s", word.text.strip()):
        return [word]
    per_char = word.w / max(len(word.text), 1)
    return [
        Word(
            token.group(),
            int(word.x + token.start() * per_char),
            word.y,
            max(1, int(len(token.group()) * per_char)),
            word.h,
            word.conf,
        )
        for token in TOKEN.finditer(word.text)
    ]


def group_rows(words):
    """Group words whose vertical centers are within half a word height."""
    rows = []

    for word in sorted(words, key=lambda w: w.cy):
        if rows and abs(word.cy - rows[-1]["cy"]) <= 0.5 * max(word.h, rows[-1]["h"]):
            row = rows[-1]
            row["words"].append(word)
            n = len(row["words"])
            row["cy"] += (word.cy - row["cy"]) / n
            row["h"] += (word.h - row["h"]) / n
        else:
            rows.append({"words": [word], "cy": word.cy, "h": word.h})

    return [sorted(row["words"], key=lambda w: w.x) for row in rows]


def find_label(row, nutrient):
    """Index of the label's distinctive word in the row, or None."""
    keys, excluded = LABELS[nutrient]

    for i, word in enumerate(row):
        # Drop a unit glued to the label, e.g. 'Sódio(mg)'.
        token = normalize(UNIT_SUFFIX.sub("", word.text))
        if len(token) < 3:
            continue
        # Learned OCR returns whole phrases without spaces:
        # 'Acucaresadicionados', 'Gordurassaturadas'.
        if any(e in token for e in excluded):
            continue
        if any(len(key) >= 5 and key in token for key in keys):
            return i
        # Abbreviations ('adic') only as a prefix: fuzzy, they match short
        # ingredient words ('dico', dev: total sugars read as added).
        if any(len(key) < 5 and token.startswith(key) for key in keys):
            return i
        score = max(similarity(token, key) for key in keys if len(key) >= 5)
        if any(similarity(token, e) > score for e in excluded):
            continue
        if score >= MIN_SIMILARITY:
            return i

    return None


def is_unit(word):
    # '(g)' often comes out as '(9)', '(6)' or '(o)'. Values are never in
    # parentheses.
    if "(" in word.text or ")" in word.text:
        return True
    return normalize(word.text) in ("", "g", "mg", "ml")


def value_span(row, index):
    """Where the values of a label start: past the label and its unit."""
    label = row[index]
    x = label.right
    for word in row[index + 1 :]:
        if not is_unit(word):
            break
        x = word.right
    return x, label.cy, label.h


def value_after(row, index):
    """First number right of the label, skipping the unit '(g)' / '(mg)'."""
    for word in row[index + 1 :]:
        if is_unit(word):
            continue
        value = parse_number(word.text)
        if value is not None:
            return value
        # Another label starts before any number: no value on this row.
        if len(normalize(word.text)) > 3:
            return None

    return None


def value_near(words, label):
    """Fallback when the label's row has no number: on curved or skewed
    tables the value drifts out of the row. Take the nearest number to the
    right within one text height."""
    candidates = [
        word
        for word in words
        if word.x > label.right
        and abs(word.cy - label.cy) <= label.h
        and not is_unit(word)
        and parse_number(word.text) is not None
    ]
    if not candidates:
        return None
    # Closest row first, then closest to the label.
    best = min(
        candidates, key=lambda w: (round(abs(w.cy - label.cy) / label.h, 1), w.x)
    )
    return parse_number(best.text)


HEADER_SOLID = {"100g", "1009"}  # 'g' is often read as '9'
HEADER_LIQUID = {"100ml", "100mi", "100m1"}


@dataclass(frozen=True)
class Header:
    """A '100 g' / '100 ml' column header."""

    cx: float
    cy: float
    w: int
    basis: Basis


def find_headers(rows):
    """'100 g' / '100 ml' column headers, as one word or two. Side-by-side
    tables have one per half."""
    headers = []

    for row in rows:
        tokens = [normalize(word.text) for word in row]
        for i, token in enumerate(tokens):
            following = tokens[i + 1] if i + 1 < len(tokens) else ""
            if token in HEADER_LIQUID or (token == "100" and following in ("ml", "mi")):
                basis = Basis.LIQUID
            elif token in HEADER_SOLID or (token == "100" and following in ("g", "9")):
                basis = Basis.SOLID
            else:
                continue

            words = row[i : i + 1] if token != "100" else row[i : i + 2]
            x0, x1 = words[0].x, words[-1].right
            headers.append(Header((x0 + x1) / 2, words[0].cy, x1 - x0, basis))

    return headers


def value_in_column(words, label, headers):
    """Number on the label's row aligned with the first '100 g' column right
    of the label. Avoids taking the per-portion or %VD value, or a number
    from a neighboring row."""
    right = [h for h in headers if h.cx > label.right and h.cy < label.cy]
    if not right:
        return None
    column = min(right, key=lambda h: h.cx)
    tolerance = max(column.w, 2 * label.h)

    candidates = [
        word
        for word in words
        if word.x > label.right
        and abs(word.cy - label.cy) <= label.h
        and abs(word.x + word.w / 2 - column.cx) <= tolerance
        and not is_unit(word)
        and parse_number(word.text) is not None
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda w: abs(w.x + w.w / 2 - column.cx))
    return parse_number(best.text)


# RDC 429/2020 allows leaving out nutrients in non-significant amounts and
# declaring them in a sentence: "Não contém quantidades significativas de
# açúcares adicionados, gorduras saturadas e sódio". Those amounts count as
# zero. Normalized text (no spaces or accents), as OCR often drops spaces.
NOT_SIGNIFICANT = "significativ"
NOT_SIGNIFICANT_SPAN = 120  # characters of the nutrient list after it
NOT_SIGNIFICANT_KEYS = {
    # 'açúcares' alone covers total and added sugars.
    Nutrient.ADDED_SUGARS: ("adicionad", "acucar"),
    # Saturated fat is part of total fat.
    Nutrient.SATURATED_FAT: ("saturad", "gordurastotais"),
    Nutrient.SODIUM: ("sodio",),
}


def row_text(row):
    return normalize(" ".join(word.text for word in row))


def not_significant(rows):
    """Nutrients declared as 'não contém quantidades significativas'."""
    text = "".join(row_text(row) for row in rows)
    found = set()
    start = text.find(NOT_SIGNIFICANT)
    while start >= 0:
        listed = text[start : start + NOT_SIGNIFICANT_SPAN]
        # The list ends at the next sentence, e.g. '*Percentual de valores'.
        listed = listed.split("percentual")[0]
        for nutrient, keys in NOT_SIGNIFICANT_KEYS.items():
            if any(key in listed for key in keys):
                found.add(nutrient)
        start = text.find(NOT_SIGNIFICANT, start + 1)
    return found


def parse_table(words) -> Reading:
    # Rows are grouped on whole OCR boxes, then each box is split in place:
    # splitting first interleaves the tokens of two boxes grouped into one
    # row ('Fibras alimentares (g)' over 'Sódio (mg)' read as 'Fibras Sódio
    # alimentares (mg) (g)', dev: 1110 mg of sodium read as 0).
    rows = [
        [token for word in row for token in split_word(word)]
        for row in group_rows(words)
    ]
    headers = find_headers(rows)
    values, matches, labels, spans, value_rows, sources = {}, {}, set(), {}, {}, {}
    # The sentence mentions the labels; it isn't a table row.
    table_rows = [row for row in rows if NOT_SIGNIFICANT not in row_text(row)]

    for nutrient in Nutrient:
        for row in table_rows:
            index = find_label(row, nutrient)
            if index is None:
                continue
            labels.add(nutrient)
            if nutrient not in spans:
                spans[nutrient] = value_span(row, index)

            # Same row first; other words only when the row has no value
            # (curved tables push values out of their row).
            label = row[index]
            value, rule = value_in_column(row[index + 1 :], label, headers), "column"
            if value is None:
                value, rule = value_after(row, index), "first_right"
            if value is None and NEARBY_ROWS:
                value, rule = value_in_column(words, label, headers), "column_rows"
            if value is None and NEARBY_ROWS:
                value, rule = value_near(words, label), "near"
            # An impossible amount is a misreading: try the other rows.
            if value is not None and plausible(nutrient, value):
                values[nutrient] = value
                value_rows[nutrient] = rows.index(row)
                sources[nutrient] = rule
                matches[nutrient] = (" ".join(w.text for w in row), value)
                break

    # Rows are top to bottom, and the sentence comes after the table rows: a
    # number for a listed nutrient on or below the sentence's first line was
    # read from the sentence itself ("gorduras saturadas, gorduras trans" as
    # a row; dev: 90 g of saturated fat in a drink declaring it
    # insignificant), so the sentence wins; a number from a row above it
    # (a table row) still wins over the sentence.
    sentence = next(
        (i for i, row in enumerate(rows) if NOT_SIGNIFICANT in row_text(row)), None
    )
    for nutrient in not_significant(rows):
        if nutrient not in values or value_rows[nutrient] >= sentence:
            values[nutrient] = 0.0
            matches[nutrient] = ("não contém quantidades significativas", 0.0)
            sources[nutrient] = "not_significant"

    facts = NutritionFacts(
        basis=headers[0].basis if headers else None,
        added_sugars_g=values.get(Nutrient.ADDED_SUGARS),
        saturated_fat_g=values.get(Nutrient.SATURATED_FAT),
        sodium_mg=values.get(Nutrient.SODIUM),
    )
    return Reading(
        facts=facts,
        matches=matches,
        labels=labels,
        value_spans=spans,
        headers=headers,
        sources=sources,
    )
