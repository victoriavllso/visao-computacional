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


# Distinctive word of each label, and words that look similar but belong to
# other rows.
LABELS = {
    Nutrient.ADDED_SUGARS: ("adicionados", ()),
    Nutrient.SATURATED_FAT: ("saturadas", ("insaturadas", "monoinsaturadas")),
    Nutrient.SODIUM: ("sodio", ()),
}
MIN_SIMILARITY = 0.75

# Letters OCR confuses with digits inside numbers.
DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"})
# Punctuation OCR attaches to numbers: quotes, bars, stars from dirt or lines.
NUMBER_JUNK = "()[]{}|:;*.,'\"“”‘’`´~^_-–—!¡"
NUMBER = re.compile(r"^[<≤]?(\d+(?:[.,]\d+)?)(mg|g|ml)?$")


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
    return float(match.group(1).replace(",", ".")) if match else None


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
    key, excluded = LABELS[nutrient]

    for i, word in enumerate(row):
        # Drop a unit glued to the label, e.g. 'Sódio(mg)'.
        token = normalize(re.sub(r"\(.*$", "", word.text))
        if len(token) < 3 or any(
            similarity(token, e) > similarity(token, key) for e in excluded
        ):
            continue
        if similarity(token, key) >= MIN_SIMILARITY:
            return i

    return None


def is_unit(word):
    # '(g)' often comes out as '(9)', '(6)' or '(o)'. Values are never in
    # parentheses.
    if "(" in word.text or ")" in word.text:
        return True
    return normalize(word.text) in ("", "g", "mg", "ml")


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
    return parse_number(min(candidates, key=lambda w: w.x - label.right).text)


HEADER_SOLID = {"100g", "1009"}  # 'g' is often read as '9'
HEADER_LIQUID = {"100ml", "100mi", "100m1"}


def find_basis(rows):
    """'100 g' / '100 ml' column header, as one word or two."""
    for row in rows:
        tokens = [normalize(word.text) for word in row]
        for i, token in enumerate(tokens):
            following = tokens[i + 1] if i + 1 < len(tokens) else ""
            if token in HEADER_LIQUID or (token == "100" and following in ("ml", "mi")):
                return Basis.LIQUID
            if token in HEADER_SOLID or (token == "100" and following in ("g", "9")):
                return Basis.SOLID

    return None


def parse_table(words) -> Reading:
    rows = group_rows(words)
    values, matches, labels = {}, {}, set()

    for nutrient in Nutrient:
        for row in rows:
            index = find_label(row, nutrient)
            if index is None:
                continue
            labels.add(nutrient)

            value = value_after(row, index)
            if value is None:
                value = value_near(words, row[index])
            if value is not None:
                values[nutrient] = value
                matches[nutrient] = (" ".join(w.text for w in row), value)
                break

    facts = NutritionFacts(
        basis=find_basis(rows),
        added_sugars_g=values.get(Nutrient.ADDED_SUGARS),
        saturated_fat_g=values.get(Nutrient.SATURATED_FAT),
        sodium_mg=values.get(Nutrient.SODIUM),
    )
    return Reading(facts=facts, matches=matches, labels=labels)
