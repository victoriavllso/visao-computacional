"""What kind of nutrition information a back photo shows, so that only
photos that can be checked against RDC 429/2020 are read, and the others get
an explicit answer instead of a guessed reading.

    TABLE       new-format table (RDC 429/2020): read it
    LINEAR      new-format linear text, on small packages: read it
    OLD_FORMAT  RDC 360/2003 table, per portion only: can't be checked
    NO_TABLE    no nutrition information (barcode, front, unreadable)

Classical features: ruled row lines from the table locator (before OCR) and
keywords in the OCR text (after OCR). Old and new formats only differ in
their text: the new one requires a 100 g/100 ml column and an "açúcares
adicionados" row.
"""

from dataclasses import dataclass
from enum import Enum

from rules.anvisa import Nutrient
from vision.back.parse import normalize


class PhotoType(Enum):
    TABLE = "table"
    LINEAR = "linear"
    OLD_FORMAT = "old_format"
    NO_TABLE = "no_table"

    @property
    def readable(self):
        return self in (PhotoType.TABLE, PhotoType.LINEAR)


# Hand label status → photo type.
STATUS_TYPES = {
    "ok": PhotoType.TABLE,
    "partial": PhotoType.TABLE,
    "linear": PhotoType.LINEAR,
    "old_format": PhotoType.OLD_FORMAT,
    "no_table": PhotoType.NO_TABLE,
}

# Distinct nutrition words; normalized (no accents, no spaces).
NUTRITION_WORDS = (
    "nutricional",
    "energetic",
    "carboidrato",
    "acucar",
    "proteina",
    "gordura",
    "fibra",
    "sodio",
)
PER_100 = ("por100g", "por100ml")

# Thresholds, to tune on dev hand labels.
MIN_NUTRITION_WORDS = 3
MIN_RULED_ROWS = 3


@dataclass(frozen=True)
class Features:
    ruled_rows: int
    nutrition_words: int
    has_100_header: bool
    has_added_sugars: bool
    has_per_100_text: bool


def features(table, words, reading) -> Features:
    text = normalize(" ".join(word.text for word in words))
    return Features(
        ruled_rows=table.ruled_rows if table is not None else 0,
        nutrition_words=sum(word in text for word in NUTRITION_WORDS),
        has_100_header=bool(reading.headers),
        has_added_sugars=Nutrient.ADDED_SUGARS in reading.labels,
        has_per_100_text=any(phrase in text for phrase in PER_100),
    )


def classify(f: Features) -> PhotoType:
    if f.nutrition_words < MIN_NUTRITION_WORDS:
        return PhotoType.NO_TABLE
    if not (f.has_100_header or f.has_added_sugars or f.has_per_100_text):
        return PhotoType.OLD_FORMAT
    if f.has_per_100_text and f.ruled_rows < MIN_RULED_ROWS:
        return PhotoType.LINEAR
    return PhotoType.TABLE
