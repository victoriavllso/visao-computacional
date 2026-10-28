"""What kind of nutrition information a back photo shows, so that only
photos that can be checked against RDC 429/2020 are read, and the others get
an explicit answer instead of a guessed reading.

    TABLE       new-format table (RDC 429/2020): read it
    LINEAR      new-format linear text, on small packages: read it
    OLD_FORMAT  RDC 360/2003 table, per portion only: can't be checked
    NO_TABLE    no nutrition information (barcode, front, unreadable)

Classical features: ruled row lines from the table locator (before OCR) and
keywords in the OCR text (after OCR). Old and new formats only differ in
their fixed wording, so the format is decided from positive evidence of each:

    new (RDC 429/2020): 100 g/100 ml column, "açúcares totais" and
        "açúcares adicionados" rows, footnote "*Percentual de valores
        diários fornecidos pela porção"
    old (RDC 360/2003): "Quantidade por porção", footnote "... com base em
        uma dieta de 2.000 kcal ou 8.400 kJ"

OCR misses words, so absence of new evidence alone doesn't make a photo old:
with no evidence either way the type is UNCERTAIN, and the pipeline reads it
again with the learned OCR before deciding.
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
    # Nutrition text without format evidence; resolved with more OCR text.
    UNCERTAIN = "uncertain"

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

# Text patterns indicating a 100g/100ml basis for RDC 429/2020
PER_100 = ("por100g", "por100ml", "100g", "100ml", "100gpor", "100mlpor")

# Distinct phrases used in linear/text declarations (e.g. "por 100 g contém...")
PER_100_EXPLICIT_LINEAR = ("por100g", "por100ml", "em100g", "em100ml", "100gcontem", "100mlcontem")

NEW_FORMAT_TEXT = ("acucarestotais", "fornecidospelaporcao", "adicionados")
OLD_FORMAT_TEXT = ("quantidadeporporcao", "dietade2000", "8400kj")

# Thresholds, to tune on dev hand labels.
MIN_NUTRITION_WORDS = 3
# Nutrition words needed for a '100 g' header to count.
MIN_HEADER_CONTEXT = 1


@dataclass(frozen=True)
class Features:
    ruled_rows: int
    nutrition_words: int
    has_100_header: bool
    has_added_sugars: bool
    has_per_100_text: bool
    has_explicit_linear: bool
    has_new_format_text: bool
    has_old_format_text: bool

    @property
    def new_format_evidence(self):
        # A lone '100 g' is often the net weight printed on the package.
        header = self.has_100_header and self.nutrition_words >= MIN_HEADER_CONTEXT
        return (
            header
            or self.has_added_sugars
            or self.has_per_100_text
            or self.has_new_format_text
        )


def features(table, words, reading) -> Features:
    text = normalize(" ".join(word.text for word in words))
    return Features(
        ruled_rows=table.ruled_rows if table is not None else 0,
        nutrition_words=sum(word in text for word in NUTRITION_WORDS),
        has_100_header=bool(reading.headers),
        has_added_sugars=Nutrient.ADDED_SUGARS in reading.labels,
        has_per_100_text=any(phrase in text for phrase in PER_100),
        has_explicit_linear=any(phrase in text for phrase in PER_100_EXPLICIT_LINEAR),
        has_new_format_text=any(phrase in text for phrase in NEW_FORMAT_TEXT),
        has_old_format_text=any(phrase in text for phrase in OLD_FORMAT_TEXT),
    )


def classify(f: Features) -> PhotoType:
    # If no nutrition context and no new format evidence -> NO_TABLE
    if f.nutrition_words < MIN_NUTRITION_WORDS and not f.new_format_evidence:
        return PhotoType.NO_TABLE

    # Priority 1: New format evidence (RDC 429/2020)
    if f.new_format_evidence:
        # Only declare as LINEAR if there is an explicit continuous linear text pattern AND no ruled lines detected
        if f.has_explicit_linear and f.ruled_rows == 0:
            return PhotoType.LINEAR
        return PhotoType.TABLE

    # Priority 2: Clear evidence of old format without any new format hints
    if f.has_old_format_text:
        return PhotoType.OLD_FORMAT

    # Priority 3: Has nutrition words but format is ambiguous -> UNCERTAIN (trigger RapidOCR)
    return PhotoType.UNCERTAIN