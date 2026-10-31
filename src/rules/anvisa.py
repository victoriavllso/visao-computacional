"""ANVISA front-of-pack nutrition labeling rules (RDC 429/2020, IN 75/2020).

Pure logic: no knowledge of Open Food Facts or images. Any source of
nutrition facts (OFF data, OCR of the nutrition table) goes through `assess`.
"""

from dataclasses import dataclass
from enum import Enum


class Nutrient(Enum):
    ADDED_SUGARS = "added_sugars"
    SATURATED_FAT = "saturated_fat"
    SODIUM = "sodium"


class Basis(Enum):
    SOLID = "100g"  # solid or semi-solid foods
    LIQUID = "100ml"


class Exemption(Enum):
    NONE = "none"
    # Annex XVI items 1-6: exempt only if no added ingredient brings added
    # sugars or significant saturated fat / sodium. Not decidable from the
    # nutrition facts alone.
    CONDITIONAL = "conditional"
    # Annex XVI items 7-19: the warning is forbidden.
    FORBIDDEN = "forbidden"


# IN 75/2020, Annex XV: the warning is required when the amount per 100 g or
# 100 ml of the food as sold is greater than or equal to the limit.
# Sugars and fat in grams, sodium in milligrams.
LIMITS = {
    Basis.SOLID: {
        Nutrient.ADDED_SUGARS: 15.0,
        Nutrient.SATURATED_FAT: 6.0,
        Nutrient.SODIUM: 600.0,
    },
    Basis.LIQUID: {
        Nutrient.ADDED_SUGARS: 7.5,
        Nutrient.SATURATED_FAT: 3.0,
        Nutrient.SODIUM: 300.0,
    },
}

WARNING_TEXT = {
    Nutrient.ADDED_SUGARS: "ALTO EM AÇÚCAR ADICIONADO",
    Nutrient.SATURATED_FAT: "ALTO EM GORDURA SATURADA",
    Nutrient.SODIUM: "ALTO EM SÓDIO",
}


@dataclass(frozen=True)
class NutritionFacts:
    """Amounts per 100 g or 100 ml; None when unknown."""

    basis: Basis | None
    added_sugars_g: float | None
    saturated_fat_g: float | None
    sodium_mg: float | None

    def amount(self, nutrient: Nutrient) -> float | None:
        return {
            Nutrient.ADDED_SUGARS: self.added_sugars_g,
            Nutrient.SATURATED_FAT: self.saturated_fat_g,
            Nutrient.SODIUM: self.sodium_mg,
        }[nutrient]


@dataclass(frozen=True)
class Assessment:
    required: frozenset[Nutrient]
    # Nutrients whose requirement can't be decided (missing amount, or an
    # amount between the liquid and solid limits with an unknown basis).
    unknown: frozenset[Nutrient]
    exemption: Exemption

    @property
    def needs_warning(self) -> bool | None:
        """True/False when decidable, None otherwise."""
        if self.exemption is Exemption.FORBIDDEN:
            return False
        if self.required:
            return None if self.exemption is Exemption.CONDITIONAL else True
        return None if self.unknown else False


def is_required(nutrient: Nutrient, amount: float, basis: Basis | None) -> bool | None:
    if basis is not None:
        return amount >= LIMITS[basis][nutrient]

    # Unknown basis: decidable when both limits agree.
    if amount >= LIMITS[Basis.SOLID][nutrient]:
        return True
    if amount < LIMITS[Basis.LIQUID][nutrient]:
        return False
    return None


def assess(facts: NutritionFacts, exemption: Exemption = Exemption.NONE) -> Assessment:
    if exemption is Exemption.FORBIDDEN:
        return Assessment(frozenset(), frozenset(), exemption)

    required, unknown = set(), set()

    for nutrient in Nutrient:
        amount = facts.amount(nutrient)
        result = None if amount is None else is_required(nutrient, amount, facts.basis)

        if result is None:
            unknown.add(nutrient)
        elif result:
            required.add(nutrient)

    return Assessment(frozenset(required), frozenset(unknown), exemption)
