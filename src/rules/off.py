"""Apply the ANVISA rules to Open Food Facts data: the reference answer for
which warnings each product should have."""

import re

import pandas as pd

from rules.anvisa import Basis, Nutrient, NutritionFacts, assess
from rules.exemptions import exemption_from_categories
from utils.constants import FILTER_PAIRS_OUTPUT_FILE, OFF_EXPECTED_WARNINGS_FILE
from utils.logger import Logger

log = Logger("rules")

LIQUID_UNITS = {"ml", "cl", "dl", "l", "lt", "litro", "litros"}
SOLID_UNITS = {"g", "gr", "kg", "mg"}
QUANTITY_UNIT = re.compile(r"\d\s*([a-z]+)\b")


def basis_from_quantity(quantity) -> Basis | None:
    """Solid or liquid from the package quantity, e.g. '500 ml' or '1kg'."""
    if not isinstance(quantity, str):
        return None

    units = set(QUANTITY_UNIT.findall(quantity.lower()))

    if units & LIQUID_UNITS and not units & SOLID_UNITS:
        return Basis.LIQUID
    if units & SOLID_UNITS and not units & LIQUID_UNITS:
        return Basis.SOLID
    return None


def _value(value):
    return None if pd.isna(value) else float(value)


def facts_from_off(product) -> NutritionFacts:
    sodium_g = _value(product.off_sodium_100g)

    return NutritionFacts(
        basis=basis_from_quantity(product.quantity),
        added_sugars_g=_value(product.off_added_sugars_100g),
        saturated_fat_g=_value(product.off_saturated_fat_100g),
        # OFF stores sodium in grams.
        sodium_mg=None if sodium_g is None else sodium_g * 1000,
    )


def expected_warnings_from_off():
    if OFF_EXPECTED_WARNINGS_FILE.exists():
        log.skip("OFF expected warnings already computed")
        log.blank()
        return

    df = pd.read_parquet(FILTER_PAIRS_OUTPUT_FILE)
    rows = []

    for product in df.itertuples(index=False):
        facts = facts_from_off(product)
        exemption = exemption_from_categories(product.categories_tags)
        assessment = assess(facts, exemption)

        row = {
            "code": product.code,
            "split": product.split,
            "basis": None if facts.basis is None else facts.basis.value,
            "exemption": exemption.value,
            "needs_warning": assessment.needs_warning,
        }
        for nutrient in Nutrient:
            row[f"requires_{nutrient.value}"] = (
                None
                if nutrient in assessment.unknown
                else nutrient in assessment.required
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_parquet(OFF_EXPECTED_WARNINGS_FILE, index=False)

    labeled = out[out.split != "unlabeled"]
    log.info(f"Labeled products (dev + val): {len(labeled):,}")
    log.info(f"  basis      : {labeled.basis.value_counts(dropna=False).to_dict()}")
    log.info(f"  exemption  : {labeled.exemption.value_counts().to_dict()}")
    log.info(
        f"  needs warning: {labeled.needs_warning.value_counts(dropna=False).to_dict()}"
    )
    for nutrient in Nutrient:
        counts = labeled[f"requires_{nutrient.value}"].value_counts(dropna=False)
        log.info(f"  requires {nutrient.value:<14}: {counts.to_dict()}")
    log.info(f"Output written to: {OFF_EXPECTED_WARNINGS_FILE}")
