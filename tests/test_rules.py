import pytest

from rules.anvisa import Basis, Exemption, Nutrient, NutritionFacts, assess
from rules.exemptions import exemption_from_categories
from rules.off import basis_from_quantity

SOLID, LIQUID = Basis.SOLID, Basis.LIQUID


def facts(basis=SOLID, sugars=0.0, fat=0.0, sodium=0.0):
    return NutritionFacts(basis, sugars, fat, sodium)


# ============================================================
# Annex XV limits
# ============================================================


@pytest.mark.parametrize(
    "nutrient, basis, below, limit",
    [
        (Nutrient.ADDED_SUGARS, SOLID, 14.9, 15.0),
        (Nutrient.SATURATED_FAT, SOLID, 5.9, 6.0),
        (Nutrient.SODIUM, SOLID, 599.0, 600.0),
        (Nutrient.ADDED_SUGARS, LIQUID, 7.4, 7.5),
        (Nutrient.SATURATED_FAT, LIQUID, 2.9, 3.0),
        (Nutrient.SODIUM, LIQUID, 299.0, 300.0),
    ],
)
def test_limit_is_inclusive(nutrient, basis, below, limit):
    def with_amount(amount):
        values = {"sugars": 0.0, "fat": 0.0, "sodium": 0.0}
        key = {
            Nutrient.ADDED_SUGARS: "sugars",
            Nutrient.SATURATED_FAT: "fat",
            Nutrient.SODIUM: "sodium",
        }[nutrient]
        values[key] = amount
        return facts(basis, **values)

    assert assess(with_amount(below)).required == frozenset()
    # Q&A 176: exactly 15 g of added sugars per 100 g requires the warning.
    assert assess(with_amount(limit)).required == {nutrient}


def test_nescau_cereal_needs_only_added_sugars():
    # Back of the package: 29 g added sugars, 1.5 g sat. fat, 367 mg sodium.
    result = assess(facts(SOLID, 29, 1.5, 367))

    assert result.required == {Nutrient.ADDED_SUGARS}
    assert result.needs_warning is True


def test_multiple_warnings():
    result = assess(facts(SOLID, 20, 8, 700))

    assert result.required == set(Nutrient)


def test_same_amounts_differ_between_solid_and_liquid():
    amounts = dict(sugars=10, fat=4, sodium=400)

    assert assess(facts(SOLID, **amounts)).needs_warning is False
    assert assess(facts(LIQUID, **amounts)).required == set(Nutrient)


# ============================================================
# Missing information
# ============================================================


def test_missing_amount_is_unknown():
    result = assess(NutritionFacts(SOLID, None, 1.0, 100.0))

    assert result.unknown == {Nutrient.ADDED_SUGARS}
    assert result.needs_warning is None


def test_one_required_nutrient_is_enough_despite_unknowns():
    result = assess(NutritionFacts(SOLID, None, None, 800.0))

    assert result.needs_warning is True


@pytest.mark.parametrize(
    "sugars, expected",
    [(16.0, True), (5.0, False), (10.0, None)],
)
def test_unknown_basis_decides_when_both_limits_agree(sugars, expected):
    result = assess(facts(None, sugars=sugars))

    assert result.needs_warning is expected


# ============================================================
# Annex XVI exemptions
# ============================================================


def test_forbidden_never_needs_warning():
    # Olive oil: 13 g sat. fat but the warning is forbidden.
    result = assess(facts(SOLID, fat=13), Exemption.FORBIDDEN)

    assert result.required == frozenset()
    assert result.needs_warning is False


def test_conditional_with_required_nutrient_is_undecided():
    # Plain cheese over the sodium limit: exempt only without added
    # ingredients bringing sodium, which the nutrition facts can't show.
    result = assess(facts(SOLID, sodium=700), Exemption.CONDITIONAL)

    assert result.required == {Nutrient.SODIUM}
    assert result.needs_warning is None


def test_conditional_under_limits_needs_no_warning():
    result = assess(facts(SOLID, sodium=100), Exemption.CONDITIONAL)

    assert result.needs_warning is False


@pytest.mark.parametrize(
    "tags, expected",
    [
        (["en:fats", "en:vegetable-oils", "en:olive-oils"], Exemption.FORBIDDEN),
        (["en:dairies", "en:milks", "en:uht-milks"], Exemption.FORBIDDEN),
        (["en:salts"], Exemption.FORBIDDEN),
        (["en:cheeses", "en:grated-cheese"], Exemption.CONDITIONAL),
        (["en:meats", "en:frozen-meats"], Exemption.CONDITIONAL),
        # Not exempt: meat products (Q&A 166), requeijão (Q&A 170),
        # peanut butter (Q&A 165), sugar.
        (["en:meats", "en:prepared-meats", "en:sausages"], Exemption.NONE),
        (["en:cheeses", "en:cream-cheeses", "pt:requeijao-cremoso"], Exemption.NONE),
        (["en:nuts", "en:peanuts", "en:spreads", "en:peanut-butters"], Exemption.NONE),
        (["en:sweeteners", "en:sugars"], Exemption.NONE),
        (["en:milks", "en:sweetened-condensed-milks"], Exemption.NONE),
        (["en:creams", "en:milks"], Exemption.NONE),
        (["en:alcoholic-beverages", "en:non-alcoholic-beverages"], Exemption.NONE),
        (None, Exemption.NONE),
    ],
)
def test_exemption_from_categories(tags, expected):
    assert exemption_from_categories(tags) is expected


# ============================================================
# OFF quantity → basis
# ============================================================


@pytest.mark.parametrize(
    "quantity, expected",
    [
        ("500 ml", LIQUID),
        ("1L", LIQUID),
        ("1 serving = 250ml", LIQUID),
        ("380 g", SOLID),
        ("1,005kg", SOLID),
        ("37.5g", SOLID),
        ("", None),
        (None, None),
        ("6 unidades", None),
    ],
)
def test_basis_from_quantity(quantity, expected):
    assert basis_from_quantity(quantity) is expected
