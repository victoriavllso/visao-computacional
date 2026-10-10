"""IN 75/2020 Annex XVI (foods where the front warning is forbidden), mapped
to Open Food Facts category tags.

This needs the product category, which the image pipeline can't provide yet,
so for now it's only used on the dataset side. The mapping is an
approximation: OFF tags are user-entered and broader or narrower than the
legal categories, so each group pairs the tags that match with tags that
rule a product out.
"""

from rules.anvisa import Exemption

# Items 1-6: forbidden unless added ingredients bring added sugars or
# significant saturated fat / sodium (Anexo IV), which the data can't tell.
CONDITIONAL_TAGS = {
    # 1. fruits, vegetables, legumes, tubers, cereals, nuts, seeds, mushrooms
    "en:fruits",
    "en:vegetables",
    "en:legumes",
    "en:legume-seeds",
    "en:cereal-grains",
    "en:nuts",
    "en:peanuts",
    "en:seeds",
    "en:mushrooms",
    # 2. flours
    "en:flours",
    # 3. packaged meat and fish, chilled or frozen
    "en:meats",
    "en:fishes",
    # 4. eggs
    "en:eggs",
    # 5. fermented milks
    "en:fermented-milk-products",
    "en:yogurts",
    # 6. cheeses
    "en:cheeses",
}

# Processed products that carry a conditional tag but aren't the plain food:
# meat products (Q&A 166), requeijão and cream cheese (Q&A 170), peanut
# butter and paçoca (Q&A 165), snacks, desserts, ready meals.
NOT_CONDITIONAL_TAGS = {
    "en:prepared-meats",
    "en:meat-preparations",
    "en:breaded-products",
    "en:sausages",
    "en:meals",
    "en:canned-fishes",
    "en:cream-cheeses",
    "pt:requeijao-cremoso",
    "en:spreads",
    "en:snacks",
    "en:desserts",
    "en:fermented-dairy-desserts",
}

# Items 7-19: forbidden.
FORBIDDEN_TAGS = {
    # 7-8. milks of all mammal species, milk powder
    "en:uht-milks",
    "en:whole-milks",
    "en:skimmed-milks",
    "en:semi-skimmed-milks",
    "en:pasteurised-milks",
    "en:milk-powders",
    "en:powdered-milks",
    # 9. olive oil and other vegetable oils
    "en:olive-oils",
    "en:vegetable-oils",
    # 10. salt
    "en:salts",
    # 11-12. infant and enteral formulas
    "en:infant-formulas",
    "en:baby-milks",
    # 13-14. weight control foods, food supplements
    "en:dietary-supplements",
    "en:bodybuilding-supplements",
    # 15. alcoholic beverages
    "en:alcoholic-beverages",
    # 18. food additives (table sweeteners included, Q&A 173). Not
    # en:sweeteners: in OFF it also covers sugar and honey.
    "en:food-additives",
    "en:sugar-substitutes",
}

# Products wrongly caught by a forbidden tag: dairy drinks and fermented
# milks get tagged as milks or supplements; cream and sweetened milks aren't
# plain milk.
NOT_FORBIDDEN_TAGS = {
    "en:creams",
    "en:dairy-drinks",
    "en:fermented-milk-products",
    "en:non-alcoholic-beverages",
    "en:sweetened-condensed-milks",
    "en:condensed-milks",
    "en:flavoured-milks",
    "en:milk-drinks",
    "en:cocoa-and-its-products",
}


def exemption_from_categories(categories_tags) -> Exemption:
    tags = set(categories_tags if categories_tags is not None else [])

    if tags & FORBIDDEN_TAGS and not tags & NOT_FORBIDDEN_TAGS:
        return Exemption.FORBIDDEN

    if tags & CONDITIONAL_TAGS and not tags & NOT_CONDITIONAL_TAGS:
        return Exemption.CONDITIONAL

    return Exemption.NONE
