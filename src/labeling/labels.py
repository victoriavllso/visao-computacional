"""Ground truth for the back pipeline: hand labels where they exist, Open
Food Facts values otherwise. Products are matched by barcode."""

import json

from rules.anvisa import Basis, NutritionFacts
from rules.off import facts_from_off
from utils.constants import BACK_LABELS_DIR

# Statuses with values to use. "no_table": the photo has no readable table;
# "old_format": RDC 360/2003 table, per portion only, no 100 g/ml column. In
# both there is no per-100 g/ml value to compare a reading with.
# "linear": linear text format (RDC 429/2020, small packages), per 100 g/ml.
USABLE_STATUSES = {"ok", "partial", "linear"}


def load_labels(directory=BACK_LABELS_DIR):
    """{barcode: label} from every person's file; for a barcode labeled more
    than once, the most recent entry wins."""
    entries = []
    for path in sorted(directory.glob("*.jsonl")) if directory.exists() else []:
        for line in path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                entry["labeler"] = path.stem
                entries.append(entry)

    labels = {}
    for entry in sorted(entries, key=lambda e: e.get("saved_at", "")):
        labels[str(entry["code"])] = entry
    return labels


def facts_from_label(label) -> NutritionFacts:
    # Empty fields mean the person couldn't read the value: unknown, not a
    # reason to fall back to OFF.
    return NutritionFacts(
        basis=Basis(label["basis"]) if label.get("basis") else None,
        added_sugars_g=label.get("added_sugars"),
        saturated_fat_g=label.get("saturated_fat"),
        sodium_mg=label.get("sodium"),
    )


def truth_facts(product, labels):
    """(facts, source) for a product of the pairs file. Source is "hand",
    "off", or the status of a hand label without usable values ("no_table",
    "old_format"), with facts None: nothing to compare with."""
    label = labels.get(str(product.code))

    if label is None:
        return facts_from_off(product), "off"
    if label.get("status") not in USABLE_STATUSES:
        return None, label.get("status", "no_table")

    facts = facts_from_label(label)
    if facts.basis is None:
        # Basis not marked by hand: keep the one from the package quantity.
        off_basis = facts_from_off(product).basis
        facts = NutritionFacts(off_basis, *_amounts(facts))
    return facts, "hand"


def _amounts(facts):
    return facts.added_sugars_g, facts.saturated_fat_g, facts.sodium_mg
