"""Ground truth for the back pipeline: hand labels where they exist, Open
Food Facts values otherwise. Products are matched by barcode.

    python src/labeling/labels.py --compact

merges every label file into hand-labeled.jsonl, one entry per barcode."""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rules.anvisa import Basis, NutritionFacts
from rules.off import facts_from_off
from utils.constants import BACK_LABELS_DIR
from utils.logger import Logger

log = Logger("labels")

DEFAULT_LABELER = "hand-labeled"

# Statuses with values to use. "no_table": the photo has no readable table;
# "old_format": RDC 360/2003 table, per portion only, no 100 g/ml column. In
# both there is no per-100 g/ml value to compare a reading with.
# "linear": linear text format (RDC 429/2020, small packages), per 100 g/ml.
USABLE_STATUSES = {"ok", "partial", "linear"}


def load_labels(directory=BACK_LABELS_DIR):
    """{barcode: label} from every label file; for a barcode labeled more
    than once, the most recent entry wins. Broken lines (a bad merge, a
    write cut short) are skipped with a warning instead of failing."""
    entries = []
    for path in sorted(directory.glob("*.jsonl")) if directory.exists() else []:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entry["code"] = str(entry["code"])
            except (json.JSONDecodeError, TypeError, KeyError):
                log.error(f"{path.name}:{number}: not a valid label, skipped")
                continue
            # Kept through compaction: who labeled it, not the file it is in.
            entry.setdefault("labeler", path.stem)
            entries.append(entry)

    labels = {}
    for entry in sorted(entries, key=lambda e: e.get("saved_at", "")):
        labels[entry["code"]] = entry
    return labels


def compact_labels(directory=BACK_LABELS_DIR):
    """Rewrite all label files as a single DEFAULT_LABELER file with one
    entry per barcode (the one load_labels uses), oldest first. Returns
    (lines before, entries after)."""
    paths = sorted(directory.glob("*.jsonl"))
    before = sum(
        1 for path in paths for line in path.read_text().splitlines() if line.strip()
    )
    labels = load_labels(directory)

    target = directory / f"{DEFAULT_LABELER}.jsonl"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in labels.values())
    )
    temporary.replace(target)
    for path in paths:
        if path != target:
            path.unlink()
    return before, len(labels)


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


def main():
    parser = argparse.ArgumentParser(description="Hand label files.")
    parser.add_argument(
        "--compact",
        action="store_true",
        help=f"merge all label files into {DEFAULT_LABELER}.jsonl, one entry per barcode",
    )
    if not parser.parse_args().compact:
        parser.print_help()
        return
    before, after = compact_labels()
    log.info(f"{before} lines → {after} labels in {DEFAULT_LABELER}.jsonl")


if __name__ == "__main__":
    main()
