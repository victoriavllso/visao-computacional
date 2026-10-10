"""Evaluate the back pipeline against Open Food Facts values.

Two levels:
- reading: is the value read from the photo the one OFF has?
- decision: does the value read lead to the same warning as OFF's value?
  This is what the final answer depends on; a reading of 29 vs 30 g of
  added sugar is wrong but harmless.

    python src/evaluation/back.py --split dev [--limit 100] [--no-locate] [--no-preprocess]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

# One Tesseract thread per process; parallelism comes from the pool.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from rules.anvisa import Nutrient, assess
from rules.off import facts_from_off
from utils.constants import DATA_DIR, FILTER_PAIRS_OUTPUT_FILE, IMAGES_OUTPUT_DIR
from utils.logger import Logger
from vision.back.pipeline import read_back

log = Logger("eval back")

EVAL_DIR = DATA_DIR / "evaluation"

# A reading matches OFF within 5% or this absolute amount (g, g, mg).
TOLERANCE = {
    Nutrient.ADDED_SUGARS: 0.1,
    Nutrient.SATURATED_FAT: 0.1,
    Nutrient.SODIUM: 5.0,
}


def matches(read, truth, nutrient):
    return abs(read - truth) <= max(0.05 * abs(truth), TOLERANCE[nutrient])


def evaluate_product(args):
    product, locate, preprocess = args
    product = SimpleNamespace(**product)
    path = IMAGES_OUTPUT_DIR / "nutrition" / f"{product.code}.jpg"
    row = {"code": product.code}

    try:
        result = read_back(str(path), locate=locate, preprocess=preprocess)
    except Exception as error:
        row["error"] = repr(error)
        return row

    read = result.reading.facts
    truth = facts_from_off(product)
    # The OFF basis comes from the package quantity; prefer the table header.
    basis = read.basis or truth.basis
    read_assessment = assess(replace(read, basis=basis))
    true_assessment = assess(replace(truth, basis=basis))

    row["basis_read"] = None if read.basis is None else read.basis.value
    for nutrient in Nutrient:
        name = nutrient.value
        read_value, true_value = read.amount(nutrient), truth.amount(nutrient)
        row[f"{name}_read"] = read_value
        row[f"{name}_off"] = true_value
        row[f"{name}_label"] = nutrient in result.reading.labels
        row[f"{name}_found"] = read_value is not None
        row[f"{name}_correct"] = read_value is not None and matches(
            read_value, true_value, nutrient
        )
        row[f"{name}_decision_correct"] = (
            nutrient not in read_assessment.unknown
            and nutrient not in true_assessment.unknown
            and (nutrient in read_assessment.required)
            == (nutrient in true_assessment.required)
        )

    return row


def summarize(results):
    log.info(
        f"Products: {len(results)}   errors: {results.get('error', pd.Series()).notna().sum()}"
    )
    log.blank()
    print(f"{'nutrient':<15} {'label':>7} {'found':>7} {'correct':>8} {'decision':>9}")
    for nutrient in Nutrient:
        name = nutrient.value
        print(
            f"{name:<15} "
            f"{results[f'{name}_label'].mean():>7.1%} "
            f"{results[f'{name}_found'].mean():>7.1%} "
            f"{results[f'{name}_correct'].mean():>8.1%} "
            f"{results[f'{name}_decision_correct'].mean():>9.1%}"
        )
    all_decisions = results[[f"{n.value}_decision_correct" for n in Nutrient]].all(
        axis=1
    )
    log.blank()
    log.info(f"All three decisions correct: {all_decisions.mean():.1%}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="dev", choices=["dev", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-locate", action="store_true")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def main():
    args = parse_args()
    products = pd.read_parquet(FILTER_PAIRS_OUTPUT_FILE)
    products = products[products.split == args.split]
    if args.limit:
        products = products.sample(min(args.limit, len(products)), random_state=0)

    locate, preprocess = not args.no_locate, not args.no_preprocess
    tasks = [(p, locate, preprocess) for p in products.to_dict("records")]

    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = pd.DataFrame(pool.map(evaluate_product, tasks, chunksize=4))
    log.info(f"Done in {time.perf_counter() - start:.0f} s")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    variant = (
        f"{'locate' if locate else 'nolocate'}_{'prep' if preprocess else 'noprep'}"
    )
    output = EVAL_DIR / f"back_{args.split}_{variant}.csv"
    results.to_csv(output, index=False)

    summarize(results)
    log.info(f"Per-product results: {output}")


if __name__ == "__main__":
    main()
