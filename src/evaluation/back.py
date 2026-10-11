"""Evaluate the back pipeline against the ground truth: hand labels where
they exist (labels/back_labels.jsonl), Open Food Facts values otherwise.

Two levels:
- reading: is the value read from the photo the true one?
- decision: does the value read lead to the same warning as the true value?
  This is what the final answer depends on; a reading of 29 vs 30 g of
  added sugar is wrong but harmless.

Each metric only counts products whose true value (or decision) is known.

    python src/evaluation/back.py --split dev [--limit 100] [--hand-only]
        [--no-locate] [--no-preprocess] [--tag name]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

# One OCR thread per process (Tesseract, ONNX Runtime); parallelism comes
# from the pool.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("ORT_THREADS", "1")

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from rules.anvisa import Nutrient, assess
from labeling.labels import load_labels, truth_facts
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
    product, labels, locate, preprocess = args
    product = SimpleNamespace(**product)
    path = IMAGES_OUTPUT_DIR / "nutrition" / f"{product.code}.jpg"
    truth, source = truth_facts(product, labels)
    row = {"code": product.code, "truth": source}

    try:
        result = read_back(str(path), locate=locate, preprocess=preprocess)
    except Exception as error:
        row["error"] = repr(error)
        return row

    read = result.reading.facts
    row["basis_read"] = None if read.basis is None else read.basis.value
    for nutrient in Nutrient:
        name = nutrient.value
        row[f"{name}_read"] = read.amount(nutrient)
        row[f"{name}_label"] = nutrient in result.reading.labels

    if truth is None:
        return row

    # Hand labels give the basis; OFF's comes from the package quantity, so
    # prefer the one read from the table header.
    basis = (
        truth.basis if source == "hand" and truth.basis else read.basis or truth.basis
    )
    read_assessment = assess(replace(read, basis=basis))
    true_assessment = assess(replace(truth, basis=basis))

    for nutrient in Nutrient:
        name = nutrient.value
        read_value, true_value = read.amount(nutrient), truth.amount(nutrient)
        row[f"{name}_true"] = true_value
        if true_value is None:
            continue
        row[f"{name}_found"] = read_value is not None
        row[f"{name}_correct"] = read_value is not None and matches(
            read_value, true_value, nutrient
        )
        if nutrient not in true_assessment.unknown:
            row[f"{name}_decision_correct"] = (
                nutrient not in read_assessment.unknown
                and (nutrient in read_assessment.required)
                == (nutrient in true_assessment.required)
            )

    return row


def rate(results, column):
    """Share of True among products where the metric applies."""
    values = results[column].dropna() if column in results else pd.Series(dtype=bool)
    return values.astype(bool).mean() if len(values) else float("nan")


def summarize(results):
    errors = results["error"].notna().sum() if "error" in results else 0
    log.info(f"Products: {len(results)}   errors: {errors}")
    log.info(f"Ground truth: {results.truth.value_counts().to_dict()}")
    log.blank()
    print(f"{'nutrient':<15} {'label':>7} {'found':>7} {'correct':>8} {'decision':>9}")
    for nutrient in Nutrient:
        name = nutrient.value
        print(
            f"{name:<15} "
            f"{rate(results, f'{name}_label'):>7.1%} "
            f"{rate(results, f'{name}_found'):>7.1%} "
            f"{rate(results, f'{name}_correct'):>8.1%} "
            f"{rate(results, f'{name}_decision_correct'):>9.1%}"
        )

    # Products where all three decisions are known and correct.
    decisions = results[[f"{n.value}_decision_correct" for n in Nutrient]]
    complete = decisions.dropna()
    log.blank()
    log.info(
        f"All three decisions correct: {complete.astype(bool).all(axis=1).mean():.1%}"
        f" (of {len(complete)} products with all three decisions known)"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="dev", choices=["dev", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--hand-only", action="store_true", help="only hand-labeled products"
    )
    parser.add_argument("--tag", default=None, help="name for the output file")
    parser.add_argument("--no-locate", action="store_true")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def main():
    args = parse_args()
    products = pd.read_parquet(FILTER_PAIRS_OUTPUT_FILE)
    products = products[products.split == args.split]
    labels = load_labels()
    if args.hand_only:
        products = products[products.code.isin(labels)]
    if args.limit:
        products = products.sample(min(args.limit, len(products)), random_state=0)

    locate, preprocess = not args.no_locate, not args.no_preprocess
    tasks = [(p, labels, locate, preprocess) for p in products.to_dict("records")]

    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = pd.DataFrame(pool.map(evaluate_product, tasks, chunksize=4))
    log.info(f"Done in {time.perf_counter() - start:.0f} s")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    variant = (
        f"{'locate' if locate else 'nolocate'}_{'prep' if preprocess else 'noprep'}"
    )
    output = EVAL_DIR / f"back_{args.split}_{args.tag or variant}.csv"
    results.to_csv(output, index=False)

    summarize(results)
    log.info(f"Per-product results: {output}")


if __name__ == "__main__":
    main()
