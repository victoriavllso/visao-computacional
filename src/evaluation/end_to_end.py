"""Evaluate the whole system as the app runs it: front and back photos →
verdict (compliant / inconclusive / non-compliant), against the verdict the
hand labels give.

The truth for each product: the back hand label (values per 100 g/ml; basis
from the package quantity when it wasn't marked, as in evaluation/back.py)
→ required warnings by the rules, the front hand label → printed warnings,
and the app's own summarize() → verdict. Products need labels on both
sides; fronts labeled "not the front" are left out, "unreadable" ones give
an unknown front (inconclusive truth).

    python src/evaluation/end_to_end.py --split dev [--limit 100]
        [--front-method yolo|ocr] [--tag name] [--workers 32]

Writes a CSV per product and a Markdown report to
data/evaluation/e2e_<split>_<timestamp>_<tag>/, linked from
data/evaluation/latest/e2e_<split>.{csv,md}.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# One OCR thread per process; parallelism comes from the pool.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("ORT_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "web"))

import pandas as pd
from tqdm import tqdm

from app import pipeline
from labeling.labels import load_labels, truth_facts
from rules.anvisa import Exemption, Nutrient, assess
from utils.constants import (
    DATA_DIR,
    FILTER_PAIRS_OUTPUT_FILE,
    FRONT_LABELS_DIR,
    IMAGES_OUTPUT_DIR,
)
from utils.logger import Logger
from vision.back import pipeline as back_pipeline
from vision.back.ocr import fetch_models

log = Logger("eval e2e")

EVAL_DIR = DATA_DIR / "evaluation"
VERDICTS = ["compliant", "inconclusive", "non_compliant"]


def true_readings(product, back_labels, front_label):
    facts, source = truth_facts(product, back_labels)
    if facts is None:
        required = {n.value: None for n in Nutrient}
        photo_type = source  # "old_format" or "no_table"
    else:
        assessment = assess(facts, Exemption.NONE)
        required = {
            n.value: None if n in assessment.unknown else n in assessment.required
            for n in Nutrient
        }
        photo_type = "table"
    readable_front = front_label["status"] == "ok"
    found = {
        n.value: bool(front_label[n.value]) if readable_front else None
        for n in Nutrient
    }
    return pipeline.Readings({}, required, found, photo_type, {})


def evaluate_product(args):
    product, back_labels, front_label = args
    code = product.code
    truth, _ = pipeline.summarize(true_readings(product, back_labels, front_label))
    row = {"code": code, "truth": truth, "error": None}
    start = time.perf_counter()
    try:
        result = pipeline.analyze(
            IMAGES_OUTPUT_DIR / "front" / f"{code}.jpg",
            IMAGES_OUTPUT_DIR / "nutrition" / f"{code}.jpg",
        )
    except Exception as error:
        row["error"] = repr(error)
        return row
    row["seconds"] = time.perf_counter() - start
    row["verdict"] = result["verdict"]
    row["reason"] = result["reasons"][0]
    row["back_photo_type"] = result["back_photo_type"]
    for n in pipeline.NUTRIENTS:
        row[f"required_{n}"] = result["required_warnings"][n]
        row[f"found_{n}"] = result["found_warnings"][n]
    return row


def report(results, args, elapsed):
    done = results[results.error.isna()]
    matrix = pd.crosstab(done.truth, done.verdict).reindex(
        index=VERDICTS, columns=VERDICTS, fill_value=0
    )
    same = (done.truth == done.verdict).mean()
    decided = done[done.verdict != "inconclusive"]
    lines = [
        "# End-to-end Evaluation Report",
        "",
        f"**Run:** {datetime.now():%Y-%m-%d %H:%M} | **Split:** `{args.split}` | "
        f"**Front:** `{pipeline.front_method()}` | **Products:** {len(results)} | "
        f"**Errors:** {results.error.notna().sum()} | **Time:** {elapsed:.0f} s "
        f"(median {done.seconds.median():.1f} s per product, one process)",
        "",
        f"**Same verdict as the hand labels:** **{same:.1%}** ({len(done)} products)",
        "",
        f"When the app decides (not inconclusive, {len(decided)} products): "
        f"{(decided.truth == decided.verdict).mean():.1%} right.",
        "",
        "| Labels \\ App | " + " | ".join(VERDICTS) + " |",
        "| :--- | " + " | ".join("---:" for _ in VERDICTS) + " |",
    ]
    for truth in VERDICTS:
        cells = [str(matrix.loc[truth, verdict]) for verdict in VERDICTS]
        lines.append(f"| **{truth}** | " + " | ".join(cells) + " |")
    for truth in ("non_compliant", "compliant"):
        rows = done[done.truth == truth]
        if len(rows):
            hit = (rows.verdict == truth).mean()
            lines += [
                "",
                f"Truly {truth}: app says {truth} on {hit:.1%} ({len(rows)}).",
            ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="dev", choices=["dev", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--front-method", choices=["yolo", "ocr"], default=None)
    parser.add_argument("--tag", default="app")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def main():
    args = parse_args()
    if args.front_method:
        # Read by pipeline.front_method() in each worker.
        os.environ["APP_FRONT_METHOD"] = args.front_method
    products = pd.read_parquet(FILTER_PAIRS_OUTPUT_FILE)
    products = products[products.split == args.split]
    back_labels = load_labels()
    front_labels = load_labels(FRONT_LABELS_DIR)
    # Plain objects: pool workers need picklable rows.
    rows = [SimpleNamespace(**row) for row in products.to_dict("records")]
    tasks = [
        (row, back_labels, front_labels[row.code])
        for row in rows
        if row.code in back_labels
        and row.code in front_labels
        and front_labels[row.code]["status"] != "not_front"
    ]
    if args.limit:
        tasks = pd.Series(tasks).sample(min(args.limit, len(tasks)), random_state=0)
        tasks = list(tasks)
    log.info(f"{len(tasks)} {args.split} products with both photos labeled")

    # RapidOCR loads its models on first use, downloading them: once here,
    # not in every worker at once (vision/back/ocr.py, fetch_models).
    models = {back_pipeline.RAPIDOCR_MODEL, back_pipeline.FALLBACK_MODEL} - {None}
    with ProcessPoolExecutor(1) as pool:
        for model in sorted(models):
            pool.submit(fetch_models, model).result()

    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        mapped = pool.map(evaluate_product, tasks, chunksize=2)
        results = pd.DataFrame(list(tqdm(mapped, total=len(tasks), unit="product")))
    elapsed = time.perf_counter() - start

    run_name = f"e2e_{args.split}_{datetime.now():%Y%m%d_%H%M%S}_{args.tag}"
    run_dir = EVAL_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / f"{run_name}.csv"
    results.to_csv(output, index=False)
    text = report(results, args, elapsed)
    output.with_suffix(".md").write_text(text)
    print(text)

    latest_dir = EVAL_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".csv", ".md"):
        link = latest_dir / f"e2e_{args.split}{ext}"
        link.unlink(missing_ok=True)
        link.symlink_to(os.path.relpath(output.with_suffix(ext), latest_dir))
    log.info(f"Results: {output}")


if __name__ == "__main__":
    main()
