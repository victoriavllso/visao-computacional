"""Evaluate the front pipeline against the hand labels of the front photo
(labels/front/, see LABELING.md): which "ALTO EM" warnings each photo shows.

Accuracy counts photos labeled "front of the package"; photos where a
warning is there but can't be read ("unreadable") are reported apart, and
photos that aren't the front are left out. The OCR runs once per photo (and
is cached in data/cache/, so changing the rules doesn't rerun it), and every
rule variant in VARIANTS is scored on the same words: each one adds a rule
to the previous, to measure what it brings.

    python src/evaluation/front.py --split dev [--limit 100] [--tag name]
        [--workers 32]

Writes a CSV per product and a Markdown report to
data/evaluation/front_<split>_<timestamp>_<tag>/, linked from
data/evaluation/latest/front_<split>.{csv,md}.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

# One OCR thread per process; parallelism comes from the pool.
os.environ.setdefault("ORT_THREADS", "1")

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import pandas as pd

from labeling.labels import load_labels
from rules.anvisa import Nutrient
from utils.constants import (
    DATA_DIR,
    FILTER_PAIRS_OUTPUT_FILE,
    FRONT_LABELS_DIR,
    IMAGES_OUTPUT_DIR,
)
from utils.logger import Logger
from vision.back.ocr import read_words_rapidocr
from vision.back.parse import Word
from vision.front.warnings import KEYS, find_warnings

log = Logger("eval front")

EVAL_DIR = DATA_DIR / "evaluation"
OCR_CACHE = DATA_DIR / "cache" / "front_ocr"
NUTRIENTS = [n.value for n in Nutrient]

# Each variant adds one rule to the previous; "anywhere" drops the panel.
FIRST_KEYS = {n: alternatives[:1] for n, alternatives in KEYS.items()}
VARIANTS = {
    "v1": dict(keys=FIRST_KEYS, reach=(10, 7), height_ratio=(0.6, 1.6), not_anchors=()),
    "+any_word": dict(reach=(10, 7), height_ratio=(0.6, 1.6), not_anchors=()),
    "+reach_height": dict(not_anchors=()),
    "+not_alto_en": dict(),
    "anywhere": dict(scope="anywhere"),
}


def ocr_words(code):
    """RapidOCR lines of the front photo, from the cache if there."""
    cached = OCR_CACHE / f"{code}.json"
    if cached.exists():
        return [Word(**w) for w in json.loads(cached.read_text())]
    image = cv2.imread(str(IMAGES_OUTPUT_DIR / "front" / f"{code}.jpg"))
    if image is None:
        raise FileNotFoundError(code)
    words = read_words_rapidocr(image)
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps([w.__dict__ for w in words], ensure_ascii=False))
    return words


def evaluate_product(args):
    code, label = args
    row = {"code": code, "status": label["status"], "error": None}
    for n in NUTRIENTS:
        row[f"true_{n}"] = bool(label[n])
    try:
        words = ocr_words(code)
    except Exception as error:
        row["error"] = repr(error)
        return row

    row["lines"] = len(words)
    for variant, params in VARIANTS.items():
        result = find_warnings(words, **params)
        row[f"{variant}_anchors"] = len(result.anchors)
        for nutrient, found in result.warnings.items():
            row[f"{variant}_{nutrient.value}"] = found
    return row


def scores(truth, pred):
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else float("nan")
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metrics(results, variant):
    """Scores on the photos labeled as the front, readable."""
    front = results[(results.status == "ok") & results.error.isna()]
    out = {"photos": len(front)}
    for n in NUTRIENTS:
        out[n] = scores(
            front[f"true_{n}"].astype(bool), front[f"{variant}_{n}"].astype(bool)
        )

    true_any = front[[f"true_{n}" for n in NUTRIENTS]].any(axis=1)
    pred_any = front[[f"{variant}_{n}" for n in NUTRIENTS]].any(axis=1)
    out["any"] = scores(true_any, pred_any)
    all_right = (
        front[[f"true_{n}" for n in NUTRIENTS]].values
        == front[[f"{variant}_{n}" for n in NUTRIENTS]].values
    ).all(axis=1)
    out["all_three"] = all_right.mean() if len(front) else float("nan")
    out["all_three_with_warning"] = (
        all_right[true_any.values].mean() if true_any.any() else float("nan")
    )
    return out


def pct(v):
    return "n/a" if v != v else f"{v:.1%}"


def report(results, args, elapsed):
    unreadable = results[results.status == "unreadable"]
    lines = [
        "# Front Evaluation Report",
        "",
        f"**Run:** {datetime.now():%Y-%m-%d %H:%M} | **Split:** `{args.split}` | "
        f"**Products:** {len(results)} | **Errors:** {results.error.notna().sum()} | "
        f"**Time:** {elapsed:.0f} s",
        "",
        f"**Labels:** `{results.status.value_counts().to_dict()}` "
        "(scores count `ok` photos: the front of the package)",
        "",
    ]
    summary = {}
    for variant in VARIANTS:
        m = metrics(results, variant)
        summary[variant] = m
        lines += [
            f"## Variant `{variant}` ({m['photos']} photos)",
            "",
            f"**All three warnings right:** **{pct(m['all_three'])}** "
            f"(photos with a warning: {pct(m['all_three_with_warning'])})",
            "",
            "| Warning | TP | FP | FN | Precision | Recall | F1 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in NUTRIENTS + ["any"]:
            s = m[name]
            lines.append(
                f"| {name} | {s['tp']} | {s['fp']} | {s['fn']} | "
                f"{pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} |"
            )
        if len(unreadable):
            seen = unreadable[[f"{variant}_{n}" for n in NUTRIENTS]].any(axis=1).mean()
            lines.append("")
            lines.append(
                f"Unreadable warnings ({len(unreadable)}): a warning found on {pct(seen)}."
            )
        lines.append("")
    return "\n".join(lines), summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="dev", choices=["dev", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="ocr")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def main():
    args = parse_args()
    products = pd.read_parquet(FILTER_PAIRS_OUTPUT_FILE)
    codes = set(products[products.split == args.split].code)
    labels = load_labels(FRONT_LABELS_DIR)
    tasks = sorted(
        (code, label)
        for code, label in labels.items()
        if code in codes and label["status"] != "not_front"
    )
    if args.limit:
        tasks = pd.Series(tasks).sample(min(args.limit, len(tasks)), random_state=0)
        tasks = list(tasks)
    log.info(f"{len(tasks)} labeled {args.split} photos")

    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = pd.DataFrame(pool.map(evaluate_product, tasks, chunksize=4))
    elapsed = time.perf_counter() - start

    run_name = f"front_{args.split}_{datetime.now():%Y%m%d_%H%M%S}_{args.tag}"
    run_dir = EVAL_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / f"{run_name}.csv"
    results.to_csv(output, index=False)
    text, _ = report(results, args, elapsed)
    output.with_suffix(".md").write_text(text)
    print(text)

    latest_dir = EVAL_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".csv", ".md"):
        link = latest_dir / f"front_{args.split}{ext}"
        link.unlink(missing_ok=True)
        link.symlink_to(os.path.relpath(output.with_suffix(ext), latest_dir))
    log.info(f"Results: {output}")


if __name__ == "__main__":
    main()
