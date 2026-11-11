"""Evaluate the front pipeline against the hand labels of the front photo
(labels/front/, see LABELING.md): which "ALTO EM" warnings each photo shows.

Accuracy counts photos labeled "front of the package"; photos where a
warning is there but can't be read ("unreadable") are reported apart, and
photos that aren't the front are left out. The OCR runs once per photo (and
is cached in data/cache/, so changing the rules doesn't rerun it), and every
rule variant in VARIANTS is scored on the same words: each one adds a rule
to the previous, to measure what it brings.

Method "yolo" (YOLO + OCR): the detector boxes the panels and RapidOCR reads
each crop; its variants (YOLO_VARIANTS) change the box score threshold and
the crop rule on the same boxes and crops. On dev each photo is scored by
the cross-validation model that didn't train on it (training/front_detector.py); on
val, by the final model. The report adds the detector's box metrics against
the labeled boxes, and the time per photo of each method.

    python src/evaluation/front.py --split dev [--method ocr|yolo]
        [--limit 100] [--tag name] [--workers 32] [--no-cache]
        [--detector models/front_detector.onnx | --cv-tag yolo26n_1024]
        [--margin 0.15] [--min-side 480]

Writes a CSV per product and a Markdown report to
data/evaluation/front_<split>_<timestamp>_<tag>/, linked from
data/evaluation/latest/front_<split>[_yolo].{csv,md}.
"""

import argparse
import hashlib
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
import numpy as np
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
from vision.front.detector import (
    DETECTOR_DIR,
    MODEL_PATH,
    Box,
    detect_panels,
    fold_of,
)
from vision.front.warnings import (
    CROP_MARGIN,
    CROP_MIN_SIDE,
    KEYS,
    find_warnings,
    find_warnings_in_panels,
    read_panel,
)

log = Logger("eval front")

EVAL_DIR = DATA_DIR / "evaluation"
OCR_CACHE = DATA_DIR / "cache" / "front_ocr"
YOLO_CACHE = DATA_DIR / "cache" / "front_yolo"
NUTRIENTS = [n.value for n in Nutrient]
# Boxes kept in the cache and for the detector's AP; the variants threshold
# on top of this.
LOWEST_SCORE = 0.05

# Each variant adds one rule to the previous; "anywhere" drops the panel.
FIRST_KEYS = {n: alternatives[:1] for n, alternatives in KEYS.items()}
VARIANTS = {
    "v1": dict(keys=FIRST_KEYS, reach=(10, 7), height_ratio=(0.6, 1.6), not_anchors=()),
    "+any_word": dict(reach=(10, 7), height_ratio=(0.6, 1.6), not_anchors=()),
    "+reach_height": dict(not_anchors=()),
    "+not_alto_en": dict(),
    "anywhere": dict(scope="anywhere"),
}
OCR_DEFAULT = "+not_alto_en"

# YOLO + OCR: box score threshold and crop rule on the same boxes and
# crops; "+ocr" also takes what the whole-photo OCR finds (costs both).
YOLO_VARIANTS = {
    "yolo_s0.05": dict(min_score=0.05),
    "yolo_s0.1": dict(min_score=0.1),
    "yolo_s0.25": dict(min_score=0.25),
    "yolo_s0.5": dict(min_score=0.5),
    "yolo_s0.1_anchor": dict(min_score=0.1, rule="anchor"),
    "yolo_s0.1+ocr": dict(min_score=0.1, union=True),
}


def load_image(code):
    image = cv2.imread(str(IMAGES_OUTPUT_DIR / "front" / f"{code}.jpg"))
    if image is None:
        raise FileNotFoundError(code)
    return image


def ocr_words(code, use_cache=True):
    """RapidOCR lines of the front photo and the seconds they took, from the
    cache if there (entries from before timing was recorded: no time)."""
    cached = OCR_CACHE / f"{code}.json"
    if use_cache and cached.exists():
        data = json.loads(cached.read_text())
        if isinstance(data, list):
            return [Word(**w) for w in data], float("nan")
        return [Word(**w) for w in data["words"]], data["seconds"]
    image = load_image(code)
    start = time.perf_counter()
    words = read_words_rapidocr(image)
    seconds = time.perf_counter() - start
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    data = {"words": [w.__dict__ for w in words], "seconds": seconds}
    cached.write_text(json.dumps(data, ensure_ascii=False))
    return words, seconds


def model_key(model):
    """Cache key of a detector: its content, so retraining under the same
    name doesn't reuse old boxes."""
    digest = hashlib.md5(Path(model).read_bytes()).hexdigest()[:10]
    return f"{Path(model).stem}_{digest}"


def yolo_panels(code, model, margin, min_side, use_cache=True):
    """Detected boxes (score >= LOWEST_SCORE), the lines read in each crop,
    and the seconds of detection and of the whole method."""
    key = f"{model_key(model)}_m{margin}_s{min_side}"
    cached = YOLO_CACHE / key / f"{code}.json"
    if use_cache and cached.exists():
        data = json.loads(cached.read_text())
    else:
        image = load_image(code)
        start = time.perf_counter()
        boxes = detect_panels(image, model, LOWEST_SCORE)
        detect = time.perf_counter() - start
        panels = [read_panel(image, box, margin, min_side) for box in boxes]
        data = {
            "boxes": [b.__dict__ for b in boxes],
            "panels": [[w.__dict__ for w in words] for words in panels],
            "detect_seconds": detect,
            "seconds": time.perf_counter() - start,
        }
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data, ensure_ascii=False))
    boxes = [Box(**b) for b in data["boxes"]]
    panels = [[Word(**w) for w in words] for words in data["panels"]]
    return boxes, panels, data["detect_seconds"], data["seconds"]


def detector_for(code, args):
    """The given model, or on dev the cross-validation model of the fold
    that didn't train on this photo."""
    if args.detector:
        return Path(args.detector)
    return DETECTOR_DIR / args.cv_tag / f"fold{fold_of(code)}.onnx"


def store(row, variant, result):
    row[f"{variant}_anchors"] = len(result.anchors)
    for nutrient, found in result.warnings.items():
        row[f"{variant}_{nutrient.value}"] = found


def yolo_variant(boxes, panels, ocr, min_score, union=False, **params):
    kept = [(b, p) for b, p in zip(boxes, panels) if b.score >= min_score]
    result = find_warnings_in_panels(
        [p for _, p in kept], [b for b, _ in kept], **params
    )
    if union:
        for nutrient in result.warnings:
            result.warnings[nutrient] |= ocr.warnings[nutrient]
    return result


def evaluate_product(task):
    code, label, args = task
    row = {"code": code, "status": label["status"], "error": None}
    for n in NUTRIENTS:
        row[f"true_{n}"] = bool(label[n])
    row["true_boxes"] = json.dumps(label.get("boxes") or [])
    try:
        # --no-cache with "yolo" times YOLO + OCR; the whole-photo OCR (for
        # the "+ocr" variant) still comes from the cache.
        use_ocr_cache = not args.no_cache or args.method == "yolo"
        words, row["ocr_seconds"] = ocr_words(code, use_ocr_cache)
        if args.method == "yolo":
            boxes, panels, row["detect_seconds"], row["yolo_seconds"] = yolo_panels(
                code,
                detector_for(code, args),
                args.margin,
                args.min_side,
                not args.no_cache,
            )
    except Exception as error:
        row["error"] = repr(error)
        return row

    row["lines"] = len(words)
    for variant, params in VARIANTS.items():
        store(row, variant, find_warnings(words, **params))
    if args.method == "yolo":
        row["pred_boxes"] = json.dumps(
            [[b.x, b.y, b.w, b.h, round(b.score, 4)] for b in boxes]
        )
        ocr = find_warnings(words, **VARIANTS[OCR_DEFAULT])
        for variant, params in YOLO_VARIANTS.items():
            store(row, variant, yolo_variant(boxes, panels, ocr, **params))
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


def iou(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - w * h
    return w * h / union if union else 0.0


def match(truth, pred, threshold=0.5):
    """Greedy matching, best score first: (score, hit) per prediction, a hit
    when it overlaps a labeled box not yet taken with IoU >= threshold."""
    used, hits = set(), []
    for p in sorted(pred, key=lambda p: -p[4]):
        overlaps = [(iou(t, p), i) for i, t in enumerate(truth) if i not in used]
        best = max(overlaps, default=(0, None))
        hit = best[0] >= threshold
        if hit:
            used.add(best[1])
        hits.append((p[4], hit))
    return hits


def detector_metrics(results, min_score):
    """Box metrics at IoU 0.5 on the photos labeled as the front (readable
    or not): AP@0.5 over every kept box (score >= LOWEST_SCORE), and box
    recall / precision and photo-level hits at `min_score`."""
    scored = results[results.status.isin(["ok", "unreadable"]) & results.error.isna()]
    hits, labeled = [], 0
    for row in scored.itertuples():
        truth = json.loads(row.true_boxes)
        labeled += len(truth)
        hits += match(truth, json.loads(row.pred_boxes))
    hits.sort(key=lambda h: -h[0])
    tp = np.cumsum([h for _, h in hits])
    fp = np.cumsum([not h for _, h in hits])
    recall = tp / max(labeled, 1)
    precision = tp / np.maximum(tp + fp, 1)
    # All-point interpolated AP (VOC 2010+), at one IoU.
    ap = 0.0
    if hits:
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        ap = float(np.sum(np.diff(np.concatenate([[0], recall])) * envelope))

    at = [h for s, h in hits if s >= min_score]
    with_panel = scored.true_boxes.map(lambda b: len(json.loads(b)) > 0)
    found = scored.pred_boxes.map(
        lambda b: any(p[4] >= min_score for p in json.loads(b))
    )
    return {
        "photos": len(scored),
        "labeled_boxes": labeled,
        "ap50": ap,
        "recall": sum(at) / max(labeled, 1),
        "precision": sum(at) / len(at) if at else float("nan"),
        "panel_found": (found & with_panel).sum() / max(with_panel.sum(), 1),
        "false_box": (found & ~with_panel).sum() / max((~with_panel).sum(), 1),
    }


def pct(v):
    return "n/a" if v != v else f"{v:.1%}"


def timing(results, column):
    if column not in results:
        return "n/a"
    seconds = results[column].dropna()
    if not len(seconds):
        return "n/a (cached)"
    return (
        f"median {seconds.median():.2f} s, mean {seconds.mean():.2f} s "
        f"({len(seconds)} photos)"
    )


def detector_report(results):
    lines = [
        "## Detector (IoU 0.5 against the labeled boxes)",
        "",
        "| Score >= | AP@0.5 | Box recall | Box precision "
        "| Photos with a panel: found | Photos without: a box |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for min_score in (0.05, 0.1, 0.25, 0.5):
        d = detector_metrics(results, min_score)
        lines.append(
            f"| {min_score} | {pct(d['ap50'])} | {pct(d['recall'])} | "
            f"{pct(d['precision'])} | {pct(d['panel_found'])} | "
            f"{pct(d['false_box'])} |"
        )
    lines += [
        "",
        f"{d['photos']} photos (`ok` + `unreadable`), "
        f"{d['labeled_boxes']} labeled boxes. AP@0.5 doesn't depend on the "
        "threshold (all boxes with score >= "
        f"{LOWEST_SCORE}).",
        "",
    ]
    return lines


def report(results, args, elapsed):
    unreadable = results[results.status == "unreadable"]
    method = f"**Method:** `{args.method}`"
    times = f"OCR of the whole photo {timing(results, 'ocr_seconds')}"
    if args.method == "yolo":
        detector = args.detector or f"cross-validation `{args.cv_tag}`"
        method += (
            f" | **Detector:** {detector} | **Crop:** margin {args.margin}, "
            f"short side >= {args.min_side} px"
        )
        times = (
            f"YOLO + OCR of the crops {timing(results, 'yolo_seconds')}, "
            f"of which detection {timing(results, 'detect_seconds')}; " + times
        )
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
        method,
        "",
        f"**Time per photo** (one process, ONNX Runtime on 1 thread): " f"{times}",
        "",
    ]
    if args.method == "yolo":
        lines += detector_report(results)

    summary = {}
    variants = list(VARIANTS)
    if args.method == "yolo":
        variants += list(YOLO_VARIANTS)
    for variant in variants:
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
    parser.add_argument("--method", default="ocr", choices=["ocr", "yolo"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default=None, help="default: the method")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--no-cache", action="store_true", help="rerun the method (to time it)"
    )
    parser.add_argument(
        "--detector",
        default=None,
        help=f"ONNX model for every photo (default on val: {MODEL_PATH})",
    )
    parser.add_argument(
        "--cv-tag",
        default="yolo26n_1024",
        help="cross-validation models on dev: data/detector/<tag>/fold<k>.onnx",
    )
    parser.add_argument("--margin", type=float, default=CROP_MARGIN)
    parser.add_argument("--min-side", type=int, default=CROP_MIN_SIDE)
    args = parser.parse_args()
    args.tag = args.tag or args.method
    if args.method == "yolo" and args.split == "val" and not args.detector:
        args.detector = str(MODEL_PATH)
    return args


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
    tasks = [(code, label, args) for code, label in tasks]

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
    suffix = "" if args.method == "ocr" else f"_{args.method}"
    for ext in (".csv", ".md"):
        link = latest_dir / f"front_{args.split}{suffix}{ext}"
        link.unlink(missing_ok=True)
        link.symlink_to(os.path.relpath(output.with_suffix(ext), latest_dir))
    log.info(f"Results: {output}")


if __name__ == "__main__":
    main()
