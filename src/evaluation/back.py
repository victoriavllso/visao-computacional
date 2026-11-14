"""Evaluate the back pipeline against the ground truth: hand labels where
they exist (labels/back_labels.jsonl), Open Food Facts values otherwise.

Two levels:
- reading: is the value read from the photo the true one?
- decision: does the value read lead to the same warning as the true value?
  This is what the final answer depends on; a reading of 29 vs 30 g of
  added sugar is wrong but harmless.

Each metric only counts products whose true value (or decision) is known.

    python src/evaluation/back.py --split dev [--limit 100] [--hand-only]
        [--no-locate] [--no-preprocess] [--tag name] [--ocr-cache]
        [--rapidocr-model small|medium]

--ocr-cache reads the OCR results stored in data/cache/back_ocr/ and stores
the missing ones (vision/back/cache.py): rule changes after the OCR are then
scored without running it again (times are then meaningless).
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# One OCR thread per process (Tesseract, ONNX Runtime); parallelism comes
# from the pool.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("ORT_THREADS", "1")

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from labeling.labels import load_labels, truth_facts
from rules.anvisa import Nutrient, NutritionFacts, assess
from utils.constants import DATA_DIR, FILTER_PAIRS_OUTPUT_FILE, IMAGES_OUTPUT_DIR
from utils.logger import Logger
from vision.back import pipeline
from vision.back.cache import OcrCache
from vision.back.ocr import fetch_models
from vision.back.classify import STATUS_TYPES, PhotoType
from vision.back.pipeline import read_back

log = Logger("eval back")

EVAL_DIR = DATA_DIR / "evaluation"
DEBUG_DIR = DATA_DIR / "debug"

# A reading matches OFF within 5% or this absolute amount (g, g, mg).
TOLERANCE = {
    Nutrient.ADDED_SUGARS: 0.1,
    Nutrient.SATURATED_FAT: 0.1,
    Nutrient.SODIUM: 5.0,
}


def matches(read, truth, nutrient):
    return abs(read - truth) <= max(0.05 * abs(truth), TOLERANCE[nutrient])


def evaluate_product(args):
    product, labels, locate, preprocess, debug, use_cache, model = args
    product = SimpleNamespace(**product)
    # Set in the worker: spawned processes don't inherit the parent's.
    pipeline.RAPIDOCR_MODEL = model
    cache = OcrCache(product.code) if use_cache else None
    debug_dir = str(EVAL_DIR.parent / "debug" / str(product.code)) if debug else None
    path = IMAGES_OUTPUT_DIR / "nutrition" / f"{product.code}.jpg"
    truth, source = truth_facts(product, labels)
    label = labels.get(str(product.code))
    true_type = STATUS_TYPES.get(label["status"]) if label else None
    row = {
        "code": product.code,
        "truth": source,
        "true_type": None if true_type is None else true_type.value,
        "photo_type": None,
        "error": None,
    }

    try:
        result = read_back(
            str(path),
            locate=locate,
            preprocess=preprocess,
            debug_dir=debug_dir,
            cache=cache,
        )
    except Exception as error:
        row["error"] = repr(error)
        log.error(f"Error reading {product.code}: {error}")
        return row
    finally:
        if cache is not None:
            cache.save()
            row.update({f"ocr_cache_{k}": v for k, v in cache.stats.items()})

    row["photo_type"] = result.photo_type.value
    row.update({f"f_{k}": v for k, v in vars(result.features).items()})

    # A photo classified as not readable gets no values: the pipeline
    # abstains, as the app would ("old label", "no table found").
    read = (
        result.reading.facts
        if result.readable
        else NutritionFacts(None, None, None, None)
    )
    row["basis_read"] = None if read.basis is None else read.basis.value
    row["basis_true"] = (
        None if truth is None or truth.basis is None else truth.basis.value
    )

    for nutrient in Nutrient:
        name = nutrient.value
        row[f"{name}_read"] = read.amount(nutrient)
        # Engine and rule that gave the value (vision.back.parse.Reading).
        row[f"{name}_source"] = result.reading.sources.get(nutrient)
        row[f"{name}_label"] = nutrient in result.reading.labels

    if truth is None:
        if debug_dir is not None:
            from vision.back.debug import update_summary_evaluation

            update_summary_evaluation(
                debug_dir=debug_dir,
                result=result,
                truth=None,
                source=source,
                true_type=true_type,
                read_assessment=None,
                true_assessment=None,
                matches_func=matches,
                tolerance_dict=TOLERANCE,
            )
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

    if debug_dir is not None:
        from vision.back.debug import update_summary_evaluation

        update_summary_evaluation(
            debug_dir=debug_dir,
            result=result,
            truth=truth,
            source=source,
            true_type=true_type,
            read_assessment=read_assessment,
            true_assessment=true_assessment,
            matches_func=matches,
            tolerance_dict=TOLERANCE,
        )

    return row


def _metric_row(results, nutrient):
    """Return a dict with the four metrics for one nutrient (None = n/a)."""
    name = nutrient.value
    cols = [
        f"{name}_label",
        f"{name}_found",
        f"{name}_correct",
        f"{name}_decision_correct",
    ]
    row = {}
    for col in cols:
        values = results[col].dropna() if col in results else pd.Series(dtype=bool)
        row[col] = values.astype(bool).mean() if len(values) else float("nan")
    return row


def compute_subset_metrics(subset: pd.DataFrame) -> dict:
    """Compute all nutrient metrics and all-three score for one subset."""
    nutrient_rows = {nutrient: _metric_row(subset, nutrient) for nutrient in Nutrient}

    all_three = float("nan")
    complete_count = 0
    columns = [f"{n.value}_decision_correct" for n in Nutrient]
    if set(columns) <= set(subset.columns):
        complete = subset[columns].dropna()
        complete_count = len(complete)
        if complete_count:
            all_three = complete.astype(bool).all(axis=1).mean()

    return {
        "count": len(subset),
        "nutrients": nutrient_rows,
        "all_three": all_three,
        "complete_count": complete_count,
    }


def _format_metric_block_md(title: str, metrics_data: dict) -> str:
    """Return a Markdown string for one metric subset."""
    lines = [f"### {title} ({metrics_data['count']} products)", ""]
    lines.append(
        f"| {'Nutrient':<15} | {'Label':>7} | {'Found':>7} | {'Correct':>8} | {'Decision':>9} |"
    )
    lines.append(f"| {'-'*15} | {'-'*7} | {'-'*7} | {'-'*8} | {'-'*9} |")

    def fmt(v):
        return f"{v:.1%}" if v == v else "n/a"  # NaN check

    for nutrient in Nutrient:
        row = metrics_data["nutrients"][nutrient]
        name = nutrient.value
        lines.append(
            f"| {name:<15} | {fmt(row[f'{name}_label']):>7} | {fmt(row[f'{name}_found']):>7}"
            f" | {fmt(row[f'{name}_correct']):>8} | {fmt(row[f'{name}_decision_correct']):>9} |"
        )

    all_three = metrics_data["all_three"]
    if all_three == all_three:  # not NaN
        lines.append("")
        lines.append(
            f"**All three decisions correct:** {all_three:.1%}"
            f" (of {metrics_data['complete_count']} products with all three decisions known)"
        )

    return "\n".join(lines)


def _metric_block_md(results, title):
    """Return a Markdown string for one metric subset (backward compatibility)."""
    return _format_metric_block_md(title, compute_subset_metrics(results))


def _format_confusion_matrix(typed: pd.DataFrame) -> list[str]:
    """Generate markdown tables for raw confusion matrix, normalized recall, and per-class metrics."""
    lines = []

    # Get all unique labels across true and predicted to ensure square matrix
    all_labels = sorted(
        list(
            set(typed["true_type"].dropna().unique())
            | set(typed["photo_type"].dropna().unique())
        )
    )

    # Build raw crosstab with complete index/columns
    cm_raw = pd.crosstab(typed["true_type"], typed["photo_type"], dropna=False).reindex(
        index=all_labels, columns=all_labels, fill_value=0
    )

    # 1. Raw Confusion Matrix Table
    lines.append("### Confusion Matrix (Raw Counts)")
    lines.append("")
    header = (
        "| True \\ Pred | "
        + " | ".join([str(c) for c in cm_raw.columns])
        + " | Total |"
    )
    sep = "| " + " | ".join(["---"] * (len(cm_raw.columns) + 2)) + " |"
    lines.extend([header, sep])

    for idx, row in cm_raw.iterrows():
        total = row.sum()
        row_str = " | ".join([f"{v:,}" for v in row])
        lines.append(f"| **{idx}** | {row_str} | **{total:,}** |")

    # Column totals
    col_totals = cm_raw.sum(axis=0)
    total_all = col_totals.sum()
    col_str = " | ".join([f"**{v:,}**" for v in col_totals])
    lines.append(f"| **Total** | {col_str} | **{total_all:,}** |")
    lines.append("")

    # 2. Per-Class Precision, Recall, and F1-Score
    lines.append("### Per-Class Classification Metrics")
    lines.append("")
    lines.append("| Photo Type | Support | Precision | Recall | F1-Score |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for label in all_labels:
        tp = (
            cm_raw.loc[label, label]
            if label in cm_raw.index and label in cm_raw.columns
            else 0
        )
        fn = cm_raw.loc[label].sum() - tp
        fp = cm_raw[label].sum() - tp
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        lines.append(
            f"| **{label}** | {support:,} | {precision:.1%} | {recall:.1%} | {f1:.1%} |"
        )

    lines.append("")
    return lines


def _format_basis_metrics(results: pd.DataFrame) -> list[str]:
    """Evaluate serving basis extraction accuracy and compare ground truth vs predicted basis."""
    if "basis_read" not in results.columns or "basis_true" not in results.columns:
        return []

    evaluated = results.dropna(subset=["basis_read", "basis_true"])
    if len(evaluated) == 0:
        return []

    correct = (evaluated["basis_read"] == evaluated["basis_true"]).sum()
    total = len(evaluated)
    acc = correct / total if total > 0 else 0.0

    all_bases = sorted(
        list(
            set(evaluated["basis_true"].unique())
            | set(evaluated["basis_read"].unique())
        )
    )

    cm_basis = pd.crosstab(
        evaluated["basis_true"], evaluated["basis_read"], dropna=False
    ).reindex(index=all_bases, columns=all_bases, fill_value=0)

    lines = [
        "## Serving Basis Extraction",
        "",
        f"**Evaluated photos with ground truth:** {total:,}  ",
        f"**Basis Extraction Accuracy:** **{acc:.1%}** ({correct}/{total})  ",
        f"**Ground Truth Basis distribution:** `{evaluated['basis_true'].value_counts().to_dict()}`  ",
        f"**Predicted Basis distribution (Evaluated):** `{evaluated['basis_read'].value_counts().to_dict()}`  ",
        f"**Total Predicted Basis distribution (All photos):** `{results['basis_read'].value_counts().to_dict()}`  ",
        "",
        "### Basis Confusion Matrix",
        "",
    ]

    header = (
        "| True \\ Pred | "
        + " | ".join([str(c) for c in cm_basis.columns])
        + " | Total |"
    )
    sep = "| " + " | ".join(["---"] * (len(cm_basis.columns) + 2)) + " |"
    lines.extend([header, sep])

    for idx, row in cm_basis.iterrows():
        total_row = row.sum()
        row_str = " | ".join([f"{v:,}" for v in row])
        lines.append(f"| **{idx}** | {row_str} | **{total_row:,}** |")

    col_totals = cm_basis.sum(axis=0)
    col_str = " | ".join([f"**{v:,}**" for v in col_totals])
    lines.append(f"| **Total** | {col_str} | **{col_totals.sum():,}** |")
    lines.append("")

    return lines


def _format_numerical_errors(results: pd.DataFrame) -> list[str]:
    """Calculate MAE and error distributions for numerical OCR extractions."""
    lines = ["## Numerical Extractions Error Analysis (MAE)", ""]
    lines.append(
        "| Nutrient | Evaluated Count | Mean Abs Error (MAE) | Median Abs Error |"
    )
    lines.append("| :--- | :---: | :---: | :---: |")

    has_data = False
    for nutrient in Nutrient:
        name = nutrient.value
        read_col, true_col = f"{name}_read", f"{name}_true"

        if read_col in results.columns and true_col in results.columns:
            valid = results.dropna(subset=[read_col, true_col]).copy()
            if len(valid):
                has_data = True
                errors = (valid[read_col] - valid[true_col]).abs()
                mae = errors.mean()
                median_err = errors.median()
                unit = "mg" if nutrient == Nutrient.SODIUM else "g"
                lines.append(
                    f"| **{name}** | {len(valid):,} | {mae:.2f} {unit} | {median_err:.2f} {unit} |"
                )

    if not has_data:
        lines = [
            "## Numerical Extractions Error Analysis (MAE)",
            "",
            "*No numerical values available for comparison.*",
            "",
        ]
        return lines

    lines.append("")
    return lines


def _format_top_outliers(results: pd.DataFrame, top_n: int = 5) -> list[str]:
    """Identify and format products with the largest numerical OCR differences between read and true values."""
    lines = [
        f"### Top {top_n} Largest Numerical Differences (Read vs Ground Truth)",
        "",
    ]

    outliers = []
    for nutrient in Nutrient:
        name = nutrient.value
        r_col, t_col = f"{name}_read", f"{name}_true"
        if r_col in results.columns and t_col in results.columns:
            valid = results.dropna(subset=[r_col, t_col]).copy()
            if len(valid) == 0:
                continue
            valid["diff"] = (valid[r_col] - valid[t_col]).abs()
            valid["nutrient"] = name
            valid["read_val"] = valid[r_col]
            valid["true_val"] = valid[t_col]
            outliers.append(valid[["code", "nutrient", "read_val", "true_val", "diff"]])

    if not outliers:
        lines.append("*No evaluated products with numerical values to compare.*")
        lines.append("")
        return lines

    df_outliers = (
        pd.concat(outliers).sort_values(by="diff", ascending=False).head(top_n)
    )
    if len(df_outliers) == 0 or df_outliers["diff"].max() == 0:
        lines.append("*All read numerical values match ground truth (no differences).*")
        lines.append("")
        return lines

    lines.extend(
        [
            "| Product Code | Nutrient | Read Value (Predicted) | True Value (Ground Truth) | Absolute Difference |",
            "| :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, row in df_outliers.iterrows():
        unit = "mg" if row["nutrient"] == "sodium" else "g"
        lines.append(
            f"| `{row['code']}` | **{row['nutrient']}** | {row['read_val']:.1f} {unit} | {row['true_val']:.1f} {unit} | **{row['diff']:.1f} {unit}** |"
        )
    lines.append("")
    return lines


def write_markdown_report(
    results: pd.DataFrame,
    output_csv: Path,
    args,
    elapsed_time: float = 0.0,
    subsets_metrics: dict | None = None,
):
    """Write a comprehensive Markdown summary of the evaluation run beside the CSV output."""
    errors_df = (
        results[results["error"].notna()] if "error" in results else pd.DataFrame()
    )
    total_products = len(results)
    error_count = len(errors_df)
    success_count = total_products - error_count

    throughput = total_products / elapsed_time if elapsed_time > 0 else 0
    avg_latency = (elapsed_time / total_products * 1000) if total_products > 0 else 0

    lines = [
        "# Back Evaluation Report",
        "",
        f"**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Split:** `{args.split}` | **Locate:** `{not args.no_locate}` | **Preprocess:** `{not args.no_preprocess}`  ",
        "",
        "## Executive Summary",
        "",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| **Total Products Evaluated** | {total_products:,} |",
        f"| **Successful Executions** | {success_count:,} ({success_count / total_products:.1%}) |",
        f"| **Pipeline Errors / Failures** | {error_count:,} ({error_count / total_products:.1%}) |",
        f"| **Total Execution Time** | {elapsed_time:.1f} s |",
        f"| **Throughput** | {throughput:.1f} products/sec ({avg_latency:.0f} ms/product) |",
        "",
        "## Dataset Composition",
        "",
        f"**Ground truth sources:** `{results.truth.value_counts().to_dict()}`  ",
        f"**Photo types detected:** `{results.photo_type.value_counts().to_dict() if 'photo_type' in results else 'N/A'}`  ",
        "",
    ]

    # Photo type accuracy & confusion matrix
    if "true_type" in results.columns and "photo_type" in results.columns:
        typed = results.dropna(subset=["true_type", "photo_type"])
        if len(typed):
            accuracy = (typed["true_type"] == typed["photo_type"]).mean()
            lines.extend(
                [
                    "## Photo Type Classification",
                    "",
                    f"**Hand-labeled products:** {len(typed):,}  ",
                    f"**Overall Classification Accuracy:** **{accuracy:.1%}**  ",
                    "",
                ]
            )
            lines.extend(_format_confusion_matrix(typed))

    # Basis accuracy
    lines.extend(_format_basis_metrics(results))

    # Numerical error analysis & top outliers
    lines.extend(_format_numerical_errors(results))
    lines.extend(_format_top_outliers(results, top_n=5))

    # Accuracy Metrics Tables
    lines.append("## Nutrient Extraction & Decision Metrics")
    lines.append("")

    if subsets_metrics is None:
        subsets_metrics = evaluate_subsets(results)

    for title, m in subsets_metrics.items():
        if m["count"] > 0:
            lines.append(_format_metric_block_md(title, m))
            lines.append("")

    # Error & Exception Breakdown
    if error_count > 0:
        lines.extend(
            [
                "## Pipeline Failure & Error Log",
                "",
                f"Total runtime errors: **{error_count}**",
                "",
                "### Exception Summary",
                "",
                "| Error Count | Sample Error Message |",
                "| :---: | :--- |",
            ]
        )
        error_counts = errors_df["error"].value_counts()
        for err_msg, count in error_counts.head(5).items():
            short_msg = str(err_msg).replace("\n", " ")[:120]
            lines.append(f"| {count} | `{short_msg}` |")
        lines.append("")

    md_path = output_csv.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Markdown report saved: {md_path}")


def rate(results, column):
    """Share of True among products where the metric applies."""
    values = results[column].dropna() if column in results else pd.Series(dtype=bool)
    return values.astype(bool).mean() if len(values) else float("nan")


def build_subsets(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute the named subsets for metric reporting."""
    readable_types = {PhotoType.TABLE.value, PhotoType.LINEAR.value}
    subsets = {
        "All photos, end-to-end (abstaining counts as wrong)": results,
    }

    if "photo_type" in results.columns:
        subsets["Predicted readable"] = results[
            results["photo_type"].isin(readable_types)
        ]

    if "true_type" in results.columns:
        typed = results.dropna(subset=["true_type"])
        if len(typed):
            subsets["Hand-labeled readable (unmasked)"] = typed[
                typed["true_type"].isin(readable_types)
            ]

    return subsets


def evaluate_subsets(results: pd.DataFrame) -> dict[str, dict]:
    """Compute metrics for all standard subsets of results."""
    return {
        title: compute_subset_metrics(subset)
        for title, subset in build_subsets(results).items()
    }


def _print_metrics_terminal(metrics_data: dict):
    def fmt(v):
        return f"{v:.1%}" if v == v else "n/a"

    print(f"{'nutrient':<15} {'label':>7} {'found':>7} {'correct':>8} {'decision':>9}")
    for nutrient in Nutrient:
        name = nutrient.value
        row = metrics_data["nutrients"][nutrient]
        print(
            f"{name:<15} "
            f"{fmt(row[f'{name}_label']):>7} "
            f"{fmt(row[f'{name}_found']):>7} "
            f"{fmt(row[f'{name}_correct']):>8} "
            f"{fmt(row[f'{name}_decision_correct']):>9}"
        )

    all_three = metrics_data["all_three"]
    if all_three == all_three:
        log.info(
            f"All three decisions correct: {all_three:.1%}"
            f" (of {metrics_data['complete_count']} products with all three decisions known)"
        )


def summarize(results: pd.DataFrame, subsets_metrics: dict | None = None):
    errors = results["error"].notna().sum() if "error" in results else 0
    log.info(f"Products: {len(results)}   errors: {errors}")
    log.info(f"Ground truth: {results.truth.value_counts().to_dict()}")

    if "photo_type" in results and results["photo_type"].notna().any():
        log.info(f"Photo types: {results['photo_type'].value_counts().to_dict()}")
    else:
        log.info("Photo types: N/A")

    if "true_type" in results and "photo_type" in results:
        typed = results.dropna(subset=["true_type", "photo_type"])
        if len(typed):
            log.blank()
            log.info(f"Photo type vs hand labels ({len(typed)} products), rows = true:")
            print(pd.crosstab(typed.true_type, typed.photo_type, margins=True))
            log.info(
                f"Photo type accuracy: {(typed.true_type == typed.photo_type).mean():.1%}"
            )

    if subsets_metrics is None:
        subsets_metrics = evaluate_subsets(results)

    for title, m in subsets_metrics.items():
        log.blank()
        log.info(f"{title}: {m['count']} products")
        _print_metrics_terminal(m)


def metrics(results):
    _print_metrics_terminal(compute_subset_metrics(results))


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
    parser.add_argument(
        "--ocr-cache",
        action="store_true",
        help="reuse the OCR results in data/cache/back_ocr/, store the missing ones",
    )
    parser.add_argument(
        "--rapidocr-model",
        default=pipeline.RAPIDOCR_MODEL,
        choices=["small", "medium"],
        help="PP-OCRv6 model size for RapidOCR",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="save intermediate CV/OCR images to data/debug/<code>/",
    )
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
    tasks = [
        (p, labels, locate, preprocess, args.debug, args.ocr_cache, args.rapidocr_model)
        for p in products.to_dict("records")
    ]

    # Models present and checked before the workers start, in a process of
    # its own (forked workers must not inherit an ONNX Runtime session).
    with ProcessPoolExecutor(1) as pool:
        pool.submit(fetch_models, args.rapidocr_model).result()

    start = time.perf_counter()
    if args.debug:
        # Sequential: clearer console output and avoids cross-process I/O
        # races writing debug images.
        log.info(f"Debug mode: processing {len(tasks)} products sequentially")
        results = pd.DataFrame(
            [evaluate_product(t) for t in tqdm(tasks, unit="product")]
        )
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            # Results come in order (a slow product holds the bar back a
            # little); the order of the CSV doesn't change.
            mapped = pool.map(evaluate_product, tasks, chunksize=4)
            results = pd.DataFrame(list(tqdm(mapped, total=len(tasks), unit="product")))
    elapsed = time.perf_counter() - start
    log.info(f"Done in {elapsed:.0f} s")
    if args.ocr_cache and "ocr_cache_hits" in results:
        totals = {
            k: int(results[f"ocr_cache_{k}"].sum())
            for k in ("hits", "misses", "changed")
        }
        log.info(
            f"OCR cache: {totals['hits']} calls reused ({totals['changed']} of them"
            f" on a slightly different input), {totals['misses']} run and stored"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = (
        f"{'locate' if locate else 'nolocate'}_{'prep' if preprocess else 'noprep'}"
    )
    if args.rapidocr_model != "small":
        variant += f"_{args.rapidocr_model}"
    run_name = f"{timestamp}_{args.tag or variant}"
    run_dir = EVAL_DIR / f"back_{args.split}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / f"back_{args.split}_{run_name}.csv"
    results.to_csv(output, index=False)

    subsets_metrics = evaluate_subsets(results)
    summarize(results, subsets_metrics=subsets_metrics)
    write_markdown_report(
        results, output, args, elapsed_time=elapsed, subsets_metrics=subsets_metrics
    )
    log.info(f"Per-product results: {output}")

    latest_dir = EVAL_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".csv", ".md"):
        link = latest_dir / f"back_{args.split}{ext}"
        target = output.with_suffix(ext)
        link.unlink(missing_ok=True)
        link.symlink_to(os.path.relpath(target, latest_dir))
    log.info(f"Updated latest symlinks in {latest_dir}")


if __name__ == "__main__":
    main()
