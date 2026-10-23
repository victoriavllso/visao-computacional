"""Save intermediate images and data from the back pipeline for debugging.

When a debug directory is set, every stage of the pipeline writes its output:

    <debug_dir>/
        01_original.jpg              — original input image
        02_located_crop.jpg          — crop after table location
        03_line_masks.jpg            — horizontal (red) + vertical (blue) lines
        04_binarized.jpg             — binarized image fed to Tesseract
        05_dewarped_crop.jpg         — crop after deskew + dewarp
        06_ocr_tesseract.jpg         — Tesseract words drawn on the crop
        07_ocr_rapidocr.jpg          — RapidOCR words drawn on the crop
        08_parsed_values.jpg         — nutrient labels + values annotated
        summary.json                 — features, classification, parsed values
"""

import json
from pathlib import Path

import cv2
import numpy as np


def _save(directory, filename, image):
    """Write an image, handling grayscale and binary."""
    if image is None:
        return
    path = directory / filename
    cv2.imwrite(str(path), image)


def _draw_words(image, words, color=(0, 200, 0), thickness=2, font_scale=0.45):
    """Draw bounding boxes and text labels on a copy of the image."""
    if image is None or not words:
        return image
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for word in words:
        x, y, w, h = word.x, word.y, word.w, word.h
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        label = word.text
        if len(label) > 30:
            label = label[:27] + "..."
        cv2.putText(
            vis,
            label,
            (x, max(y - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def _draw_parsed(image, reading, color_label=(255, 100, 0), color_value=(0, 180, 255)):
    """Annotate the image with nutrient labels and their read values."""
    if image is None:
        return image
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    from rules.anvisa import Nutrient, LIMITS, Basis

    names = {
        Nutrient.ADDED_SUGARS: "added_sugars",
        Nutrient.SATURATED_FAT: "saturated_fat",
        Nutrient.SODIUM: "sodium",
    }
    basis = reading.facts.basis or Basis.SOLID
    limits = LIMITS.get(basis, {})

    for nutrient, (x, cy, th) in reading.value_spans.items():
        name = names.get(nutrient, str(nutrient))
        value = reading.facts.amount(nutrient)
        value_str = f"{value}" if value is not None else "?"
        limit = limits.get(nutrient)
        is_high = value is not None and limit is not None and value >= limit

        # Draw a marker at the label position.
        ix, iy = int(x), int(cy)
        cv2.circle(vis, (ix, iy), 6, (0, 0, 255) if is_high else color_label, -1)
        tag = f"{name}: {value_str}" + (" [ALTO!]" if is_high else "")
        val_color = (0, 0, 255) if is_high else color_value
        cv2.putText(
            vis,
            tag,
            (ix + 10, iy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            val_color,
            2,
            cv2.LINE_AA,
        )

    # Draw headers.
    for header in reading.headers:
        hx, hy = int(header.cx), int(header.cy)
        cv2.drawMarker(vis, (hx, hy), (0, 255, 0), cv2.MARKER_DIAMOND, 12, 2)
        cv2.putText(
            vis,
            f"100 {header.basis.value}",
            (hx + 8, hy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def save_original(debug_dir, image):
    _save(debug_dir, "01_original.jpg", image)


def save_located_crop(debug_dir, crop, box, angle, inverted):
    _save(debug_dir, "02_located_crop.jpg", crop)


def save_line_masks(debug_dir, gray, h_lines, v_lines):
    """Overlay horizontal (red) and vertical (blue) line masks on the image."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    overlay = vis.copy()
    overlay[h_lines > 0] = (0, 0, 255)  # red for horizontal
    overlay[v_lines > 0] = (255, 0, 0)  # blue for vertical
    blended = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
    _save(debug_dir, "03_line_masks.jpg", blended)


def save_binarized(debug_dir, binarized):
    _save(debug_dir, "04_binarized.jpg", binarized)


def save_dewarped(debug_dir, dewarped):
    _save(debug_dir, "05_dewarped_crop.jpg", dewarped)


def save_ocr_tesseract(debug_dir, image, words):
    vis = _draw_words(image, words, color=(0, 200, 0))
    _save(debug_dir, "06_ocr_tesseract.jpg", vis)


def save_ocr_rapidocr(debug_dir, image, words):
    vis = _draw_words(image, words, color=(200, 0, 200))
    _save(debug_dir, "07_ocr_rapidocr.jpg", vis)


def save_parsed_values(debug_dir, image, reading):
    vis = _draw_parsed(image, reading)
    _save(debug_dir, "08_parsed_values.jpg", vis)


def save_summary(debug_dir, result):
    """Write a JSON summary of classification, features, and parsed values."""
    from dataclasses import replace
    from rules.anvisa import Nutrient, Basis, LIMITS, WARNING_TEXT, assess

    debug_dir = Path(debug_dir)
    basis = result.reading.facts.basis or Basis.SOLID
    assessment = (
        assess(replace(result.reading.facts, basis=basis)) if result.readable else None
    )
    limits = LIMITS.get(basis, {})

    anvisa_flags = {
        "basis": basis.value if basis else None,
        "warnings_triggered": (
            [WARNING_TEXT[n] for n in assessment.required] if assessment else []
        ),
        "nutrients": {
            n.value: {
                "amount": (result.reading.facts.amount(n) if result.readable else None),
                "unit": "mg" if n is Nutrient.SODIUM else "g",
                "threshold": limits.get(n),
                "flagged": (n in assessment.required) if assessment else False,
                "warning": (
                    WARNING_TEXT.get(n)
                    if assessment and n in assessment.required
                    else None
                ),
            }
            for n in Nutrient
        },
    }

    data = {
        "photo_type": result.photo_type.value,
        "readable": result.readable,
        "features": {k: v for k, v in vars(result.features).items()},
        "facts": {
            "basis": (
                result.reading.facts.basis.value if result.reading.facts.basis else None
            ),
            "added_sugars_g": result.reading.facts.added_sugars_g,
            "saturated_fat_g": result.reading.facts.saturated_fat_g,
            "sodium_mg": result.reading.facts.sodium_mg,
        },
        "anvisa_flags": anvisa_flags,
        "labels_found": [n.value for n in result.reading.labels],
        "matches": {
            n.value: {"row_text": row, "value": val}
            for n, (row, val) in result.reading.matches.items()
        },
        "headers": [
            {"cx": h.cx, "cy": h.cy, "basis": h.basis.value}
            for h in result.reading.headers
        ],
        "table": (
            {
                "box": result.table.box,
                "angle": result.table.angle,
                "text_height": result.table.text_height,
                "inverted": result.table.inverted,
                "ruled_rows": result.table.ruled_rows,
                "text_blobs": result.table.text_blobs,
            }
            if result.table
            else None
        ),
    }
    path = debug_dir / "summary.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def update_summary_evaluation(
    debug_dir,
    result,
    truth,
    source,
    true_type,
    read_assessment,
    true_assessment,
    matches_func,
    tolerance_dict,
):
    """Enrich summary.json with ground truth comparison and ANVISA flag evaluation."""
    from rules.anvisa import Nutrient, LIMITS, WARNING_TEXT, Basis

    debug_dir = Path(debug_dir)
    path = debug_dir / "summary.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    else:
        data = {}

    if truth is None:
        data["evaluation"] = {
            "ground_truth_available": False,
            "source": source,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return

    read = result.reading.facts if result.readable else None
    basis = (
        truth.basis
        if (source == "hand" and truth.basis)
        else (read.basis if read else None) or truth.basis or Basis.SOLID
    )
    limits = LIMITS.get(basis, {})

    all_decisions_correct = True
    all_values_correct = True
    nutrients_eval = {}

    for nutrient in Nutrient:
        name = nutrient.value
        read_val = read.amount(nutrient) if read else None
        truth_val = truth.amount(nutrient)
        unit = "mg" if nutrient is Nutrient.SODIUM else "g"
        threshold = limits.get(nutrient)
        limit_tol = tolerance_dict.get(nutrient, 0.1)

        val_correct = None
        diff = None
        tolerance = None
        if truth_val is not None:
            tolerance = round(max(0.05 * abs(truth_val), limit_tol), 3)
            if read_val is not None:
                diff = round(read_val - truth_val, 3)
                val_correct = bool(matches_func(read_val, truth_val, nutrient))
            else:
                val_correct = False
            if not val_correct:
                all_values_correct = False

        # ANVISA warning flag evaluation:
        read_flagged = (
            bool(nutrient in read_assessment.required) if read_assessment else False
        )
        truth_flagged = (
            bool(nutrient in true_assessment.required) if true_assessment else None
        )

        flag_correct = None
        flag_status = "unknown"
        if true_assessment and (nutrient not in true_assessment.unknown):
            flag_correct = read_flagged == truth_flagged
            if not flag_correct:
                all_decisions_correct = False
            if read_flagged and truth_flagged:
                flag_status = "true_positive"
            elif not read_flagged and not truth_flagged:
                flag_status = "true_negative"
            elif read_flagged and not truth_flagged:
                flag_status = "false_positive"
            elif not read_flagged and truth_flagged:
                flag_status = "false_negative"
        elif truth_val is None:
            flag_status = "unlabeled"

        # Context details:
        match_info = result.reading.matches.get(nutrient)
        row_text = match_info[0] if match_info else None
        label_found = nutrient in result.reading.labels

        nutrients_eval[name] = {
            "read_value": read_val,
            "truth_value": truth_val,
            "unit": unit,
            "value_correct": val_correct,
            "difference": diff,
            "tolerance": tolerance,
            "threshold": threshold,
            "model_flagged": read_flagged,
            "truth_flagged": truth_flagged,
            "flag_decision_correct": flag_correct,
            "flag_status": flag_status,
            "warning_text": (
                WARNING_TEXT.get(nutrient) if (read_flagged or truth_flagged) else None
            ),
            "context": {
                "row_text": row_text,
                "label_found": label_found,
                "basis_applied": (
                    basis.value if hasattr(basis, "value") else str(basis)
                ),
            },
        }

    read_warnings = (
        [WARNING_TEXT[n] for n in read_assessment.required] if read_assessment else []
    )
    truth_warnings = (
        [WARNING_TEXT[n] for n in true_assessment.required] if true_assessment else []
    )

    data["evaluation"] = {
        "ground_truth_available": True,
        "source": source,
        "photo_type": {
            "predicted": result.photo_type.value,
            "truth": true_type.value if true_type else None,
            "correct": (bool(result.photo_type == true_type) if true_type else None),
        },
        "all_flag_decisions_correct": all_decisions_correct,
        "all_values_correct": all_values_correct,
        "warnings": {
            "model_flagged_warnings": read_warnings,
            "truth_flagged_warnings": truth_warnings,
            "warnings_match": (set(read_warnings) == set(truth_warnings)),
        },
        "nutrients": nutrients_eval,
    }

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
