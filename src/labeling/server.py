"""Hand-labeling tool for both photos of each product.

/       back: the nutrition photo, and the values read by a person (added
        sugars, saturated fat, sodium per 100 g/ml) as the ground truth for
        the back pipeline, replacing the noisy Open Food Facts values.
/front  front: which "ALTO EM" warnings the front photo shows, and a box
        around each warning panel, as the ground truth for the front
        pipeline (and training data for a detector).

    python src/labeling/server.py [--labeler <name>] [--port 8765]

Labels are appended to labels/{back,front}/<name>.jsonl (default:
hand-labeled.jsonl) as they are saved, so closing the page loses nothing.
See LABELING.md.
"""

import argparse
import json
import random
import re
import sys
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import pandas as pd

from labeling.labels import DEFAULT_LABELER, load_labels
from utils.constants import (
    BACK_LABELS_DIR,
    DATA_DIR,
    FILTER_PAIRS_OUTPUT_FILE,
    FRONT_LABELS_DIR,
    IMAGES_OUTPUT_DIR,
)
from utils.logger import Logger
from vision.back.locate import locate_table

log = Logger("labeling")

PAGE = Path(__file__).parent / "page.html"
FRONT_PAGE = Path(__file__).parent / "front.html"
FRONT_STATUSES = {"ok", "unreadable", "not_front"}
EVAL_RESULTS = DATA_DIR / "evaluation" / "latest" / "back_dev.csv"
NUTRIENTS = ("added_sugars", "saturated_fat", "sodium")


def build_queues():
    products = pd.read_parquet(FILTER_PAIRS_OUTPUT_FILE)
    products = products[products.split != "unlabeled"]

    off = {
        row.code: {
            "added_sugars": row.off_added_sugars_100g,
            "saturated_fat": row.off_saturated_fat_100g,
            # OFF stores sodium in grams.
            "sodium": row.off_sodium_100g * 1000,
        }
        for row in products.itertuples()
    }

    queues = {"val": sorted(products[products.split == "val"].code)}

    # Dev products where the pipeline disagrees with OFF on some nutrient:
    # either OFF or the pipeline is wrong, a person settles it.
    if EVAL_RESULTS.exists():
        results = pd.read_csv(EVAL_RESULTS, dtype={"code": str})
        disagree = ~results[[f"{n}_correct" for n in NUTRIENTS]].all(axis=1)
        queues["dev_disagreements"] = shuffled(results[disagree].code)
    queues["dev"] = shuffled(products[products.split == "dev"].code)

    # Upload time of each nutrition photo, to find labels made on an older one.
    photo_times = dict(zip(products.code, products.nutrition_uploaded_t))

    return queues, off, photo_times


def shuffled(codes):
    """Fixed random order: barcodes cluster by manufacturer, so a sorted
    queue would make any partly labeled prefix a biased sample."""
    codes = sorted(codes)
    random.Random(0).shuffle(codes)
    return codes


REVIEW_NOTES = ("100g header", "100 g header", "per portion", "por porção")


def latest_photo_types():
    """{barcode: photo type} from the most recent evaluation that classified
    photos, or {} if there is none."""
    runs = [
        path
        for path in (DATA_DIR / "evaluation" / "latest").glob("back_*.csv")
        if "photo_type" in path.open().readline().split(",")
    ]
    if not runs:
        return {}
    latest = max(runs, key=lambda path: path.stat().st_mtime)
    results = pd.read_csv(latest, dtype={"code": str}, usecols=["code", "photo_type"])
    return dict(zip(results.code, results.photo_type))


def review_queue(labels, photo_times=None):
    """{barcode: reason} for labels to check again:
    - saved before the product's nutrition photo was uploaded: the photo was
      replaced after labeling (photo_times: {barcode: upload time, epoch s});
    - readable without a basis marked: a 100 g/ml column must be there, else
      it's old format;
    - a note doubting the 100 g header;
    - labeled "no table" where the pipeline found and read a table."""
    readable = {"ok", "partial", "linear"}
    photo_types = latest_photo_types()
    reasons = {}

    for code, label in labels.items():
        status, note = label.get("status"), (label.get("note") or "").lower()
        uploaded = (photo_times or {}).get(code)
        # saved_at is the labeler's local time; uploads are hours or days apart.
        if uploaded and label.get("saved_at", "") < time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(uploaded)
        ):
            reasons[code] = "photo replaced after it was labeled"
        elif status in readable and not label.get("basis"):
            reasons[code] = "readable, but no 100 g / 100 ml marked"
        elif status in readable and any(n in note for n in REVIEW_NOTES):
            reasons[code] = "note doubts the 100 g header"
        elif status == "no_table" and photo_types.get(code) in ("table", "linear"):
            reasons[code] = (
                f"labeled no table, but the pipeline read a {photo_types[code]}"
            )

    # Replaced photos first: the label is surely stale, and the other reasons
    # can list hundreds of labels.
    replaced = "photo replaced after it was labeled"
    order = sorted(reasons.items(), key=lambda item: (item[1] != replaced, item[0]))
    return dict(order)


def valid_code(code):
    return code.isdigit() and len(code) <= 20


def valid_front_label(entry):
    """status, one bool per warning, boxes as [x, y, width, height] in pixels
    of the original photo."""
    boxes = entry.get("boxes")
    return (
        entry.get("status") in FRONT_STATUSES
        and all(isinstance(entry.get(n), bool) for n in NUTRIENTS)
        and isinstance(boxes, list)
        and all(
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(v, (int, float)) and v >= 0 for v in box)
            for box in boxes
        )
    )


@lru_cache(maxsize=64)
def table_crop(code):
    image = cv2.imread(str(IMAGES_OUTPUT_DIR / "nutrition" / f"{code}.jpg"))
    if image is None:
        return None
    crop = locate_table(image).image
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes() if ok else None


class Handler(BaseHTTPRequestHandler):
    queues, off, photo_times, labels_file, front_labels_file = (None,) * 5

    def log_message(self, *args):
        pass

    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        self.send(status, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        parts = url.path.strip("/").split("/")

        if url.path == "/":
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/front":
            return self.send(200, FRONT_PAGE.read_bytes(), "text/html; charset=utf-8")

        if url.path == "/api/front/queue":
            name = parse_qs(url.query).get("name", ["val"])[0]
            labels = load_labels(FRONT_LABELS_DIR)
            # Front queues: the same products as the back ones, no review.
            names = [q for q in self.queues if q != "dev_disagreements"]
            items = [
                {"code": c, "label": labels.get(c)} for c in self.queues.get(name, [])
            ]
            return self.send_json({"queues": names, "items": items})

        if url.path == "/api/queue":
            name = parse_qs(url.query).get("name", ["val"])[0]
            labels = load_labels()
            # Computed per request: fixed labels drop out right away.
            reasons = review_queue(labels, self.photo_times) if name == "review" else {}
            codes = list(reasons) if name == "review" else self.queues.get(name, [])
            items = [
                {
                    "code": c,
                    "off": self.off.get(c),
                    "label": labels.get(c),
                    "reason": reasons.get(c),
                }
                for c in codes
            ]
            return self.send_json({"queues": [*self.queues, "review"], "items": items})

        if len(parts) == 3 and parts[0] == "img" and parts[2].endswith(".jpg"):
            kind, code = parts[1], parts[2][:-4]
            if not valid_code(code):
                return self.send(400, b"bad code", "text/plain")
            if kind == "table":
                body = table_crop(code)
            elif kind in ("photo", "front"):
                side = "nutrition" if kind == "photo" else "front"
                path = IMAGES_OUTPUT_DIR / side / f"{code}.jpg"
                body = path.read_bytes() if path.exists() else None
            else:
                body = None
            if body is None:
                return self.send(404, b"not found", "text/plain")
            return self.send(200, body, "image/jpeg")

        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        files = {
            "/api/label": self.labels_file,
            "/api/front/label": self.front_labels_file,
        }
        if self.path not in files:
            return self.send(404, b"not found", "text/plain")

        entry = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not valid_code(str(entry.get("code", ""))):
            return self.send_json({"error": "bad code"}, 400)
        if self.path == "/api/front/label" and not valid_front_label(entry):
            return self.send_json({"error": "bad front label"}, 400)

        entry["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        labels_file = files[self.path]
        labels_file.parent.mkdir(parents=True, exist_ok=True)
        with labels_file.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.send_json({"ok": True})


def main():
    parser = argparse.ArgumentParser(description="Hand-label nutrition values.")
    parser.add_argument(
        "--labeler",
        default=DEFAULT_LABELER,
        help=f"label file name (default: {DEFAULT_LABELER})",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    labeler = re.sub(r"[^A-Za-z0-9_-]", "_", args.labeler)

    Handler.queues, Handler.off, Handler.photo_times = build_queues()
    Handler.labels_file = BACK_LABELS_DIR / f"{labeler}.jsonl"
    Handler.front_labels_file = FRONT_LABELS_DIR / f"{labeler}.jsonl"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log.info(f"Labeling tool at http://localhost:{args.port} (front: /front)")
    log.info(f"Labels saved to {Handler.labels_file}")
    server.serve_forever()


if __name__ == "__main__":
    main()
