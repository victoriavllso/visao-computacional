"""Hand-labeling tool for the nutrition table values.

Shows each product's nutrition photo and saves the values read by a person
(added sugars, saturated fat, sodium per 100 g/ml) as the ground truth for
the back pipeline, replacing the noisy Open Food Facts values.

    python src/labeling/server.py --labeler <your-name> [--port 8765]

Labels are appended to labels/back/<your-name>.jsonl as they are saved, so
closing the page loses nothing. See LABELING.md.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import pandas as pd

from labeling.labels import load_labels
from utils.constants import (
    BACK_LABELS_DIR,
    DATA_DIR,
    FILTER_PAIRS_OUTPUT_FILE,
    IMAGES_OUTPUT_DIR,
)
from utils.logger import Logger
from vision.back.locate import locate_table

log = Logger("labeling")

PAGE = Path(__file__).parent / "page.html"
EVAL_RESULTS = DATA_DIR / "evaluation" / "back_dev_locate_prep.csv"
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
        queues["dev_disagreements"] = sorted(results[disagree].code)
    queues["dev"] = sorted(products[products.split == "dev"].code)

    return queues, off


def valid_code(code):
    return code.isdigit() and len(code) <= 20


@lru_cache(maxsize=64)
def table_crop(code):
    image = cv2.imread(str(IMAGES_OUTPUT_DIR / "nutrition" / f"{code}.jpg"))
    if image is None:
        return None
    crop = locate_table(image).image
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes() if ok else None


class Handler(BaseHTTPRequestHandler):
    queues, off, labels_file = None, None, None

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

        if url.path == "/api/queue":
            name = parse_qs(url.query).get("name", ["val"])[0]
            codes = self.queues.get(name, [])
            labels = load_labels()
            items = [
                {"code": c, "off": self.off.get(c), "label": labels.get(c)}
                for c in codes
            ]
            return self.send_json({"queues": list(self.queues), "items": items})

        if len(parts) == 3 and parts[0] == "img" and parts[2].endswith(".jpg"):
            kind, code = parts[1], parts[2][:-4]
            if not valid_code(code):
                return self.send(400, b"bad code", "text/plain")
            if kind == "table":
                body = table_crop(code)
            elif kind == "photo":
                path = IMAGES_OUTPUT_DIR / "nutrition" / f"{code}.jpg"
                body = path.read_bytes() if path.exists() else None
            else:
                body = None
            if body is None:
                return self.send(404, b"not found", "text/plain")
            return self.send(200, body, "image/jpeg")

        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/label":
            return self.send(404, b"not found", "text/plain")

        entry = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not valid_code(str(entry.get("code", ""))):
            return self.send_json({"error": "bad code"}, 400)

        entry["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.labels_file.parent.mkdir(parents=True, exist_ok=True)
        with self.labels_file.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.send_json({"ok": True})


def default_labeler():
    """The git user name, if set."""
    try:
        name = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return None
    return name or None


def main():
    parser = argparse.ArgumentParser(description="Hand-label nutrition values.")
    parser.add_argument(
        "--labeler", default=default_labeler(), help="your name (default: git user)"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not args.labeler:
        parser.error("pass --labeler <your-name>")
    labeler = re.sub(r"[^A-Za-z0-9_-]", "_", args.labeler)

    Handler.queues, Handler.off = build_queues()
    Handler.labels_file = BACK_LABELS_DIR / f"{labeler}.jsonl"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log.info(f"Labeling tool at http://localhost:{args.port}")
    log.info(f"Labels saved to {Handler.labels_file}")
    server.serve_forever()


if __name__ == "__main__":
    main()
