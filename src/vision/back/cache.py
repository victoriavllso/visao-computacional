"""OCR results stored per product, so that changes to the rules after the OCR
(parse, classify, combine, orientation choice) are scored on the same reading
without running the OCR again: a full dev evaluation from the cache takes
about a minute instead of a server run.

One JSON file per product, data/cache/back_ocr/<code>.json, holding one entry
per OCR call: Tesseract's words on the preprocessed crop, RapidOCR's lines
(text, score and the four corners of each line's box) on the crop and on the
whole photo, and Tesseract OSD's turn, for each rotation the pipeline tried.

The OCR input depends on the classical steps before it (locate, dewarp,
preprocess). An entry is used only if its input image had the same size;
any other change to those steps must bump VERSION, which empties every
entry. Each entry also keeps a hash of its input: when it differs from the
current input (same size), the entry is still used but counted as `changed`,
to tell whether crops made on another machine are identical.
"""

import hashlib
import json
import os

import numpy as np

from utils.constants import DATA_DIR

CACHE_DIR = DATA_DIR / "cache" / "back_ocr"
# Bump when the OCR input changes (locate, dewarp, preprocessing, OSD input).
VERSION = 1


def image_hash(image):
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()[:16]


class OcrCache:
    """The OCR calls of one product. `get` returns the stored result or
    computes it; `save` writes the file if anything was computed."""

    def __init__(self, code, directory=CACHE_DIR):
        self.path = directory / f"{code}.json"
        self.entries = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except json.JSONDecodeError:  # a write cut short
                data = {}
            if data.get("version") == VERSION:
                self.entries = data.get("entries", {})
        self.dirty = False
        self.hits = self.misses = self.changed = 0

    def get(self, key, image, compute):
        """compute()'s JSON-serializable result for `image`, stored under
        `key`; the stored one when it was made from an image of this size."""
        shape = list(image.shape[:2])
        digest = image_hash(image)
        entry = self.entries.get(key)
        if entry is not None and entry["shape"] == shape:
            self.hits += 1
            self.changed += entry["hash"] != digest
            return entry["value"]

        self.misses += 1
        value = compute()
        self.entries[key] = {"shape": shape, "hash": digest, "value": value}
        self.dirty = True
        return value

    def save(self):
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {"version": VERSION, "entries": self.entries}, ensure_ascii=False
            )
        )
        temporary.replace(self.path)
        self.dirty = False

    @property
    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "changed": self.changed}
