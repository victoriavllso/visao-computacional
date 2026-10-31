import hashlib
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from utils.constants import (
    FILTER_BRAZIL_OUTPUT_FILE,
    FILTER_PAIRS_OUTPUT_DIR,
    FILTER_PAIRS_OUTPUT_FILE,
    FRONT_IMAGE_KEY,
    NUTRITION_IMAGE_KEY,
    REGULATION_CUTOFF,
    IMAGE_END_CUTOFF,
    GROUND_TRUTH_NUTRIENTS,
    VAL_FRACTION,
)
from utils.logger import Logger

log = Logger("filter")

COLUMNS = [
    "code",
    "product_name",
    "brands",
    "categories_tags",
    "quantity",
    "serving_size",
    "nutrition_data_per",
    "images",
    "nutriments",
]


def get_selected_image(images, image_key):
    """Return the selected image entry plus the upload time of its raw photo.

    Selected entries (e.g. front_pt) carry no uploaded_t; it lives on the raw
    upload they point to through imgid.
    """
    if images is None:
        return None

    raw = {image["key"]: image for image in images if image["key"].isdigit()}

    for image in images:
        if image["key"] != image_key or image["imgid"] is None:
            continue

        raw_image = raw.get(str(image["imgid"]))
        if raw_image is None or raw_image["uploaded_t"] is None:
            return None

        return {
            "imgid": image["imgid"],
            "rev": image["rev"],
            "uploaded_t": raw_image["uploaded_t"],
        }

    return None


def get_ground_truth(nutriments):
    """Return {nutrient: value per 100 g/ml} for the ground-truth nutrients."""
    values = {name: None for name in GROUND_TRUTH_NUTRIENTS}

    for nutriment in nutriments or []:
        if nutriment["name"] in values:
            values[nutriment["name"]] = nutriment["100g"]

    return values


def get_split(code, has_ground_truth):
    """Stable dev/val split for labeled products; the rest is unlabeled."""
    if not has_ground_truth:
        return "unlabeled"

    digest = hashlib.sha256(code.encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32

    return "val" if bucket < VAL_FRACTION else "dev"


def build_pair(product, cutoff_t, end_cutoff_t):
    front = get_selected_image(product["images"], FRONT_IMAGE_KEY)
    nutrition = get_selected_image(product["images"], NUTRITION_IMAGE_KEY)

    if front is None or nutrition is None:
        return None

    if min(front["uploaded_t"], nutrition["uploaded_t"]) < cutoff_t:
        return None

    if max(front["uploaded_t"], nutrition["uploaded_t"]) >= end_cutoff_t:
        return None

    truth = get_ground_truth(product["nutriments"])
    has_ground_truth = all(value is not None for value in truth.values())
    code = str(product["code"]).strip()

    pair = {key: product[key] for key in COLUMNS if key not in ("images", "nutriments")}
    pair["code"] = code

    for prefix, image in (("front", front), ("nutrition", nutrition)):
        pair[f"{prefix}_imgid"] = image["imgid"]
        pair[f"{prefix}_rev"] = image["rev"]
        pair[f"{prefix}_uploaded_t"] = image["uploaded_t"]

    for name, value in truth.items():
        pair[f"off_{name.replace('-', '_')}_100g"] = value

    pair["has_ground_truth"] = has_ground_truth
    pair["split"] = get_split(code, has_ground_truth)

    return pair


def filter_pairs():
    if FILTER_PAIRS_OUTPUT_FILE.exists():
        log.skip("pairs filter already completed")
        log.blank()
        return

    cutoff_t = (
        datetime.fromisoformat(REGULATION_CUTOFF)
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )

    end_cutoff_t = (
        datetime.fromisoformat(IMAGE_END_CUTOFF)
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )

    products = pq.read_table(FILTER_BRAZIL_OUTPUT_FILE, columns=COLUMNS).to_pylist()
    pairs = [
        pair
        for product in products
        if (pair := build_pair(product, cutoff_t, end_cutoff_t))
    ]

    FILTER_PAIRS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(pairs), FILTER_PAIRS_OUTPUT_FILE)

    splits = {}
    for pair in pairs:
        splits[pair["split"]] = splits.get(pair["split"], 0) + 1

    log.info(f"Products in Brazil dataset : {len(products):,}")
    log.info(
        f"With {FRONT_IMAGE_KEY} + {NUTRITION_IMAGE_KEY} since {REGULATION_CUTOFF}: {len(pairs):,}"
    )
    for split, count in sorted(splits.items()):
        log.info(f"  {split:<10}: {count:,}")
    log.info(f"Output written to: {FILTER_PAIRS_OUTPUT_FILE}")
