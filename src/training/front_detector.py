"""Train the warning panel detector (YOLO, ultralytics) on the boxes of the
front labels, and export it to ONNX for detector.py.

Only dev photos train it; val stays for the final score. To score the
detector and YOLO+OCR on dev without testing on training photos, dev is
split into FOLDS folds by barcode: the model of fold k trains on the other
folds and is scored on fold k (cross-validation), so every dev photo gets a
prediction from a model that didn't see it. The final model ("all") trains
on all of dev with the same settings and is the one used on val.

No model selection on the held-out photos: each model trains a fixed number
of epochs and keeps the last weights; the held-out fold is only scored.

    pixi run -e detector train-detector [--folds 0 1 2 3 4 all]
        [--model yolo26n.pt] [--imgsz 1024] [--epochs 100] [--device 0]

Writes the dataset and runs to data/detector/, the fold models to
data/detector/<tag>/fold<k>.onnx and the final model to
models/front_detector.onnx.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import pandas as pd

from labeling.labels import load_labels
from utils.constants import (
    FILTER_PAIRS_OUTPUT_FILE,
    FRONT_LABELS_DIR,
    IMAGES_OUTPUT_DIR,
)
from utils.logger import Logger
from vision.front.detector import DETECTOR_DIR, FOLDS, MODEL_PATH, fold_of

log = Logger("train detector")

DATASET_DIR = DETECTOR_DIR / "dataset"
# Training images are re-encoded at this size: enough for imgsz up to 1280,
# and it bakes the EXIF orientation in (the boxes were drawn on the
# oriented photo).
MAX_SIDE = 1280
NUTRIENTS = ("added_sugars", "saturated_fat", "sodium")
SEED = 0


def training_photos():
    """Dev photos usable to train: labeled as the front (readable or not),
    and not a warning left without boxes (it would teach "no panel")."""
    products = pd.read_parquet(FILTER_PAIRS_OUTPUT_FILE)
    dev = set(products[products.split == "dev"].code)
    photos = {}
    for code, label in load_labels(FRONT_LABELS_DIR).items():
        if code not in dev or label["status"] == "not_front":
            continue
        boxes = label.get("boxes") or []
        if any(label[n] for n in NUTRIENTS) and not boxes:
            continue
        photos[code] = boxes
    return photos


def write_dataset(photos):
    """YOLO layout: images/<code>.jpg and labels/<code>.txt (class cx cy w h,
    normalized). Photos without boxes are negatives (empty label file)."""
    (DATASET_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels").mkdir(parents=True, exist_ok=True)
    for code, boxes in photos.items():
        target = DATASET_DIR / "images" / f"{code}.jpg"
        image = cv2.imread(str(IMAGES_OUTPUT_DIR / "front" / f"{code}.jpg"))
        height, width = image.shape[:2]
        if not target.exists():
            scale = min(MAX_SIDE / max(height, width), 1.0)
            small = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(str(target), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for x, y, w, h in boxes:
            x, y = max(x, 0), max(y, 0)
            w, h = min(w, width - x), min(h, height - y)
            if w > 0 and h > 0:
                lines.append(
                    f"0 {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                    f"{w / width:.6f} {h / height:.6f}"
                )
        label = DATASET_DIR / "labels" / f"{code}.txt"
        if not label.exists() or label.read_text() != "\n".join(lines):
            label.write_text("\n".join(lines))


def write_split(name, train, held_out):
    """data.yaml for one model. `held_out` is only scored after training;
    the final model has none, so it is scored on its training photos (a
    sanity check, not an estimate)."""
    split_dir = DETECTOR_DIR / "splits" / name
    split_dir.mkdir(parents=True, exist_ok=True)
    for part, codes in (("train", train), ("val", held_out or train)):
        paths = [str(DATASET_DIR / "images" / f"{c}.jpg") for c in sorted(codes)]
        (split_dir / f"{part}.txt").write_text("\n".join(paths) + "\n")
    config = split_dir / "data.yaml"
    config.write_text(
        f"path: {split_dir}\ntrain: train.txt\nval: val.txt\nnames:\n  0: panel\n"
    )
    return config


def train(name, config, args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=SEED,
        deterministic=True,
        project=str(DETECTOR_DIR / "runs" / args.tag),
        name=name,
        exist_ok=True,
        val=False,  # no model selection: the last weights are kept
        # Mirrored text doesn't occur on packages; panels are often tilted.
        fliplr=0.0,
        degrees=10.0,
        plots=False,
        verbose=False,
        cache="ram",  # the shared machine is short of CPU for decoding
    )
    weights = DETECTOR_DIR / "runs" / args.tag / name / "weights" / "last.pt"
    model = YOLO(str(weights))
    scores = model.val(
        data=str(config), imgsz=args.imgsz, device=args.device, plots=False
    )
    onnx = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    target = DETECTOR_DIR / args.tag / f"{name}.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, target)
    return {
        "map50": float(scores.box.map50),
        "map50_95": float(scores.box.map),
        "precision": float(scores.box.mp),
        "recall": float(scores.box.mr),
        "onnx": str(target),
        "onnx_mb": round(target.stat().st_size / 2**20, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", nargs="+", default=[*map(str, range(FOLDS)), "all"])
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="yolo26n_1024")
    parser.add_argument(
        "--install", action="store_true", help=f"copy the final model to {MODEL_PATH}"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    photos = training_photos()
    write_dataset(photos)
    positives = sum(bool(b) for b in photos.values())
    log.info(f"{len(photos)} dev photos, {positives} with panels")

    # One JSON per model: folds can train in parallel processes.
    run_dir = DETECTOR_DIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in args.folds:
        if name == "all":
            config = write_split("all", list(photos), [])
        else:
            k = int(name)
            held_out = [c for c in photos if fold_of(c) == k]
            name = f"fold{k}"
            config = write_split(name, [c for c in photos if fold_of(c) != k], held_out)
        log.info(f"Training {name}")
        scores = {"args": vars(args), **train(name, config, args)}
        log.info(f"{name}: {scores}")
        (run_dir / f"{name}.json").write_text(json.dumps(scores, indent=2))

    if args.install:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(run_dir / "all.onnx", MODEL_PATH)
        log.info(f"Final model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
