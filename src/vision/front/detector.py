"""Locate the "ALTO EM" warning panels on the front photo with a YOLO
detector, exported to ONNX so it runs on ONNX Runtime (already used by the
OCR): ultralytics and torch are only needed to train it
(train_detector.py, pixi environment `detector`).

One class: the warning panel (the black box with the magnifier, "ALTO EM"
and up to three nutrients), as boxed in the front labels.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.constants import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "front_detector.onnx"
# Low on purpose: the OCR of the crop decides, so a doubtful box costs one
# OCR call, a missed one a missed warning (dev cross-validation, README).
MIN_SCORE = 0.1
NMS_IOU = 0.5
# Dev photos are split into folds by barcode for cross-validation: each
# photo is scored by the model of the fold that didn't train on it.
FOLDS = 5


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float


def fold_of(code, folds=FOLDS):
    """Stable fold of a barcode (Python's hash() changes between runs)."""
    return int(hashlib.md5(str(code).encode()).hexdigest(), 16) % folds


class Detector:
    def __init__(self, path=MODEL_PATH):
        import onnxruntime

        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path}: train the detector first (README, Front pipeline)"
            )
        options = onnxruntime.SessionOptions()
        threads = int(os.environ.get("ORT_THREADS", 0))
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[2]  # square, fixed at export

    def letterbox(self, image):
        """Resize keeping the aspect ratio and pad to a square, as in
        training (ultralytics' LetterBox, centered, gray 114)."""
        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, scale, left, top

    def __call__(self, image, min_score=MIN_SCORE) -> list[Box]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        canvas, scale, left, top = self.letterbox(image)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        output = self.session.run(None, {self.input.name: blob})[0][0]
        return self.decode(output, scale, left, top, image.shape, min_score)

    @staticmethod
    def decode(output, scale, left, top, shape, min_score):
        if output.shape[-1] == 6:
            # End-to-end export (YOLO26): rows of x1, y1, x2, y2, score, class,
            # already without duplicates.
            x1, y1, x2, y2, scores = output[:, :5].T
        else:
            # Classic head: (4 + classes, anchors) of cx, cy, w, h, scores.
            cx, cy, w, h = output[:4]
            scores = output[4:].max(axis=0)
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        keep = scores >= min_score
        x1, y1, x2, y2, scores = (v[keep] for v in (x1, y1, x2, y2, scores))
        if output.shape[-1] != 6 and len(scores):
            rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
            kept = cv2.dnn.NMSBoxes(rects, scores.tolist(), min_score, NMS_IOU)
            kept = np.array(kept, dtype=int).reshape(-1)
            x1, y1, x2, y2, scores = (v[kept] for v in (x1, y1, x2, y2, scores))

        height, width = shape[:2]
        boxes = []
        for a, b, c, d, s in zip(x1, y1, x2, y2, scores):
            a = int(np.clip((a - left) / scale, 0, width))
            b = int(np.clip((b - top) / scale, 0, height))
            c = int(np.clip((c - left) / scale, 0, width))
            d = int(np.clip((d - top) / scale, 0, height))
            if c > a and d > b:
                boxes.append(Box(a, b, c - a, d - b, float(s)))
        return sorted(boxes, key=lambda box: -box.score)


_detectors = {}


def detect_panels(image, model=MODEL_PATH, min_score=MIN_SCORE) -> list[Box]:
    """Warning panels on the photo, best first (one model per process)."""
    model = str(model)
    if model not in _detectors:
        _detectors[model] = Detector(model)
    return _detectors[model](image, min_score)
