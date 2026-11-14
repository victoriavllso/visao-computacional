"""OCR of the located nutrition table with Tesseract.

Classical preprocessing prepares the crop for the OCR engine: scale the text
to the height Tesseract reads best, binarize, and erase the ruled lines,
which Tesseract otherwise reads as characters or uses to split words.

Tried and dropped: reading the table row by row, with rows cut along the
ruled lines. Curved or wrinkled packages bend the lines, rows get merged,
and the decision accuracy on dev fell from 27% to 12%.
"""

import os

import cv2
import numpy as np
import pytesseract

from vision.back.locate import binarize, estimate_text_height, line_masks
from vision.back.parse import Word

TARGET_TEXT_HEIGHT = 30  # pixels; best of 24, 30, 40, 50 on dev
MAX_SIDE = 4000
# Sparse text: table cells, not paragraphs. Best of psm 3, 4, 6, 11 and 12
# on dev.
TESSERACT_CONFIG = "--psm 11"
# Noisy images (paper texture binarized into specks) can take Tesseract
# minutes; past this, the table is treated as unreadable.
TIMEOUT_SECONDS = 20


# Photometric corrections before binarization. Measured on the full dev set
# and off by default: none improved the decisions (normalization −0.3 pts,
# Sauvola −1.5, both +0.3, CLAHE −6). Kept to rerun the comparison.
NORMALIZE_ILLUMINATION = False
LOCAL_CONTRAST = False  # CLAHE
BINARIZATION = "mean"  # "mean" (adaptive mean) or "sauvola"
SAUVOLA_K, SAUVOLA_R = 0.2, 128


def normalize_illumination(gray, text_height):
    """Divide by the background brightness: flattens shadows, glare gradients
    and creases. The background is estimated by closing (removes dark text
    thinner than the kernel) and blurring."""
    size = int(2 * text_height) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (size, size), 0)
    normalized = gray.astype(np.float32) / np.maximum(background, 1) * 255
    return np.clip(normalized, 0, 255).astype(np.uint8)


def local_contrast(gray, text_height):
    """CLAHE with tiles of a few text lines: lifts faint text inside glare."""
    tiles = max(2, int(max(gray.shape) / (8 * text_height)))
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tiles, tiles)).apply(gray)


def sauvola(gray, window):
    """Sauvola thresholding: ink only where there is local contrast, so flat
    textured backgrounds don't turn into specks. Ink → 255."""
    gray = gray.astype(np.float64)
    mean = cv2.boxFilter(gray, -1, (window, window))
    mean_sq = cv2.boxFilter(gray * gray, -1, (window, window))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    threshold = mean * (1 + SAUVOLA_K * (std / SAUVOLA_R - 1))
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def prepare(image):
    """Grayscale crop → black text on white, text ~30 px tall, no lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    text_height = estimate_text_height(binarize(gray))
    scale = TARGET_TEXT_HEIGHT / text_height
    scale = min(scale, MAX_SIDE / max(gray.shape))
    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )

    if NORMALIZE_ILLUMINATION:
        gray = normalize_illumination(gray, TARGET_TEXT_HEIGHT)
    if LOCAL_CONTRAST:
        gray = local_contrast(gray, TARGET_TEXT_HEIGHT)

    if BINARIZATION == "sauvola":
        ink = sauvola(gray, int(2 * TARGET_TEXT_HEIGHT) | 1)
    else:
        ink = binarize(gray)
    h_lines, v_lines = line_masks(ink, TARGET_TEXT_HEIGHT)
    lines = cv2.dilate(h_lines | v_lines, np.ones((3, 3), np.uint8))
    ink[lines > 0] = 0

    # Drop specks smaller than a period.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return 255 - ink


# Single line, full alphabet: a digits whitelist turns junk such as '*:'
# into digits ('88'), while the parser already skips junk words.
ROW_CONFIG = "--psm 7"


def read_row(clean, x0, y0, x1, y1, lang="por") -> list[Word]:
    """Re-read one row's value region as a single text line."""
    h, w = clean.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return []

    pad = TARGET_TEXT_HEIGHT // 2
    strip = cv2.copyMakeBorder(
        clean[y0:y1, x0:x1], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
    )
    try:
        data = pytesseract.image_to_data(
            strip,
            lang=lang,
            config=ROW_CONFIG,
            output_type=pytesseract.Output.DICT,
            timeout=TIMEOUT_SECONDS,
        )
    except RuntimeError:
        return []

    return [
        Word(text, x + x0 - pad, y + y0 - pad, ww, hh, float(conf))
        for text, x, y, ww, hh, conf in zip(
            data["text"],
            data["left"],
            data["top"],
            data["width"],
            data["height"],
            data["conf"],
        )
        if text.strip() and float(conf) >= 0
    ]


# One RapidOCR engine per model size, created on first use.
_rapidocr = {}
_threads_limited = False


def _limit_onnx_threads():
    """RapidOCR doesn't expose ONNX Runtime's thread count, and by default
    each process uses every core. ORT_THREADS (e.g. 1 per worker when
    evaluating in parallel) is applied to the sessions it creates."""
    global _threads_limited
    threads = int(os.environ.get("ORT_THREADS", 0))
    if not threads or _threads_limited:
        return
    _threads_limited = True

    import onnxruntime

    original = onnxruntime.SessionOptions

    def session_options():
        options = original()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        return options

    onnxruntime.SessionOptions = session_options


def _engine(model):
    """PP-OCRv6 detection and recognition models: "small" (RapidOCR's
    default) or "medium" (~4.7x slower per call on one thread, 12 dev
    crops)."""
    if model not in _rapidocr:
        _limit_onnx_threads()
        import logging

        from rapidocr import ModelType, RapidOCR

        if model == "small":
            engine = RapidOCR()
        elif model == "medium":
            engine = RapidOCR(
                params={
                    "Det.model_type": ModelType.MEDIUM,
                    "Rec.model_type": ModelType.MEDIUM,
                }
            )
        else:
            raise ValueError(f"unknown RapidOCR model: {model}")
        _rapidocr[model] = engine

        # Set RapidOCR's logger to ERROR level to avoid excessive logging output
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    return _rapidocr[model]


def rapidocr_lines(image, model="small") -> list:
    """RapidOCR's raw output as plain lists, [[box, text, score], ...], box
    being the four corners [[x, y], ...] of the text line (it may be
    tilted). Plain data, so it can be cached (vision/back/cache.py)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output = _engine(model)(image)
    # No text found: boxes is None.
    if output.boxes is None:
        return []
    return [
        [[[float(x), float(y)] for x, y in box], text, float(score)]
        for box, text, score in zip(output.boxes, output.txts, output.scores)
    ]


def words_from_lines(lines) -> list[Word]:
    """One Word per RapidOCR text line, on the line's upright bounding box."""
    words = []
    for box, text, score in lines:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        x, y = int(min(xs)), int(min(ys))
        words.append(
            Word(text, x, y, int(max(xs)) - x, int(max(ys)) - y, 100 * float(score))
        )
    return words


def read_words_rapidocr(image, model="small") -> list[Word]:
    """Learned OCR (PaddleOCR models through RapidOCR / ONNX Runtime), on the
    color crop: it is trained on photos, binarizing first makes it worse.
    Returns one Word per detected text line, e.g. 'Gorduras saturadas (g)'."""
    return words_from_lines(rapidocr_lines(image, model))


def read_words(image, preprocess=True, lang="por") -> list[Word]:
    if preprocess:
        image = prepare(image)

    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
            timeout=TIMEOUT_SECONDS,
        )
    except RuntimeError:  # pytesseract's timeout
        return []

    return [
        Word(text, x, y, w, h, float(conf))
        for text, x, y, w, h, conf in zip(
            data["text"],
            data["left"],
            data["top"],
            data["width"],
            data["height"],
            data["conf"],
        )
        if text.strip() and float(conf) >= 0
    ]
