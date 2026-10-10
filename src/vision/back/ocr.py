"""OCR of the located nutrition table with Tesseract.

Classical preprocessing prepares the crop for the OCR engine: scale the text
to the height Tesseract reads best, binarize, and erase the ruled lines,
which Tesseract otherwise reads as characters or uses to split words.

Tried and dropped: reading the table row by row, with rows cut along the
ruled lines. Curved or wrinkled packages bend the lines, rows get merged,
and the decision accuracy on dev fell from 27% to 12%.
"""

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

    ink = binarize(gray)
    h_lines, v_lines = line_masks(ink, TARGET_TEXT_HEIGHT)
    lines = cv2.dilate(h_lines | v_lines, np.ones((3, 3), np.uint8))
    ink[lines > 0] = 0

    # Drop specks smaller than a period.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return 255 - ink


def read_words(image, preprocess=True, lang="por") -> list[Word]:
    if preprocess:
        image = prepare(image)

    data = pytesseract.image_to_data(
        image, lang=lang, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT
    )

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
