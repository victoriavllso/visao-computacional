"""Photos taken with the package on its side or upside down.

The parser reads a table row by row, left to right: on a rotated photo it
takes the nutrients for a row and the numbers of the wrong row (on dev, 7 of
the 17 products the back made the app wrongly call non-compliant). Two
classical cues, cheapest first:

    Tesseract OSD   orientation of the whole photo (0/90/180/270), with a
                    confidence; it misses small tables in large photos
    text lines      most lines read by the learned OCR are taller than wide:
                    the text runs vertically, the table is on its side; the
                    direction (90 or 270) is the one the parser reads best
"""

import cv2
import pytesseract

# Degrees to turn the photo clockwise so that it is upright (OSD's "rotate").
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
# On 40 upright dev photos OSD answered 0, or another angle with confidence
# below 1; on the sideways ones it was right whenever confident (2 to 10).
MIN_OSD_CONFIDENCE = 2.0
# Downscaled for speed; a higher resolution didn't find more rotations.
OSD_MAX_SIDE = 2600
# A text line is vertical when its box is clearly taller than wide; only
# lines of a few characters count (a lone digit is as tall as it is wide).
TALL_RATIO = 1.5
MIN_LINE_CHARS = 4
MIN_VERTICAL_SHARE = 0.5
MIN_LINES = 5


def rotate(image, degrees):
    return cv2.rotate(image, ROTATIONS[degrees]) if degrees else image


def osd_rotation(image):
    """Clockwise degrees that make the photo upright, when Tesseract is
    confident; 0 otherwise (or when there is too little text)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = OSD_MAX_SIDE / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        osd = pytesseract.image_to_osd(
            gray, output_type=pytesseract.Output.DICT, config="--psm 0"
        )
    except pytesseract.TesseractError:
        return 0
    if osd["orientation_conf"] < MIN_OSD_CONFIDENCE:
        return 0
    return osd["rotate"] if osd["rotate"] in ROTATIONS else 0


def vertical_text(words):
    """Whether most text lines run vertically (Word boxes from the learned
    OCR, which reads them anyway)."""
    lines = [w for w in words if len(w.text.strip()) >= MIN_LINE_CHARS]
    if len(lines) < MIN_LINES:
        return False
    tall = sum(1 for w in lines if w.h > TALL_RATIO * w.w)
    return tall / len(lines) >= MIN_VERTICAL_SHARE
