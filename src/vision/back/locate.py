"""Locate and straighten the nutrition table in a photo of the back of a
package, with classical image processing.

The new-format table (RDC 429/2020) is black text on a white background with
ruled lines. Ruled lines are much longer than any letter, so morphological
opening with long thin kernels keeps the lines and drops the text. Kernel
lengths are multiples of the estimated text height, which makes them
independent of the image resolution and of how much of the photo the table
takes up.
"""

from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIZE = 1600  # longest side of the working copy, in pixels

# Line lengths, in text heights.
HORIZONTAL_LINE = 8
VERTICAL_LINE = 4
# Gap between lines of the same table, in text heights.
TABLE_GAP = 2
DARK_BACKGROUND = 100  # median gray below this → light text on dark


@dataclass
class Table:
    image: np.ndarray  # straightened crop of the table, full resolution
    box: tuple[int, int, int, int] | None  # x, y, w, h in the original image
    angle: float  # rotation applied, in degrees
    text_height: float  # estimated in the original image, in pixels
    inverted: bool  # light text on a dark background, crop already inverted


def binarize(gray):
    """Dark ink → white, background → black."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )


def estimate_text_height(bw):
    """Median height of blobs shaped like characters."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    height = stats[1:, cv2.CC_STAT_HEIGHT]
    width = stats[1:, cv2.CC_STAT_WIDTH]
    area = stats[1:, cv2.CC_STAT_AREA]

    is_char = (
        (height >= 6)
        & (height <= bw.shape[0] / 10)
        & (width <= 3 * height)
        & (area >= 15)
    )
    return float(np.median(height[is_char])) if is_char.any() else 12.0


def line_masks(bw, text_height):
    horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(HORIZONTAL_LINE * text_height), 1)
    )
    vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, int(VERTICAL_LINE * text_height))
    )
    return (
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical),
    )


def find_table_box(h_lines, v_lines, text_height):
    """Bounding box and score (line pixels) of the connected group of lines
    with the most line pixels, among groups that have both horizontal and
    vertical lines."""
    gap = int(TABLE_GAP * text_height)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    joined = cv2.morphologyEx(h_lines | v_lines, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    best, best_score = None, 0

    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        component = labels[y : y + h, x : x + w] == i
        h_pixels = np.count_nonzero(h_lines[y : y + h, x : x + w][component])
        v_pixels = np.count_nonzero(v_lines[y : y + h, x : x + w][component])

        if h_pixels and v_pixels and h_pixels + v_pixels > best_score:
            best, best_score = (x, y, w, h), h_pixels + v_pixels

    if best is None:
        return None, 0

    # Margin, larger on top: the title row usually has no ruled line above.
    x, y, w, h = best
    image_h, image_w = h_lines.shape
    margin = int(text_height)
    x0 = max(0, x - margin)
    y0 = max(0, y - 3 * margin)
    x1, y1 = min(image_w, x + w + margin), min(image_h, y + h + margin)
    return (x0, y0, x1 - x0, y1 - y0), best_score


def skew_angle(h_lines):
    """Median angle of the horizontal ruled lines, in degrees."""
    lines = cv2.HoughLinesP(
        h_lines,
        1,
        np.pi / 720,
        threshold=80,
        minLineLength=h_lines.shape[1] // 4,
        maxLineGap=10,
    )
    if lines is None:
        return 0.0

    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0]
    ]
    angles = [a for a in angles if abs(a) < 20]
    return float(np.median(angles)) if angles else 0.0


def rotate(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# Straighten curved/creased tables along their ruled lines.
DEWARP = True
# A horizontal line must span this fraction of the table to guide dewarping.
DEWARP_LINE_SPAN = 0.4
DEWARP_DEGREE = 3


def dewarp(image):
    """Remap the table so its horizontal ruled lines become straight.

    Text-line dewarping (Ulges et al. 2004; Pugliese et al.) applied to the
    ruled lines, which are easier to detect than text lines: fit a cubic
    y = f(x) to each line, then shift every column vertically so each line
    lands on its mean height, interpolating the shift between lines.
    """
    scale = min(1.0, WORK_SIZE / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    bw = binarize(gray)
    h_lines, _ = line_masks(bw, estimate_text_height(bw))
    height, width = h_lines.shape

    n, labels, stats, _ = cv2.connectedComponentsWithStats(h_lines)
    fits = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_WIDTH] < DEWARP_LINE_SPAN * width:
            continue
        ys, xs = np.nonzero(labels == i)
        poly = np.polynomial.Polynomial.fit(xs, ys, DEWARP_DEGREE)
        columns = np.clip(np.arange(width), xs.min(), xs.max())  # no extrapolation
        curve = poly(columns)
        fits.append((float(np.median(ys)), curve - np.median(ys)))

    if len(fits) < 3:
        return image

    fits.sort(key=lambda f: f[0])
    targets = np.array([t for t, _ in fits])
    shifts = np.stack([d for _, d in fits])  # lines × columns
    if np.abs(shifts).max() > 0.15 * height:
        return image  # implausible fit

    rows = np.arange(height, dtype=np.float32)
    map_y = np.empty((height, width), np.float32)
    for x in range(width):
        map_y[:, x] = rows + np.interp(rows, targets, shifts[:, x])
    map_x = np.tile(np.arange(width, dtype=np.float32), (height, 1))

    full_h, full_w = image.shape[:2]
    map_x = cv2.resize(map_x, (full_w, full_h)) / scale
    map_y = cv2.resize(map_y, (full_w, full_h)) / scale
    return cv2.remap(
        image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def locate_table(image) -> Table:
    """Crop and straighten the nutrition table. Falls back to the whole
    image when no ruled table is found (e.g. tables without lines)."""
    scale = min(1.0, WORK_SIZE / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    bw = binarize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    text_height = estimate_text_height(bw)
    h_lines, v_lines = line_masks(bw, text_height)
    box, _ = find_table_box(h_lines, v_lines, text_height)

    if box is not None:
        x, y, w, h = (int(round(v / scale)) for v in box)
        crop = image[y : y + h, x : x + w]
        angle = skew_angle(h_lines[box[1] : box[1] + box[3], box[0] : box[0] + box[2]])
        box = (x, y, w, h)
    else:
        crop = image
        angle = skew_angle(h_lines)

    # Light text on a dark background: the table background is the bulk of
    # the crop, so its median brightness tells the polarity.
    inverted = bool(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < DARK_BACKGROUND)
    if inverted:
        crop = 255 - crop

    crop = rotate(crop, angle) if abs(angle) > 0.2 else crop
    if DEWARP:
        crop = dewarp(crop)

    return Table(
        image=crop,
        box=box,
        angle=angle,
        text_height=text_height / scale,
        inverted=inverted,
    )
