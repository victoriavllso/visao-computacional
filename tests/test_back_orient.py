import numpy as np

from vision.back import orient
from vision.back.orient import rotate, vertical_text
from vision.back.parse import Word


def lines(n, w, h, text="Gorduras saturadas"):
    return [Word(text, 10, 40 * i, w, h) for i in range(n)]


def test_vertical_text_when_most_lines_are_tall():
    assert vertical_text(lines(8, 30, 200))
    assert not vertical_text(lines(8, 200, 30))
    # Short tokens (digits) don't count: a lone '0' is as tall as wide.
    assert not vertical_text(lines(8, 200, 30) + lines(8, 10, 30, text="0"))
    # Too few lines to tell.
    assert not vertical_text(lines(3, 30, 200))


def test_rotate_turns_clockwise():
    image = np.zeros((2, 3), np.uint8)
    image[0, 0] = 1  # top-left
    assert rotate(image, 0) is image
    assert rotate(image, 90).shape == (3, 2) and rotate(image, 90)[0, 1] == 1
    assert rotate(image, 180)[1, 2] == 1
    assert rotate(image, 270)[2, 0] == 1


def test_osd_needs_confidence(monkeypatch):
    def fake_osd(rotate, confidence):
        return lambda *a, **k: {"rotate": rotate, "orientation_conf": confidence}

    image = np.full((100, 100, 3), 255, np.uint8)
    monkeypatch.setattr(orient.pytesseract, "image_to_osd", fake_osd(270, 5.0))
    assert orient.osd_rotation(image) == 270
    monkeypatch.setattr(orient.pytesseract, "image_to_osd", fake_osd(180, 0.4))
    assert orient.osd_rotation(image) == 0
