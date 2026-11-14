import numpy as np

from vision.back import cache as cache_module
from vision.back import pipeline
from vision.back.cache import OcrCache
from vision.back.parse import Word


def counter(value):
    calls = []

    def compute():
        calls.append(1)
        return value

    return compute, calls


def test_entry_is_reused_after_saving(tmp_path):
    image = np.zeros((20, 30), np.uint8)
    compute, calls = counter([["Sódio", 1, 2, 3, 4, 95.5]])

    first = OcrCache("123", tmp_path)
    assert first.get("r0/tesseract", image, compute) == [["Sódio", 1, 2, 3, 4, 95.5]]
    first.save()

    second = OcrCache("123", tmp_path)
    assert second.get("r0/tesseract", image, compute) == [["Sódio", 1, 2, 3, 4, 95.5]]
    assert len(calls) == 1
    assert second.stats == {"hits": 1, "misses": 0, "changed": 0}


def test_input_of_another_size_runs_again(tmp_path):
    compute, calls = counter(1)
    cache = OcrCache("123", tmp_path)
    cache.get("osd", np.zeros((20, 30), np.uint8), compute)
    cache.get("osd", np.zeros((21, 30), np.uint8), compute)
    assert len(calls) == 2


def test_same_size_other_pixels_is_reused_but_counted(tmp_path):
    compute, calls = counter(1)
    cache = OcrCache("123", tmp_path)
    cache.get("osd", np.zeros((20, 30), np.uint8), compute)
    cache.get("osd", np.ones((20, 30), np.uint8), compute)
    assert len(calls) == 1
    assert cache.changed == 1


def test_new_version_empties_the_cache(tmp_path, monkeypatch):
    image = np.zeros((20, 30), np.uint8)
    compute, calls = counter(1)
    cache = OcrCache("123", tmp_path)
    cache.get("osd", image, compute)
    cache.save()

    monkeypatch.setattr(cache_module, "VERSION", cache_module.VERSION + 1)
    OcrCache("123", tmp_path).get("osd", image, compute)
    assert len(calls) == 2


def test_cached_words_equal_fresh_ones(tmp_path, monkeypatch):
    """Words come back identical through JSON: same parse, same result."""
    words = [Word("Sódio", 10, 20, 50, 12, 91.0), Word("0,5", 70, 21, 20, 12, 88.5)]
    lines = [[[[10.0, 20.0], [60.0, 20.5], [60.0, 32.0], [10.0, 31.5]], "Sódio", 0.91]]
    monkeypatch.setattr(pipeline, "read_words", lambda image, preprocess: words)
    monkeypatch.setattr(pipeline, "rapidocr_lines", lambda image, model: lines)
    image = np.zeros((40, 80, 3), np.uint8)

    fresh = (
        pipeline._tesseract(None, "t", image),
        pipeline._rapidocr(None, "r", image),
    )
    cache = OcrCache("123", tmp_path)
    pipeline._tesseract(cache, "t", image)
    pipeline._rapidocr(cache, "r", image)
    cache.save()
    reloaded = OcrCache("123", tmp_path)
    cached = (
        pipeline._tesseract(reloaded, "t", image),
        pipeline._rapidocr(reloaded, "r", image),
    )
    assert cached == fresh
    assert reloaded.stats["misses"] == 0
