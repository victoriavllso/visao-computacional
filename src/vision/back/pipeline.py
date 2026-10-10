"""Back of the package → nutrition facts per 100 g/ml.

photo ─► locate + straighten table ─► OCR ─► parse rows ─► NutritionFacts
"""

from dataclasses import dataclass

import cv2

from vision.back.locate import Table, locate_table
from vision.back.ocr import read_words
from vision.back.parse import Reading, parse_table


@dataclass
class BackResult:
    reading: Reading
    table: Table | None


def read_back(image, locate=True, preprocess=True) -> BackResult:
    """`locate` and `preprocess` switch the classical steps off, to measure
    what each one adds."""
    if isinstance(image, str):
        path, image = image, cv2.imread(image)
        if image is None:
            raise FileNotFoundError(path)

    table = locate_table(image) if locate else None
    crop = table.image if table is not None else image

    words = read_words(crop, preprocess=preprocess)
    return BackResult(reading=parse_table(words), table=table)
