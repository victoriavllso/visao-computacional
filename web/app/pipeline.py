"""Adapter between the app and the vision pipelines: the only module the
backend calls to analyze a product.

    analyze(front_path, back_path) -> dict

front_path None: only the nutrition table is read, and the verdict is "info"
(which warnings the product must carry) instead of a compliance decision.

`read_photos` runs the real pipelines: back (nutrition table) → rules
(required warnings) → front (printed warnings). The front uses YOLO + OCR when
the detector weights (models/front_detector.onnx) are there, OCR on the whole
photo otherwise; APP_FRONT_METHOD=ocr|yolo forces one.

APP_PIPELINE=mock switches to `read_mock`, which makes up the readings from a
hash of the two files (fast; used by the app's tests and to demo every
verdict). With the mock, APP_MOCK_VERDICT=compliant|inconclusive|
non_compliant forces a verdict.

Returned structure (also what the JSON export contains, under "result"):

    {
      "mode": "full" | "table_only",
      "verdict": "compliant" | "inconclusive" | "non_compliant" | "info",
      "reasons": [str, ...],          # plain Portuguese, shown to the user
      "nutrition": {                  # per 100 g or 100 ml, as sold
        "basis": "100g" | "100ml" | None,
        "added_sugars_g": float | None,
        "saturated_fat_g": float | None,
        "sodium_mg": float | None,
      },
      "required_warnings": {"added_sugars": bool | None,   # None: undecidable
                            "saturated_fat": bool | None,
                            "sodium": bool | None},
      "found_warnings": {...same keys...},  # None: unreadable or no front
      "notes": [str, ...],  # when a warning is required: limits for solid or
                            # liquid foods, and the exempt categories
      "back_photo_type": "table" | "linear" | "old_format" | "no_table"
                         | "uncertain",
      "versions": {"back": str, "front": str, "rules": str},
      "mock": bool,
    }

`read_photos` returns a `Readings`; `summarize` turns required vs found into
the verdict and its reasons. Where each part lives:

- Back (src/vision/back/pipeline.py): `read_back(image)` with a BGR image
  (cv2.imread) returns a BackResult; `result.reading.facts` is a
  rules.anvisa.NutritionFacts (basis: Basis | None, added_sugars_g,
  saturated_fat_g, sodium_mg), `result.photo_type` a PhotoType (its .value
  goes in back_photo_type). When `not result.readable` (old format, no
  table) the required warnings are all None.
- Rules (src/rules/anvisa.py): `assess(facts, exemption)` returns an
  Assessment: a nutrient in `required` → True, in `unknown` → None,
  otherwise False. The exemption (src/rules/exemptions.py) needs the
  product category, which the app doesn't ask for yet: use
  Exemption.NONE; an `assessment.needs_warning is None` from a CONDITIONAL
  exemption should make the verdict inconclusive.
- Front (src/vision/front/warnings.py): `read_front(image, method)` gives,
  per nutrient, whether its "ALTO EM" warning is printed. A panel found by
  the detector with no nutrient read means a warning is there but can't be
  read: all None (inconclusive), not "no warning" (a false violation).

The pipelines need src/ on sys.path (as tests/conftest.py does), and their
heavy imports (OCR models) should happen once, at the first call.
"""

import hashlib
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

from rules.anvisa import LIMITS as RULE_LIMITS  # noqa: E402 (needs src/)

NUTRIENTS = ("added_sugars", "saturated_fat", "sodium")
VERDICTS = ("compliant", "inconclusive", "non_compliant")  # with the front

# IN 75/2020 Annex XV, keyed like the rest of the app: {"100g": {"sodium":
# 600, ...}, "100ml": {...}}; sugars and fat in g, sodium in mg.
LIMITS = {
    basis.value: {nutrient.value: limit for nutrient, limit in limits.items()}
    for basis, limits in RULE_LIMITS.items()
}
FOOD_KIND = {
    "100g": "alimento sólido ou semissólido (tabela por 100 g)",
    "100ml": "alimento líquido (tabela por 100 ml)",
}
# The app doesn't ask for the category (a hindrance to the user): instead,
# when a warning is required, it lists the categories that change that
# (IN 75/2020 Annex XVI).
EXEMPTIONS_NOTE = (
    "Algumas categorias não seguem esta regra (IN 75/2020, Anexo XVI). "
    "Nunca levam o alerta: leites, azeite e óleos vegetais, sal, fórmulas "
    "infantis e enterais, alimentos para controle de peso, suplementos, "
    "bebidas alcoólicas e aditivos (como adoçantes de mesa). Só levam se um "
    "ingrediente adicionado trouxer o nutriente: frutas, hortaliças, "
    "leguminosas, cereais, castanhas, sementes, cogumelos, farinhas, carnes e "
    "peixes embalados, ovos, leites fermentados e queijos. Se o produto for "
    "de uma dessas categorias, o resultado pode não se aplicar."
)

WARNING_TEXT = {
    "added_sugars": "ALTO EM AÇÚCAR ADICIONADO",
    "saturated_fat": "ALTO EM GORDURA SATURADA",
    "sodium": "ALTO EM SÓDIO",
}

PHOTO_TYPE_REASON = {
    "old_format": "A tabela nutricional do verso está no formato antigo "
    "(por porção), anterior à RDC 429/2020: não dá para verificar.",
    "no_table": "Não foi encontrada uma tabela nutricional na foto do verso.",
    "uncertain": "Não foi possível identificar o formato da tabela "
    "nutricional do verso.",
}


@dataclass
class Readings:
    """What the pipelines read from the two photos."""

    nutrition: dict
    required: dict  # nutrient -> bool | None
    found: dict  # nutrient -> bool | None
    photo_type: str
    versions: dict
    mock: bool = False
    table_only: bool = False  # no front photo: found is all None


def analyze(front_path: Path | None, back_path: Path) -> dict:
    front_path = Path(front_path) if front_path is not None else None
    if os.environ.get("APP_PIPELINE") == "mock":
        readings = read_mock(front_path, Path(back_path))
    else:
        with _PIPELINE_LOCK:
            readings = read_photos(front_path, Path(back_path))
    verdict, reasons = summarize(readings)
    return {
        "mode": "table_only" if readings.table_only else "full",
        "verdict": verdict,
        "reasons": reasons,
        "notes": notes(readings, verdict),
        "nutrition": readings.nutrition,
        "required_warnings": readings.required,
        "found_warnings": readings.found,
        "back_photo_type": readings.photo_type,
        "versions": readings.versions,
        "mock": readings.mock,
    }


def summarize(readings: Readings) -> tuple[str, list[str]]:
    """Verdict and the reasons behind it. Any definite mismatch is a
    violation even when other nutrients are undecidable: the product is
    already wrong."""
    if readings.table_only:
        return "info", required_reasons(readings)
    if readings.photo_type in PHOTO_TYPE_REASON:
        return "inconclusive", [PHOTO_TYPE_REASON[readings.photo_type]]

    violations, unknown = [], []
    for nutrient in NUTRIENTS:
        required, found = readings.required[nutrient], readings.found[nutrient]
        text = WARNING_TEXT[nutrient]
        if required is True and found is False:
            violations.append(f"Deveria ter {text} e não tem.")
        elif required is False and found is True:
            violations.append(f"Tem {text}, mas o produto não atinge o limite.")
        elif required is None:
            unknown.append(f"Não foi possível ler o valor para decidir {text}.")
        elif found is None:
            unknown.append(f"Não foi possível ver na frente se há {text}.")

    if violations:
        return "non_compliant", violations
    if unknown:
        return "inconclusive", unknown
    if any(readings.required.values()):
        return "compliant", ["Todos os alertas exigidos estão na frente."]
    return "compliant", ["Nenhum alerta é exigido e nenhum foi impresso."]


def required_reasons(readings: Readings) -> list[str]:
    """What the table says the front must show, for the table-only mode."""
    if readings.photo_type in PHOTO_TYPE_REASON:
        return [PHOTO_TYPE_REASON[readings.photo_type]]
    reasons = []
    for nutrient in NUTRIENTS:
        required, text = readings.required[nutrient], WARNING_TEXT[nutrient]
        if required:
            reasons.append(f"Deve ter o alerta {text}.")
        elif required is None:
            reasons.append(f"Não foi possível ler o valor para decidir {text}.")
    return reasons or ["Não precisa de nenhum alerta “ALTO EM”."]


def notes(readings: Readings, verdict: str) -> list[str]:
    """When a warning is required, or a violation was found: which limits
    applied (solid or liquid food), and the categories exempt from them."""
    required = any(readings.required.values())
    if verdict != "non_compliant" and not (verdict == "info" and required):
        return []
    basis = readings.nutrition.get("basis")
    if basis in FOOD_KIND:
        kind = (
            f"O produto foi tratado como {FOOD_KIND[basis]}: o alerta é "
            f"exigido a partir de {_limits_text(LIMITS[basis])} por "
            f"{basis.replace('100', '100 ')}."
        )
    else:
        kind = (
            "Não foi possível saber pela tabela se o alimento é sólido ou "
            "líquido. Limites: por 100 g (sólidos e semissólidos) "
            + _limits_text(LIMITS["100g"])
            + "; por 100 ml (líquidos) "
            + _limits_text(LIMITS["100ml"])
            + "."
        )
    return [kind, EXEMPTIONS_NOTE]


def _limits_text(limits):
    number = lambda value: f"{value:g}".replace(".", ",")  # noqa: E731
    return (
        f"{number(limits['added_sugars'])} g de açúcares adicionados, "
        f"{number(limits['saturated_fat'])} g de gorduras saturadas ou "
        f"{number(limits['sodium'])} mg de sódio"
    )


# The OCR engines and the detector are module-level singletons, not meant
# for concurrent calls: one analysis at a time.
_PIPELINE_LOCK = threading.Lock()


def front_method():
    from vision.front.detector import MODEL_PATH

    method = os.environ.get("APP_FRONT_METHOD")
    if method in ("ocr", "yolo"):
        return method
    return "yolo" if MODEL_PATH.exists() else "ocr"


def read_photos(front_path: Path | None, back_path: Path) -> Readings:
    """The real pipelines. Heavy imports (OCR models, detector) happen here,
    at the first analysis, and stay loaded."""
    import cv2

    from rules.anvisa import Exemption, Nutrient, assess
    from vision.back.pipeline import ENGINE, read_back
    from vision.front.warnings import read_front

    back = read_back(_imread(back_path))
    facts = back.reading.facts
    readable = back.readable
    nutrition = {
        "basis": facts.basis.value if readable and facts.basis else None,
        # An old-format table is per portion: its numbers aren't per 100 g.
        "added_sugars_g": facts.added_sugars_g if readable else None,
        "saturated_fat_g": facts.saturated_fat_g if readable else None,
        "sodium_mg": facts.sodium_mg if readable else None,
    }
    if readable:
        # The app doesn't ask for the product category yet: no exemption.
        assessment = assess(facts, Exemption.NONE)
        required = {
            n.value: (None if n in assessment.unknown else n in assessment.required)
            for n in Nutrient
        }
    else:
        required = {n.value: None for n in Nutrient}

    versions = {
        "back": f"locate + Tesseract/RapidOCR ({ENGINE})",
        "rules": "IN 75/2020, Anexo XV",
    }
    if front_path is None:
        found = {n.value: None for n in Nutrient}
        return Readings(
            nutrition, required, found, back.photo_type.value, versions, table_only=True
        )

    method = front_method()
    front = read_front(_imread(front_path), method=method)
    found = {n.value: bool(present) for n, present in front.warnings.items()}
    if front.boxes and not front.has_warning:
        found = {n: None for n in found}
    versions["front"] = "YOLO + OCR" if method == "yolo" else "OCR"
    return Readings(nutrition, required, found, back.photo_type.value, versions)


def _imread(path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"not an image: {path}")
    return image


# ---------------------------------------------------------------- MOCK ---
# Fake readings, for the app's tests and demos (APP_PIPELINE=mock).

# Copy of IN 75/2020 Annex XV (src/rules/anvisa.py LIMITS), solid foods only,
# so that the fake numbers look consistent with the fake warnings.
MOCK_LIMITS = {"added_sugars": 15.0, "saturated_fat": 6.0, "sodium": 600.0}
MOCK_VERSIONS = {"back": "mock", "front": "mock", "rules": "mock"}


def read_mock(front_path: Path | None, back_path: Path) -> Readings:
    """MOCK: deterministic fake readings derived from the files' bytes."""
    front_bytes = front_path.read_bytes() if front_path else b""
    digest = hashlib.sha256(front_bytes + back_path.read_bytes())
    seed = digest.digest()
    forced = os.environ.get("APP_MOCK_VERDICT")
    verdict = forced if forced in VERDICTS else VERDICTS[seed[0] % 3]

    nutrition = {
        "basis": "100g",
        "added_sugars_g": round(seed[1] / 255 * 30, 1),
        "saturated_fat_g": round(seed[2] / 255 * 12, 1),
        "sodium_mg": float(seed[3] * 5),
    }
    required = {
        n: nutrition[key] >= MOCK_LIMITS[n]
        for n, key in zip(NUTRIENTS, ("added_sugars_g", "saturated_fat_g", "sodium_mg"))
    }
    found = dict(required)
    photo_type = "table"

    if verdict == "non_compliant":
        # Drop a required warning (the common real case), or, when none is
        # required, print one that isn't.
        candidates = [n for n in NUTRIENTS if required[n]] or list(NUTRIENTS)
        nutrient = candidates[seed[4] % len(candidates)]
        found[nutrient] = not required[nutrient]
    elif verdict == "inconclusive":
        if seed[5] % 2:
            photo_type = "old_format"
            nutrition = {key: None for key in nutrition}
            required = {n: None for n in NUTRIENTS}
        else:
            nutrition["sodium_mg"] = None
            required["sodium"] = None

    if front_path is None:
        found = {n: None for n in NUTRIENTS}
    return Readings(
        nutrition,
        required,
        found,
        photo_type,
        MOCK_VERSIONS,
        mock=True,
        table_only=front_path is None,
    )
