from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
VENV_DIR: Path = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE: Path = PROJECT_ROOT / "requirements.txt"
DEV_REQUIREMENTS_FILE: Path = PROJECT_ROOT / "requirements-dev.txt"
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_DIR: Path = DATA_DIR / "product-database"

HF_REPO_ID: str = "openfoodfacts/product-database"
HF_REPO_TYPE: str = "dataset"
HF_FOOD_FILE: str = "food.parquet"
HF_BEAUTY_FILE: str = "beauty.parquet"

FILTER_BRAZIL_OUTPUT_DIR: Path = DATA_DIR / "brazil"
FILTER_BRAZIL_OUTPUT_FILE: Path = FILTER_BRAZIL_OUTPUT_DIR / "food_brazil.parquet"
FILTER_FRONT_OUTPUT_DIR: Path = DATA_DIR / "brazil_with_front_pt"
FILTER_FRONT_OUTPUT_FILE: Path = FILTER_FRONT_OUTPUT_DIR / "products_with_front_pt.parquet"
IMAGES_FRONT_OUTPUT_DIR:Path = FILTER_FRONT_OUTPUT_DIR / "images"