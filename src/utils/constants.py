from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_DIR: Path = DATA_DIR / "product-database"

HF_REPO_ID: str = "openfoodfacts/product-database"
HF_REPO_TYPE: str = "dataset"
# The dataset is updated daily: pin the snapshot (2026-10-18 18:29 UTC, the
# last one before IMAGE_END_CUTOFF was set) so everyone regenerates the same
# products. It has every hand-labeled product (2,571 pairs).
HF_REVISION: str = "e7f6e1c9187651f6cfdffa051d6a8c99a187ecc6"
HF_FOOD_FILE: str = "food.parquet"
HF_BEAUTY_FILE: str = "beauty.parquet"

FILTER_BRAZIL_OUTPUT_DIR: Path = DATA_DIR / "brazil"
FILTER_BRAZIL_OUTPUT_FILE: Path = FILTER_BRAZIL_OUTPUT_DIR / "food_brazil.parquet"
FILTER_PAIRS_OUTPUT_DIR: Path = DATA_DIR / "brazil_pairs"
FILTER_PAIRS_OUTPUT_FILE: Path = FILTER_PAIRS_OUTPUT_DIR / "products_pairs.parquet"
IMAGES_OUTPUT_DIR: Path = FILTER_PAIRS_OUTPUT_DIR / "images"
OFF_EXPECTED_WARNINGS_FILE: Path = (
    FILTER_PAIRS_OUTPUT_DIR / "off_expected_warnings.parquet"
)

# Hand labels are versioned (outside data/), to share with the team.
BACK_LABELS_DIR: Path = PROJECT_ROOT / "labels" / "back"
# Warnings seen on the front photo, and the boxes around them.
FRONT_LABELS_DIR: Path = PROJECT_ROOT / "labels" / "front"

# Selected-image keys used for each side of the package. Open Food Facts has
# no "back" key: the back of the package is the nutrition table photo.
FRONT_IMAGE_KEY = "front_pt"
NUTRITION_IMAGE_KEY = "nutrition_pt"

# End of the transition period for products already on the market
# (RDC 429/2020). Both photos must have been uploaded on or after this date.
REGULATION_CUTOFF = "2023-10-09"

# Strict upper bound on image capture time. Both photos must have been uploaded
# before this date.
IMAGE_END_CUTOFF = "2026-10-18"

# Ground-truth nutrients (per 100 g/ml) needed to evaluate the back pipeline.
GROUND_TRUTH_NUTRIENTS = ("added-sugars", "saturated-fat", "sodium")

# Labeled products are split into dev/val by a stable hash of the barcode.
VAL_FRACTION = 0.3


AWS_BASE_URL = "https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/data"
OFF_BASE_URL = "https://images.openfoodfacts.org/images/products"
USER_AGENT = "visao-computacional/1.0 " "(Open Food Facts image dataset)"
# "full" keeps the original resolution, needed to OCR the nutrition table.
IMAGE_SIZE = "full"
MAX_CONCURRENT_DOWNLOADS = 20
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
