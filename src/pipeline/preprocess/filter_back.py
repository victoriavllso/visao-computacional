import pandas as pd

from utils.constants import (
    FILTER_BRAZIL_OUTPUT_FILE,
    FILTER_BACK_OUTPUT_FILE,
    FILTER_BACK_OUTPUT_DIR,
)
from utils.logger import Logger

log = Logger("filter")

def has_back_pt(images):
    return any(
        isinstance(image, dict) and image.get("key") == "back_pt"
        for image in images
    )


def filter_back():
    if FILTER_BACK_OUTPUT_FILE.exists():
        log.info("Back image filter already completed. Skipping.")
        log.blank()
        return

    df = pd.read_parquet(FILTER_BRAZIL_OUTPUT_FILE)

    df = df[df["images"].apply(has_back_pt)].copy()

    FILTER_BACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FILTER_BACK_OUTPUT_FILE, index=False)

    log.info(f"Products with back_pt image: {len(df)}")


if __name__ == "__main__":
    filter_back()