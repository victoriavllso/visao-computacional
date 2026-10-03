import pandas as pd
from utils.constants import FILTER_BRAZIL_OUTPUT_FILE, FILTER_FRONT_OUTPUT_FILE, FILTER_FRONT_OUTPUT_DIR

#python3 -m pipeline.preprocess.filter_data

from utils.constants import (
    FILTER_BRAZIL_OUTPUT_FILE,
    FILTER_FRONT_OUTPUT_FILE,
    FILTER_FRONT_OUTPUT_DIR,
)

from utils.logger import Logger

log = Logger("filter")

def has_front_pt(images):
    return any(
        isinstance(image, dict) and image.get("key") == "front_pt"
        for image in images
    )


def filter_front():
    if FILTER_FRONT_OUTPUT_FILE.exists():
        log.info("Front image filter already completed. Skipping.")
        log.blank()
        return
    df = pd.read_parquet(FILTER_BRAZIL_OUTPUT_FILE)

    df = df[df["images"].apply(has_front_pt)].copy()

    FILTER_FRONT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FILTER_FRONT_OUTPUT_FILE, index=False)
    log.info(f"Products with front_pt image: {len(df)}")

if __name__ == "__main__":
    filter_front()
