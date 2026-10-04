import pandas as pd

from utils.constants import (
    FILTER_BRAZIL_OUTPUT_FILE,
    FILTER_FRONT_OUTPUT_FILE,
    #FILTER_BACK_OUTPUT_FILE,
    FILTER_FRONT_OUTPUT_DIR,
    #FILTER_BACK_OUTPUT_DIR,
)
from utils.logger import Logger

log = Logger("filter")

def has_image(images, image_type):
    return any(
        isinstance(image, dict) and image.get("key") == image_type
        for image in images
    )


def filter_images(image_type, output_file, output_dir):
    if output_file.exists():
        log.skip("filter already completed")
        log.blank()
        return

    df = pd.read_parquet(FILTER_BRAZIL_OUTPUT_FILE)

    df = df[df["images"].apply(
        lambda images: has_image(images, image_type)
    )].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)

    print(
        f"Products with {image_type} image: {len(df)}"
    )


def filter_front():
    filter_images(
        image_type="front_pt",
        output_file=FILTER_FRONT_OUTPUT_FILE,
        output_dir=FILTER_FRONT_OUTPUT_DIR,
    )


#def filter_back():
#    filter_images(
#        image_type="back_pt",
#        output_file=FILTER_BACK_OUTPUT_FILE,
#        output_dir=FILTER_BACK_OUTPUT_DIR,
#    )