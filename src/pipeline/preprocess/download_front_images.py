import pandas as pd
import requests

from utils.constants import (
    FILTER_FRONT_OUTPUT_FILE,
    IMAGES_FRONT_OUTPUT_DIR,
)
from utils.logger import Logger

log = Logger("download")

def get_front_pt(images):
    return next(
        image
        for image in images
        if isinstance(image, dict) and image.get("key") == "front_pt"
    )


def get_front_image_url(code, front_image, size="400"):
    rev = int(front_image["rev"])

    code = str(code).zfill(13)

    return (
        f"https://images.openfoodfacts.org/images/products/"
        f"{code[:3]}/{code[3:6]}/{code[6:9]}/{code[9:]}/"
        f"front_pt.{rev}.{size}.jpg"
    )


def download_image(url, output_path):
    response = requests.get(url)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        file.write(response.content)


def download_front_images():
    df = pd.read_parquet(FILTER_FRONT_OUTPUT_FILE)

    IMAGES_FRONT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    for _, product in df.iterrows():
        output_path = IMAGES_FRONT_OUTPUT_DIR / f"{product['code']}.jpg"

        if output_path.exists():
            continue

        front = get_front_pt(product["images"])

        url = get_front_image_url(
            product["code"],
            front,
        )

        download_image(url, output_path)

        downloaded += 1

    log.info(f"Images downloaded: {downloaded}")
    log.blank()


if __name__ == "__main__":
    download_front_images()