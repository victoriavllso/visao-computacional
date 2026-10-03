import pandas as pd
import requests

from utils.constants import (
    FILTER_BACK_OUTPUT_FILE, # SE FORMOS USAR, PRECISA CRIAR RAPAZIADA
    IMAGES_BACK_OUTPUT_DIR,
)
from utils.logger import Logger

log = Logger("download")

def get_back_pt(images):
    return next(
        image
        for image in images
        if isinstance(image, dict) and image.get("key") == "back_pt"
    )


def get_back_image_url(code, back_image, size="400"):
    rev = int(back_image["rev"])

    code = str(code).zfill(13)

    return (
        f"https://images.openfoodfacts.org/images/products/"
        f"{code[:3]}/{code[3:6]}/{code[6:9]}/{code[9:]}/"
        f"back_pt.{rev}.{size}.jpg"
    )


def download_image(url, output_path):
    response = requests.get(url)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        file.write(response.content)


def download_back_images():
    df = pd.read_parquet(FILTER_BACK_OUTPUT_FILE)

    IMAGES_BACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    for _, product in df.iterrows():
        output_path = (
            IMAGES_BACK_OUTPUT_DIR / f"{product['code']}.jpg"
        )

        # Permite continuar de onde parou
        if output_path.exists():
            continue

        back = get_back_pt(product["images"])

        url = get_back_image_url(
            product["code"],
            back,
        )

        download_image(url, output_path)

        downloaded += 1

    log.info(f"Images downloaded: {downloaded}")


if __name__ == "__main__":
    download_back_images()