import asyncio

import httpx
import pandas as pd
from tqdm import tqdm

from utils.constants import (
    AWS_BASE_URL,
    OFF_BASE_URL,
    USER_AGENT,
    FILTER_PAIRS_OUTPUT_FILE,
    IMAGES_OUTPUT_DIR,
    IMAGE_SIZE,
    MAX_CONCURRENT_DOWNLOADS,
    MAX_RETRIES,
    REQUEST_TIMEOUT
)
from utils.logger import Logger

log = Logger("pipeline")


# ============================================================
# Dados das imagens
# ============================================================

def get_product_path(code):
    code = str(code).strip().zfill(13)

    return (
        f"{code[:3]}/"
        f"{code[3:6]}/"
        f"{code[6:9]}/"
        f"{code[9:]}"
    )


def get_raw_image_filename(imgid):
    # Raw uploads, not the selected crops: same file on both servers and
    # closest to a photo taken with a phone.
    if IMAGE_SIZE == "full":
        return f"{int(imgid)}.jpg"

    return f"{int(imgid)}.{IMAGE_SIZE}.jpg"


def get_image_urls(code, imgid):
    path = f"{get_product_path(code)}/{get_raw_image_filename(imgid)}"

    return f"{AWS_BASE_URL}/{path}", f"{OFF_BASE_URL}/{path}"

async def download_from_url(
    client,
    url,
    output_path,
):

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.get(url)

            # Arquivo não existe nessa orige

            if response.status_code == 404:
                return False

            # Erros temp.

            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** (attempt - 1)

                    await asyncio.sleep(wait_time)

                    continue

                return False

            response.raise_for_status()

            output_path.write_bytes(response.content)

            return True

        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ):
            if attempt < MAX_RETRIES:
                wait_time = 2 ** (attempt - 1)

                await asyncio.sleep(wait_time)

                continue

            return False

        except Exception:
            return False

    return False


async def download_product_image(
    client,
    semaphore,
    code,
    imgid,
    output_dir,
):

    output_path = output_dir / f"{code}.jpg"

    if output_path.exists():
        return "exists"

    if imgid is None:
        return "no_image"

    try:
        aws_url, off_url = get_image_urls(
            code=code,
            imgid=imgid,
        )
    except (TypeError, ValueError):
        return "invalid_image_data"

    async with semaphore:

        #AWS S3

        downloaded = await download_from_url(
            client=client,
            url=aws_url,
            output_path=output_path,
        )

        if downloaded:
            return "aws"

        #Fallback Open Food Facts

        downloaded = await download_from_url(
            client=client,
            url=off_url,
            output_path=output_path,
        )

        if downloaded:
            return "off"

    return "failed"



# ============================================================
# Pipeline de downloads
# ============================================================

def create_http_client():
    timeout = httpx.Timeout(
        REQUEST_TIMEOUT
    )

    limits = httpx.Limits(
        max_connections=MAX_CONCURRENT_DOWNLOADS,
        max_keepalive_connections=MAX_CONCURRENT_DOWNLOADS,
    )

    headers = {
        "User-Agent": USER_AGENT,
    }

    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        headers=headers,
        follow_redirects=True,
    )


def create_download_tasks(
    batch,
    client,
    semaphore,
    image_type,
    output_dir,
):
    tasks = []

    for product in batch.itertuples(index=False):

        code = str(product.code).strip()

        tasks.append(
            asyncio.create_task(
                download_product_image(
                    client=client,
                    semaphore=semaphore,
                    code=code,
                    imgid=getattr(product, f"{image_type}_imgid"),
                    output_dir=output_dir,
                )
            )
        )

    return tasks


def update_download_stats(
    result,
    stats,
):
    stats[result] = stats.get(result, 0) + 1


async def process_download_tasks(
    tasks,
    stats,
    progress,
):
    for task in asyncio.as_completed(tasks):

        try:
            result = await task

        except Exception:
            result = "failed"

        update_download_stats(
            result=result,
            stats=stats,
        )

        progress.update(1)

        progress.set_postfix(
            AWS=stats["aws"],
            OFF=stats["off"],
            existing=stats["exists"],
            failed=stats["failed"],
        )


async def download_images(
    input_file,
    image_type,
    output_dir,
):
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_parquet(input_file)

    log.blank()
    print(f"Downloading {image_type}")
    print(f"Products: {len(df)}")
    print(f"Output:   {output_dir}")
    print(f"Workers:  {MAX_CONCURRENT_DOWNLOADS}")
    log.blank()

    semaphore = asyncio.Semaphore(
        MAX_CONCURRENT_DOWNLOADS
    )

    stats = {
        "aws": 0,
        "off": 0,
        "exists": 0,
        "no_image": 0,
        "invalid_image_data": 0,
        "failed": 0,
    }

    batch_size = MAX_CONCURRENT_DOWNLOADS * 10

    async with create_http_client() as client:

        with tqdm(
            total=len(df),
            desc=f"Downloading {image_type}",
            unit="img",
        ) as progress:

            for start in range(
                0,
                len(df),
                batch_size,
            ):
                batch = df.iloc[
                    start:start + batch_size
                ]

                tasks = create_download_tasks(
                    batch=batch,
                    client=client,
                    semaphore=semaphore,
                    image_type=image_type,
                    output_dir=output_dir,
                )

                await process_download_tasks(
                    tasks=tasks,
                    stats=stats,
                    progress=progress,
                )

    log.summary(
        image_type=image_type,
        stats=stats,
    )

# ============================================================
# Front + nutrition
# ============================================================

def download_pair_images():

    for image_type in ("front", "nutrition"):
        asyncio.run(
            download_images(
                input_file=FILTER_PAIRS_OUTPUT_FILE,
                image_type=image_type,
                output_dir=IMAGES_OUTPUT_DIR / image_type,
            )
        )
