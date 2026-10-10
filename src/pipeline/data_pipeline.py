import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from ingestion.download_images import download_pair_images
from utils.constants import DATASET_DIR
from utils.logger import Logger
from preprocessing.filter_brazil import filter_brazil
from preprocessing.filter_images import filter_pairs
import time

log = Logger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full processing pipeline.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Directory containing the raw dataset files (default: {DATASET_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(" Open Food Facts — Pipeline")
    print("=" * 60)

    log.info("Stage 1 — Filter Brazil")
    log.blank()
    filter_brazil(args.dataset_dir)

    log.info("Stage 2 — Filter front + nutrition pairs")
    log.blank()
    filter_pairs()

    log.info("Stage 3 — Download front + nutrition images")
    log.blank()
    inicio = time.perf_counter()
    download_pair_images()
    fim = time.perf_counter()
    print(f"tempo de execução: {fim-inicio}")

    log.blank()
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
