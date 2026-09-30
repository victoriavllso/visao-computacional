import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from utils.constants import DATASET_DIR
from utils.logger import Logger
from preprocess.filter_brazil import filter_brazil

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

    log.info("Stage 1 — Preprocessing")
    log.blank()
    filter_brazil(args.dataset_dir)

    log.blank()
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
