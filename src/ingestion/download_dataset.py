import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.constants import DATASET_DIR, HF_REPO_ID, HF_REPO_TYPE
from utils.logger import Logger

log = Logger("dataset download")


def check_hf_available() -> None:
    """Abort early if the `hf` CLI is not found on PATH."""
    if shutil.which("hf") is None:
        log.error(
            "`hf` CLI not found. Install it with: pip install huggingface_hub[cli]"
        )
        sys.exit(1)


def login(token: str | None) -> None:
    """Log in to Hugging Face if a token was supplied."""
    if token:
        log.info("Logging in to Hugging Face …")
        subprocess.check_call(
            ["hf", "login", "--token", token, "--add-to-git-credential"]
        )
        log.info("Login successful.")
    else:
        log.info("No token supplied — using cached credentials (if any).")


def download_dataset(local_dir: Path) -> None:
    """Download the dataset with the huggingface-cli."""
    local_dir.mkdir(parents=True, exist_ok=True)

    log.blank()
    log.info(f"Downloading {HF_REPO_ID} → {local_dir.resolve()} …")
    log.info("This may take a while — the dataset is large.")
    log.blank()

    cmd = [
        "hf",
        "download",
        "--repo-type",
        HF_REPO_TYPE,
        HF_REPO_ID,
        "--local-dir",
        str(local_dir),
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        log.blank()
        log.error(
            "Download failed. If the dataset is private, pass --token with a valid HF token."
        )
        sys.exit(result.returncode)

    log.blank()
    log.info(f"Dataset downloaded successfully to {local_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up the Open Food Facts HF dataset project."
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (optional for public datasets).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Local directory to download the dataset into (default: {DATASET_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(" Open Food Facts — HF Dataset Setup")
    print("=" * 60)

    check_hf_available()
    login(args.token)
    download_dataset(args.local_dir)

    log.blank()
    log.info("Setup complete. You're ready to work with the dataset.")


if __name__ == "__main__":
    main()
