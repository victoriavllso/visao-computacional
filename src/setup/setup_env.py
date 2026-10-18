import argparse
import subprocess
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent))

from utils.constants import (
    VENV_DIR,
    REQUIREMENTS_FILE,
    DEV_REQUIREMENTS_FILE,
    NODEPS_REQUIREMENTS_FILE,
    DATA_DIR,
)
from utils.logger import Logger

log = Logger("setup")


def _venv_is_valid(venv_path: Path) -> bool:
    """Return True only when the directory contains a functioning Python interpreter
    and the current process has permission to execute it."""
    python = (
        venv_path / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv_path / "bin" / "python"
    )
    return python.exists() and os.access(python, os.X_OK)


def create_venv(venv_dir: Path) -> Path:
    """Create a virtual environment if one doesn't already exist (or is invalid)."""
    if not venv_dir.exists():
        log.info(f"Creating virtual environment at {venv_dir} …")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        log.info("Virtual environment created.")
        return venv_dir

    if _venv_is_valid(venv_dir):
        log.info(
            f"Virtual environment already exists at {venv_dir} — skipping creation."
        )
        return venv_dir

    log.info(
        f"Directory {venv_dir} exists but is not a valid virtual environment — recreating …"
    )
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    log.info("Virtual environment created.")

    return venv_dir


def get_pip(venv_path: Path) -> Path:
    """Return the path to pip inside the virtual environment."""
    pip = (
        venv_path / "Scripts" / "pip.exe"
        if sys.platform == "win32"
        else venv_path / "bin" / "pip"
    )

    if not pip.exists():
        log.error(f"pip not found at {pip} — aborting.")
        sys.exit(1)

    return pip


def upgrade_pip(pip: Path) -> None:
    """Upgrade pip itself before installing project packages."""
    log.blank()
    log.info("Upgrading pip …")
    subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
    log.info("pip upgraded.")


def install_requirements(pip: Path, requirements: Path, no_deps: bool = False) -> None:
    """Install packages from a requirements file."""
    req_path = requirements

    if not req_path.exists():
        log.error(f"{req_path} not found — aborting.")
        sys.exit(1)

    log.blank()
    log.info(f"Installing packages from {req_path} …")
    log.blank()
    command = [str(pip), "install", "-r", str(req_path)]
    if no_deps:
        command.append("--no-deps")
    subprocess.check_call(command)
    log.blank()
    log.info("All packages installed successfully.")


def install_pre_commit_hooks(venv_path: Path) -> None:
    """Install pre-commit hooks into the local .git directory."""
    log.blank()
    log.info("Installing pre-commit hooks...")

    pre_commit_path = venv_path / "bin" / "pre-commit"
    if not pre_commit_path.exists():
        log.error(f"pre-commit not found at {pre_commit_path} — aborting.")
        sys.exit(1)

    subprocess.check_call([str(pre_commit_path), "install"])
    log.info("pre-commit hooks installed.")


def ensure_data_dir() -> None:
    """Create the project-level data/ directory if it doesn't exist."""
    data_dir = DATA_DIR
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        log.info(f"Created data directory at {data_dir}.")
    else:
        log.info(f"Data directory already exists at {data_dir} — skipping.")


def print_activation_hint(venv_path: Path) -> None:
    """Print the activation command for the current platform."""
    if sys.platform == "win32":
        activate = venv_path / "Scripts" / "activate.bat"
        cmd = f"    {activate}"
    else:
        activate = venv_path / "bin" / "activate"
        cmd = f"    source {activate}"

    log.blank()
    log.info(f"To activate the environment, run:\n{cmd}")
    log.blank()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a venv and install project dependencies."
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=VENV_DIR,
        help=f"Directory for the virtual environment (default: {VENV_DIR}).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install dev dependencies (requirements-dev.txt) and set up pre-commit hooks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(" Project Environment Setup")
    print("=" * 60)

    venv_path = create_venv(args.venv_dir)
    pip = get_pip(venv_path)
    upgrade_pip(pip)

    if args.dev:
        install_requirements(pip, DEV_REQUIREMENTS_FILE)
        install_pre_commit_hooks(venv_path)
    else:
        install_requirements(pip, REQUIREMENTS_FILE)
    install_requirements(pip, NODEPS_REQUIREMENTS_FILE, no_deps=True)

    ensure_data_dir()
    print_activation_hint(venv_path)

    log.info("Environment ready.")


if __name__ == "__main__":
    main()
