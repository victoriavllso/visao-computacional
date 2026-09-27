# Visão Computacional

This repository contains tools and scripts for computer vision and multimodal analysis using the **Open Food Facts** product database.

---

## 📋 Overview

The project leverages the Open Food Facts product dataset hosted on Hugging Face ([`openfoodfacts/product-database`](https://huggingface.co/datasets/openfoodfacts/product-database)), which contains structured product records, ingredient listings, nutritional facts, packaging details, and label references in Parquet format (`food.parquet` and `beauty.parquet`).

---

## ⚙️ Prerequisites

- **Python 3.14**
- **Disk Space**: At least **10–15 GB** of free disk space (the `food.parquet` split is ~7.9 GB).
- *(Optional)* A **Hugging Face Account & User Access Token** if you experience rate limits (the dataset is public and can be downloaded without authentication).

This project was tested extensively under Python 3.14. Other versions may work, but are not guaranteed to.

---

## 🚀 Environment Setup

Run the included setup script from the project root:

```bash
# Regular setup (runtime dependencies only)
python3 src/setup_env.py

# Developer setup (includes black, pre-commit, and git hooks)
python3 src/setup_env.py --dev
```

The setup script automatically:
1. Creates a Python virtual environment (`.venv/` at the project root).
2. Upgrades `pip` to the latest version.
3. Installs all required packages.
4. Creates the `data/` directory for the dataset.
5. *(With `--dev`)* Installs developer tooling and registers the pre-commit git hook.

#### Options:
- `--venv-dir <path>`: Specify a custom virtual environment folder (default: `.venv`).
- `--dev`: Install developer dependencies and set up pre-commit hooks.

#### Activate the Virtual Environment:
After the setup script finishes, activate the environment in your shell:

```bash
source .venv/bin/activate
```

---

## 📥 Downloading the Dataset

> Ensure your virtual environment is **activated** before running the download script so the `hf` CLI tool is available.

Download the dataset into `data/product-database/`:

```bash
python src/download_dataset.py
```

If the dataset files are already present in the destination directory, the download is skipped automatically.

#### Optional Arguments:
- `--local-dir <path>`: Destination folder (default: `data/product-database`).
- `--token <your_hf_token>`: Hugging Face access token (optional, useful for higher API rate limits).

```bash
# Example with a custom target directory and token
python src/download_dataset.py --local-dir ./my-data --token hf_xxxxxxxxxxxxxxxxxxxxx
```

---

## 📊 Dataset Details

The dataset files are stored in Apache Parquet format:

| Split | Filename | Approximate Size | Description |
| :--- | :--- | :--- | :--- |
| **Food** | `food.parquet` | ~7.89 GB | Complete food product catalogue including labels, ingredients, nutriscore, and packaging data |
| **Beauty** | `beauty.parquet` | ~61.4 MB | Personal care and beauty products dataset |

---

## 🧹 Code Formatting

This project uses [black](https://github.com/psf/black) for code formatting, enforced via a pre-commit hook. Running `setup_env.py --dev` installs and registers it automatically.

Before committing, run black across all files to avoid the hook aborting your commit:

```bash
pre-commit run --all-files
git add .
git commit -m "your message"
```
