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

## ⚡ Quick Start

All entry points are available as `make` targets from the project root:

| Command | Description |
| :--- | :--- |
| `make setup` | Create `.venv` and install runtime dependencies |
| `make setup-dev` | Same as above, plus dev tools and pre-commit hooks |
| `make download` | Download the Open Food Facts dataset |
| `make pipeline` | Run the full processing pipeline |

Typical first-time flow:

```bash
make setup
source .venv/bin/activate
make download
make pipeline
```

---

## 🚀 Environment Setup

Run the setup target from the project root:

```bash
# Regular setup (runtime dependencies only)
make setup

# Developer setup (includes black, pre-commit, and git hooks)
make setup-dev
```

The setup script automatically:
1. Creates a Python virtual environment (`.venv/` at the project root).
2. Upgrades `pip` to the latest version.
3. Installs all required packages.
4. Creates the `data/` directory for the dataset.
5. *(With `make setup-dev`)* Installs developer tooling and registers the pre-commit git hook.

#### Options (advanced):
If you need a custom venv location, call the script directly:
- `--venv-dir <path>`: Specify a custom virtual environment folder (default: `.venv`).
- `--dev`: Install developer dependencies and set up pre-commit hooks.

```bash
# Example with a custom venv directory
python3 src/setup/setup_env.py --venv-dir ./my-venv

# Developer setup with a custom venv directory
python3 src/setup/setup_env.py --dev --venv-dir ./my-venv
```

#### Activate the Virtual Environment:
After setup finishes, activate the environment in your shell:

```bash
source .venv/bin/activate
```

---

## 📥 Downloading the Dataset

> Ensure your virtual environment is **activated** before downloading so the `hf` CLI tool is available.

```bash
make download
```

If the dataset files are already present in `data/product-database/`, the download is skipped automatically.

#### Options (advanced):
If you need a custom directory or token, call the script directly:
- `--local-dir <path>`: Destination folder (default: `data/product-database`).
- `--token <your_hf_token>`: Hugging Face access token (optional, useful for higher API rate limits).

```bash
# Example with a custom target directory and token
python src/setup/download_dataset.py --local-dir ./my-data --token hf_xxxxxxxxxxxxxxxxxxxxx
```

---

## 🔁 Running the Pipeline

With the dataset downloaded and the virtual environment activated:

```bash
make pipeline
```

#### Options (advanced):
- `--dataset-dir <path>`: Directory containing the raw dataset files (default: `data/product-database`).

```bash
python src/pipeline/pipeline.py --dataset-dir ./my-data
```

The pipeline runs the following stages in order:

### Stage 1 — Preprocessing

**Filter Brazil** (`src/pipeline/preprocess/filter_brazil.py`)

Filters `food.parquet` to products whose `countries_tags` field ends with `:brazil`, producing a much smaller dataset scoped to products sold in Brazil. Output is written to `data/brazil/food_brazil.parquet`. A before/after record count is printed to stdout at the end of the stage.

The filter step can also be run standalone:

```bash
python src/pipeline/preprocess/filter_brazil.py
python src/pipeline/preprocess/filter_brazil.py --dataset-dir ./my-data
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

This project uses [black](https://github.com/psf/black) for code formatting, enforced via a pre-commit hook. Running `make setup-dev` installs and registers it automatically.

Before committing, run black across all files to avoid the hook aborting your commit:

```bash
pre-commit run --all-files
git add .
git commit -m "your message"
```