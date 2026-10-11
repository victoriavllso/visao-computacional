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
| `make test` | Run the unit tests (needs `make setup-dev`) |
| `make label` | Open the hand-labeling tool (see [LABELING.md](LABELING.md)) |

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
python src/ingestion/download_dataset.py --local-dir ./my-data --token hf_xxxxxxxxxxxxxxxxxxxxx
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
python src/pipeline/data_pipeline.py --dataset-dir ./my-data
```

The pipeline runs the following stages in order:

### Stage 1 — Filter Brazil

**`src/preprocessing/filter_brazil.py`**

Filters `food.parquet` to products whose `countries_tags` field ends with `:brazil`. Output: `data/brazil/food_brazil.parquet`.

### Stage 2 — Filter front + nutrition pairs

**`src/preprocessing/filter_images.py`**

Keeps products that have both a `front_pt` and a `nutrition_pt` selected photo (Open Food Facts has no "back" key; the back of the package is the nutrition photo), and whose raw photos were both uploaded on or after `REGULATION_CUTOFF` (2023-10-09, end of the RDC 429/2020 transition for products already on the market).

Output: `data/brazil_pairs/products_pairs.parquet`, one row per product with:
- `front_*` / `nutrition_*`: image id, revision and upload time of each photo;
- `off_added_sugars_100g`, `off_saturated_fat_100g`, `off_sodium_100g`: Open Food Facts values, used as ground truth for the back pipeline (sodium in g);
- `split`: `dev` / `val` for products with all three ground-truth values (stable hash of the barcode, 70/30), `unlabeled` otherwise.

### Stage 3 — Download images

**`src/ingestion/download_images.py`**

Downloads the raw (uncropped) photos at full resolution, from the AWS mirror with the Open Food Facts server as fallback, to `data/brazil_pairs/images/{front,nutrition}/{code}.jpg`. Full resolution is needed to read the nutrition table.

### Stage 4 — Expected warnings from OFF data

**`src/rules/off.py`**

Applies the ANVISA rules (below) to the Open Food Facts values of each product. Output: `data/brazil_pairs/off_expected_warnings.parquet` with, per product, the basis (`100g`/`100ml`, from the package quantity), the Annex XVI exemption, `requires_<nutrient>` and `needs_warning` (`None` when undecidable). This is the reference answer for the image pipelines.

### Current numbers (dataset of 2026-10-10)

| Step | Products |
| :--- | ---: |
| Full dataset | 4,788,297 |
| Brazil | 36,446 |
| `front_pt` + `nutrition_pt`, both since 2023-10-09 | 2,532 |
| — `dev` | 944 |
| — `val` | 395 |
| — `unlabeled` | 1,193 |

---

## ⚖️ ANVISA Rules

**`src/rules/`** implements the front-of-pack warning rules of RDC 429/2020 and IN 75/2020. It is pure logic: Open Food Facts data and, later, the nutrition table read from the back photo both go through `rules.anvisa.assess`.

**Annex XV** — the warning is required when the amount per 100 g or 100 ml of the food *as sold* is **greater than or equal to** the limit:

| Nutrient | Solids / semi-solids (per 100 g) | Liquids (per 100 ml) | Warning |
| :--- | :---: | :---: | :--- |
| Added sugars | 15 g | 7.5 g | ALTO EM AÇÚCAR ADICIONADO |
| Saturated fat | 6 g | 3 g | ALTO EM GORDURA SATURADA |
| Sodium | 600 mg | 300 mg | ALTO EM SÓDIO |

**Annex XVI** — foods where the warning is forbidden (`rules/exemptions.py`, mapped to OFF categories):
- *Forbidden*: milks, milk powder, olive and vegetable oils, salt, infant and enteral formulas, weight control foods, supplements, alcoholic beverages, additives (table sweeteners included), and others.
- *Conditional*: fruits, vegetables, cereals, nuts, seeds, flours, plain meat and fish, eggs, fermented milks and cheeses are exempt **unless** added ingredients bring added sugars or significant saturated fat / sodium. The data can't tell, so these come out as undecided when a limit is exceeded. Meat products, requeijão and peanut butter are not exempt (ANVISA Q&A 165, 166, 170).

Sources: [IN 75/2020](https://www.in.gov.br/en/web/dou/-/instrucao-normativa-in-n-75-de-8-de-outubro-de-2020-282071143), [ANVISA Q&A on nutrition labeling, 2nd ed.](https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/alimentos/perguntas-e-respostas-arquivos/rotulagem-nutricional_2a-edicao.pdf/@@display-file/file)

---

## 🔎 Back Pipeline (nutrition table)

**`src/vision/back/`** reads the nutrition table from a photo of the back of the package:

```
photo ─► locate ─► almost no text? ─────────────────────────────► NO_TABLE (no OCR)
         │
         └─► Tesseract ─► classify ─► old format? ────────────────► OLD_FORMAT (abstain)
                                    └─► RapidOCR ─► combine ─► classify ─► TABLE / LINEAR ─► values ─► rules.anvisa.assess
                                                                        └─► NO_TABLE / OLD_FORMAT (abstain)
```

- **Classify** (`classify.py`): what the photo shows, so only photos that can be checked against RDC 429/2020 are read and the others get an explicit answer instead of a guessed reading. `TABLE` / `LINEAR` (new format, read), `OLD_FORMAT` (RDC 360/2003, per portion only, no 100 g/ml column: can't be checked), `NO_TABLE`. Classical features: ruled row lines and character-shaped blobs before OCR; after OCR, positive evidence of each format in the text: new (100 g/ml header, *açúcares totais / adicionados* rows, *"Percentual de valores diários fornecidos pela porção"*), old (*"Quantidade por porção"*, *"dieta de 2.000 kcal ou 8.400 kJ"*). Without evidence either way the photo is read again with the learned OCR before deciding: the fast OCR reading little text isn't evidence of no table. A lone *100 g* only counts next to nutrition words (it is often the net weight).

- **Locate** (classical): estimate the text height, keep the ruled lines with morphological opening (kernels of 8× / 4× the text height, so any resolution works), take the connected group of lines with the most line pixels, deskew from the angle of the horizontal lines (Hough), and **dewarp** curved/creased tables: fit a cubic to each horizontal ruled line and remap so the lines become straight (text-line dewarping, applied to ruled lines). Tables with light text on a dark background are inverted.
- **OCR**: scale the text to ~30 px, adaptive binarization, erase the ruled lines, Tesseract `--psm 11` in Portuguese, with a timeout for pathological images.
- **Parse**: match each row on its distinctive word (*adicionados* / *adic*, *saturadas*, *sódio*) with fuzzy matching, take the number aligned with the `100 g` header when it is read, else the first number to the right of the label (the 100 g/ml column comes first). Tolerates OCR errors such as `(g)` read as `(9)`; handles side-by-side and linear-text layouts.
- **Refine**: when a label is found without a value, re-read just that row's value region as a single text line.
- **Learned OCR, combined** (`pipeline.ENGINE = "combined"`, default): RapidOCR (PaddleOCR models on ONNX Runtime, CPU) reads the located and dewarped **color** crop; each nutrient takes RapidOCR's value, or Tesseract's where RapidOCR has none. RapidOCR is more accurate when it reads a value, but its text detector misses isolated small cells such as a lone `0`, which Tesseract reads. Classical binarization helps Tesseract but makes RapidOCR worse (it is trained on photos), so each engine gets its own input.
- **"Não contém quantidades significativas de …"**: nutrients listed in this sentence (RDC 429/2020 simplified tables) count as 0.

**Evaluation** against the ground truth (hand labels where they exist, see [LABELING.md](LABELING.md); Open Food Facts values otherwise):

```bash
python src/evaluation/back.py --split dev [--hand-only] [--limit N] [--no-locate] [--no-preprocess] [--tag name]
```

Each metric counts only products whose true value or decision is known. Per nutrient: *label* (row found), *found* (value read), *correct* (within 5%), *decision* (same warning decision as the truth). Photos classified as not readable get no values (the pipeline abstains). The report has the photo type confusion matrix against the hand labels and the metrics on three subsets: all photos end-to-end, predicted readable, and **hand-labeled readable** (the unmasked score).

### Results on hand-labeled dev photos (unmasked)

255 dev photos labeled by hand (187 tables, 20 linear, 15 old format, 33 without a readable table):

| | Value |
| :--- | ---: |
| Photo type accuracy | 84.7% |
| Readable tables wrongly rejected (no answer) | 4 of 187 |
| Old-format labels wrongly checked as tables | 2 of 15 |
| Readable photos with all three warning decisions right | **66.5%** |
| Decision right: added sugars / saturated fat / sodium | 84.1% / 83.5% / 82.5% |

Photos with no format evidence even after both OCR engines are treated as old format (abstain): of the 10 such photos, 6 were old format and none of the 3 tables among them was read right. With 176 labels the headline was 68.2%; the difference is within noise.

The OFF-based figures below count unreadable photos (a quarter of them in the hand labels) and OFF errors as failures, so they understate the pipeline.

### Results (full dev set, 944 products, OFF ground truth)

Share of products whose warning decision matches for all three nutrients, each step added to the previous one:

| Step | All three | Δ |
| :--- | ---: | ---: |
| Raw Tesseract on the photo (no locate, no preprocessing)¹ | 12% | |
| + locate table, scale, binarize, erase lines¹ | 27% | +15 |
| + parser fixes (units read as digits, drifted values, header variants) | 31.3% | +4 |
| + column alignment with the `100 g` header | 30.6% | −0.7 (kept: neutral) |
| + re-read missing values row by row | 33.6% | +3.0 |
| + dewarping along the ruled lines | 35.8% | +2.2 |
| + values from the label's own row first | 36.5% | +0.7 |
| + learned OCR (RapidOCR) combined with Tesseract | **44.5%** | +8.0 |

¹ measured on 150 dev products.

Per nutrient (current): added sugars 63.6%, saturated fat 68.2%, sodium 69.8% correct decisions.

The combination rule was chosen on 300 dev products; on the other 644 it gives 42.6% vs 35.1% for Tesseract alone (+7.5), so it isn't fitted to the sample. On the 300: Tesseract 41%, RapidOCR alone 23%, combined 51%. When both engines read a value and agree, it is right 71–85% of the time, a confidence signal for later. Cost: two OCR engines per photo.

Tried and dropped (all on full dev, relative to the step they were tested on):

| Technique | Result | Why |
| :--- | :--- | :--- |
| Other Tesseract layouts (`psm 3/4/6/12`) | −8 to −18 pts | `psm 11` (sparse) handles table cells best |
| OCR row by row cut along the ruled lines | −15 pts | bent lines merge rows |
| Widening the box for cut-off labels | −2 pts | locating isn't the bottleneck |
| Illumination normalization (divide by background) | −0.3 pts | |
| Sauvola thresholding | −1.5 pts | |
| Normalization + Sauvola | +0.3 pts | within noise |
| CLAHE for glare | −6 pts | amplifies texture into false ink |
| Re-reading **every** value row by row | −9 to −21 pts | single-line reading is noisier than the sparse one |
| Text heights 24 / 40 / 50 px | −1 to −4 pts | 30 px is best |
| RapidOCR alone | −18 pts vs Tesseract (300 dev) | detector misses isolated small numbers (added sugars is often `0`) |
| RapidOCR on the binarized image | worse on samples | trained on photos, binarization merges digits |
| Engine preference per nutrient (Tesseract for sugars) | −1 pt vs RapidOCR-first | |

**Where errors come from** (per nutrient reading, dev): about 40% correct, 7–15% the label isn't found, 13–21% a wrong number is picked, and 23–39% the true value is never produced by the OCR. The last group (OCR misreading digits, decimal commas) is the main limit of Tesseract on these photos, and includes errors in the OFF values themselves: in a manual check of 12 disagreements, 3 were OFF errors and the photo was read correctly. That's why the ground truth is being corrected by hand.

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