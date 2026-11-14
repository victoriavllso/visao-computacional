# Visão Computacional

This repository contains tools and scripts for computer vision and multimodal analysis using the **Open Food Facts** product database.

---

## 📋 Overview

The project leverages the Open Food Facts product dataset hosted on Hugging Face ([`openfoodfacts/product-database`](https://huggingface.co/datasets/openfoodfacts/product-database)), which contains structured product records, ingredient listings, nutritional facts, packaging details, and label references in Parquet format (`food.parquet` and `beauty.parquet`).

---

## ⚙️ Prerequisites

- **[pixi](https://pixi.sh)**: installs Python 3.13, Tesseract (with the Portuguese model) and every Python package, at the exact versions in `pixi.lock`. No root needed; works on Linux, macOS (Apple Silicon, 14+) and Windows.
- **Disk Space**: At least **15 GB** of free disk space (the `food.parquet` split is ~7.9 GB, the images ~5 GB).
- *(Optional)* A **Hugging Face Account & User Access Token** if you experience rate limits (the dataset is public and can be downloaded without authentication).

Install pixi once (see [pixi.sh](https://pixi.sh) for other systems):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

---

## ⚡ Quick Start

Everything runs inside the pixi environment, through `pixi run <task>` (tasks in `pixi.toml`). The `make` targets are shortcuts for the same tasks:

| Command | Description |
| :--- | :--- |
| `pixi install` (`make setup`) | Create the environment in `.pixi/` from `pixi.lock` |
| `make setup-dev` | Same as above, plus black, pre-commit and the git hook |
| `pixi run download` | Download the Open Food Facts dataset |
| `pixi run pipeline` | Run the full processing pipeline |
| `pixi run test` | Run the unit tests |
| `pixi run eval --split dev --hand-only` | Evaluate the back pipeline (options below) |
| `pixi run eval-front --split dev [--method yolo]` | Evaluate the front pipeline ("ALTO EM" warnings) |
| `pixi run -e detector train-detector` | Train the front warning detector (GPU; environment `detector`, Linux) |
| `pixi run label` | Open the hand-labeling tool (see [LABELING.md](LABELING.md)) |
| `pixi run compact-labels` | Merge the hand label files into one, one entry per product |
| `pixi run -e web app` | Open the web app at http://localhost:8000 (see [web/README.md](web/README.md)) |

Typical first-time flow:

```bash
pixi install
pixi run download
pixi run pipeline
```

Other scripts run the same way, e.g. `pixi run python src/evaluation/back.py --help`, or open a shell inside the environment with `pixi shell`.

---

## 🚀 Environment

`pixi.toml` is the only definition of the environment, and `pixi.lock` pins the exact version of every package (conda and PyPI) for each platform, so every machine runs the same code with the same libraries. The OCR results depend on them: library versions changed the score by several points.

- Add or update a dependency by editing `pixi.toml` and running `pixi install`, which updates `pixi.lock`; commit both.
- The environment lives in `.pixi/` (git-ignored); delete it to start over.

### On the UFSC JupyterHub server

The server can't run containers, but pixi needs no root. Only `~/privado` survives a container restart, so keep pixi and the project there:

```bash
cd ~/privado
curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=~/privado/.pixi PIXI_NO_PATH_UPDATE=1 sh
export PATH=~/privado/.pixi/bin:$PATH   # in every new terminal
git clone https://github.com/victoriavllso/visao-computacional.git
cd visao-computacional
pixi install && pixi run download && pixi run pipeline
```

The machine is shared: pass `--workers` to the evaluation (e.g. `--workers 32`) instead of the default, which uses every core.

---

## 📥 Downloading the Dataset

```bash
pixi run download
```

If the dataset files are already present in `data/product-database/`, the download is skipped automatically.

The dataset is updated daily, so the download is pinned to one snapshot (`HF_REVISION` in `src/utils/constants.py`, 2026-10-18, 2,571 product pairs, with every hand-labeled product): everyone gets the same products. If your copy came from another snapshot, `pixi run download` replaces it; then delete the filtered files and run `pixi run pipeline` again (images already downloaded are kept):

```bash
rm -r data/brazil data/brazil_pairs/*.parquet
pixi run pipeline
```

#### Options (advanced):
If you need a custom directory or token, pass the options to the task:
- `--local-dir <path>`: Destination folder (default: `data/product-database`).
- `--token <your_hf_token>`: Hugging Face access token (optional, useful for higher API rate limits).

```bash
# Example with a custom target directory and token
pixi run download --local-dir ./my-data --token hf_xxxxxxxxxxxxxxxxxxxxx
```

---

## 🔁 Running the Pipeline

With the dataset downloaded:

```bash
pixi run pipeline
```

#### Options (advanced):
- `--dataset-dir <path>`: Directory containing the raw dataset files (default: `data/product-database`).

```bash
pixi run pipeline --dataset-dir ./my-data
```

The pipeline runs the following stages in order:

### Stage 1 — Filter Brazil

**`src/preprocessing/filter_brazil.py`**

Filters `food.parquet` to products whose `countries_tags` field ends with `:brazil`. Output: `data/brazil/food_brazil.parquet`.

### Stage 2 — Filter front + nutrition pairs

**`src/preprocessing/filter_images.py`**

Keeps products that have both a `front_pt` and a `nutrition_pt` selected photo (Open Food Facts has no "back" key; the back of the package is the nutrition photo), and whose raw photos were both uploaded:

- on or after `REGULATION_CUTOFF` (2023-10-09, end of the RDC 429/2020 transition for products already on the market), and
- before `IMAGE_END_CUTOFF` (2026-10-18).

The upper bound freezes the set of images at a fixed calendar date. That keeps the dataset reproducible (re-running the filter later yields the same pairs) and gives a stable pool of photos for the manual labeling of the back of the package, so labels do not drift as new images keep arriving on Open Food Facts.

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
pixi run eval --split dev [--hand-only] [--limit N] [--no-locate] [--no-preprocess] [--tag name] [--debug] [--ocr-cache] [--rapidocr-model small|medium]
```

`--ocr-cache` stores every OCR result (Tesseract words, RapidOCR lines with their four-corner boxes, Tesseract OSD) in `data/cache/back_ocr/<code>.json` and reuses it on later runs (`vision/back/cache.py`). A change to the rules after the OCR (parse, classify, combine, orientation choice) is then scored on the same reading without running the OCR again: on 50 dev products, 3 s instead of 113 s, with identical results. Bump `cache.VERSION` after changing anything before the OCR (locate, dewarp, preprocessing). Building the cache on the server and copying it here lets rule changes be measured on the whole dev set locally:

```bash
pixi run eval --split dev --hand-only --workers 32 --ocr-cache   # on the server
tar czf back_ocr_cache.tar.gz -C data/cache back_ocr             # bring this file back
tar xzf back_ocr_cache.tar.gz -C data/cache                      # locally
```

`--rapidocr-model medium` uses PP-OCRv6's medium detection and recognition models instead of the small ones (about 4.7× slower per RapidOCR call); its cache entries are kept apart from the small model's.

Results are saved to `data/evaluation/back_<split>_<timestamp>_<tag>/` (CSV + Markdown report). Symlinks in `data/evaluation/latest/` (`back_dev.csv`, `back_val.csv` and `.md` reports) always point to the most recent run for each split. Other tools (labeling server, confusion matrix) read from `latest/` by default; pass an explicit path to use a specific run.

Use `--debug` to save intermediate images and a JSON summary for every product to `data/debug/<code>/`. The output per product is:

| File | Contents |
| :--- | :--- |
| `01_original.jpg` | Original input image |
| `02_located_crop.jpg` | Crop after table location (before dewarp) |
| `03_line_masks.jpg` | Horizontal (red) + vertical (blue) line masks |
| `04_binarized.jpg` | Binarized image fed to Tesseract |
| `05_dewarped_crop.jpg` | Crop after deskew + dewarp |
| `06_ocr_tesseract.jpg` | Tesseract word bounding boxes |
| `07_ocr_rapidocr.jpg` | RapidOCR word bounding boxes |
| `08_parsed_values.jpg` | Nutrient labels + read values annotated (flagged high in red) |
| `summary.json` | Classification, features, parsed values, ANVISA warning flags, and evaluation comparison context |

Debug mode runs sequentially (no multiprocessing) for cleaner output. Combine with `--limit` to debug a few products quickly:

```bash
pixi run eval --split dev --hand-only --limit 5 --debug
```

Each metric counts only products whose true value or decision is known. Per nutrient: *label* (row found), *found* (value read), *correct* (within 5%), *decision* (same warning decision as the truth). Photos classified as not readable get no values (the pipeline abstains). The report has the photo type confusion matrix against the hand labels and the metrics on three subsets: all photos end-to-end, predicted readable, and **hand-labeled readable** (the unmasked score).

### Final score: validation set (held out, hand-labeled)

All 399 val products labeled by hand (284 tables, 36 linear, 46 old format, 33 without a readable table), on the pinned 2026-10-18 snapshot, with the 2026-11-12 back pipeline (parser fixes, rotated photos, no nearby-row guesses; all decided on dev). Nothing was tuned on val, so this is the honest estimate:

| | 2026-11-04 | **2026-11-12** |
| :--- | ---: | ---: |
| **Readable photos with all three warning decisions right** | 71.6% | **71.8%** |
| Decision right: added sugars / saturated fat / sodium | 85.2% / 81.6% / 82.7% | 86.6% / 83.3% / 82.2% |
| When the pipeline says "readable": all three right | 73.3% | 73.2% |
| Photo type accuracy | 81.9% | 81.7% |
| Readable tables left without an answer | 2 of 282 | 2 of 284 |
| Old-format labels wrongly checked as tables | 17 of 46 | 18 of 46 |

The 2026-11-12 changes count an unread value as an error here (they turn guesses into unknowns), so the back's score barely moves; their effect shows in the verdicts below ("Whole system").

The learned OCR library made the difference: on the same photos, labels and code, `rapidocr_onnxruntime` 1.2.3 gave 57.1% (dev: 62.8%) and `rapidocr` 3.10 gives 71.6% (dev: 77.4%), +14.5 points. The gain is in reading values (added sugars read right: 62% → 81%); the old library's text detector merged neighboring cells into one number (`338 | 338` → `338338`). Old-format detection is still the weakest point.

Val is about 6 points below dev: the decisions made on dev errors fit dev better than new photos. Val also has far more old-format labels (12%, against 2% in dev), and old-format detection is what generalizes worst. Improvements are developed on dev; val is only for checking them at the end.

### Results on hand-labeled dev photos (unmasked)

929 dev photos labeled by hand (837 tables, 27 linear, 15 old format, 50 without a readable table), same snapshot and environment as val:

| | Value |
| :--- | ---: |
| Photo type accuracy | 87.0% |
| Readable tables wrongly rejected (no answer) | 52 of 837 |
| Old-format labels wrongly checked as tables | 4 of 15 |
| Readable photos with all three warning decisions right | **77.4%** |
| Decision right: added sugars / saturated fat / sodium | 88.2% / 84.1% / 80.4% |

Photos with no format evidence even after both OCR engines are treated as old format (abstain).

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

## 🔖 Front Pipeline ("ALTO EM" warnings)

**`src/vision/front/`** answers: which of the three RDC 429/2020 warnings (*açúcar adicionado*, *gordura saturada*, *sódio*) does the front photo show? The warning has a fixed design (a black panel with a magnifier, "ALTO EM" and the nutrients, in one bold font), and two methods read it behind one function:

```python
from vision.front.warnings import read_front

result = read_front("front.jpg", method="yolo")  # or method="ocr"
result.warnings   # {Nutrient.ADDED_SUGARS: True, Nutrient.SATURATED_FAT: False, ...}
result.boxes      # panels found by the detector ("yolo")
```

```
"ocr"   photo ─► RapidOCR (whole photo) ─► "ALTO EM" anchor ─► lines near it, same text size ─► nutrient words
"yolo"  photo ─► YOLO (panel boxes) ─► crop + margin, upscale ─► RapidOCR (crop) ─► nutrient words
```

- **OCR** (`warnings.py`): RapidOCR reads the whole photo; a line reading "ALTO EM" (fuzzy) is the anchor, and the lines near it with about its text height are the panel (ingredients and claims elsewhere are in other sizes). Each nutrient is any word of its phrase (*adicionad* / *açúcar*, *saturad* / *gordura*, *sódio*), matched with tolerance for OCR errors (similarity ≥ 0.8). Panels with Spanish words (*grasa*, *azúcares*, *calorías*: imported products with the Spanish warning) don't count.
- **YOLO + OCR** (`detector.py` + `warnings.py`): a YOLO26 nano detector, trained on the hand-drawn panel boxes, finds the panels (one class; one box per panel, not per nutrient); RapidOCR reads each crop (15% margin, short side scaled up to ≥ 480 px), and the nutrients are matched in the crops' text. Inside a detected panel the only words are "ALTO EM" and the nutrients, so the match is looser (similarity ≥ 0.7) and "ALTO EM" doesn't have to be read. The detector runs on ONNX Runtime (already used by the OCR), so the default environment runs it; ultralytics and torch are only needed to train it.

Why a detector, against the classical-first rule of the project: most of the whole-photo OCR's misses are panels where "ALTO EM" isn't read (small or tilted panels; RapidOCR shrinks photos to 2,000 px on the long side, so a small panel's text gets too small), and a crop of the panel at full resolution fixes that. The panel has a fixed design and its boxes were cheap to label (268 products), the best case for a detector. A classical locator of the black panel (threshold + contours + aspect ratio) wasn't tried in this round; it is the natural comparison to add.

**Evaluation** against the front hand labels (see [LABELING.md](LABELING.md), section 3b): scores count the photos labeled "front of the package" (`ok`), whatever they show (no masking); photos whose warning can't be read (`unreadable`) are reported apart, and photos that aren't the front are left out.

```bash
pixi run eval-front --split dev [--method ocr|yolo] [--workers 32] [--limit N] [--no-cache]
```

Results go to `data/evaluation/front_<split>_<timestamp>_<tag>/` (CSV per product + Markdown report), linked from `data/evaluation/latest/front_<split>.{csv,md}` (`front_<split>_yolo.*` for YOLO). The OCR lines are cached in `data/cache/front_ocr/` and the detector boxes + crop lines in `data/cache/front_yolo/`, so rule variants are scored on the same reading; `--no-cache` reruns them (to time them). The YOLO report adds the detector's box metrics (IoU 0.5 against the labeled boxes), and every report the time per photo.

**Training the detector** (GPU; the `detector` environment, Linux only, has ultralytics + torch with CUDA, several GB):

```bash
pixi run -e detector train-detector                         # 5 cross-validation folds + the final model
pixi run -e detector train-detector --folds all --install   # final model only → models/front_detector.onnx
```

Only dev photos train it (494 labeled the front, 134 with panels; photos without panels are negatives). To score it on dev without testing on training photos, dev is split in 5 folds by an MD5 hash of the barcode: fold *k*'s model trains on the other four and predicts fold *k*, so every dev photo is scored by a model that didn't see it (`--cv-tag` in the evaluation). No model selection on held-out photos: each model trains a fixed 100 epochs (YOLO26n from COCO weights, `imgsz=1024`, batch 16, seed 0, no horizontal flips since mirrored text doesn't occur, ±10° rotation) and keeps the last weights. The final model trains the same way on all of dev and is the one scored on val. A model takes ~10 min on one H100 of the UFSC server; the ONNX file is 9.6 MB. The weights aren't in git yet (the trained model is on the server, `~/privado/visao-computacional/data/detector/yolo26n_1024/all.onnx`): retrain with the command above, or copy that file to `models/front_detector.onnx`. Training isn't bit-exact (images cached in RAM), so a retrained model can differ by a photo or two.

### Final score: validation set (held out, hand-labeled)

391 val photos labeled by hand (386 the front, 5 with an unreadable warning; 127 show a warning). Everything was decided on dev; val was run once per method, at the end:

| | OCR | YOLO + OCR |
| :--- | ---: | ---: |
| **All three warnings right** | **93.3%** | **97.7%** |
| Photos with a warning: all three right | 81.9% | 94.5% |
| Added sugars: precision / recall / F1 | 92.7 / 92.7 / 92.7% | 98.7 / 95.1 / 96.9% |
| Saturated fat: precision / recall / F1 | 93.7 / 84.3 / 88.7% | 98.6 / 97.1 / 97.8% |
| Sodium: precision / recall / F1 | 93.8 / 78.9 / 85.7% | 100.0 / 89.5 / 94.4% |
| Any warning: precision / recall | 97.3 / 86.6% | 98.4 / 94.5% |
| Time per photo, median (mean) | 5.05 s (5.27 s) | 0.39 s (1.38 s) |
| Detector: AP@0.5 / box recall / box precision | | 94.2 / 93.4 / 92.7% |

Times: one process, ONNX Runtime on one thread, CPU, on the shared server (load far above its cores, so absolute numbers are pessimistic; the two methods ran one after the other). YOLO + OCR is faster because RapidOCR reads only the small crops: photos without a panel cost only the detection (~0.35 s), a photo with a panel ~3 s.

The OCR method loses 3.2 points from dev to val (96.5% → 93.3%), YOLO + OCR none (97.6% → 97.7%): the OCR rules were fitted to dev's errors. On val, 14 of the OCR's 26 errors are panels where "ALTO EM" isn't read, 3 miss a nutrient next to a read anchor, and 9 add a nutrient (3 of them on photos labeled without a warning). YOLO + OCR's 9 val errors: 3 panels not detected, 4 detected but misread (a 43 × 38 px panel, a thin 311 × 36 one, a vertical one), and 2 nutrients found on panels labeled without a warning (partly visible panels).

### Results on dev (cross-validation)

490 dev photos labeled the front, 4 unreadable; 130 show a warning. YOLO + OCR on dev is the out-of-fold prediction (each photo scored by the model that didn't train on it).

| | OCR | YOLO + OCR |
| :--- | ---: | ---: |
| **All three warnings right** | **96.5%** | **97.6%** |
| Photos with a warning: all three right | 88.5% | 92.3% |
| Added sugars: precision / recall / F1 | 97.4 / 90.4 / 93.8% | 100.0 / 94.0 / 96.9% |
| Saturated fat: precision / recall / F1 | 97.0 / 94.1 / 95.5% | 96.9 / 92.6 / 94.7% |
| Sodium: precision / recall / F1 | 95.7 / 95.7 / 95.7% | 100.0 / 87.0 / 93.0% |
| Any warning: precision / recall | 98.3 / 91.5% | 98.4 / 93.8% |
| Time per photo, median (mean) | 6.41 s (6.85 s) | 0.44 s (1.33 s) |

Detector, out of fold (494 photos, 134 labeled boxes; ultralytics' mAP@0.5 per held-out fold: 0.96 / 0.98 / 0.99 / 0.99 / 0.96):

| Score ≥ | AP@0.5 | Box recall | Box precision | Photos with a panel: found | Photos without: a box |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 95.3% | 95.5% | 85.9% | 96.3% | 3.3% |
| **0.1** (used) | 95.3% | 94.8% | 92.0% | 95.5% | 1.4% |
| 0.25 | 95.3% | 94.0% | 96.2% | 94.8% | 0.8% |
| 0.5 | 95.3% | 93.3% | 99.2% | 93.3% | 0.3% |

**OCR rules**, each added to the previous one (dev):

| Rule | All three | Δ | Any warning P / R |
| :--- | ---: | ---: | :--- |
| Baseline: anchor + panel, distinctive word per nutrient, reach 10 heights, height ratio 0.6–1.6 | 95.7% | | 98.3 / 91.5% |
| + any word of the phrase (*gordura* read without *saturada*) | 95.7% | 0 | 97.5 / 91.5% |
| + reach 14 heights, height ratio 0.5–2.0 (horizontal panels, perspective) | 96.3% | +0.6 | 97.5 / 91.5% |
| + Spanish panels (by their words) don't count | **96.5%** | +0.2 | 98.3 / 91.5% |

**YOLO + OCR settings** (dev, all three right; the chosen settings give 97.6%):

| Setting | All three | Note |
| :--- | ---: | :--- |
| Box score ≥ 0.05 / **0.1** / 0.25 / 0.5 | 97.8 / **97.6** / 97.6 / 97.6% | 0.05: +1 photo for 14% false boxes (an OCR call each) |
| Match similarity in crops 0.8 / **0.7** / 0.6 | 97.3 / **97.6** / 82.2% | 0.7 reads *CATURA*, *ADICUCNADO*; 0.6 matches almost anything |
| Crop margin 0.05 / **0.15** / 0.3 | 97.6 / **97.6** / 97.8% | within noise (1 photo) |
| Crop short side: no upscale / **480** / 960 px | 97.3 / **97.6** / 97.1% | within noise |
| Detector at `imgsz` 640 instead of **1024** | 97.6% | AP@0.5 94.8 vs 95.3%, more false boxes (4.4% vs 1.4% of photos without a panel); detection 0.14 s vs 0.39 s |

Tried and dropped:

| Technique | Result (dev, all three right) | Why |
| :--- | :--- | :--- |
| Reject "ALTO EN" as an anchor (Spanish warning) | 96.3% → 96.1% | the OCR reads the real "ALTO EM" as *ALTOEN* (2 warnings lost, 1 Spanish one avoided); replaced by Spanish words in the panel |
| Nutrient words anywhere on a photo with an anchor | 90.6% | ingredients and claims: sodium precision 56% |
| YOLO crops: count only crops where "ALTO EM" is read | 97.6% → 96.1% | the anchor is misread more often than the nutrients |
| YOLO + OCR **or** whole-photo OCR (union) | 97.8% (+0.2) on dev, 96.4% on val (−1.3) | costs both methods and adds the OCR's false positives (val was only scored, not used to drop it) |

**Where errors remain** (YOLO + OCR, dev, 12 photos): 5 panels not detected (2 of them vertical, one 64 × 17 px), 5 detected but misread by the OCR (*PALTOEM* alone, *ANMSUAR*, *SÓO1O*), and 2 false positives (one on a partly visible panel the hand label marks without a warning).

---

## ✅ Whole system: photos → verdict (the app)

The app (`web/`, see its README) runs the back pipeline, the rules and the front pipeline (YOLO + OCR) on a pair of photos and gives a verdict: compliant, inconclusive or non-compliant. `pixi run eval-e2e --split dev` compares that verdict with the one the hand labels of both sides give (same rules, same `summarize()`), on every product with both sides labeled.

Dev, 494 products (2026-11-12, server, 32 workers; history of the changes below):

| Labels \ App | compliant | inconclusive | non-compliant |
| :--- | ---: | ---: | ---: |
| **compliant** (309) | **244** | 58 | 7 |
| **inconclusive** (126) | 53 | **66** | 7 |
| **non-compliant** (59) | 3 | 9 | **47** |

- Same verdict as the labels: **72.3%**; when the app decides (not inconclusive, 361 products): **80.6%**.
- Violations: 47 of 59 caught (79.7%), only 3 called compliant.
- False accusations: 7 of 309 compliant products (2.3%) called non-compliant; of the app's 61 "non-compliant" answers, 47 are right (77%).
- Truly inconclusive products (old-format tables, values nobody could read) called compliant: 51 of 126, the old-format detection weakness of the back pipeline.

**The false accusations come from the back.** Of the first run's 21, 20 were back misreadings (the front warnings were read right) and 1 a missed front warning. Their causes: values from the wrong column or row (%VD or per portion, ~9), physically impossible values (477 g of saturated fat per 100 g, 411,415 mg of sodium, ~6), thousands read as decimals ("1.180 mg" as 1.18, 2), values not read (~3). Two parser fixes, measured on dev:

| Change | Back: all three right | E2E: same verdict | False accusations | Violations caught |
| :--- | ---: | ---: | ---: | ---: |
| Before | 77.4% | 72.1% | 21 | 48 / 59 |
| + "1.180" read as 1180 (dot + three digits = thousands); amounts above 100 g or 40,000 mg of sodium per 100 g/ml dropped (another row, or unknown) | 77.6% | **72.5%** | **18** | 47 / 59 |
| + photos on their side or upside down turned (Tesseract OSD on the whole photo, OSD's angle applied directly) | 79.5% | 73.3% | 12 | 47 / 59 |
| + a turn kept only if the photo reads better turned (OSD, or 90/270 when most text lines are taller than wide) | **79.6%** | **73.7%** | 13 | 47 / 59 |
| + no values from beyond the label's row (nearby rows, nearest number, row re-read): unknown instead (inconclusive) | 77.7% | 72.3% | **7** | 47 / 59 |

Of the 20 readings it changed on dev, 5 are thousands fixed and 15 impossible values dropped; 3 false accusations went (2 now right, 1 inconclusive), none appeared. Fewer than the ~8 expected: %VD values such as 100, 90 or 60 are physically possible and pass the limit, and some dropped values were replaced by another wrong row. The violation lost had been "caught" by an impossible reading (359 g of sugar). A closer look at the 17 that came from the back: 7 were tables photographed on their side (the parser took the nutrients as a row and the numbers of another row), ~5 were values Tesseract filled in where RapidOCR read none ("3,2" as "32;"), 1–2 a sentence "não contém quantidades significativas" read as a table row, and only a few the per-portion or %VD column. Turning rotated photos (vision/back/orient.py) fixed 6 of the 7. Applied directly, OSD also turned 2 upright photos upside down (180 degrees, confidence up to 4.6); with every turn checked by the reading, 15 dev products read better and none worse. The sentence now wins over a number read on or below its first line (its own text), while a table row above it still wins as before.

**Where the values come from.** Recorded per value (`<nutrient>_source` in the back CSV), on readable dev photos: the label's own row is right 90–95% of the time (RapidOCR and Tesseract alike: Tesseract's 224 values gave 214 right decisions and 10 wrong, so its fallback stays), the "não contém" sentence 97–100%, but the rules beyond the row only 12–36%: the `100 g` column on nearby rows, the nearest number, and the row-by-row re-read. Those caused 6 of the 13 remaining false accusations, and their "right" decisions were mostly a wrong value on the right side of the limit. Turned off (`parse.NEARBY_ROWS`, `pipeline.REFINE`), measured end to end: false accusations 13 → 7, violations caught unchanged (and one fewer called compliant), 13 more compliant products answered inconclusive, and the reading ~25% faster. The back's "all three right" drops (79.6 → 77.7%) because it counts an unknown value as wrong; the verdict, which says "inconclusive" for it, is what the app shows.

**Val** (held out, once, 2026-11-12), 391 products with both sides labeled:

| Labels \ App | compliant | inconclusive | non-compliant |
| :--- | ---: | ---: | ---: |
| **compliant** (271) | **188** | 67 | 16 |
| **inconclusive** (89) | 7 | **74** | 8 |
| **non-compliant** (31) | 3 | 8 | **20** |

- Same verdict as the labels: **72.1%**; when the app decides (242 products): **86.0%** right.
- Violations: 20 of 31 caught (64.5%), 3 called compliant.
- False accusations: 16 of 271 compliant products (5.9%), more than on dev (2.3%); of the app's 44 "non-compliant" answers, 20 are right (45%). The changes that cut them on dev were chosen from dev's errors, and val's photos have other ones: they are the next thing to look at, on dev.
- More compliant products come out inconclusive than on dev (25% vs 19%), and far fewer truly inconclusive ones (old-format tables) come out compliant (7 of 89 vs 53 of 126 on dev).

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