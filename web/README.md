# Web app

Send a photo of the **front** and of the **back** of a food package and see
whether its "ALTO EM" warnings follow ANVISA's front-of-pack rule
(RDC 429/2020, IN 75/2020):

- ✅ **De acordo com a ANVISA**: every required warning is printed, and no other.
- ⚠️ **Não foi possível decidir**: a value or a warning couldn't be read, or the
  back photo has no new-format table (old per-portion format, no table).
- ❌ **Viola as regras**: a required warning is missing (e.g. "Deveria ter ALTO
  EM SÓDIO e não tem"), or a warning is printed below the limit.

The result shows the reasons, the warnings required vs printed, the nutrition
table read from the back photo, the photos, and a JSON export. Past analyses
stay in the **Histórico**, where they can be reopened or deleted.

The analysis runs the project's pipelines (see "Pipelines" below): a few
seconds per product on a laptop CPU, the first one slower while the models
load.

## Running

```bash
pixi run -e web app          # http://localhost:8000
pixi run -e web test-web     # API tests (FastAPI TestClient)
```

The `web` environment adds FastAPI, uvicorn and python-multipart to the
default one, which stays light. History and photos are kept in `data/app/`
(git-ignored; `APP_DATA_DIR` changes it): `history.sqlite3` and
`photos/<id>/`. Delete the folder to start over.

The server listens on 127.0.0.1 only. To open it on a phone on the same
network, run `pixi run -e web uvicorn app.main:app --app-dir web --host 0.0.0.0`
and open `http://<computer's IP>:8000`. Note that browsers only offer the
camera button on phones; on desktop both buttons open the file picker.

### On the UFSC JupyterHub server

Each person runs the app on their own JupyterHub server and opens it through
JupyterHub's proxy, so it only needs a browser (also on a phone, after
signing in to JupyterHub there). The app needs only the repository (the
detector weights are in `models/`), not the dataset. In a JupyterLab
terminal:

```bash
export PATH=~/privado/.pixi/bin:$PATH     # pixi set up as in the main README
cd ~/privado/visao-computacional          # the clone, in privado/ (persists)
git pull
pixi install -e web
nohup pixi run -e web app > ~/privado/app.log 2>&1 &
```

Then open `https://jupyter.vlab.ufsc.br/user/<your user>/proxy/8000/` (the
trailing slash matters). Only the owner of the JupyterHub server can open
that address: each colleague runs their own copy, with their own history.
The app stops when the container stops; run the last command again after
signing in. To stop it: `pkill -f "uvicorn app.main:app"`.

## Layout

| File | Role |
| :--- | :--- |
| `app/main.py` | FastAPI app: the page and the JSON API |
| `app/pipeline.py` | **Adapter to the vision pipelines** (and a mock for tests) |
| `app/storage.py` | History in SQLite, photos on disk |
| `app/images.py` | Upload checks (JPEG/PNG/WebP by content, 20 MB), smaller copies |
| `static/` | The page: plain HTML/CSS/JS, no build step |
| `tests/` | API and verdict tests |

API: `POST /api/analyses` (multipart: `front`, `back`, optional `name`),
`GET /api/analyses`, `GET|DELETE /api/analyses/<id>`,
`GET /api/analyses/<id>/export` (JSON download),
`GET /api/analyses/<id>/photos/<front|back>/<view|thumb>`.

## Plugging in the pipelines

The backend calls only `app.pipeline.analyze(front_path, back_path)`, which
returns:

```python
{
  "verdict": "compliant" | "inconclusive" | "non_compliant",
  "reasons": ["Deveria ter ALTO EM SÓDIO e não tem.", ...],
  "nutrition": {"basis": "100g" | "100ml" | None,
                "added_sugars_g": float | None,
                "saturated_fat_g": float | None,
                "sodium_mg": float | None},
  "required_warnings": {"added_sugars": bool | None, "saturated_fat": ..., "sodium": ...},
  "found_warnings":    {"added_sugars": bool | None, "saturated_fat": ..., "sodium": ...},
  "back_photo_type": "table" | "linear" | "old_format" | "no_table" | "uncertain",
  "versions": {"back": str, "front": str, "rules": str},
  "mock": bool,
}
```

`None` means "couldn't tell". `read_photos()` runs the pipelines and
`summarize()` turns required vs found into the verdict and its reasons:

- **back**: `vision.back.pipeline.read_back` → photo type and the nutrition
  table per 100 g/ml (none when the photo isn't a new-format table: old
  format, no table);
- **rules**: `rules.anvisa.assess(facts, Exemption.NONE)` → required
  warnings (True / False / None when a value couldn't be read). The app
  doesn't ask for the product category yet, so no exemption is assumed;
- **front**: `vision.front.warnings.read_front` → printed warnings. It uses
  YOLO + OCR when `models/front_detector.onnx` is there, OCR on the whole
  photo otherwise; `APP_FRONT_METHOD=ocr|yolo` forces one. A panel found with
  no nutrient read counts as "couldn't tell", not as "no warning".

Analyses run one at a time (the OCR models aren't shared across threads).

`APP_PIPELINE=mock` replaces the readings with made-up ones derived from a
hash of the two photos (instant; the app's tests use it), and the page shows
a "resultado simulado" notice; `APP_MOCK_VERDICT=compliant|inconclusive|
non_compliant` then forces a verdict.

## Known limitations

- HEIC photos (iPhone's format) are rejected; iOS Safari usually converts
  them to JPEG when uploading through the page, other apps may not.
- No accounts: everyone who can open the page sees the same history.
- The analysis runs inside the request; a slow pipeline makes the page wait
  (with a "Analisando…" notice) rather than queuing the job.
