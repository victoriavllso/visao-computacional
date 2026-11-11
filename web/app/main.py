"""Web app: send the front and back photos of a package, see whether its
"ALTO EM" warnings follow ANVISA's rules, and browse past analyses.

    pixi run -e web app        # http://localhost:8000

The page (static/) is plain HTML/CSS/JS talking to the JSON API below.
"""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, HTTPException, Path as PathParam, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import images, pipeline, storage
from app.images import InvalidImage

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SIDE_NAMES = {"front": "da frente", "back": "do verso"}
# Ids are uuid4 hex: also keeps them from reaching the filesystem as paths.
AnalysisId = PathParam(pattern=r"^[0-9a-f]{32}$")

app = FastAPI(title="Rotulagem ANVISA")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


def with_photos(entry: dict) -> dict:
    """Links to the photos the analysis has (no front when only the table
    was checked)."""
    # Relative: the app may be served under a prefix, e.g. behind
    # JupyterHub's proxy (/user/<name>/proxy/8000/).
    base = f"api/analyses/{entry['id']}/photos"
    entry["photos"] = {
        side: {variant: f"{base}/{side}/{variant}" for variant in ("view", "thumb")}
        for side in storage.SIDES
        if entry["extensions"].get(side)
    }
    return entry


def read_upload(upload: UploadFile, side: str) -> tuple[bytes, object]:
    # One byte over the limit is enough to reject, without reading it all.
    data = upload.file.read(images.MAX_UPLOAD_BYTES + 1)
    try:
        return data, images.decode(data)
    except InvalidImage as error:
        raise HTTPException(422, f"Foto {SIDE_NAMES[side]}: {error}")


def store_photos(analysis_id: str, uploads: dict) -> dict:
    storage.photo_dir(analysis_id).mkdir(parents=True)
    extensions = {}
    for side, (data, image) in uploads.items():
        ext = images.extension(data)
        storage.photo_path(analysis_id, side, "original", ext).write_bytes(data)
        for variant, size in (("view", images.VIEW_SIZE), ("thumb", images.THUMB_SIZE)):
            small = images.encode_jpeg(images.downscale(image, size))
            storage.photo_path(analysis_id, side, variant).write_bytes(small)
        extensions[side] = ext
    return extensions


# Plain `def`: FastAPI runs it in a thread pool, so a slow pipeline doesn't
# block the other requests.
@app.post("/api/analyses", status_code=201)
def create_analysis(
    back: UploadFile, front: UploadFile | None = None, name: str = Form("")
):
    """Without the front photo, only the table is read: the result says which
    warnings the product must carry ("info"), not whether it complies."""
    uploads = {"back": read_upload(back, "back")}
    if front is not None:
        uploads["front"] = read_upload(front, "front")
    analysis_id = storage.new_id()
    try:
        extensions = store_photos(analysis_id, uploads)
        front_path = (
            storage.photo_path(analysis_id, "front", "original", extensions["front"])
            if "front" in extensions
            else None
        )
        result = pipeline.analyze(
            front_path,
            storage.photo_path(analysis_id, "back", "original", extensions["back"]),
        )
        entry = storage.save(analysis_id, name.strip()[:120], extensions, result)
    except Exception:
        storage.delete(analysis_id)
        raise
    return with_photos(entry)


@app.get("/api/analyses")
def list_analyses():
    return [with_photos(entry) for entry in storage.list_all()]


def get_or_404(analysis_id: str) -> dict:
    entry = storage.get(analysis_id)
    if entry is None:
        raise HTTPException(404, "Análise não encontrada.")
    return entry


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str = AnalysisId):
    return with_photos(get_or_404(analysis_id))


@app.delete("/api/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str = AnalysisId):
    if not storage.delete(analysis_id):
        raise HTTPException(404, "Análise não encontrada.")
    return Response(status_code=204)


@app.get("/api/analyses/{analysis_id}/export")
def export_analysis(analysis_id: str = AnalysisId):
    entry = get_or_404(analysis_id)
    entry.pop("extensions")
    filename = f"analise-{entry['created_at'][:10]}-{analysis_id[:8]}.json"
    return JSONResponse(
        entry, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/api/analyses/{analysis_id}/photos/{side}/{variant}")
def get_photo(
    side: Literal["front", "back"],
    variant: Literal["view", "thumb"],
    analysis_id: str = AnalysisId,
):
    path = storage.photo_path(analysis_id, side, variant)
    if not path.exists():
        raise HTTPException(404, "Foto não encontrada.")
    return FileResponse(path, media_type="image/jpeg")
