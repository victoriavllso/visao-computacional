"""History of analyses: one SQLite row per analysis, photos on disk.

    <data dir>/history.sqlite3
    <data dir>/photos/<id>/{front,back}.<ext>        original upload
    <data dir>/photos/<id>/{front,back}_view.jpg     result page
    <data dir>/photos/<id>/{front,back}_thumb.jpg    history list

The data dir is data/app/ (git-ignored) unless APP_DATA_DIR is set.
"""

import json
import os
import shutil
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDES = ("front", "back")
VARIANTS = ("original", "view", "thumb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    verdict TEXT NOT NULL,
    front_ext TEXT NOT NULL,
    back_ext TEXT NOT NULL,
    result TEXT NOT NULL
)
"""


def data_dir() -> Path:
    return Path(os.environ.get("APP_DATA_DIR", REPO_ROOT / "data" / "app"))


def connect() -> sqlite3.Connection:
    # A connection per call: requests run in a thread pool, and sqlite3
    # connections can't be shared across threads.
    data_dir().mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(data_dir() / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    connection.execute(SCHEMA)
    return connection


def new_id() -> str:
    return uuid.uuid4().hex


def photo_dir(analysis_id: str) -> Path:
    return data_dir() / "photos" / analysis_id


def photo_path(analysis_id: str, side: str, variant: str, ext=".jpg") -> Path:
    name = f"{side}{ext}" if variant == "original" else f"{side}_{variant}.jpg"
    return photo_dir(analysis_id) / name


def save(analysis_id: str, name: str, extensions: dict, result: dict) -> dict:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(connect()) as connection, connection:
        connection.execute(
            "INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                analysis_id,
                created_at,
                name,
                result["verdict"],
                extensions["front"],
                extensions["back"],
                json.dumps(result, ensure_ascii=False),
            ),
        )
    return get(analysis_id)


def as_dict(row: sqlite3.Row, full=True) -> dict:
    entry = {
        "id": row["id"],
        "created_at": row["created_at"],
        "name": row["name"],
        "verdict": row["verdict"],
        "extensions": {"front": row["front_ext"], "back": row["back_ext"]},
    }
    if full:
        entry["result"] = json.loads(row["result"])
    return entry


def get(analysis_id: str) -> dict | None:
    with closing(connect()) as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    return as_dict(row) if row else None


def list_all() -> list[dict]:
    with closing(connect()) as connection:
        rows = connection.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
    return [as_dict(row, full=False) for row in rows]


def delete(analysis_id: str) -> bool:
    with closing(connect()) as connection, connection:
        deleted = connection.execute(
            "DELETE FROM analyses WHERE id = ?", (analysis_id,)
        ).rowcount
    shutil.rmtree(photo_dir(analysis_id), ignore_errors=True)
    return bool(deleted)
