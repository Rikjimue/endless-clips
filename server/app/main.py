import sqlite3
import subprocess
import uuid
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

DATA_DIR = Path("/data")
RAW_DIR = DATA_DIR / "raw"
EDITED_DIR = DATA_DIR / "edited"
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "clips.db"

for d in (RAW_DIR, EDITED_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'raw',
            slug TEXT UNIQUE,
            published INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            slug TEXT UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ---------- Upload endpoints ----------

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".mp4"
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = RAW_DIR / new_name

    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    conn = get_db()
    cur = conn.execute(
        "INSERT INTO clips (filename, status) VALUES (?, 'raw')",
        (new_name,)
    )
    conn.commit()
    clip_id = cur.lastrowid
    conn.close()

    return {"id": clip_id, "filename": new_name}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".png"
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = IMAGES_DIR / new_name

    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    slug = uuid.uuid4().hex[:8]
    conn = get_db()
    conn.execute(
        "INSERT INTO images (filename, slug) VALUES (?, ?)",
        (new_name, slug)
    )
    conn.commit()
    conn.close()

    return {"url": f"/i/{slug}"}


# ---------- Library ----------

@app.get("/")
async def library(request: Request):
    conn = get_db()
    clips = conn.execute(
        "SELECT * FROM clips ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(
        "library.html", {"request": request, "clips": clips}
    )


# ---------- Editor ----------

@app.get("/clip/{clip_id}/edit")
async def edit_page(request: Request, clip_id: int):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not clip:
        raise HTTPException(404, "Clip not found")

    source = clip["filename"] if clip["status"] == "raw" else clip["filename"]
    subdir = "raw" if clip["status"] != "edited" else "edited"

    return templates.TemplateResponse(
        "editor.html",
        {"request": request, "clip": clip, "subdir": subdir}
    )


@app.post("/clip/{clip_id}/trim")
async def trim_clip(clip_id: int, start: str = Form(...), end: str = Form(...)):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if not clip:
        conn.close()
        raise HTTPException(404, "Clip not found")

    src_dir = RAW_DIR if clip["status"] != "edited" else EDITED_DIR
    src_path = src_dir / clip["filename"]

    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = EDITED_DIR / out_name

    # QSV-accelerated trim. -ss/-to placed after -i for accurate frame seeking.
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "qsv", "-hwaccel_output_format", "qsv",
        "-i", str(src_path),
        "-ss", start, "-to", end,
        "-c:v", "h264_qsv", "-global_quality", "20",
        "-c:a", "aac",
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        conn.close()
        raise HTTPException(500, f"ffmpeg failed: {result.stderr[-500:]}")

    conn.execute(
        "UPDATE clips SET filename = ?, status = 'edited' WHERE id = ?",
        (out_name, clip_id)
    )
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/clip/{clip_id}/edit", status_code=303)


# ---------- Publish / Share ----------

@app.post("/clip/{clip_id}/publish")
async def publish_clip(clip_id: int):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if not clip:
        conn.close()
        raise HTTPException(404, "Clip not found")

    slug = clip["slug"] or uuid.uuid4().hex[:8]
    conn.execute(
        "UPDATE clips SET slug = ?, published = 1 WHERE id = ?",
        (slug, clip_id)
    )
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/c/{slug}", status_code=303)


@app.get("/c/{slug}")
async def share_clip(request: Request, slug: str):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    if not clip:
        raise HTTPException(404, "Clip not found")

    subdir = "edited" if clip["status"] == "edited" else "raw"
    return templates.TemplateResponse(
        "share.html",
        {"request": request, "clip": clip, "subdir": subdir}
    )


@app.get("/i/{slug}")
async def share_image(request: Request, slug: str):
    conn = get_db()
    img = conn.execute("SELECT * FROM images WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    if not img:
        raise HTTPException(404, "Image not found")

    return templates.TemplateResponse(
        "share_image.html",
        {"request": request, "image": img}
    )