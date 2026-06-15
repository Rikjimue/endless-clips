import sqlite3
import subprocess
import secrets
import os
import re
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

DATA_DIR = Path("/data")
RAW_DIR = DATA_DIR / "raw"
EDITED_DIR = DATA_DIR / "edited"
IMAGES_DIR = DATA_DIR / "images"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "clips.db"

for d in (RAW_DIR, EDITED_DIR, IMAGES_DIR, THUMBNAIL_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
ADMIN_USERNAME = os.environ.get("CLIPS_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("CLIPS_PASSWORD", "changeme")

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


class AuthRedirect(Exception):
    pass


@app.exception_handler(AuthRedirect)
async def auth_redirect_handler(request: Request, exc: AuthRedirect):
    return RedirectResponse(url="/login")


def require_auth(request: Request):
    if not request.session.get("authenticated"):
        raise AuthRedirect()
    return True


# ---------- Helpers ----------

def format_duration(seconds):
    if not seconds:
        return "--:--"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


templates.env.filters["duration"] = format_duration


def get_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def generate_thumbnail(video_path: Path):
    thumb_name = f"{uuid.uuid4().hex}.jpg"
    thumb_path = THUMBNAIL_DIR / thumb_name
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "00:00:01", "-i", str(video_path),
         "-frames:v", "1", "-vf", "scale=480:-1", str(thumb_path)],
        capture_output=True
    )
    if thumb_path.exists() and thumb_path.stat().st_size > 0:
        return thumb_name
    return None


def detect_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    return "other"


def media_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    types = {
        ".mp4": "video/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
    }
    return types.get(ext, "application/octet-stream")


# ---------- DB ----------

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
            display_name TEXT,
            status TEXT NOT NULL DEFAULT 'raw',
            slug TEXT UNIQUE,
            published INTEGER DEFAULT 0,
            thumbnail TEXT,
            duration REAL DEFAULT 0,
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT,
            display_name TEXT,
            kind TEXT,
            thumbnail TEXT,
            slug TEXT UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(clips)").fetchall()]
    if "thumbnail" not in existing_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN thumbnail TEXT")
    if "duration" not in existing_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN duration REAL DEFAULT 0")
    if "display_name" not in existing_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN display_name TEXT")

    conn.commit()
    conn.close()


def backfill_metadata():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM clips WHERE thumbnail IS NULL OR duration IS NULL OR duration = 0"
    ).fetchall()
    for row in rows:
        subdir = "edited" if row["status"] == "edited" else "raw"
        base_dir = EDITED_DIR if subdir == "edited" else RAW_DIR
        path = base_dir / row["filename"]
        if not path.exists():
            continue
        duration = get_duration(path)
        thumbnail = row["thumbnail"] or generate_thumbnail(path)
        conn.execute(
            "UPDATE clips SET duration = ?, thumbnail = ? WHERE id = ?",
            (duration, thumbnail, row["id"])
        )
    conn.commit()
    conn.close()


init_db()
backfill_metadata()


# ---------- Media serving (range-request aware) ----------

CHUNK_SIZE = 1024 * 1024


@app.get("/media/{subpath:path}")
async def serve_media(request: Request, subpath: str):
    file_path = DATA_DIR / subpath
    if not file_path.is_file():
        raise HTTPException(404, "File not found")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    media_type = media_type_for(file_path)

    if range_header:
        range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            end = min(end, file_size - 1)
            chunk_size = end - start + 1

            def iterfile():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        data = f.read(min(CHUNK_SIZE, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            }
            return StreamingResponse(iterfile(), status_code=206, headers=headers, media_type=media_type)

    def iterfile_full():
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                yield chunk

    return StreamingResponse(
        iterfile_full(),
        headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"},
        media_type=media_type,
    )


# ---------- Auth ----------

@app.get("/login")
async def login_page(request: Request):
    if request.session.get("authenticated"):
        return RedirectResponse(url="/")
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(password, ADMIN_PASSWORD):
        request.session["authenticated"] = True
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request, "login.html", {"error": "Incorrect username or password."}, status_code=401
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ---------- Client upload endpoints (used by watcher) ----------

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".mp4"
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = RAW_DIR / new_name

    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    duration = get_duration(dest)
    thumbnail = generate_thumbnail(dest)

    conn = get_db()
    cur = conn.execute(
        "INSERT INTO clips (filename, display_name, status, thumbnail, duration) VALUES (?, ?, 'raw', ?, ?)",
        (new_name, file.filename, thumbnail, duration)
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
    conn.execute("INSERT INTO images (filename, slug) VALUES (?, ?)", (new_name, slug))
    conn.commit()
    conn.close()

    return {"url": f"/i/{slug}"}


# ---------- Library ----------

@app.get("/")
async def library(request: Request, user: bool = Depends(require_auth)):
    conn = get_db()
    clips = conn.execute("SELECT * FROM clips ORDER BY created_at DESC").fetchall()
    images = conn.execute("SELECT * FROM images ORDER BY created_at DESC LIMIT 24").fetchall()
    conn.close()
    return templates.TemplateResponse(
        request, "library.html", {"clips": clips, "images": images, "active": "library"}
    )


# ---------- Editor ----------

@app.get("/clip/{clip_id}/edit")
async def edit_page(request: Request, clip_id: int, user: bool = Depends(require_auth)):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not clip:
        raise HTTPException(404, "Clip not found")

    subdir = "edited" if clip["status"] == "edited" else "raw"

    return templates.TemplateResponse(
        request, "editor.html", {"clip": clip, "subdir": subdir, "active": "library"}
    )


@app.post("/clip/{clip_id}/trim")
async def trim_clip(clip_id: int, start: str = Form(...), end: str = Form(...), user: bool = Depends(require_auth)):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if not clip:
        conn.close()
        raise HTTPException(404, "Clip not found")

    src_dir = RAW_DIR if clip["status"] != "edited" else EDITED_DIR
    src_path = src_dir / clip["filename"]

    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = EDITED_DIR / out_name

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-ss", start, "-to", end,
        "-c:v", "h264_qsv", "-global_quality", "20",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        conn.close()
        raise HTTPException(500, f"ffmpeg failed: {result.stderr[-500:]}")

    duration = get_duration(out_path)
    thumbnail = generate_thumbnail(out_path)

    conn.execute(
        "UPDATE clips SET filename = ?, status = 'edited', thumbnail = ?, duration = ? WHERE id = ?",
        (out_name, thumbnail, duration, clip_id)
    )
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/clip/{clip_id}/edit", status_code=303)


@app.post("/clip/{clip_id}/rename")
async def rename_clip(clip_id: int, name: str = Form(...), user: bool = Depends(require_auth)):
    conn = get_db()
    conn.execute("UPDATE clips SET display_name = ? WHERE id = ?", (name.strip() or None, clip_id))
    conn.commit()
    conn.close()
    return RedirectResponse(url=f"/clip/{clip_id}/edit", status_code=303)


# ---------- Publish / Share ----------

@app.post("/clip/{clip_id}/publish")
async def publish_clip(clip_id: int, user: bool = Depends(require_auth)):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if not clip:
        conn.close()
        raise HTTPException(404, "Clip not found")

    slug = clip["slug"] or uuid.uuid4().hex[:8]
    conn.execute("UPDATE clips SET slug = ?, published = 1 WHERE id = ?", (slug, clip_id))
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/clip/{clip_id}/edit", status_code=303)


@app.get("/c/{slug}")
async def share_clip(request: Request, slug: str):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    if not clip:
        raise HTTPException(404, "Clip not found")

    subdir = "edited" if clip["status"] == "edited" else "raw"
    return templates.TemplateResponse(
        request, "share.html", {"clip": clip, "subdir": subdir}
    )


@app.get("/i/{slug}")
async def share_image(request: Request, slug: str):
    conn = get_db()
    img = conn.execute("SELECT * FROM images WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    if not img:
        raise HTTPException(404, "Image not found")

    return templates.TemplateResponse(
        request, "share_image.html", {"image": img}
    )


# ---------- Manual uploads ----------

@app.get("/uploads")
async def uploads_page(request: Request, user: bool = Depends(require_auth)):
    conn = get_db()
    uploads = conn.execute("SELECT * FROM uploads ORDER BY created_at DESC").fetchall()
    conn.close()
    return templates.TemplateResponse(
        request, "uploads.html", {"uploads": uploads, "active": "uploads"}
    )


@app.post("/uploads/add")
async def add_upload(file: UploadFile = File(...), user: bool = Depends(require_auth)):
    original_name = file.filename
    ext = Path(original_name).suffix
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOADS_DIR / new_name

    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    kind = detect_kind(original_name)
    thumbnail = generate_thumbnail(dest) if kind == "video" else None
    slug = uuid.uuid4().hex[:8]

    conn = get_db()
    conn.execute(
        "INSERT INTO uploads (filename, original_name, display_name, kind, thumbnail, slug) VALUES (?,?,?,?,?,?)",
        (new_name, original_name, original_name, kind, thumbnail, slug)
    )
    conn.commit()
    conn.close()

    return {"ok": True, "slug": slug}


@app.post("/uploads/{upload_id}/rename")
async def rename_upload(upload_id: int, name: str = Form(...), user: bool = Depends(require_auth)):
    conn = get_db()
    conn.execute("UPDATE uploads SET display_name = ? WHERE id = ?", (name.strip() or None, upload_id))
    conn.commit()
    conn.close()
    return RedirectResponse(url="/uploads", status_code=303)


@app.post("/uploads/{upload_id}/delete")
async def delete_upload(upload_id: int, user: bool = Depends(require_auth)):
    conn = get_db()
    row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if row:
        path = UPLOADS_DIR / row["filename"]
        if path.exists():
            path.unlink()
        if row["thumbnail"]:
            tp = THUMBNAIL_DIR / row["thumbnail"]
            if tp.exists():
                tp.unlink()
        conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
        conn.commit()
    conn.close()
    return RedirectResponse(url="/uploads", status_code=303)


@app.get("/u/{slug}")
async def share_upload(request: Request, slug: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM uploads WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Not found")
    return templates.TemplateResponse(request, "share_upload.html", {"upload": row})