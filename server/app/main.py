import asyncio
import os
import re
import secrets
import sqlite3
import subprocess
import urllib.parse
import uuid
from email.utils import formatdate
from pathlib import Path

import aiofiles
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
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

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
CHUNK_SIZE = 1024 * 1024
FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_PATH", "ffprobe")


# ---------- Auth ----------

class AuthRedirect(Exception):
    pass


@app.exception_handler(AuthRedirect)
async def auth_redirect_handler(request: Request, exc: AuthRedirect):
    return RedirectResponse(url="/login")


def require_auth(request: Request):
    if not request.session.get("authenticated"):
        raise AuthRedirect()
    return True


def is_authenticated(request: Request) -> bool:
    return bool(request.session.get("authenticated"))


# ---------- ffmpeg helpers ----------

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


# Software path is the guaranteed fallback. veryfast/crf 21 is a good
# size/speed/quality balance for short gameplay clips.
SOFTWARE_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]

# Hardware candidates, best-first. Each is verified with a real test encode at
# startup, so a machine that *lists* an encoder but can't actually use it
# (the QSV situation here) is skipped instead of failing every trim.
HW_CANDIDATES = [
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]),
    ("h264_qsv", ["-c:v", "h264_qsv", "-global_quality", "23"]),
    ("h264_amf", ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "23", "-qp_p", "23"]),
    ("h264_vaapi", ["-vaapi_device", "/dev/dri/renderD128",
                    "-vf", "format=nv12,hwupload",
                    "-c:v", "h264_vaapi", "-qp", "23"]),
]


def _test_encoder(enc_args) -> bool:
    try:
        r = _run([
            FFMPEG, "-hide_banner", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=0.2:size=256x256:rate=10",
            *enc_args, "-f", "null", "-",
        ])
        return r.returncode == 0
    except Exception:
        return False


def detect_encoder():
    try:
        listed = _run([FFMPEG, "-hide_banner", "-encoders"]).stdout
    except Exception:
        listed = ""
    for name, args in HW_CANDIDATES:
        if name in listed and _test_encoder(args):
            return name, args
    return "libx264", SOFTWARE_ARGS


ENCODER_NAME, ENCODER_ARGS = detect_encoder()
print(f"[encoder] using {ENCODER_NAME}")


def get_duration(path: Path) -> float:
    try:
        result = _run([
            FFPROBE, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ])
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def generate_thumbnail(video_path: Path):
    thumb_name = f"{uuid.uuid4().hex}.jpg"
    thumb_path = THUMBNAIL_DIR / thumb_name
    # -ss before -i = fast seek; clamp to ~1s but tolerate very short clips.
    _run([
        FFMPEG, "-hide_banner", "-y", "-ss", "00:00:01", "-i", str(video_path),
        "-frames:v", "1", "-vf", "scale=480:-1", str(thumb_path),
    ])
    if thumb_path.exists() and thumb_path.stat().st_size > 0:
        return thumb_name
    # Fallback: grab the very first frame (clip shorter than 1s).
    _run([
        FFMPEG, "-hide_banner", "-y", "-i", str(video_path),
        "-frames:v", "1", "-vf", "scale=480:-1", str(thumb_path),
    ])
    if thumb_path.exists() and thumb_path.stat().st_size > 0:
        return thumb_name
    return None


def run_trim(source_path: Path, out_path: Path, start: float, duration: float):
    """Trim with the detected encoder; fall back to libx264 if it fails.

    Returns (ok: bool, error_message: str).
    """
    attempts = [ENCODER_ARGS]
    if ENCODER_ARGS is not SOFTWARE_ARGS:
        attempts.append(SOFTWARE_ARGS)

    last_err = "ffmpeg failed"
    for enc_args in attempts:
        cmd = [
            FFMPEG, "-hide_banner", "-y",
            "-ss", f"{start:.3f}",        # input seek: fast AND frame-accurate on modern ffmpeg
            "-i", str(source_path),
            "-t", f"{duration:.3f}",
            *enc_args,
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",   # moov atom up front => instant scrub/playback
            str(out_path),
        ]
        r = _run(cmd)
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True, ""
        last_err = (r.stderr.strip().splitlines() or ["ffmpeg failed"])[-1]
        if out_path.exists():
            out_path.unlink()
    return False, last_err


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
        ".webm": "video/webm", ".avi": "video/x-msvideo", ".flv": "video/x-flv",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
    }
    return types.get(ext, "application/octet-stream")


def find_source_path(filename: str):
    for subdir, base in (("raw", RAW_DIR), ("edited", EDITED_DIR)):
        p = base / filename
        if p.exists():
            return p, subdir
    return None, None


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


# ---------- DB ----------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL = concurrent reads while a trim/upload writes; big responsiveness win.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            source_filename TEXT,
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

    cols = [r[1] for r in conn.execute("PRAGMA table_info(clips)").fetchall()]
    if "thumbnail" not in cols:
        conn.execute("ALTER TABLE clips ADD COLUMN thumbnail TEXT")
    if "duration" not in cols:
        conn.execute("ALTER TABLE clips ADD COLUMN duration REAL DEFAULT 0")
    if "display_name" not in cols:
        conn.execute("ALTER TABLE clips ADD COLUMN display_name TEXT")
    if "source_filename" not in cols:
        conn.execute("ALTER TABLE clips ADD COLUMN source_filename TEXT")
        conn.execute("UPDATE clips SET source_filename = filename WHERE source_filename IS NULL")

    conn.commit()
    conn.close()


def backfill_metadata():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM clips WHERE thumbnail IS NULL OR duration IS NULL OR duration = 0"
    ).fetchall()
    for row in rows:
        path, _ = find_source_path(row["filename"])
        if path is None:
            continue
        duration = get_duration(path)
        thumbnail = row["thumbnail"] or generate_thumbnail(path)
        conn.execute(
            "UPDATE clips SET duration = ?, thumbnail = ? WHERE id = ?",
            (duration, thumbnail, row["id"]),
        )
    conn.commit()
    conn.close()


init_db()
backfill_metadata()


# ---------- Media serving (async + range + cache) ----------

@app.get("/media/{subpath:path}")
async def serve_media(request: Request, subpath: str):
    file_path = (DATA_DIR / subpath).resolve()
    data_root = DATA_DIR.resolve()
    if data_root not in file_path.parents or not file_path.is_file():
        raise HTTPException(404, "File not found")

    st = file_path.stat()
    file_size = st.st_size
    etag = f'"{st.st_mtime_ns:x}-{file_size:x}"'
    media_type = media_type_for(file_path)
    base_headers = {
        "Accept-Ranges": "bytes",
        "ETag": etag,
        "Last-Modified": formatdate(st.st_mtime, usegmt=True),
        # Filenames are content-unique (uuid), so a long immutable cache is safe
        # and means the browser stops re-downloading bytes while scrubbing.
        "Cache-Control": "public, max-age=31536000, immutable",
    }

    if etag in (request.headers.get("if-none-match") or ""):
        return Response(status_code=304, headers=base_headers)

    range_header = request.headers.get("range")
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                return Response(status_code=416,
                                headers={**base_headers, "Content-Range": f"bytes */{file_size}"})
            length = end - start + 1

            async def iter_range():
                async with aiofiles.open(file_path, "rb") as f:
                    await f.seek(start)
                    remaining = length
                    while remaining > 0:
                        data = await f.read(min(CHUNK_SIZE, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            return StreamingResponse(
                iter_range(), status_code=206, media_type=media_type,
                headers={**base_headers,
                         "Content-Range": f"bytes {start}-{end}/{file_size}",
                         "Content-Length": str(length)},
            )

    async def iter_full():
        async with aiofiles.open(file_path, "rb") as f:
            while True:
                data = await f.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    return StreamingResponse(
        iter_full(), media_type=media_type,
        headers={**base_headers, "Content-Length": str(file_size)},
    )


# ---------- Auth routes ----------

@app.get("/login")
async def login_page(request: Request):
    if request.session.get("authenticated"):
        return RedirectResponse(url="/library")
    return templates.TemplateResponse(request, "login.html", {"error": None, "authenticated": False})


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(password, ADMIN_PASSWORD):
        request.session["authenticated"] = True
        return RedirectResponse(url="/library", status_code=303)
    return templates.TemplateResponse(
        request, "login.html", {"error": "Incorrect username or password.", "authenticated": False},
        status_code=401,
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


# ---------- Client upload endpoints (watcher) ----------

async def _save_upload(file: UploadFile, dest: Path):
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(CHUNK_SIZE):
            await f.write(chunk)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".mp4"
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = RAW_DIR / new_name
    await _save_upload(file, dest)

    duration = await run_in_threadpool(get_duration, dest)
    thumbnail = await run_in_threadpool(generate_thumbnail, dest)

    conn = get_db()
    cur = conn.execute(
        "INSERT INTO clips (filename, source_filename, display_name, status, thumbnail, duration) "
        "VALUES (?, ?, ?, 'raw', ?, ?)",
        (new_name, new_name, file.filename, thumbnail, duration),
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
    await _save_upload(file, dest)

    slug = uuid.uuid4().hex[:8]
    conn = get_db()
    conn.execute("INSERT INTO images (filename, slug) VALUES (?, ?)", (new_name, slug))
    conn.commit()
    conn.close()
    return {"url": f"/i/{slug}"}


# ---------- Library ----------

@app.get("/library")
async def library(request: Request, user: bool = Depends(require_auth)):
    conn = get_db()
    clips = conn.execute("SELECT * FROM clips ORDER BY created_at DESC").fetchall()
    images = conn.execute("SELECT * FROM images ORDER BY created_at DESC LIMIT 60").fetchall()
    conn.close()
    return templates.TemplateResponse(
        request, "library.html",
        {"clips": clips, "images": images, "active": "library", "authenticated": True},
    )


# ---------- Editor ----------

@app.get("/clip/{clip_id}/edit")
async def edit_page(request: Request, clip_id: int, user: bool = Depends(require_auth)):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not clip:
        raise HTTPException(404, "Clip not found")

    source_filename = clip["source_filename"] or clip["filename"]
    source_path, source_subdir = find_source_path(source_filename)
    if source_path is None:
        source_filename = clip["filename"]
        source_path, source_subdir = find_source_path(source_filename)
        if source_subdir is None:
            source_subdir = "edited" if clip["status"] == "edited" else "raw"

    return templates.TemplateResponse(
        request, "editor.html",
        {
            "clip": clip,
            "source_filename": source_filename,
            "source_subdir": source_subdir,
            "active": "library",
            "authenticated": True,
            "error": request.query_params.get("error"),
        },
    )


@app.post("/clip/{clip_id}/trim")
async def trim_clip(clip_id: int, start: str = Form(...), end: str = Form(...),
                    user: bool = Depends(require_auth)):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if not clip:
        conn.close()
        raise HTTPException(404, "Clip not found")

    def _redirect_err(msg):
        return RedirectResponse(
            url=f"/clip/{clip_id}/edit?error=" + urllib.parse.quote(msg), status_code=303)

    try:
        start_f = max(0.0, float(start))
        end_f = float(end)
    except ValueError:
        conn.close()
        return _redirect_err("Invalid trim range.")
    if end_f <= start_f:
        conn.close()
        return _redirect_err("End must be after start.")

    source_filename = clip["source_filename"] or clip["filename"]
    source_path, _ = find_source_path(source_filename)
    if source_path is None:
        conn.close()
        return _redirect_err("Source file is missing.")

    prev_filename = clip["filename"]
    prev_status = clip["status"]
    prev_thumb = clip["thumbnail"]

    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = EDITED_DIR / out_name

    ok, err = await run_in_threadpool(run_trim, source_path, out_path, start_f, end_f - start_f)
    if not ok:
        conn.close()
        return _redirect_err(err or "ffmpeg failed")

    duration = await run_in_threadpool(get_duration, out_path)
    thumbnail = await run_in_threadpool(generate_thumbnail, out_path)

    conn.execute(
        "UPDATE clips SET filename = ?, status = 'edited', thumbnail = ?, duration = ? WHERE id = ?",
        (out_name, thumbnail, duration, clip_id),
    )
    conn.commit()
    conn.close()

    # Clean up the now-orphaned previous edited file + thumbnail (non-destructive
    # to the raw source, which is never touched).
    if prev_status == "edited" and prev_filename != source_filename:
        old = EDITED_DIR / prev_filename
        if old.exists():
            old.unlink()
    if prev_thumb and prev_thumb != thumbnail:
        tp = THUMBNAIL_DIR / prev_thumb
        if tp.exists():
            tp.unlink()

    return RedirectResponse(url=f"/clip/{clip_id}/edit", status_code=303)


@app.post("/clip/{clip_id}/rename")
async def rename_clip(clip_id: int, name: str = Form(...), user: bool = Depends(require_auth)):
    name = name.strip() or None
    conn = get_db()
    conn.execute("UPDATE clips SET display_name = ? WHERE id = ?", (name, clip_id))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "name": name})


@app.post("/clip/{clip_id}/delete")
async def delete_clip(clip_id: int, user: bool = Depends(require_auth)):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if clip:
        names = {clip["filename"], clip["source_filename"]}
        for name in names:
            if not name:
                continue
            for base in (RAW_DIR, EDITED_DIR):
                p = base / name
                if p.exists():
                    p.unlink()
        if clip["thumbnail"]:
            tp = THUMBNAIL_DIR / clip["thumbnail"]
            if tp.exists():
                tp.unlink()
        conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
        conn.commit()
    conn.close()
    return RedirectResponse(url="/library", status_code=303)


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


@app.post("/clip/{clip_id}/unpublish")
async def unpublish_clip(clip_id: int, user: bool = Depends(require_auth)):
    conn = get_db()
    conn.execute("UPDATE clips SET published = 0 WHERE id = ?", (clip_id,))
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
        request, "share.html",
        {"clip": clip, "subdir": subdir, "authenticated": is_authenticated(request)},
    )


@app.get("/i/{slug}")
async def share_image(request: Request, slug: str):
    conn = get_db()
    img = conn.execute("SELECT * FROM images WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    if not img:
        raise HTTPException(404, "Image not found")
    return templates.TemplateResponse(
        request, "share_image.html", {"image": img, "authenticated": is_authenticated(request)},
    )


@app.post("/image/{image_id}/delete")
async def delete_image(image_id: int, user: bool = Depends(require_auth)):
    conn = get_db()
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if row:
        p = IMAGES_DIR / row["filename"]
        if p.exists():
            p.unlink()
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
    conn.close()
    return RedirectResponse(url="/library", status_code=303)


# ---------- Public uploads page ----------

@app.get("/")
async def uploads_page(request: Request):
    conn = get_db()
    uploads = conn.execute("SELECT * FROM uploads ORDER BY created_at DESC").fetchall()
    conn.close()
    return templates.TemplateResponse(
        request, "uploads.html",
        {"uploads": uploads, "active": "uploads", "authenticated": is_authenticated(request)},
    )


@app.post("/uploads/add")
async def add_upload(file: UploadFile = File(...)):
    original_name = file.filename
    ext = Path(original_name).suffix
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOADS_DIR / new_name
    await _save_upload(file, dest)

    kind = detect_kind(original_name)
    thumbnail = await run_in_threadpool(generate_thumbnail, dest) if kind == "video" else None
    slug = uuid.uuid4().hex[:8]

    conn = get_db()
    conn.execute(
        "INSERT INTO uploads (filename, original_name, display_name, kind, thumbnail, slug) "
        "VALUES (?,?,?,?,?,?)",
        (new_name, original_name, original_name, kind, thumbnail, slug),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "slug": slug}


@app.post("/uploads/{upload_id}/rename")
async def rename_upload(upload_id: int, name: str = Form(...), user: bool = Depends(require_auth)):
    name = name.strip() or None
    conn = get_db()
    conn.execute("UPDATE uploads SET display_name = ? WHERE id = ?", (name, upload_id))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "name": name})


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
    return RedirectResponse(url="/", status_code=303)


@app.get("/u/{slug}")
async def share_upload(request: Request, slug: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM uploads WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Not found")
    return templates.TemplateResponse(
        request, "share_upload.html", {"upload": row, "authenticated": is_authenticated(request)},
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True, "encoder": ENCODER_NAME}