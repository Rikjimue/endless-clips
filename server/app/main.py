import base64
import hashlib
import hmac
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import uuid
from email.utils import formatdate
from pathlib import Path

import aiofiles
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, HTTPException,
                     Request, UploadFile)
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

# ---------- Paths ----------
DATA_DIR = Path("/data")
RAW_DIR = DATA_DIR / "raw"
EDITED_DIR = DATA_DIR / "edited"
IMAGES_DIR = DATA_DIR / "images"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
UPLOADS_DIR = DATA_DIR / "uploads"
HLS_DIR = DATA_DIR / "hls"
TMP_DIR = DATA_DIR / "tmp"
DB_PATH = DATA_DIR / "clips.db"

for d in (RAW_DIR, EDITED_DIR, IMAGES_DIR, THUMBNAIL_DIR, UPLOADS_DIR, HLS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
ADMIN_USERNAME = os.environ.get("CLIPS_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("CLIPS_PASSWORD", "changeme")
FFMPEG = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_PATH", "ffprobe")

COOKIE_SECURE = os.environ.get("CLIPS_COOKIE_SECURE", "0") in ("1", "true", "True")
TRUST_PROXY = os.environ.get("CLIPS_TRUST_PROXY", "0") in ("1", "true", "True")
OPEN_REGISTRATION = os.environ.get("CLIPS_OPEN_REGISTRATION", "1") in ("1", "true", "True")
REQUIRE_LOGIN_UPLOAD = os.environ.get("CLIPS_REQUIRE_LOGIN_TO_UPLOAD", "0") in ("1", "true", "True")
UPLOAD_TOKEN = os.environ.get("CLIPS_UPLOAD_TOKEN", "")  # shared secret for the capture-PC watcher
MAX_UPLOAD_BYTES = int(os.environ.get("CLIPS_MAX_UPLOAD_MB", "4096")) * 1024 * 1024
GLOBAL_RATE_PER_MIN = int(os.environ.get("CLIPS_GLOBAL_RATE_PER_MIN", "600"))

VISIBILITIES = {"private", "unlisted", "public"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
CHUNK_SIZE = 1024 * 1024
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,20}$")

# ---------- HLS / scaling config ----------
HLS_SEGMENT_SECONDS = int(os.environ.get("CLIPS_HLS_SEGMENT", "4"))
HLS_ENCODER = os.environ.get("CLIPS_HLS_ENCODER", "libx264")
# (height, video_kbps, audio_kbps). Native top rung is kept whenever the source
# is taller than the highest standard rung, so 1440p/4K sources stream at source
# resolution. Rungs above the source are skipped (never upscale).
STANDARD_RUNGS = [
    (2160, 16000, 192), (1440, 9000, 160), (1080, 5000, 128),
    (720, 2800, 128), (480, 1200, 96),
]
MAX_RUNGS = int(os.environ.get("CLIPS_HLS_MAX_RUNGS", "4"))

TRANSCODE_CONCURRENCY = max(
    1, int(os.environ.get("CLIPS_TRANSCODE_CONCURRENCY", str(max(1, (os.cpu_count() or 2) // 2))))
)
_transcode_sem = threading.BoundedSemaphore(TRANSCODE_CONCURRENCY)

CSP = (
    "default-src 'self'; "
    "img-src 'self' data: blob:; "
    "media-src 'self' blob:; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.plyr.io; "
    "font-src 'self' https://fonts.gstatic.com; "
    "script-src 'self' 'unsafe-inline' https://cdn.plyr.io https://cdn.jsdelivr.net; "
    "worker-src 'self' blob:; child-src blob:; connect-src 'self' blob:; "
    "frame-ancestors 'self'; base-uri 'self'; form-action 'self'"
)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
STATIC_DIR = Path(__file__).resolve().parent / "static"
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    SessionMiddleware, secret_key=SESSION_SECRET,
    same_site="lax", https_only=COOKIE_SECURE, max_age=14 * 24 * 3600,
)


# ---------- Rate limiting ----------

_rl_lock = threading.Lock()
_rl_store: dict = {}


def rate_ok(key: str, limit: int, window: int) -> bool:
    """Sliding-window limiter (per worker process)."""
    now = time.time()
    with _rl_lock:
        q = [t for t in _rl_store.get(key, ()) if t > now - window]
        if len(q) >= limit:
            _rl_store[key] = q
            return False
        q.append(now)
        _rl_store[key] = q
        return True


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def gate(request: Request, call_next):
    path = request.url.path
    # Skip the generic flood limiter for static/media (HLS pulls many segments)
    # and chunk uploads (a single large file is many sequential requests).
    if not (path.startswith("/media") or path.startswith("/static") or path == "/uploads/chunk"):
        if not rate_ok(f"g:{client_ip(request)}", GLOBAL_RATE_PER_MIN, 60):
            return JSONResponse({"detail": "Too many requests"}, status_code=429)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers.setdefault("Content-Security-Policy", CSP)
    if COOKIE_SECURE:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


# ---------- Auth primitives ----------

class AuthRedirect(Exception):
    pass


@app.exception_handler(AuthRedirect)
async def auth_redirect_handler(request: Request, exc: AuthRedirect):
    return RedirectResponse(url="/login")


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return f"pbkdf2_sha256$200000${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _algo, iters, salt_b64, hash_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, int(iters))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def current_user(request: Request):
    uid = request.session.get("uid")
    if not uid:
        return None
    return {
        "id": uid,
        "username": request.session.get("username"),
        "is_admin": bool(request.session.get("is_admin")),
    }


def require_user(request: Request):
    user = current_user(request)
    if not user:
        raise AuthRedirect()
    return user


def is_authenticated(request: Request) -> bool:
    return bool(request.session.get("uid"))


def _back(request: Request, fallback: str):
    """Redirect to the same-origin path of the referer, else a fallback."""
    ref = request.headers.get("referer")
    if ref:
        p = urllib.parse.urlparse(ref)
        if p.path:
            return RedirectResponse(p.path + (("?" + p.query) if p.query else ""), status_code=303)
    return RedirectResponse(fallback, status_code=303)


def owns_or_admin(user, row) -> bool:
    if user and user.get("is_admin"):
        return True
    try:
        owner = row["user_id"]
    except Exception:
        owner = None
    return bool(user and owner is not None and user["id"] == owner)


# ---------- ffmpeg helpers ----------

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


SOFTWARE_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]
HW_CANDIDATES = [
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]),
    ("h264_qsv", ["-c:v", "h264_qsv", "-global_quality", "23"]),
    ("h264_amf", ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "23", "-qp_p", "23"]),
    ("h264_vaapi", ["-vaapi_device", "/dev/dri/renderD128", "-vf", "format=nv12,hwupload",
                    "-c:v", "h264_vaapi", "-qp", "23"]),
]


def _test_encoder(enc_args) -> bool:
    try:
        r = _run([FFMPEG, "-hide_banner", "-y", "-f", "lavfi",
                  "-i", "testsrc=duration=0.2:size=256x256:rate=10",
                  *enc_args, "-f", "null", "-"])
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
        r = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def probe_dimensions(path: Path):
    r = _run([FFPROBE, "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(path)])
    try:
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return 0, 0


def generate_thumbnail(video_path: Path):
    thumb_name = f"{uuid.uuid4().hex}.jpg"
    thumb_path = THUMBNAIL_DIR / thumb_name
    for ss in ("00:00:01", None):
        cmd = [FFMPEG, "-hide_banner", "-y"]
        if ss:
            cmd += ["-ss", ss]
        cmd += ["-i", str(video_path), "-frames:v", "1", "-vf", "scale=640:-1", str(thumb_path)]
        _run(cmd)
        if thumb_path.exists() and thumb_path.stat().st_size > 0:
            return thumb_name
    return None


def optimize_faststart(path: Path):
    if not path.exists() or path.suffix.lower() != ".mp4":
        return
    tmp = path.with_name(path.stem + ".opt.mp4")
    r = _run([FFMPEG, "-hide_banner", "-y", "-i", str(path),
              "-c", "copy", "-movflags", "+faststart", str(tmp)])
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        os.replace(tmp, path)
    elif tmp.exists():
        tmp.unlink()


def run_trim(source_path: Path, out_path: Path, start: float, duration: float):
    attempts = [ENCODER_ARGS]
    if ENCODER_ARGS is not SOFTWARE_ARGS:
        attempts.append(SOFTWARE_ARGS)
    last_err = "ffmpeg failed"
    for enc_args in attempts:
        cmd = [FFMPEG, "-hide_banner", "-y", "-ss", f"{start:.3f}", "-i", str(source_path),
               "-t", f"{duration:.3f}", *enc_args, "-c:a", "aac", "-b:a", "160k",
               "-movflags", "+faststart", str(out_path)]
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
    types = {
        ".mp4": "video/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
        ".webm": "video/webm", ".avi": "video/x-msvideo", ".flv": "video/x-flv",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
        ".m3u8": "application/vnd.apple.mpegurl", ".ts": "video/mp2t",
    }
    return types.get(path.suffix.lower(), "application/octet-stream")


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
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


templates.env.filters["duration"] = format_duration


# ---------- Background metadata ----------

def build_clip_metadata(clip_id: int, filename: str):
    path, _ = find_source_path(filename)
    if path is None:
        return
    duration = get_duration(path)
    thumbnail = generate_thumbnail(path)
    conn = get_db()
    conn.execute("UPDATE clips SET duration = ?, thumbnail = ? WHERE id = ?",
                 (duration, thumbnail, clip_id))
    conn.commit()
    conn.close()


def build_upload_metadata(upload_id: int, filename: str, kind: str):
    path = UPLOADS_DIR / filename
    if not path.exists() or kind != "video":
        return
    optimize_faststart(path)
    thumbnail = generate_thumbnail(path)
    conn = get_db()
    conn.execute("UPDATE uploads SET thumbnail = ? WHERE id = ?", (thumbnail, upload_id))
    conn.commit()
    conn.close()


# ---------- HLS ----------

def build_ladder(src_w: int, src_h: int):
    def even(n):
        n = int(round(n))
        return n + (n % 2)
    if src_h <= 0:
        return [(720, 2800, 128, None)]
    chosen = [r for r in STANDARD_RUNGS if r[0] <= src_h]
    if not chosen:
        return [(src_h, max(800, STANDARD_RUNGS[-1][1] // 2), 96, even(src_w) if src_w else None)]
    if len(chosen) > MAX_RUNGS:
        chosen = chosen[:MAX_RUNGS - 1] + [chosen[-1]]  # keep native top + a low floor
    out = []
    for (h, vk, ak) in chosen:
        w = even(src_w * h / src_h) if (src_w and src_h) else None
        out.append((h, vk, ak, w))
    return out


def _encode_rung(src_path, rung_dir, h, vk, ak, known_height, encoder) -> bool:
    rung_dir.mkdir(parents=True, exist_ok=True)
    vf = f"scale=-2:{h}" if known_height else f"scale=-2:'min(ih,{h})'"
    if encoder and encoder != "libx264":
        venc = ["-c:v", encoder]
    else:
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-profile:v", "high"]
    cmd = [FFMPEG, "-hide_banner", "-y", "-i", str(src_path), "-vf", vf, *venc,
           "-b:v", f"{vk}k", "-maxrate", f"{int(vk * 1.07)}k", "-bufsize", f"{int(vk * 1.5)}k",
           "-force_key_frames", f"expr:gte(t,n_forced*{HLS_SEGMENT_SECONDS})",
           "-c:a", "aac", "-b:a", f"{ak}k", "-ac", "2",
           "-f", "hls", "-hls_time", str(HLS_SEGMENT_SECONDS),
           "-hls_playlist_type", "vod", "-hls_flags", "independent_segments",
           "-hls_segment_filename", str(rung_dir / "seg_%04d.ts"), str(rung_dir / "index.m3u8")]
    r = _run(cmd)
    return r.returncode == 0 and (rung_dir / "index.m3u8").exists()


def generate_hls(src_path: Path, out_dir: Path) -> bool:
    if not src_path.exists():
        return False
    src_w, src_h = probe_dimensions(src_path)
    ladder = build_ladder(src_w, src_h)
    known = src_h > 0
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    with _transcode_sem:
        for (h, vk, ak, w) in ladder:
            rung_dir = out_dir / f"{h}p"
            ok = _encode_rung(src_path, rung_dir, h, vk, ak, known, HLS_ENCODER)
            if not ok and HLS_ENCODER != "libx264":
                ok = _encode_rung(src_path, rung_dir, h, vk, ak, known, "libx264")
            if not ok:
                shutil.rmtree(rung_dir, ignore_errors=True)
                continue
            variants.append((h, int(vk * 1.07) * 1000 + ak * 1000, w))
    if not variants:
        return False
    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
    for (h, bandwidth, w) in variants:
        res = f",RESOLUTION={w}x{h}" if w else ""
        lines.append(f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth}{res},CODECS="avc1.4d401f,mp4a.40.2"')
        lines.append(f"{h}p/index.m3u8")
    (out_dir / "master.m3u8").write_text("\n".join(lines) + "\n")
    return True


def cleanup_hls_dir(dirname):
    if dirname:
        p = HLS_DIR / dirname
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def _rebuild_hls(table: str, row_id: int, src_path: Path, old_dir):
    conn = get_db()
    if src_path is None or not src_path.exists():
        conn.execute(f"UPDATE {table} SET hls_status='error' WHERE id=?", (row_id,))
        conn.commit()
        conn.close()
        return
    conn.execute(f"UPDATE {table} SET hls_status='pending' WHERE id=?", (row_id,))
    conn.commit()
    conn.close()
    new_dir = uuid.uuid4().hex
    ok = generate_hls(src_path, HLS_DIR / new_dir)
    conn = get_db()
    if ok:
        conn.execute(f"UPDATE {table} SET hls_status='ready', hls_dir=? WHERE id=?", (new_dir, row_id))
        conn.commit()
        conn.close()
        cleanup_hls_dir(old_dir)
    else:
        cleanup_hls_dir(new_dir)
        conn.execute(f"UPDATE {table} SET hls_status='error' WHERE id=?", (row_id,))
        conn.commit()
        conn.close()


def build_clip_hls(clip_id: int):
    conn = get_db()
    clip = conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    conn.close()
    if not clip:
        return
    src, _ = find_source_path(clip["filename"])
    _rebuild_hls("clips", clip_id, src, clip["hls_dir"])


def build_upload_hls(upload_id: int):
    conn = get_db()
    up = conn.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
    conn.close()
    if not up or up["kind"] != "video":
        return
    _rebuild_hls("uploads", upload_id, UPLOADS_DIR / up["filename"], up["hls_dir"])


# ---------- DB ----------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _add_col(conn, table, col, ddl):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        return True
    return False


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, source_filename TEXT, display_name TEXT,
            status TEXT NOT NULL DEFAULT 'raw', slug TEXT UNIQUE,
            published INTEGER DEFAULT 0, visibility TEXT DEFAULT 'private',
            user_id INTEGER, thumbnail TEXT, duration REAL DEFAULT 0,
            hls_status TEXT DEFAULT 'none', hls_dir TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, slug TEXT UNIQUE,
            visibility TEXT DEFAULT 'unlisted', user_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, original_name TEXT, display_name TEXT,
            kind TEXT, thumbnail TEXT, slug TEXT UNIQUE,
            visibility TEXT DEFAULT 'unlisted', user_id INTEGER,
            hls_status TEXT DEFAULT 'none', hls_dir TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ip TEXT, ts REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, kind TEXT, ref_id INTEGER,
            slug TEXT, name TEXT, url TEXT, src TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrations for pre-existing databases.
    _add_col(conn, "clips", "thumbnail", "thumbnail TEXT")
    _add_col(conn, "clips", "duration", "duration REAL DEFAULT 0")
    _add_col(conn, "clips", "display_name", "display_name TEXT")
    if _add_col(conn, "clips", "source_filename", "source_filename TEXT"):
        conn.execute("UPDATE clips SET source_filename = filename WHERE source_filename IS NULL")
    if _add_col(conn, "clips", "visibility", "visibility TEXT DEFAULT 'private'"):
        conn.execute("UPDATE clips SET visibility = CASE WHEN published = 1 THEN 'unlisted' ELSE 'private' END")
    _add_col(conn, "clips", "hls_status", "hls_status TEXT DEFAULT 'none'")
    _add_col(conn, "clips", "hls_dir", "hls_dir TEXT")
    _add_col(conn, "clips", "user_id", "user_id INTEGER")
    _add_col(conn, "images", "visibility", "visibility TEXT DEFAULT 'unlisted'")
    _add_col(conn, "images", "user_id", "user_id INTEGER")
    _add_col(conn, "images", "display_name", "display_name TEXT")
    _add_col(conn, "uploads", "visibility", "visibility TEXT DEFAULT 'unlisted'")
    _add_col(conn, "uploads", "hls_status", "hls_status TEXT DEFAULT 'none'")
    _add_col(conn, "uploads", "hls_dir", "hls_dir TEXT")
    _add_col(conn, "uploads", "user_id", "user_id INTEGER")

    # Seed the admin account (first run) and adopt any orphaned content.
    has_user = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
    if not has_user:
        conn.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?,?,1)",
                     (ADMIN_USERNAME, hash_password(ADMIN_PASSWORD)))
    admin = conn.execute("SELECT id FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1").fetchone()
    if admin:
        for t in ("clips", "images", "uploads"):
            conn.execute(f"UPDATE {t} SET user_id = ? WHERE user_id IS NULL", (admin["id"],))
    # Every clip should have a slug so it always has a (visibility-gated) view URL.
    for row in conn.execute("SELECT id FROM clips WHERE slug IS NULL").fetchall():
        conn.execute("UPDATE clips SET slug = ? WHERE id = ?", (uuid.uuid4().hex[:8], row["id"]))
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
        conn.execute("UPDATE clips SET duration = ?, thumbnail = ? WHERE id = ?",
                     (get_duration(path), row["thumbnail"] or generate_thumbnail(path), row["id"]))
    conn.commit()
    conn.close()


def admin_user_id():
    conn = get_db()
    row = conn.execute("SELECT id FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1").fetchone()
    conn.close()
    return row["id"] if row else None


def get_user_by_name(username):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def login_locked(ip: str) -> bool:
    conn = get_db()
    n = conn.execute("SELECT COUNT(*) FROM login_attempts WHERE ip = ? AND ts > ?",
                     (ip, time.time() - 900)).fetchone()[0]
    conn.close()
    return n >= 10


def record_login_fail(ip: str):
    conn = get_db()
    conn.execute("INSERT INTO login_attempts (ip, ts) VALUES (?, ?)", (ip, time.time()))
    conn.execute("DELETE FROM login_attempts WHERE ts < ?", (time.time() - 3600,))
    conn.commit()
    conn.close()


def clear_login_fails(ip: str):
    conn = get_db()
    conn.execute("DELETE FROM login_attempts WHERE ip = ?", (ip,))
    conn.commit()
    conn.close()


def add_notification(user_id, kind, ref_id, slug, name, url, src=None):
    if not user_id:
        return
    conn = get_db()
    conn.execute(
        "INSERT INTO notifications (user_id, kind, ref_id, slug, name, url, src) VALUES (?,?,?,?,?,?,?)",
        (user_id, kind, ref_id, slug, name, url, src))
    conn.execute("DELETE FROM notifications WHERE created_at < datetime('now','-1 day')")
    conn.commit()
    conn.close()


init_db()
backfill_metadata()

# Clear orphaned chunk-upload temp files (failed/abandoned resumable uploads).
try:
    cutoff = time.time() - 12 * 3600
    for p in TMP_DIR.glob("*.part"):
        if p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)
except Exception:
    pass


# ---------- Upload helper ----------

async def _save_upload(file: UploadFile, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES):
    total = 0
    try:
        async with aiofiles.open(dest, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(413, "File too large")
                await f.write(chunk)
    except HTTPException:
        if dest.exists():
            dest.unlink()
        raise


def require_upload_token(request: Request):
    if UPLOAD_TOKEN and not secrets.compare_digest(request.headers.get("x-upload-token", ""), UPLOAD_TOKEN):
        raise HTTPException(401, "Invalid upload token")


# ---------- Media serving ----------

@app.get("/media/{subpath:path}")
async def serve_media(request: Request, subpath: str):
    file_path = (DATA_DIR / subpath).resolve()
    if DATA_DIR.resolve() not in file_path.parents or not file_path.is_file():
        raise HTTPException(404, "File not found")

    st = file_path.stat()
    file_size = st.st_size
    etag = f'"{st.st_mtime_ns:x}-{file_size:x}"'
    media_type = media_type_for(file_path)
    cache = "public, max-age=60" if file_path.suffix.lower() == ".m3u8" else \
        "public, max-age=31536000, immutable"
    base_headers = {
        "Accept-Ranges": "bytes", "ETag": etag,
        "Last-Modified": formatdate(st.st_mtime, usegmt=True), "Cache-Control": cache,
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

            return StreamingResponse(iter_range(), status_code=206, media_type=media_type,
                                     headers={**base_headers,
                                              "Content-Range": f"bytes {start}-{end}/{file_size}",
                                              "Content-Length": str(length)})

    async def iter_full():
        async with aiofiles.open(file_path, "rb") as f:
            while True:
                data = await f.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    return StreamingResponse(iter_full(), media_type=media_type,
                             headers={**base_headers, "Content-Length": str(file_size)})


# ---------- Auth routes ----------

@app.get("/login")
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/library")
    return templates.TemplateResponse(request, "login.html",
                                      {"error": None, "authenticated": False,
                                       "open_registration": OPEN_REGISTRATION})


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = client_ip(request)
    if login_locked(ip) or not rate_ok(f"login:{ip}", 10, 300):
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Too many attempts. Try again in a few minutes.",
             "authenticated": False, "open_registration": OPEN_REGISTRATION},
            status_code=429)
    user = get_user_by_name(username)
    if user and verify_password(password, user["password_hash"]):
        clear_login_fails(ip)
        request.session["uid"] = user["id"]
        request.session["username"] = user["username"]
        request.session["is_admin"] = bool(user["is_admin"])
        return RedirectResponse(url="/library", status_code=303)
    record_login_fail(ip)
    return templates.TemplateResponse(
        request, "login.html",
        {"error": "That username and password don't match.",
         "authenticated": False, "open_registration": OPEN_REGISTRATION},
        status_code=401)


@app.get("/register")
async def register_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/library")
    if not OPEN_REGISTRATION:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse(request, "register.html",
                                      {"error": None, "authenticated": False})


@app.post("/register")
async def register_submit(request: Request, username: str = Form(...),
                          password: str = Form(...), confirm: str = Form("")):
    if not OPEN_REGISTRATION:
        raise HTTPException(403, "Registration is closed")
    ip = client_ip(request)
    if not rate_ok(f"reg:{ip}", 5, 3600):
        return templates.TemplateResponse(
            request, "register.html",
            {"error": "Too many sign-ups from here. Try again later.", "authenticated": False},
            status_code=429)

    username = username.strip()
    err = None
    if not USERNAME_RE.match(username):
        err = "Usernames are 3–20 characters: letters, numbers, underscore."
    elif len(password) < 8:
        err = "Use a password of at least 8 characters."
    elif confirm and confirm != password:
        err = "Those passwords don't match."
    elif get_user_by_name(username):
        err = "That username is taken."
    if err:
        return templates.TemplateResponse(request, "register.html",
                                          {"error": err, "authenticated": False}, status_code=400)

    conn = get_db()
    cur = conn.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?,?,0)",
                       (username, hash_password(password)))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    request.session["uid"] = uid
    request.session["username"] = username
    request.session["is_admin"] = False
    return RedirectResponse(url="/library", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


# ---------- Capture-PC (watcher) endpoints ----------

@app.post("/upload")
async def upload_video(request: Request, background: BackgroundTasks, file: UploadFile = File(...)):
    require_upload_token(request)
    if not rate_ok(f"up:{client_ip(request)}", 120, 60):
        raise HTTPException(429, "Too many uploads")
    ext = Path(file.filename).suffix.lower() or ".mp4"
    new_name = f"{uuid.uuid4().hex}{ext}"
    await _save_upload(file, RAW_DIR / new_name)
    owner = admin_user_id()
    slug = uuid.uuid4().hex[:8]
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO clips (filename, source_filename, display_name, status, visibility, slug, duration, user_id) "
        "VALUES (?, ?, ?, 'raw', 'private', ?, 0, ?)",
        (new_name, new_name, file.filename, slug, owner))
    conn.commit()
    clip_id = cur.lastrowid
    conn.close()
    background.add_task(build_clip_metadata, clip_id, new_name)
    # Notify the owner's browser that a new clip is ready to edit.
    add_notification(owner, "clip", clip_id, slug, file.filename or "Clip", f"/clip/{clip_id}/edit")
    return {"id": clip_id, "filename": new_name}


@app.post("/upload-image")
async def upload_image(request: Request, file: UploadFile = File(...)):
    require_upload_token(request)
    if not rate_ok(f"up:{client_ip(request)}", 120, 60):
        raise HTTPException(429, "Too many uploads")
    ext = Path(file.filename).suffix.lower() or ".png"
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, "Unsupported image type")
    new_name = f"{uuid.uuid4().hex}{ext}"
    await _save_upload(file, IMAGES_DIR / new_name)
    slug = uuid.uuid4().hex[:8]
    owner = admin_user_id()
    conn = get_db()
    cur = conn.execute("INSERT INTO images (filename, display_name, slug, visibility, user_id) "
                       "VALUES (?, ?, ?, 'unlisted', ?)",
                       (new_name, file.filename, slug, owner))
    conn.commit()
    image_id = cur.lastrowid
    conn.close()
    add_notification(owner, "image", image_id, slug, file.filename or "Screenshot",
                     f"/i/{slug}", src=f"/media/images/{new_name}")
    return {"url": f"/i/{slug}"}


# ---------- Home (public gallery) ----------

@app.get("/")
async def home(request: Request):
    authed = is_authenticated(request)
    user = current_user(request)
    conn = get_db()
    gallery_clips = conn.execute(
        "SELECT c.*, u.username AS owner_name FROM clips c LEFT JOIN users u ON u.id = c.user_id "
        "WHERE c.visibility = 'public' AND c.slug IS NOT NULL ORDER BY c.created_at DESC"
    ).fetchall()
    gallery_uploads = conn.execute(
        "SELECT p.*, u.username AS owner_name FROM uploads p LEFT JOIN users u ON u.id = p.user_id "
        "WHERE p.visibility = 'public' ORDER BY p.created_at DESC"
    ).fetchall()
    gallery_images = conn.execute(
        "SELECT i.*, u.username AS owner_name FROM images i LEFT JOIN users u ON u.id = i.user_id "
        "WHERE i.visibility = 'public' ORDER BY i.created_at DESC"
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(
        request, "home.html",
        {"gallery_clips": gallery_clips, "gallery_uploads": gallery_uploads,
         "gallery_images": gallery_images, "active": "home",
         "authenticated": authed, "user": user})


# ---------- Library ----------

@app.get("/library")
async def library(request: Request, user: dict = Depends(require_user)):
    conn = get_db()
    if user["is_admin"]:
        clips = conn.execute("SELECT * FROM clips ORDER BY created_at DESC").fetchall()
        images = conn.execute("SELECT * FROM images ORDER BY created_at DESC LIMIT 120").fetchall()
        uploads = conn.execute("SELECT * FROM uploads ORDER BY created_at DESC").fetchall()
    else:
        clips = conn.execute("SELECT * FROM clips WHERE user_id = ? ORDER BY created_at DESC",
                             (user["id"],)).fetchall()
        images = conn.execute("SELECT * FROM images WHERE user_id = ? ORDER BY created_at DESC LIMIT 120",
                             (user["id"],)).fetchall()
        uploads = conn.execute("SELECT * FROM uploads WHERE user_id = ? ORDER BY created_at DESC",
                              (user["id"],)).fetchall()
    conn.close()
    return templates.TemplateResponse(
        request, "library.html",
        {"clips": clips, "images": images, "uploads": uploads,
         "active": "library", "authenticated": True, "user": user})


# ---------- Editor ----------

def _load_owned_clip(conn, clip_id, user):
    clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if not clip:
        raise HTTPException(404, "Clip not found")
    if not owns_or_admin(user, clip):
        raise HTTPException(403, "Not your clip")
    return clip


@app.get("/clip/{clip_id}/edit")
async def edit_page(request: Request, clip_id: int, user: dict = Depends(require_user)):
    conn = get_db()
    clip = _load_owned_clip(conn, clip_id, user)
    conn.close()
    source_filename = clip["source_filename"] or clip["filename"]
    source_path, source_subdir = find_source_path(source_filename)
    if source_path is None:
        source_filename = clip["filename"]
        source_path, source_subdir = find_source_path(source_filename)
        if source_subdir is None:
            source_subdir = "edited" if clip["status"] == "edited" else "raw"
    return templates.TemplateResponse(
        request, "editor.html",
        {"clip": clip, "source_filename": source_filename, "source_subdir": source_subdir,
         "active": "library", "authenticated": True, "user": user,
         "error": request.query_params.get("error")})


@app.post("/clip/{clip_id}/trim")
async def trim_clip(background: BackgroundTasks, clip_id: int, start: str = Form(...),
                    end: str = Form(...), save_as: str = Form("save"),
                    request: Request = None, user: dict = Depends(require_user)):
    conn = get_db()
    clip = _load_owned_clip(conn, clip_id, user)

    def _err(msg):
        return RedirectResponse(url=f"/clip/{clip_id}/edit?error=" + urllib.parse.quote(msg),
                                status_code=303)
    try:
        start_f = max(0.0, float(start))
        end_f = float(end)
    except ValueError:
        conn.close()
        return _err("Enter a valid trim range.")
    if end_f <= start_f:
        conn.close()
        return _err("The end point must come after the start.")

    source_filename = clip["source_filename"] or clip["filename"]
    source_path, _ = find_source_path(source_filename)
    if source_path is None:
        conn.close()
        return _err("The source file is missing.")

    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = EDITED_DIR / out_name
    ok, err = await run_in_threadpool(run_trim, source_path, out_path, start_f, end_f - start_f)
    if not ok:
        conn.close()
        return _err(err or "The trim couldn't be processed.")

    duration = await run_in_threadpool(get_duration, out_path)
    thumbnail = await run_in_threadpool(generate_thumbnail, out_path)

    # Save as copy: a new, independent clip (the original — even if published —
    # is left exactly as it was). The copy is self-contained (its own source).
    if save_as == "copy":
        base_name = clip["display_name"] or clip["filename"]
        new_slug = uuid.uuid4().hex[:8]
        cur = conn.execute(
            "INSERT INTO clips (filename, source_filename, display_name, status, visibility, slug, "
            "thumbnail, duration, user_id) VALUES (?, ?, ?, 'edited', 'private', ?, ?, ?, ?)",
            (out_name, out_name, f"{base_name} (copy)", new_slug, thumbnail, duration, clip["user_id"]))
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return RedirectResponse(url=f"/clip/{new_id}/edit", status_code=303)

    # Save: replace this clip's current cut.
    prev_filename, prev_status, prev_thumb, prev_hls = \
        clip["filename"], clip["status"], clip["thumbnail"], clip["hls_dir"]
    conn.execute("UPDATE clips SET filename=?, status='edited', thumbnail=?, duration=?, "
                 "hls_status='none', hls_dir=NULL WHERE id=?",
                 (out_name, thumbnail, duration, clip_id))
    conn.commit()
    conn.close()

    if prev_status == "edited" and prev_filename != source_filename:
        old = EDITED_DIR / prev_filename
        if old.exists():
            old.unlink()
    if prev_thumb and prev_thumb != thumbnail:
        tp = THUMBNAIL_DIR / prev_thumb
        if tp.exists():
            tp.unlink()
    cleanup_hls_dir(prev_hls)
    if clip["visibility"] in ("unlisted", "public"):
        background.add_task(build_clip_hls, clip_id)
    return RedirectResponse(url=f"/clip/{clip_id}/edit", status_code=303)


@app.post("/clip/{clip_id}/rename")
async def rename_clip(clip_id: int, name: str = Form(...), user: dict = Depends(require_user)):
    conn = get_db()
    _load_owned_clip(conn, clip_id, user)
    name = name.strip() or None
    conn.execute("UPDATE clips SET display_name = ? WHERE id = ?", (name, clip_id))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "name": name})


@app.post("/clip/{clip_id}/visibility")
async def set_clip_visibility(request: Request, background: BackgroundTasks, clip_id: int,
                              value: str = Form(...), user: dict = Depends(require_user)):
    if value not in VISIBILITIES:
        raise HTTPException(400, "Bad visibility")
    conn = get_db()
    clip = _load_owned_clip(conn, clip_id, user)
    slug = clip["slug"] or uuid.uuid4().hex[:8]
    published = 1 if value in ("unlisted", "public") else 0
    conn.execute("UPDATE clips SET visibility=?, slug=?, published=? WHERE id=?",
                 (value, slug, published, clip_id))
    conn.commit()
    conn.close()
    if value in ("unlisted", "public") and (clip["hls_status"] or "none") not in ("pending", "ready"):
        background.add_task(build_clip_hls, clip_id)
    if request.headers.get("accept", "").startswith("application/json"):
        return JSONResponse({"ok": True, "visibility": value, "slug": slug})
    return _back(request, f"/clip/{clip_id}/edit")


@app.post("/clip/{clip_id}/delete")
async def delete_clip(request: Request, clip_id: int, user: dict = Depends(require_user)):
    conn = get_db()
    clip = _load_owned_clip(conn, clip_id, user)
    for name in {clip["filename"], clip["source_filename"]}:
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
    cleanup_hls_dir(clip["hls_dir"])
    conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
    conn.commit()
    conn.close()
    return _back(request, "/library")


@app.post("/clip/{clip_id}/publish")
async def publish_clip(background: BackgroundTasks, clip_id: int, user: dict = Depends(require_user)):
    conn = get_db()
    clip = _load_owned_clip(conn, clip_id, user)
    slug = clip["slug"] or uuid.uuid4().hex[:8]
    conn.execute("UPDATE clips SET slug=?, published=1, visibility='unlisted' WHERE id=?", (slug, clip_id))
    conn.commit()
    conn.close()
    if (clip["hls_status"] or "none") not in ("pending", "ready"):
        background.add_task(build_clip_hls, clip_id)
    return RedirectResponse(url=f"/clip/{clip_id}/edit", status_code=303)


# ---------- Share routes ----------

@app.get("/c/{slug}")
async def share_clip(request: Request, slug: str):
    conn = get_db()
    clip = conn.execute("SELECT c.*, u.username AS owner_name FROM clips c "
                        "LEFT JOIN users u ON u.id = c.user_id WHERE c.slug = ?", (slug,)).fetchone()
    conn.close()
    if not clip:
        raise HTTPException(404, "Clip not found")
    if clip["visibility"] == "private" and not owns_or_admin(current_user(request), clip):
        raise AuthRedirect()
    subdir = "edited" if clip["status"] == "edited" else "raw"
    hls_url = f"/media/hls/{clip['hls_dir']}/master.m3u8" \
        if clip["hls_status"] == "ready" and clip["hls_dir"] else None
    return templates.TemplateResponse(
        request, "share.html",
        {"clip": clip, "subdir": subdir, "hls_url": hls_url,
         "authenticated": is_authenticated(request)})


@app.get("/i/{slug}")
async def share_image(request: Request, slug: str):
    conn = get_db()
    img = conn.execute("SELECT i.*, u.username AS owner_name FROM images i "
                       "LEFT JOIN users u ON u.id = i.user_id WHERE i.slug = ?", (slug,)).fetchone()
    conn.close()
    if not img:
        raise HTTPException(404, "Image not found")
    if img["visibility"] == "private" and not owns_or_admin(current_user(request), img):
        raise AuthRedirect()
    return templates.TemplateResponse(request, "share_image.html",
                                      {"image": img, "authenticated": is_authenticated(request)})


@app.get("/image/{image_id}/edit")
async def image_edit_page(request: Request, image_id: int, user: dict = Depends(require_user)):
    conn = get_db()
    img = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    conn.close()
    if not img:
        raise HTTPException(404, "Screenshot not found")
    if not owns_or_admin(user, img):
        raise HTTPException(403, "Not your screenshot")
    return templates.TemplateResponse(request, "image_editor.html",
                                      {"image": img, "active": "library", "authenticated": True, "user": user})


@app.post("/image/{image_id}/rename")
async def rename_image(image_id: int, name: str = Form(...), user: dict = Depends(require_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row or not owns_or_admin(user, row):
        conn.close()
        raise HTTPException(403, "Not your screenshot")
    name = name.strip() or None
    conn.execute("UPDATE images SET display_name = ? WHERE id = ?", (name, image_id))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "name": name})


@app.post("/image/{image_id}/visibility")
async def set_image_visibility(request: Request, image_id: int, value: str = Form(...), user: dict = Depends(require_user)):
    if value not in VISIBILITIES:
        raise HTTPException(400, "Bad visibility")
    conn = get_db()
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row or not owns_or_admin(user, row):
        conn.close()
        raise HTTPException(403, "Not your screenshot")
    conn.execute("UPDATE images SET visibility = ? WHERE id = ?", (value, image_id))
    conn.commit()
    conn.close()
    return _back(request, "/library")


@app.post("/image/{image_id}/delete")
async def delete_image(request: Request, image_id: int, user: dict = Depends(require_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if row and owns_or_admin(user, row):
        p = IMAGES_DIR / row["filename"]
        if p.exists():
            p.unlink()
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
    conn.close()
    return _back(request, "/library")


# ---------- Manual uploads ----------

async def _ingest_web(background, request, original_name, owner, place):
    """Create the right record from an incoming web upload. `place(dest)` is an
    awaitable that writes the bytes to dest (used by both single and chunked paths)."""
    kind = detect_kind(original_name)
    ext = Path(original_name).suffix.lower()
    owner_id = owner["id"] if owner else None
    new_name = f"{uuid.uuid4().hex}{ext}"
    slug = uuid.uuid4().hex[:8]

    if kind == "video" and owner:
        await place(RAW_DIR / new_name)
        conn = get_db()
        cur = conn.execute(
            "INSERT INTO clips (filename, source_filename, display_name, status, visibility, slug, duration, user_id) "
            "VALUES (?, ?, ?, 'raw', 'private', ?, 0, ?)",
            (new_name, new_name, original_name, slug, owner_id))
        conn.commit()
        clip_id = cur.lastrowid
        conn.close()
        background.add_task(build_clip_metadata, clip_id, new_name)
        return {"ok": True, "kind": "clip", "edit": f"/clip/{clip_id}/edit"}

    if kind == "image":
        await place(IMAGES_DIR / new_name)
        conn = get_db()
        conn.execute(
            "INSERT INTO images (filename, display_name, slug, visibility, user_id) VALUES (?,?,?, 'unlisted', ?)",
            (new_name, original_name, slug, owner_id))
        conn.commit()
        conn.close()
        return {"ok": True, "kind": "image", "slug": slug,
                "src": f"/media/images/{new_name}", "view": f"/i/{slug}"}

    await place(UPLOADS_DIR / new_name)
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO uploads (filename, original_name, display_name, kind, slug, visibility, user_id) "
        "VALUES (?,?,?,?,?, 'unlisted', ?)",
        (new_name, original_name, original_name, kind, slug, owner_id))
    conn.commit()
    upload_id = cur.lastrowid
    conn.close()
    background.add_task(build_upload_metadata, upload_id, new_name, kind)
    if kind == "video":
        background.add_task(build_upload_hls, upload_id)
    return {"ok": True, "kind": kind, "slug": slug, "view": f"/u/{slug}"}


@app.post("/uploads/add")
async def add_upload(request: Request, background: BackgroundTasks, file: UploadFile = File(...)):
    if not rate_ok(f"up:{client_ip(request)}", 120, 60):
        raise HTTPException(429, "Too many uploads")
    owner = current_user(request)
    if REQUIRE_LOGIN_UPLOAD and not owner:
        raise HTTPException(401, "Sign in to upload")
    original_name = file.filename or "file"
    return await _ingest_web(background, request, original_name, owner,
                             lambda dest: _save_upload(file, dest))


# ---- Chunked uploads (keeps every request under the Cloudflare 100MB body cap) ----

_UPLOAD_ID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")


def _part_path(upload_id: str) -> Path:
    if not upload_id or not _UPLOAD_ID_RE.match(upload_id):
        raise HTTPException(400, "Bad upload id")
    return TMP_DIR / f"{upload_id}.part"


def _chunk_auth(request: Request) -> bool:
    """True if this caller may upload. Capture-PC token or web session."""
    if UPLOAD_TOKEN and secrets.compare_digest(request.headers.get("x-upload-token", ""), UPLOAD_TOKEN):
        return True
    if REQUIRE_LOGIN_UPLOAD and not is_authenticated(request):
        raise HTTPException(401, "Sign in to upload")
    return False


@app.post("/uploads/chunk")
async def upload_chunk(request: Request):
    _chunk_auth(request)
    if not rate_ok(f"up:{client_ip(request)}", 1200, 60):
        raise HTTPException(429, "Too many requests")
    dest = _part_path(request.headers.get("x-upload-id", ""))
    size = dest.stat().st_size if dest.exists() else 0
    try:
        async with aiofiles.open(dest, "ab") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Upload exceeds maximum size")
                await f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return {"ok": True, "received": size}


@app.post("/uploads/finish")
async def upload_finish(request: Request, background: BackgroundTasks,
                        upload_id: str = Form(...), filename: str = Form(...),
                        target: str = Form("web")):
    token_ok = bool(UPLOAD_TOKEN) and secrets.compare_digest(
        request.headers.get("x-upload-token", ""), UPLOAD_TOKEN)
    part = _part_path(upload_id)
    if not part.exists():
        raise HTTPException(400, "No upload data for that id")

    # Capture-PC video -> a clip owned by admin (same result as /upload).
    if token_ok and target == "clip":
        ext = Path(filename).suffix.lower() or ".mp4"
        new_name = f"{uuid.uuid4().hex}{ext}"
        await run_in_threadpool(shutil.move, str(part), str(RAW_DIR / new_name))
        owner = admin_user_id()
        slug = uuid.uuid4().hex[:8]
        conn = get_db()
        cur = conn.execute(
            "INSERT INTO clips (filename, source_filename, display_name, status, visibility, slug, duration, user_id) "
            "VALUES (?, ?, ?, 'raw', 'private', ?, 0, ?)",
            (new_name, new_name, filename, slug, owner))
        conn.commit()
        clip_id = cur.lastrowid
        conn.close()
        background.add_task(build_clip_metadata, clip_id, new_name)
        add_notification(owner, "clip", clip_id, slug, filename or "Clip", f"/clip/{clip_id}/edit")
        return {"id": clip_id, "filename": new_name}

    # Web context (cookie session), mirrors /uploads/add routing.
    owner = current_user(request)
    if REQUIRE_LOGIN_UPLOAD and not owner:
        part.unlink(missing_ok=True)
        raise HTTPException(401, "Sign in to upload")

    async def place(dest):
        await run_in_threadpool(shutil.move, str(part), str(dest))

    return await _ingest_web(background, request, filename, owner, place)


def _load_owned_upload(conn, upload_id, user):
    row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Upload not found")
    if not owns_or_admin(user, row):
        raise HTTPException(403, "Not your upload")
    return row


@app.post("/uploads/{upload_id}/rename")
async def rename_upload(upload_id: int, name: str = Form(...), user: dict = Depends(require_user)):
    conn = get_db()
    _load_owned_upload(conn, upload_id, user)
    name = name.strip() or None
    conn.execute("UPDATE uploads SET display_name = ? WHERE id = ?", (name, upload_id))
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "name": name})


@app.post("/uploads/{upload_id}/visibility")
async def set_upload_visibility(request: Request, background: BackgroundTasks, upload_id: int,
                                value: str = Form(...), user: dict = Depends(require_user)):
    if value not in VISIBILITIES:
        raise HTTPException(400, "Bad visibility")
    conn = get_db()
    up = _load_owned_upload(conn, upload_id, user)
    conn.execute("UPDATE uploads SET visibility = ? WHERE id = ?", (value, upload_id))
    conn.commit()
    conn.close()
    if up["kind"] == "video" and value in ("unlisted", "public") \
            and (up["hls_status"] or "none") not in ("pending", "ready"):
        background.add_task(build_upload_hls, upload_id)
    return _back(request, "/library")


@app.post("/uploads/{upload_id}/delete")
async def delete_upload(request: Request, upload_id: int, user: dict = Depends(require_user)):
    conn = get_db()
    row = _load_owned_upload(conn, upload_id, user)
    path = UPLOADS_DIR / row["filename"]
    if path.exists():
        path.unlink()
    if row["thumbnail"]:
        tp = THUMBNAIL_DIR / row["thumbnail"]
        if tp.exists():
            tp.unlink()
    cleanup_hls_dir(row["hls_dir"])
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()
    return _back(request, "/library")


@app.get("/u/{slug}")
async def share_upload(request: Request, slug: str):
    conn = get_db()
    row = conn.execute("SELECT p.*, u.username AS owner_name FROM uploads p "
                       "LEFT JOIN users u ON u.id = p.user_id WHERE p.slug = ?", (slug,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Not found")
    if row["visibility"] == "private" and not owns_or_admin(current_user(request), row):
        raise AuthRedirect()
    hls_url = f"/media/hls/{row['hls_dir']}/master.m3u8" \
        if row["kind"] == "video" and row["hls_status"] == "ready" and row["hls_dir"] else None
    return templates.TemplateResponse(request, "share_upload.html",
                                      {"upload": row, "hls_url": hls_url,
                                       "authenticated": is_authenticated(request)})


@app.get("/notifications/poll")
async def notifications_poll(after: int = None, user: dict = Depends(require_user)):
    conn = get_db()
    if after is None:
        # Baseline call: tell the client the current high-water mark, no items.
        row = conn.execute("SELECT COALESCE(MAX(id),0) AS m FROM notifications WHERE user_id=?",
                           (user["id"],)).fetchone()
        conn.close()
        return {"last": row["m"], "items": []}
    rows = conn.execute(
        "SELECT id, kind, ref_id, slug, name, url, src FROM notifications "
        "WHERE user_id=? AND id>? ORDER BY id ASC LIMIT 20", (user["id"], after)).fetchall()
    conn.close()
    items = [dict(r) for r in rows]
    return {"last": items[-1]["id"] if items else after, "items": items}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "encoder": ENCODER_NAME, "hls_encoder": HLS_ENCODER,
            "hls_segment_seconds": HLS_SEGMENT_SECONDS, "transcode_concurrency": TRANSCODE_CONCURRENCY,
            "open_registration": OPEN_REGISTRATION}