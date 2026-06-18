import os
import time
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import pyperclip
except Exception:  # clipboard is optional (e.g. headless)
    pyperclip = None

# ------------------------------------------------------------------ Config ---
# All values can be overridden with environment variables.
VIDEO_FOLDER = os.environ.get("CLIPS_VIDEO_FOLDER", r"C:\Users\rikji\Videos\Endless Clips")
SCREENSHOT_FOLDER = os.environ.get("CLIPS_SCREENSHOT_FOLDER", r"C:\Users\rikji\Pictures\Endless Clips")
FFMPEG_PATH = os.environ.get(
    "CLIPS_FFMPEG_PATH",
    r"C:\Users\rikji\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
)

SERVER = os.environ.get("CLIPS_SERVER", "https://clips.jaioleeming.com")
VIDEO_UPLOAD_URL = f"{SERVER}/upload"
IMAGE_UPLOAD_URL = f"{SERVER}/upload-image"
UPLOAD_TOKEN = os.environ.get("CLIPS_UPLOAD_TOKEN", "").strip()  # strip stray .env whitespace
AUTH_HEADERS = {"X-Upload-Token": UPLOAD_TOKEN} if UPLOAD_TOKEN else {}

REMUX_TO_MP4 = os.environ.get("CLIPS_REMUX", "1") == "1"
# Delete local files once the server confirms the upload. This is the behaviour
# that was previously broken/leaky.
DELETE_AFTER_UPLOAD = os.environ.get("CLIPS_DELETE_AFTER_UPLOAD", "1") == "1"
PROCESS_EXISTING_ON_START = os.environ.get("CLIPS_PROCESS_EXISTING", "0") == "1"

VIDEO_EXTS = (".mp4", ".mkv", ".flv", ".mov", ".webm", ".avi")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# ------------------------------------------------------------- HTTP session ---
session = requests.Session()
session.mount("http://", HTTPAdapter(max_retries=Retry(
    total=4, backoff_factor=1.5, status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"])))

# In-flight dedupe so a create + modify event for the same file is processed once.
_seen = {}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def wait_until_stable(path: Path, checks=3, interval=0.5, timeout=120):
    """Wait until the file size stops changing AND the file is openable.

    The 'openable' check matters on Windows, where the recorder may still hold
    an exclusive lock briefly after it finishes writing.
    """
    last_size = -1
    stable = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not path.exists():
            time.sleep(interval)
            continue
        size = path.stat().st_size
        if size == last_size and size > 0:
            stable += 1
            if stable >= checks and _is_unlocked(path):
                return True
        else:
            stable = 0
        last_size = size
        time.sleep(interval)
    return path.exists()


def _is_unlocked(path: Path) -> bool:
    try:
        with open(path, "rb"):
            return True
    except (PermissionError, OSError):
        return False


def safe_delete(path: Path, attempts=12, delay=0.5) -> bool:
    """Delete with retries — Windows often keeps a handle open for a moment."""
    for _ in range(attempts):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError):
            time.sleep(delay)
    log(f"Could not delete {path.name} (still locked)")
    return False


def remux_to_mp4(path: Path) -> Path:
    out_path = path.with_suffix(".mp4")
    if out_path.exists() and out_path != path:
        out_path = path.with_name(f"{path.stem}_remux.mp4")
    cmd = [
        FFMPEG_PATH, "-y", "-i", str(path),
        "-c", "copy", "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def upload_video(path: Path) -> bool:
    """Chunked upload so no single request exceeds the proxy body cap (Cloudflare 100MB).

    Chunks upload in parallel and the server writes each at its byte offset, so the
    transfer is faster and no server-side stitching pass is needed.
    """
    chunk_url = f"{SERVER}/uploads/chunk"
    finish_url = f"{SERVER}/uploads/finish"
    upload_id = uuid.uuid4().hex
    chunk_bytes = 90 * 1024 * 1024
    workers = int(os.environ.get("CLIPS_UPLOAD_WORKERS", "3"))
    try:
        size = path.stat().st_size
        offsets = list(range(0, max(size, 1), chunk_bytes))
        sent = 0
        sent_lock = threading.Lock()

        def send_chunk(offset):
            nonlocal sent
            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read(chunk_bytes)
            headers = dict(AUTH_HEADERS)
            headers["X-Upload-Id"] = upload_id
            headers["X-Chunk-Offset"] = str(offset)
            headers["Content-Type"] = "application/octet-stream"
            resp = session.post(chunk_url, data=data, headers=headers, timeout=600)
            resp.raise_for_status()
            with sent_lock:
                sent += len(data)
                log(f"  {path.name}: {sent}/{size} bytes")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(send_chunk, off) for off in offsets]
            for fut in as_completed(futures):
                fut.result()  # re-raise the first failure

        resp = session.post(
            finish_url,
            data={"upload_id": upload_id, "filename": path.name, "target": "clip", "size": size},
            headers=AUTH_HEADERS, timeout=120)
        resp.raise_for_status()
        log(f"Uploaded video {path.name} ({resp.status_code})")
        return True
    except Exception as e:
        log(f"Video upload failed for {path.name}: {e}")
        return False


def upload_image(path: Path):
    try:
        with open(path, "rb") as f:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            files = {"file": (path.name, f, mime)}
            resp = session.post(IMAGE_UPLOAD_URL, files=files, headers=AUTH_HEADERS, timeout=120)
        resp.raise_for_status()
        link = resp.json().get("url")
        if link:
            full = link if link.startswith("http") else f"{SERVER}{link}"
            if pyperclip:
                try:
                    pyperclip.copy(full)
                except Exception:
                    pass
            log(f"Uploaded image {path.name}, link: {full}")
        return True
    except Exception as e:
        log(f"Image upload failed for {path.name}: {e}")
        return False


def process_video(path: Path):
    if not wait_until_stable(path):
        log(f"Gave up waiting for {path.name} to finish writing")
        return

    upload_path = path
    remuxed = None
    if REMUX_TO_MP4 and path.suffix.lower() != ".mp4":
        try:
            remuxed = remux_to_mp4(path)
            upload_path = remuxed
        except subprocess.CalledProcessError as e:
            log(f"Remux failed for {path.name}, uploading original: {e}")
            upload_path = path

    ok = upload_video(upload_path)

    # Only delete after a confirmed upload, so a failed upload never loses data.
    if ok and DELETE_AFTER_UPLOAD:
        safe_delete(path)
        if remuxed is not None and remuxed != path:
            safe_delete(remuxed)
    elif not ok and remuxed is not None and remuxed != path:
        # Upload failed: clean up our temp remux but keep the user's original.
        safe_delete(remuxed)


def process_image(path: Path):
    if not wait_until_stable(path):
        log(f"Gave up waiting for {path.name} to finish writing")
        return
    if upload_image(path) and DELETE_AFTER_UPLOAD:
        safe_delete(path)


def dispatch(path: Path):
    # Dedupe: skip if we just handled this exact path+mtime.
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return
    now = time.time()
    if _seen.get(str(path)) == key and now - _seen.get("_t_" + str(path), 0) < 5:
        return
    _seen[str(path)] = key
    _seen["_t_" + str(path)] = now

    ext = path.suffix.lower()
    try:
        if ext in VIDEO_EXTS:
            log(f"New video: {path.name}")
            process_video(path)
        elif ext in IMAGE_EXTS:
            log(f"New screenshot: {path.name}")
            process_image(path)
    except Exception as e:
        log(f"Error processing {path.name}: {e}")


class ClipHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            dispatch(Path(event.src_path))


def process_existing():
    for folder, exts in ((VIDEO_FOLDER, VIDEO_EXTS), (SCREENSHOT_FOLDER, IMAGE_EXTS)):
        for p in Path(folder).iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                dispatch(p)


if __name__ == "__main__":
    Path(VIDEO_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(SCREENSHOT_FOLDER).mkdir(parents=True, exist_ok=True)

    if PROCESS_EXISTING_ON_START:
        process_existing()

    handler = ClipHandler()
    observer = Observer()
    observer.schedule(handler, VIDEO_FOLDER, recursive=False)
    observer.schedule(handler, SCREENSHOT_FOLDER, recursive=False)
    observer.start()
    log(f"Watching {VIDEO_FOLDER} and {SCREENSHOT_FOLDER} -> {SERVER}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()