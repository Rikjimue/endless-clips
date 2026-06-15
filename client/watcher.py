import time
import subprocess
import requests
import pyperclip
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Config ---
VIDEO_FOLDER = r"C:\Users\rikji\Videos\Endless Clips"
SCREENSHOT_FOLDER = r"C:\Users\rikji\Pictures\Endless Clips"
FFMPEG_PATH = r"C:\Users\rikji\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"

VIDEO_UPLOAD_URL = "http://192.168.1.248:8000/upload"
IMAGE_UPLOAD_URL = "http://192.168.1.248:8000/upload-image"

REMUX_TO_MP4 = True
VIDEO_EXTS = (".mp4", ".mkv", ".flv")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def wait_until_stable(path: Path, checks=3, interval=0.5):
    last_size = -1
    stable_count = 0
    while stable_count < checks:
        if not path.exists():
            time.sleep(interval)
            continue
        size = path.stat().st_size
        if size == last_size:
            stable_count += 1
        else:
            stable_count = 0
        last_size = size
        time.sleep(interval)


def remux_to_mp4(path: Path) -> Path:
    if path.suffix.lower() == ".mp4":
        return path
    out_path = path.with_suffix(".mp4")
    cmd = [
    FFMPEG_PATH, "-y", "-i", str(path),
    "-c", "copy", "-movflags", "+faststart",
    str(out_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def upload_video(path: Path):
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "video/mp4")}
        resp = requests.post(VIDEO_UPLOAD_URL, files=files)
    resp.raise_for_status()
    print(f"Uploaded video {path.name}: {resp.status_code}")


def upload_image(path: Path):
    with open(path, "rb") as f:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        files = {"file": (path.name, f, mime)}
        resp = requests.post(IMAGE_UPLOAD_URL, files=files)
    resp.raise_for_status()

    data = resp.json()
    link = data.get("url")  # adjust key to match your API's response

    if link:
        pyperclip.copy(link)
        print(f"Uploaded image {path.name}, link copied: {link}")
    else:
        print(f"Uploaded image {path.name}, but no URL returned: {data}")


class ClipHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        ext = path.suffix.lower()

        try:
            if ext in VIDEO_EXTS:
                print(f"New video: {path.name}")
                wait_until_stable(path)

                if REMUX_TO_MP4:
                    final_path = remux_to_mp4(path)
                    if final_path != path:
                        path.unlink()
                else:
                    final_path = path

                upload_video(final_path)

            elif ext in IMAGE_EXTS:
                print(f"New screenshot: {path.name}")
                wait_until_stable(path)
                upload_image(path)

        except Exception as e:
            print(f"Error processing {path.name}: {e}")


if __name__ == "__main__":
    Path(VIDEO_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(SCREENSHOT_FOLDER).mkdir(parents=True, exist_ok=True)

    handler = ClipHandler()
    observer = Observer()
    observer.schedule(handler, VIDEO_FOLDER, recursive=False)
    observer.schedule(handler, SCREENSHOT_FOLDER, recursive=False)
    observer.start()

    print(f"Watching {VIDEO_FOLDER} and {SCREENSHOT_FOLDER}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()