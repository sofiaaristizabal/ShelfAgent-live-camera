"""
Uploads frames to the backend's POST /events/analyze without ever blocking the
camera loop. The backend only analyzes the image (YOLO + DeepFace) and stores
the snapshot. Right after a successful upload, the recorder (db_writer.py) saves
the movement, stock and alerts straight to the database.

We deliberately send no camera_id: some backend versions write their own stock
rows when they get one, and the recorder is the only thing that should do that.
The recorder fills in each detection's zone itself.
"""

import logging
import queue
import threading
import time

import cv2
import requests

from config import API_URL, REQUEST_TIMEOUT

logger = logging.getLogger("event_sender")

_jobs: "queue.Queue" = queue.Queue(maxsize=50)
_callback = None
_recorder = None
_worker = None
_worker_lock = threading.Lock()

CONNECT_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2.0


def set_result_callback(fn) -> None:
    """fn(info) is called from the worker thread after every upload.
    info = {"time", "reason", "ok", "lines", "raw"}"""
    global _callback
    _callback = fn


def set_recorder(fn) -> None:
    """fn(result, movement) -> dict, called after each successful upload (see db_writer.record)."""
    global _recorder
    _recorder = fn


def pending() -> int:
    """Uploads queued or in progress."""
    return _jobs.unfinished_tasks


def send_event(frame, reason: str, movement=None) -> bool:
    """Queue a frame for upload. The caller must not modify `frame` afterwards."""
    _ensure_worker()
    try:
        _jobs.put_nowait((frame, reason, movement))
    except queue.Full:
        logger.error("Upload queue full - dropping: %s", reason)
        return False
    logger.info("Queued upload: %s", reason)
    return True


def summarize(result: dict, extra: dict | None = None) -> list[str]:
    """Turn the backend's response (and what the recorder saved) into the short lines shown on screen."""
    extra = extra or {}
    counts = result.get("counts") or {}
    lines = ["Items: " + (", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) or "none detected")]

    faces = result.get("faces") or []
    if faces:
        who = [f["employee_id"] if f.get("recognized") and f.get("employee_id") else "unrecognized" for f in faces]
        lines.append("People: " + ", ".join(who))
    else:
        lines.append("People: none")

    if extra.get("error"):
        lines.append("DATABASE ERROR: " + extra["error"])
        return lines
    mv = extra.get("movement")
    if mv:
        lines.append(f"Movement saved: {mv['item']} {mv['movement_type']} "
                     f"{mv.get('zone_from') or '-'} -> {mv.get('zone_to') or '-'} "
                     f"by {mv.get('employee_id') or 'nobody'}")
    if extra:
        lines.append("Stock updated")
    for msg in extra.get("alert_messages") or []:
        lines.append("ALERT: " + msg)
    return lines


def _upload(frame):
    """Returns (response_json, None) or (None, error_text)."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None, "could not encode the frame"

    data = {"source": "live_camera"}

    error = "unknown error"
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            resp = requests.post(
                f"{API_URL}/events/analyze",
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                data=data, timeout=REQUEST_TIMEOUT,
            )
        except requests.Timeout:
            # Do NOT retry: the server may still be processing it, and a retry would duplicate it.
            return None, f"timed out after {REQUEST_TIMEOUT:.0f}s (the server may still have saved it)"
        except requests.ConnectionError:
            error = f"cannot reach {API_URL}"
            if attempt < CONNECT_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
            continue
        if resp.ok:
            return resp.json(), None
        return None, f"server error {resp.status_code}: {resp.text[:160]}"
    return None, error


def _run() -> None:
    while True:
        frame, reason, movement = _jobs.get()
        try:
            result, error = _upload(frame)
            if error:
                info = {"time": time.time(), "reason": reason, "ok": False,
                        "lines": ["UPLOAD FAILED: " + error], "raw": None}
            else:
                extra = None
                if _recorder:
                    try:
                        extra = _recorder(result, movement)
                    except Exception as exc:  # record() is built not to raise, but never fail silently
                        logger.exception("Recorder crashed")
                        extra = {"error": f"{exc.__class__.__name__}: {exc}"}
                info = {"time": time.time(), "reason": reason, "ok": not (extra or {}).get("error"),
                        "lines": summarize(result, extra), "raw": result}
            logger.info("Result [%s]\n    %s", reason, "\n    ".join(info["lines"]))
            if _callback:
                _callback(info)
        except Exception:
            logger.exception("Unexpected error uploading: %s", reason)
        finally:
            _jobs.task_done()


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, daemon=True, name="uploader")
            _worker.start()
