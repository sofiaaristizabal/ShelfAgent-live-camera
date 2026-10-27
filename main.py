"""
ShelfAgent live camera.

  - Shows the camera with the zone lines drawn on it, live item boxes and counts.
  - YOLO runs in a background thread, so the video stays smooth.
  - A CAPTURE & UPLOAD button (or the SPACE key) sends the current frame to the
    backend; the result stays in the side panel for RESULT_DISPLAY_SECONDS and is
    also printed in the terminal.
  - When an item moves between zones (or is taken out of view), the frame is
    uploaded automatically. Every upload uses update_stock=true.
  - A frame is also uploaded every SNAPSHOT_INTERVAL_HOURS.

The backend only analyzes each image (YOLO + DeepFace) and stores the snapshot.
After every upload this app writes the movement, the stock counts and the
stockout alerts to the database itself (db_writer.py).
"""

import logging
import threading
import time
from collections import Counter

import cv2
from ultralytics import YOLO

import config as cfg
import overlay
from db_writer import DbWriter
from event_sender import pending, send_event, set_recorder, set_result_callback
from zone_monitor import ZoneMonitor
from zone_resolver import resolve_zone

WINDOW = "ShelfAgent Live Camera"
logger = logging.getLogger("live_camera")


class SharedState:
    """What the camera thread, the YOLO thread and the uploader thread share."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.detections: list[dict] = []
        self.result = None
        self.capture_requested = False
        self.last_capture = 0.0
        self.stop = False

    def set_result(self, info):
        self.result = info


def detect(model, frame) -> list[dict]:
    results = model(frame, verbose=False)[0]
    dets = []
    for box in (results.boxes if results.boxes is not None else []):
        bbox = [float(v) for v in box.xyxy[0].tolist()]
        dets.append({
            "class_name": model.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "bbox": bbox,
            "zone": resolve_zone(bbox, cfg.ZONES),
        })
    return dets


def inference_loop(model, state: SharedState, monitor: ZoneMonitor) -> None:
    last_run = 0.0
    last_snapshot = time.time()
    while not state.stop:
        time.sleep(0.02)
        now = time.time()
        if now - last_run < cfg.PROCESS_INTERVAL_SECONDS:
            continue
        with state.lock:
            frame = None if state.frame is None else state.frame.copy()
        if frame is None:
            continue
        last_run = now
        try:
            dets = detect(model, frame)
            counts = Counter((d["class_name"], d["zone"]) for d in dets)
            movements = monitor.update(dict(counts), now, frame)
            with state.lock:
                state.detections = dets
            for m in movements:
                logger.info("MOVEMENT: %s", m.reason)
                send_event(m.frame if m.frame is not None else frame, m.reason, m)
            if now - last_snapshot >= cfg.SNAPSHOT_INTERVAL_HOURS * 3600:
                last_snapshot = now
                send_event(frame, "scheduled_snapshot")
        except Exception:
            logger.exception("Error in the detection loop - continuing")


def open_camera():
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    # MJPG + a 1-frame buffer fixes the lag/low FPS many USB webcams have on Linux.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def on_mouse(event, x, y, flags, state: SharedState):
    if event == cv2.EVENT_LBUTTONDOWN and overlay.in_button(x, y, cfg.FRAME_WIDTH):
        state.capture_requested = True


def request_capture(frame, state: SharedState) -> None:
    now = time.time()
    if now - state.last_capture < 1.0:  # ignore double clicks
        return
    state.last_capture = now
    send_event(frame.copy(), "manual_capture")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = YOLO(cfg.MODEL_PATH)
    state = SharedState()
    monitor = ZoneMonitor(cfg.ITEM_HOME_ZONES, cfg.CONFIRM_TICKS, cfg.PAIR_WINDOW_SECONDS, cfg.WARMUP_TICKS)
    set_result_callback(state.set_result)

    class_names = list(model.names.values())
    for name, zone in cfg.ITEM_HOME_ZONES.items():  # catch typos early
        if name not in class_names:
            logger.warning("ITEM_HOME_ZONES: '%s' is not a YOLO class. Classes are: %s", name, class_names)
        if zone not in {z["zone_id"] for z in cfg.ZONES}:
            logger.warning("ITEM_HOME_ZONES: '%s' points to unknown zone '%s'", name, zone)
    writer = DbWriter(cfg.DATABASE_URL, cfg.ZONES, class_names)
    if writer.ensure_ready():
        for line in writer.report:
            logger.info("Database: %s", line)
    else:
        logger.error("Database not reachable (%s). The camera and uploads still run; "
                     "it will keep retrying on every upload.", writer.error)
    set_recorder(writer.record)

    threading.Thread(target=inference_loop, args=(model, state, monitor), daemon=True).start()

    cap = open_camera()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse, state)
    warned_size = False
    logger.info("Running. Click CAPTURE & UPLOAD or press SPACE to upload a frame. Esc quits.")

    while True:
        ok, frame = cap.read()
        if not ok:  # runs unattended: never crash on a dropped camera
            logger.warning("Camera read failed - reopening in 2s")
            cap.release()
            time.sleep(2)
            cap = open_camera()
            continue

        if frame.shape[1] != cfg.FRAME_WIDTH or frame.shape[0] != cfg.FRAME_HEIGHT:
            if not warned_size:
                logger.warning("Camera gives %dx%d; resizing to %dx%d so the zones line up.",
                               frame.shape[1], frame.shape[0], cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)
                warned_size = True
            frame = cv2.resize(frame, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))

        with state.lock:
            state.frame = frame
            dets = list(state.detections)

        canvas = overlay.render(frame, cfg.ZONES, cfg.ITEM_HOME_ZONES, dets,
                                pending(), state.result, time.time(), cfg.RESULT_DISPLAY_SECONDS, writer.status())
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
        if key == ord(" ") or key == ord("c") or state.capture_requested:
            state.capture_requested = False
            request_capture(frame, state)

    state.stop = True
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
