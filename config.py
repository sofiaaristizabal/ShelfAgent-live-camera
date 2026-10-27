import os

from dotenv import load_dotenv

load_dotenv()

# Backend on the server. No API key: security is out of scope for this project.
API_URL = os.getenv("API_URL", "http://<server-ip>:8000").rstrip("/")

# The database the camera app writes to directly (the same Supabase database the backend uses).
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Id of this camera in the `camera` table (created for you on first run).
CAMERA_ID = os.getenv("CAMERA_ID", "cam_01")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
MODEL_PATH = os.getenv("YOLO_WEIGHTS_PATH", "best.pt")

# Every frame is forced to this size. The zone polygons below AND the ones saved
# in the backend's camera.zone_config are in these pixel units, so they must match.
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))

# --- Timing -----------------------------------------------------------------
PROCESS_INTERVAL_SECONDS = float(os.getenv("PROCESS_INTERVAL_SECONDS", "0.5"))  # how often YOLO runs
CONFIRM_TICKS = int(os.getenv("CONFIRM_TICKS", "3"))        # a change must last this many YOLO runs to count
PAIR_WINDOW_SECONDS = float(os.getenv("PAIR_WINDOW_SECONDS", "8"))  # item gone, then seen elsewhere within this = "moved"
WARMUP_TICKS = int(os.getenv("WARMUP_TICKS", "4"))          # first YOLO runs just establish the baseline
SNAPSHOT_INTERVAL_HOURS = float(os.getenv("SNAPSHOT_INTERVAL_HOURS", "24"))
RESULT_DISPLAY_SECONDS = float(os.getenv("RESULT_DISPLAY_SECONDS", "180"))  # how long an upload result stays on screen
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))  # first upload is slow while DeepFace loads


# --- Zones ------------------------------------------------------------------
def _grid_zones(width, height, cols, rows):
    """Split the frame into cols x rows rectangles named zone_1, zone_2, ...
    numbered left to right, top to bottom."""
    zones, n = [], 1
    cell_w, cell_h = width / cols, height / rows
    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * cell_w), int(r * cell_h)
            x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            zones.append({"zone_id": f"zone_{n}",
                          "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
            n += 1
    return zones


# 2x2 grid = four quadrants:   | zone_1 | zone_2 |
#                              | zone_3 | zone_4 |
ZONES = _grid_zones(FRAME_WIDTH, FRAME_HEIGHT, cols=2, rows=2)

# --- Where each item BELONGS ---------------------------------------------------
# Keys must match your YOLO class names exactly. Find them with:
#   python -c "from ultralytics import YOLO; print(YOLO('best.pt').names)"
# Items not listed here have no designated zone (movements are still logged).
ITEM_HOME_ZONES = {
    "Gafas de seguridad": "zone_1",
    "Llave": "zone_2",
    "Martillo": "zone_3",
    "Sierra": "zone_4",
    "Taladro": "zone_1",
    "Tuerca": "zone_2"
    
}
