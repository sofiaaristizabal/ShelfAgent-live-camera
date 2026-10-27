"""Everything drawn on screen. Pure functions: no camera, no network."""

import textwrap
import unicodedata
from datetime import datetime

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
SIDEBAR_WIDTH = 340
BUTTON = (10, 10, SIDEBAR_WIDTH - 10, 60)  # x1, y1, x2, y2 inside the sidebar
WRAP = 40  # characters per line in the side panel

GREEN, RED, WHITE = (0, 200, 0), (0, 0, 255), (255, 255, 255)
GRAY, YELLOW, BLUE = (170, 170, 170), (0, 220, 255), (200, 120, 30)


def ascii_text(text: str) -> str:
    """OpenCV's fonts can't draw accents or n-tilde (they show as '???')."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()


def put_text(img, text, org, color=WHITE, scale=0.5, thickness=1):
    text = ascii_text(text)
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)  # outline
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def in_button(x: int, y: int, frame_width: int) -> bool:
    """Is a click at window coordinates (x, y) on the CAPTURE button?"""
    x1, y1, x2, y2 = BUTTON
    return frame_width + x1 <= x <= frame_width + x2 and y1 <= y <= y2


def is_misplaced(class_name, zone, home_zones) -> bool:
    home = home_zones.get(class_name)
    return home is not None and zone != home


def draw_zones(img, zones):
    for z in zones:
        pts = np.array(z["polygon"], np.int32)
        cv2.polylines(img, [pts], True, WHITE, 1, cv2.LINE_AA)
        x, y = pts.min(axis=0)
        put_text(img, z["zone_id"], (int(x) + 6, int(y) + 18), WHITE, 0.5)


def draw_detections(img, dets, home_zones):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        bad = is_misplaced(d["class_name"], d["zone"], home_zones)
        color = RED if bad else GREEN
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        if bad:
            label += f' (belongs {home_zones[d["class_name"]]})'
        (tw, _), _ = cv2.getTextSize(ascii_text(label), FONT, 0.45, 1)
        lx = max(0, min(x1, img.shape[1] - tw - 4))  # keep the label inside the video
        put_text(img, label, (lx, max(y1 - 6, 12)), color, 0.45)


def _zone_label(zone) -> str:
    return "-" if zone is None else str(zone).replace("zone_", "z")


def build_sidebar(height, dets, home_zones, pending, result, now, result_seconds, db_status=("", True)):
    bar = np.full((height, SIDEBAR_WIDTH, 3), 32, np.uint8)

    x1, y1, x2, y2 = BUTTON
    cv2.rectangle(bar, (x1, y1), (x2, y2), BLUE, -1)
    put_text(bar, "CAPTURE & UPLOAD", (x1 + 40, y1 + 33), WHITE, 0.65, 2)
    put_text(bar, "or press SPACE.  Esc quits.", (x1, y2 + 18), GRAY, 0.4)

    if pending:
        put_text(bar, f"Uploading... ({pending} in queue)", (x1, 100), YELLOW, 0.5)
    else:
        put_text(bar, "Ready", (x1, 100), GREEN, 0.5)

    if db_status[0]:
        put_text(bar, db_status[0], (x1, 114), GREEN if db_status[1] else RED, 0.4)

    # --- live counts per item ---
    put_text(bar, "LIVE COUNTS", (x1, 138), GRAY, 0.45)
    by_class: dict[str, dict] = {}
    for d in dets:
        zones = by_class.setdefault(d["class_name"], {})
        zones[d["zone"]] = zones.get(d["zone"], 0) + 1
    y, names = 156, sorted(by_class)
    if not names:
        put_text(bar, "nothing detected", (x1, y), GRAY, 0.45)
    for i, name in enumerate(names):
        if i == 8:
            put_text(bar, f"+{len(names) - 8} more", (x1, y), GRAY, 0.45)
            break
        zones = by_class[name]
        detail = " ".join(f"{_zone_label(z)}:{n}" for z, n in sorted(zones.items(), key=lambda kv: str(kv[0])))
        bad = any(is_misplaced(name, z, home_zones) for z in zones)
        put_text(bar, f"{name}: {sum(zones.values())}  [{detail}]", (x1, y), RED if bad else WHITE, 0.45)
        y += 16

    # --- last upload result ---
    top = 292
    put_text(bar, "LAST UPLOAD", (x1, top), GRAY, 0.45)
    if result and now - result["time"] <= result_seconds:
        stamp = datetime.fromtimestamp(result["time"]).strftime("%H:%M:%S")
        color = GREEN if result["ok"] else RED
        put_text(bar, f"{stamp}  {'OK' if result['ok'] else 'FAILED'}", (x1 + 110, top), color, 0.45)
        lines = textwrap.wrap(result["reason"], WRAP) or [""]
        rows = [(ln, GRAY) for ln in lines[:2]]
        for text in result["lines"]:
            tcolor = RED if (text.startswith(("ALERT", "DATABASE")) or not result["ok"]) else WHITE
            rows += [(ln, tcolor) for ln in textwrap.wrap(text, WRAP)]
        y, limit = top + 20, height - 8
        for text, tcolor in rows:
            if y > limit:
                put_text(bar, "...", (x1, y - 16), GRAY, 0.45)
                break
            put_text(bar, text, (x1, y), tcolor, 0.45)
            y += 16
    else:
        put_text(bar, "no recent uploads", (x1, top + 20), GRAY, 0.45)
    return bar


def render(frame, zones, home_zones, dets, pending, result, now, result_seconds, db_status=("", True)):
    view = frame.copy()
    draw_zones(view, zones)
    draw_detections(view, dets, home_zones)
    bar = build_sidebar(frame.shape[0], dets, home_zones, pending, result, now, result_seconds, db_status)
    return np.hstack([view, bar])
