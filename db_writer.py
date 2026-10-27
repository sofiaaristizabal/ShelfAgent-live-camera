"""
Everything the camera app reads from / writes to the database.

The backend (/events/analyze) is left exactly as it is. It runs YOLO + DeepFace
on the uploaded frame and saves the snapshot, its detections and its faces.
After each upload this module finishes the job:

  - fills in item and zone on each detection row, and the camera on the snapshot
  - saves the item movement, credited to the recognized employee (or to the
    "unrecognized" placeholder employee when no registered face was found)
  - updates the stock counts from what the frame shows
  - opens a stockout alert for any item that is missing, and closes it when the
    item is seen again

Your tables are never altered. At startup it looks at them and adapts:
  * item_stock with a zone_id column  -> one row per item AND zone, written only when a count changes
  * item_stock without zone_id        -> one row per item, updated in place
  * item has a class_name column      -> items are matched on it, otherwise on item.name

Run `python db_writer.py` to check the connection and see what it found.
"""

import json
import logging
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import MetaData, and_, create_engine, event, insert, select, update
from sqlalchemy.exc import OperationalError

import config as cfg
from zone_resolver import resolve_zone

logger = logging.getLogger("db_writer")

UNRECOGNIZED = "unrecognized"  # employee id used when a movement has no recognized face
REQUIRED_TABLES = ["item", "zone", "employee", "camera", "snapshot", "detection",
                   "item_stock", "employee_movement_event", "stock_out_event", "alert"]


def _now() -> datetime:
    """UTC wall-clock time, the way the backend stores it (timestamp without time zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _id() -> str:
    return str(uuid.uuid4())


def make_engine(url: str):
    kwargs = {"pool_pre_ping": True, "pool_recycle": 240}  # the app runs 24/7; connections go stale
    if url.startswith("postgresql"):
        kwargs["connect_args"] = {"connect_timeout": 10}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):  # behave like Postgres: enforce foreign keys
        @event.listens_for(engine, "connect")
        def _foreign_keys_on(conn, _record):
            conn.execute("PRAGMA foreign_keys=ON")
    return engine


class DbWriter:
    def __init__(self, url: str, zones: list, class_names=()):
        self.url = url
        self.zones = zones
        self.class_names = list(class_names)  # YOLO class names; each becomes an item if missing
        self.engine = None
        self.t: dict = {}
        self.ready = False
        self.error = ""
        self.stock_layout = ""   # "per_item" | "per_zone"
        self.has_class_col = False
        self.report: list[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------ setup
    def ensure_ready(self) -> bool:
        """Connect and seed if we haven't yet. Safe to call before every write:
        if the database was down at startup, it recovers on its own."""
        if self.ready:
            return True
        with self._lock:
            if self.ready:
                return True
            try:
                self.report = self._bootstrap()
            except Exception as exc:
                self.error = f"{exc.__class__.__name__}: {str(exc).strip().splitlines()[0][:160]}"
                logger.error("Database not ready: %s", self.error)
                return False
        return True

    def status(self) -> tuple[str, bool]:
        """(text for the side panel, is_ok)"""
        if self.ready:
            return "DB connected", True
        return "DB OFFLINE: " + (self.error or "not connected yet")[:34], False

    def _bootstrap(self) -> list[str]:
        if not self.url:
            raise RuntimeError("DATABASE_URL is not set")
        if self.engine is None:
            self.engine = make_engine(self.url)
        md = MetaData()
        md.reflect(self.engine, only=REQUIRED_TABLES)  # raises if one of the tables is missing
        self.t = md.tables
        self.has_class_col = "class_name" in self.t["item"].c
        self.stock_layout = "per_zone" if "zone_id" in self.t["item_stock"].c else "per_item"

        report = [f"item_stock layout: {self.stock_layout.replace('_', ' ')}",
                  "items matched on " + ("item.class_name" if self.has_class_col else "item.name")]
        with self.engine.begin() as conn:
            report += self._seed(conn)
        self.ready = True
        self.error = ""
        return report

    def _only(self, table: str, **values) -> dict:
        """Keep only the columns this table actually has."""
        cols = self.t[table].c
        return {k: v for k, v in values.items() if k in cols}

    def _seed(self, conn) -> list[str]:
        """Create what the app needs if it's missing. Never touches existing rows."""
        t, report = self.t, []

        have = {r[0] for r in conn.execute(select(t["zone"].c.id))}
        for z in self.zones:
            if z["zone_id"] not in have:
                conn.execute(insert(t["zone"]).values(**self._only(
                    "zone", id=z["zone_id"], name=z["zone_id"].replace("_", " ").title())))
                report.append(f"created zone {z['zone_id']}")

        emp = t["employee"]
        if conn.execute(select(emp.c.id).where(emp.c.id == UNRECOGNIZED)).first() is None:
            conn.execute(insert(emp).values(**self._only("employee", id=UNRECOGNIZED, name="Unrecognized person")))
            report.append("created the 'unrecognized' employee")

        cam = t["camera"]
        row = conn.execute(select(cam.c.zone_config).where(cam.c.id == cfg.CAMERA_ID)).first()
        if row is None:
            conn.execute(insert(cam).values(**self._only(
                "camera", id=cfg.CAMERA_ID, name="Main camera", zone_config=self.zones)))
            report.append(f"created camera {cfg.CAMERA_ID} with the zones from config.py")
        elif row[0] != self.zones:
            report.append(f"WARNING: camera {cfg.CAMERA_ID} in the database has different zones than config.py "
                          "(only matters for dashboard uploads that pass a camera_id)")

        by_key, _ = self._items(conn)
        for cls in self.class_names:
            if cls not in by_key:
                conn.execute(insert(t["item"]).values(**self._only(
                    "item", id=_id(), name=cls, class_name=cls, created_at=_now())))
                report.append(f"created item '{cls}'")
        return report

    def _items(self, conn):
        """-> ({YOLO class name: item_id}, {item_id: display name})"""
        item = self.t["item"]
        cols = [item.c.id, item.c.name] + ([item.c.class_name] if self.has_class_col else [])
        by_key, names = {}, {}
        for row in conn.execute(select(*cols)):
            names[row[0]] = row[1]
            if self.has_class_col and row[2]:
                by_key[row[2]] = row[0]
            by_key.setdefault(row[1], row[0])
        return by_key, names

    # ----------------------------------------------------------- writing
    def record(self, result: dict, movement=None) -> dict:
        """Called after every successful upload. Never raises.
        Returns {"movement": {...}|None, "alert_messages": [...]} or {"error": "..."}."""
        if not self.ensure_ready():
            return {"error": f"database not available ({self.error})"}
        try:
            return self._record(result, movement)
        except OperationalError:
            # Stale connection or a brief outage. Everything runs in one transaction,
            # so nothing was half-written; reconnect and try once more.
            try:
                self.engine.dispose()
                return self._record(result, movement)
            except Exception as exc:
                logger.exception("Database write failed")
                return {"error": f"{exc.__class__.__name__}: {str(exc).strip().splitlines()[0][:140]}"}
        except Exception as exc:
            logger.exception("Database write failed")
            return {"error": f"{exc.__class__.__name__}: {str(exc).strip().splitlines()[0][:140]}"}

    def _record(self, result: dict, movement) -> dict:
        t, now = self.t, _now()
        snapshot_id = result["snapshot_id"]
        out = {"movement": None, "alert_messages": []}

        with self.engine.begin() as conn:
            by_key, names = self._items(conn)
            employees = {r[0] for r in conn.execute(select(t["employee"].c.id))}
            zone_ids = {r[0] for r in conn.execute(select(t["zone"].c.id))}

            # 1) the backend leaves item/zone empty on detections: complete them
            totals: Counter = Counter()    # item_id -> how many are in the frame
            by_zone: Counter = Counter()   # (item_id, zone_id) -> how many
            det = t["detection"]
            rows = conn.execute(select(det.c.id, det.c.class_name, det.c.bbox)
                                .where(det.c.snapshot_id == snapshot_id)).all()
            for row in rows:
                bbox = json.loads(row.bbox) if isinstance(row.bbox, str) else row.bbox
                zone = resolve_zone(bbox, self.zones) if bbox else None
                zone = zone if zone in zone_ids else None
                item_id = by_key.get(row.class_name)
                conn.execute(update(det).where(det.c.id == row.id).values(item_id=item_id, zone_id=zone))
                if item_id:
                    totals[item_id] += 1
                    if zone:
                        by_zone[(item_id, zone)] += 1
            snap = t["snapshot"]
            conn.execute(update(snap).where(and_(snap.c.id == snapshot_id, snap.c.camera_id.is_(None)))
                         .values(camera_id=cfg.CAMERA_ID))

            # 2) item movement
            if movement is not None:
                item_id = by_key.get(movement.class_name)
                if item_id is None:
                    logger.warning("Movement for '%s' skipped: no such item", movement.class_name)
                else:
                    best = self._best_face(result.get("faces") or [], employees)
                    employee_id = best["employee_id"] if best else (UNRECOGNIZED if UNRECOGNIZED in employees else None)
                    zone_from = movement.zone_from if movement.zone_from in zone_ids else None
                    zone_to = movement.zone_to if movement.zone_to in zone_ids else None
                    movement_id = _id()
                    conn.execute(insert(t["employee_movement_event"]).values(**self._only(
                        "employee_movement_event", id=movement_id, employee_id=employee_id, item_id=item_id,
                        zone_from_id=zone_from, zone_to_id=zone_to, movement_type=movement.movement_type,
                        snapshot_id=snapshot_id, date_time=now,
                        confidence=(1 - best["distance"]) if best else None)))
                    out["movement"] = {"id": movement_id, "item": names[item_id],
                                       "movement_type": movement.movement_type,
                                       "zone_from": zone_from, "zone_to": zone_to, "employee_id": employee_id}

            # 3) stock: this frame is a full view of the warehouse
            if self.stock_layout == "per_zone":
                self._stock_per_zone(conn, names, by_zone, now)
            else:
                self._stock_per_item(conn, names, totals, now)

            # 4) stockout alerts: one open alert per item
            out["alert_messages"] = self._alerts(conn, names, totals, now)
        return out

    @staticmethod
    def _best_face(faces: list, employees: set):
        best = None
        for f in faces:
            if f.get("recognized") and f.get("employee_id") in employees:
                if best is None or f["distance"] < best["distance"]:
                    best = f
        return best

    def _stock_per_item(self, conn, names, totals, now):
        s = self.t["item_stock"]
        for item_id in names:
            amount = totals.get(item_id, 0)  # not in the frame -> 0
            res = conn.execute(update(s).where(s.c.item_id == item_id)
                               .values(**self._only("item_stock", amount=amount, taken_at=now, source="detected")))
            if res.rowcount == 0:
                conn.execute(insert(s).values(**self._only(
                    "item_stock", id=_id(), item_id=item_id, amount=amount, taken_at=now, source="detected")))

    def _stock_per_zone(self, conn, names, by_zone, now):
        s = self.t["item_stock"]
        latest = {}
        for r in conn.execute(select(s.c.item_id, s.c.zone_id, s.c.amount).order_by(s.c.taken_at)):
            latest[(r.item_id, r.zone_id)] = r.amount
        for item_id in names:
            for z in self.zones:
                zone_id = z["zone_id"]
                amount, prev = by_zone.get((item_id, zone_id), 0), latest.get((item_id, zone_id))
                if prev == amount or (prev is None and amount == 0):
                    continue  # unchanged: don't add a row
                conn.execute(insert(s).values(**self._only(
                    "item_stock", id=_id(), item_id=item_id, zone_id=zone_id,
                    amount=amount, taken_at=now, source="detected")))

    def _alerts(self, conn, names, totals, now) -> list[str]:
        a, so = self.t["alert"], self.t["stock_out_event"]
        messages = []
        for item_id, name in names.items():
            open_alert = conn.execute(
                select(a.c.id).select_from(a.join(so, so.c.id == a.c.stock_out_event_id))
                .where(and_(so.c.item_id == item_id, a.c.alert_type == "stockout", a.c.is_resolved.is_(False)))
            ).first()
            if totals.get(item_id, 0) == 0 and open_alert is None:
                event_id = _id()
                conn.execute(insert(so).values(**self._only(
                    "stock_out_event", id=event_id, item_id=item_id, zone_id=None, date_time=now)))
                text = f"{name} is out of stock"
                conn.execute(insert(a).values(**self._only(
                    "alert", id=_id(), stock_out_event_id=event_id, description=text,
                    alert_type="stockout", is_resolved=False, created_at=now)))
                messages.append(text)
            elif totals.get(item_id, 0) > 0 and open_alert is not None:
                conn.execute(update(a).where(a.c.id == open_alert[0]).values(is_resolved=True, resolved_at=now))
        return messages

    # ------------------------------------------------------------- check
    def describe(self) -> list[str]:
        """Readable summary of what's in the database (python db_writer.py)."""
        lines = list(self.report)
        with self.engine.begin() as conn:
            by_key, names = self._items(conn)
            s, a = self.t["item_stock"], self.t["alert"]
            lines.append(f"items ({len(names)}): " + ", ".join(sorted(names.values())))
            zones = [r[0] for r in conn.execute(select(self.t["zone"].c.id))]
            lines.append(f"zones ({len(zones)}): " + ", ".join(sorted(zones)))
            lines.append(f"stock rows: {len(conn.execute(select(s.c.id)).all())}")
            open_n = len(conn.execute(select(a.c.id).where(a.c.is_resolved.is_(False))).all())
            lines.append(f"open alerts: {open_n}")
        return lines


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    writer = DbWriter(cfg.DATABASE_URL, cfg.ZONES)  # no class names: a check should not create items
    if writer.ensure_ready():
        print("Connected to the database.")
        for line in writer.describe():
            print("  " + line)
    else:
        print("COULD NOT CONNECT:", writer.error)
