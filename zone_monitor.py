"""
Decides when an item has MOVED, from per-zone counts alone.

It does not rely on YOLO's object tracker. Trackers lose an object's ID when a
hand covers it (exactly what happens when someone moves an item), so
"same ID, different zone" misses real moves. Counting is more robust:

  1. Each tick we get how many of each item are in each zone.
  2. A change only counts after it lasts CONFIRM_TICKS in a row (ignores YOLO
     missing an item for a frame, and boxes flickering across a zone line).
  3. A confirmed DROP in one zone is held for PAIR_WINDOW seconds:
       - same item confirmed UP in another zone in time  -> "relocated" / "returned"
       - same item confirmed UP in the same zone         -> nothing (put back)
       - nothing shows up before the window ends         -> "taken"
     A confirmed rise with no drop waiting is reported as a placement.
"""

from dataclasses import dataclass
from typing import Any

Key = tuple[str, "str | None"]  # (class_name, zone_id)


@dataclass
class Movement:
    class_name: str
    zone_from: "str | None"
    zone_to: "str | None"
    movement_type: str  # taken | returned | relocated
    frame: Any = None   # frame to upload (None = the current one)

    @property
    def reason(self) -> str:
        return (f"item_moved: {self.class_name} {self.movement_type} "
                f"({self.zone_from or '-'} -> {self.zone_to or '-'})")


@dataclass
class _Removal:
    class_name: str
    zone: "str | None"
    time: float
    frame: Any


class ZoneMonitor:
    def __init__(self, home_zones: dict, confirm_ticks: int = 3,
                 pair_window: float = 8.0, warmup_ticks: int = 4):
        self.home_zones = home_zones
        self.confirm_ticks = confirm_ticks
        self.pair_window = pair_window
        self.warmup_ticks = warmup_ticks
        self.stable: dict[Key, int] = {}
        self._candidates: dict[Key, tuple[int, int]] = {}
        self._removals: list[_Removal] = []
        self._ticks = 0

    def update(self, counts: dict, now: float, frame: Any = None) -> list[Movement]:
        self._ticks += 1
        if self._ticks <= self.warmup_ticks:  # settle the baseline, report nothing
            self.stable = dict(counts)
            return []

        confirmed: list[tuple[Key, int]] = []
        keys = sorted(set(self.stable) | set(counts), key=lambda k: (k[0], k[1] or ""))
        for key in keys:
            current, settled = counts.get(key, 0), self.stable.get(key, 0)
            if current == settled:
                self._candidates.pop(key, None)
                continue
            value, ticks = self._candidates.get(key, (None, 0))
            ticks = ticks + 1 if value == current else 1
            if ticks >= self.confirm_ticks:
                confirmed.append((key, current - settled))
                self.stable[key] = current
                self._candidates.pop(key, None)
            else:
                self._candidates[key] = (current, ticks)

        movements: list[Movement] = []
        # Drops first, so a drop and a rise confirmed on the same tick pair up.
        for (cls, zone), delta in confirmed:
            if delta < 0:
                self._removals.extend(_Removal(cls, zone, now, frame) for _ in range(-delta))
        for (cls, zone), delta in confirmed:
            for _ in range(max(delta, 0)):
                movements.extend(self._placed(cls, zone, frame))
        movements.extend(self._expire(now))
        return movements

    def _placed(self, cls: str, zone: "str | None", frame: Any) -> list[Movement]:
        mine = [r for r in self._removals if r.class_name == cls]
        # Prefer a drop from a different zone (a real move) over the same zone (a put-back).
        match = next((r for r in mine if r.zone != zone), None) or (mine[0] if mine else None)
        if match is not None:
            self._removals.remove(match)
            if match.zone == zone:
                return []  # picked up and put back in the same place
        kind = "returned" if self.home_zones.get(cls) == zone else "relocated"
        return [Movement(cls, match.zone if match else None, zone, kind, frame)]

    def _expire(self, now: float) -> list[Movement]:
        out, keep = [], []
        for r in self._removals:
            if now - r.time >= self.pair_window:
                out.append(Movement(r.class_name, r.zone, None, "taken", r.frame))
            else:
                keep.append(r)
        self._removals = keep
        return out
