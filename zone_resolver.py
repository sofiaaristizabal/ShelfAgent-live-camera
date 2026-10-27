from shapely.geometry import Point, Polygon


def resolve_zone(bbox: list[float], zones: list[dict]) -> str | None:
    """Return the zone_id whose polygon contains the bbox's center, else None."""
    if not zones:
        return None
    x1, y1, x2, y2 = bbox
    center = Point((x1 + x2) / 2, (y1 + y2) / 2)
    for zone in zones:
        if Polygon(zone["polygon"]).contains(center):
            return zone["zone_id"]
    return None
