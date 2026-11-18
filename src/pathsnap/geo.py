"""Geographic utilities.

Why per-trip UTM:
  * lon/lat are angles; 1 degree of longitude is 111 km at the equator but ~63 km at 55°N,
    so Euclidean distance in degrees is meaningless for metres-based features.
  * The 90 trips are spread over several continents, so a single projected CRS would be badly
    distorted for most trips.
  * A trip covers only a few km, so the UTM zone of its centroid keeps scale error far below GPS
    noise (checked numerically in scripts/01_eda.py -> results/eda/projection_check.csv).
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Geod, Transformer

_GEOD = Geod(ellps="WGS84")


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


class LocalProjector:
    """Projects WGS84 lon/lat to the UTM zone of a reference point (metres)."""

    def __init__(self, ref_lon: float, ref_lat: float):
        self.epsg = utm_epsg(ref_lon, ref_lat)
        self._fwd = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(self.epsg), always_xy=True)
        self._inv = Transformer.from_crs(CRS.from_epsg(self.epsg), CRS.from_epsg(4326), always_xy=True)

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x), np.asarray(y))
        return np.asarray(lon), np.asarray(lat)


def projector_for_track(lon: np.ndarray, lat: np.ndarray) -> LocalProjector:
    return LocalProjector(float(np.median(lon)), float(np.median(lat)))


def geodesic_dist(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Exact ellipsoidal distance in metres (reference for validating the projection)."""
    _, _, d = _GEOD.inv(lon1, lat1, lon2, lat2)
    return np.asarray(d)


def bearing_xy(dx, dy) -> np.ndarray:
    """Compass bearing in degrees [0, 360) of a vector in projected (east=x, north=y) coordinates."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a, b) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment_distance(px, py, x1, y1, x2, y2):
    """Vectorised perpendicular distance from points to segments.

    Returns (distance, t) where t in [0, 1] is the clipped projection parameter along the segment.
    """
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - x1) * dx + (py - y1) * dy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = x1 + t * dx, y1 + t * dy
    return np.hypot(px - cx, py - cy), t
