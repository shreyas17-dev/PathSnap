"""Per-observation reference labels derived from the ordered .route file.

Problem: .route gives the arcs the vehicle traversed *in order*, but not which GPS point belongs to
which arc. We need one reference arc per GPS observation for supervised training and for
Top-1 accuracy.

Method (monotonic alignment, a dynamic-programming / DTW-style alignment):
  Find j(0) <= j(1) <= ... <= j(N-1)  (indices into the route sequence) that minimises
      sum_i  dist(gps_i, route_arc[j(i)])
  * Order-preserving: the vehicle cannot go back along the route sequence. This resolves cases the
    naive "nearest route arc" rule gets wrong: both directions of a two-way street, loops and
    revisited roads, crossings of the route with itself.
  * Jumps forward are free, so skipped short arcs (between two 1 Hz samples) are allowed.
  * Exact optimum in O(N*M) time, O(N*M) int16/int32 memory for back-pointers.

The naive rule is also implemented so the two can be compared (evidence for the decision).
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment_distance
from .network import TripGeo


def _route_dist_matrix_row(g: TripGeo, r: np.ndarray, i: int) -> np.ndarray:
    d, _ = point_segment_distance(g.px[i], g.py[i], g.ax1[r], g.ay1[r], g.ax2[r], g.ay2[r])
    return d


def naive_labels(g: TripGeo) -> tuple[np.ndarray, np.ndarray]:
    """Nearest route arc per point, ignoring order. Returns (route_index, distance)."""
    r = g.trip.route
    idx = np.empty(g.n_points, np.int64)
    dist = np.empty(g.n_points)
    for i in range(g.n_points):
        d = _route_dist_matrix_row(g, r, i)
        idx[i] = int(np.argmin(d))
        dist[i] = d[idx[i]]
    return idx, dist


def monotonic_labels(g: TripGeo) -> tuple[np.ndarray, np.ndarray]:
    """Order-preserving alignment of GPS points onto the route. Returns (route_index, distance)."""
    r = g.trip.route
    N, M = g.n_points, len(r)
    ptr_dtype = np.int16 if M < np.iinfo(np.int16).max else np.int32
    back = np.empty((N, M), dtype=ptr_dtype)
    ar = np.arange(M)

    prev = _route_dist_matrix_row(g, r, 0)
    back[0] = ar
    for i in range(1, N):
        # running minimum of the previous row and where it was attained (k <= j)
        cmin = np.minimum.accumulate(prev)
        at = np.maximum.accumulate(np.where(prev == cmin, ar, 0))
        back[i] = at
        prev = cmin + _route_dist_matrix_row(g, r, i)

    j = np.empty(N, np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    a = r[j]
    dist, _ = point_segment_distance(g.px, g.py, g.ax1[a], g.ay1[a], g.ax2[a], g.ay2[a])
    return j, dist


def label_frame(g: TripGeo, unreliable_dist_m: float):
    """Monotonic labels + naive comparison for one trip, as a dict of arrays."""
    r = g.trip.route
    j_mono, d_mono = monotonic_labels(g)
    j_naive, d_naive = naive_labels(g)
    ref_arc = g.canon[r[j_mono]]
    naive_arc = g.canon[r[j_naive]]
    return dict(
        route_pos=j_mono,
        naive_route_pos=j_naive,
        ref_arc=ref_arc,
        ref_dist=d_mono,
        naive_arc=naive_arc,
        naive_dist=d_naive,
        naive_order_violations=int((np.diff(j_naive) < 0).sum()),
        label_unreliable=d_mono > unreliable_dist_m,
    )
