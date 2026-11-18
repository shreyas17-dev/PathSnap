"""Projected road network + GPS track for one trip.

Network-cleaning decisions (evidence in results/eda/data_quality.csv):
  * Self-loop arcs (from_node == to_node) have zero length and no direction -> never used as candidates.
  * Duplicate arcs (same from_node, to_node) are geometrically identical, so no model can tell them
    apart. Each duplicate group is mapped to one canonical arc id (the smallest id) for candidates,
    labels and scoring. The mapping is kept so original ids can still be reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geo import LocalProjector, bearing_xy, projector_for_track
from .io import Trip


@dataclass
class TripGeo:
    trip: Trip
    proj: LocalProjector
    px: np.ndarray            # GPS x (m)
    py: np.ndarray            # GPS y (m)
    t: np.ndarray             # GPS time (s)
    nx: np.ndarray            # node x
    ny: np.ndarray            # node y
    u: np.ndarray             # arc from-node
    v: np.ndarray             # arc to-node
    ax1: np.ndarray = field(init=False)
    ay1: np.ndarray = field(init=False)
    ax2: np.ndarray = field(init=False)
    ay2: np.ndarray = field(init=False)
    arc_len: np.ndarray = field(init=False)
    arc_bearing: np.ndarray = field(init=False)
    canon: np.ndarray = field(init=False)       # arc_id -> canonical arc_id
    usable: np.ndarray = field(init=False)      # bool: canonical & non-zero length
    has_reverse: np.ndarray = field(init=False)  # bool: arc (v,u) exists
    out_deg: np.ndarray = field(init=False)     # out-degree of each node (unique arcs)
    in_deg: np.ndarray = field(init=False)

    def __post_init__(self):
        self.ax1, self.ay1 = self.nx[self.u], self.ny[self.u]
        self.ax2, self.ay2 = self.nx[self.v], self.ny[self.v]
        dx, dy = self.ax2 - self.ax1, self.ay2 - self.ay1
        self.arc_len = np.hypot(dx, dy)
        self.arc_bearing = bearing_xy(dx, dy)

        n_nodes = len(self.nx)
        key = self.u * n_nodes + self.v
        order = np.lexsort((np.arange(len(key)), key))  # by key, then by arc id
        first = np.empty_like(order)
        ks = key[order]
        is_new = np.r_[True, ks[1:] != ks[:-1]]
        grp_first = order[np.maximum.accumulate(np.where(is_new, np.arange(len(order)), 0))]
        first[order] = grp_first
        self.canon = first
        self.usable = (self.canon == np.arange(len(self.u))) & (self.u != self.v) & (self.arc_len > 0)

        rev_key = self.v * n_nodes + self.u
        self.has_reverse = np.isin(rev_key, key)
        cu, cv = self.u[self.usable], self.v[self.usable]
        self.out_deg = np.bincount(cu, minlength=n_nodes)
        self.in_deg = np.bincount(cv, minlength=n_nodes)

    @property
    def n_points(self):
        return len(self.px)

    @property
    def route_canon(self):
        r = self.trip.route
        return None if r is None else self.canon[r]


def build_trip_geo(trip: Trip) -> TripGeo:
    tr, no, ar = trip.track, trip.nodes, trip.arcs
    proj = projector_for_track(tr.lon.to_numpy(), tr.lat.to_numpy())
    px, py = proj.to_xy(tr.lon.to_numpy(), tr.lat.to_numpy())
    nx, ny = proj.to_xy(no.lon.to_numpy(), no.lat.to_numpy())
    return TripGeo(trip, proj, px, py, tr.t.to_numpy(), nx, ny, ar.u.to_numpy(), ar.v.to_numpy())


def network_stats(g: TripGeo) -> dict:
    """Connectivity statistics used in EDA."""
    n = len(g.nx)
    m = len(g.u)
    used = np.zeros(n, bool)
    used[g.u] = True
    used[g.v] = True
    A = coo_matrix((np.ones(m), (g.u, g.v)), shape=(n, n)).tocsr()
    n_weak, lab_w = connected_components(A, directed=True, connection="weak")
    n_strong, lab_s = connected_components(A, directed=True, connection="strong")
    # components restricted to nodes that appear in at least one arc
    weak_sizes = np.bincount(lab_w[used])
    weak_sizes = weak_sizes[weak_sizes > 0]
    strong_sizes = np.bincount(lab_s[used])
    strong_sizes = strong_sizes[strong_sizes > 0]
    return dict(
        n_nodes=n,
        n_arcs=m,
        isolated_nodes=int((~used).sum()),
        self_loops=int((g.u == g.v).sum()),
        duplicate_arcs=int((g.canon != np.arange(m)).sum()),
        two_way_frac=float(g.has_reverse.mean()),
        weak_components=int(len(weak_sizes)),
        largest_weak_frac=float(weak_sizes.max() / used.sum()),
        strong_components=int(len(strong_sizes)),
        largest_strong_frac=float(strong_sizes.max() / used.sum()),
        arc_len_median_m=float(np.median(g.arc_len)),
        arc_len_p99_m=float(np.percentile(g.arc_len, 99)),
        arc_len_max_m=float(g.arc_len.max()),
        total_road_km=float(g.arc_len[g.usable].sum() / 1000),
    )


def route_stats(g: TripGeo) -> dict:
    r = g.trip.route
    if r is None:
        return {}
    conn = g.v[r[:-1]] == g.u[r[1:]]
    return dict(
        route_len=len(r),
        route_breaks=int((~conn).sum()),
        route_connected_frac=float(conn.mean()) if len(conn) else 1.0,
        route_repeated_arcs=int(len(r) - len(np.unique(r))),
        route_uses_duplicate_arc=int((g.canon[r] != r).sum()),
        route_uses_self_loop=int((g.u[r] == g.v[r]).sum()),
        route_km=float(g.arc_len[r].sum() / 1000),
    )


def track_frame(g: TripGeo, gap_threshold_s: float) -> pd.DataFrame:
    """Per-point kinematics computed WITHOUT crossing GPS gaps."""
    dt = np.diff(g.t)
    step = np.hypot(np.diff(g.px), np.diff(g.py))
    speed = np.where(dt > 0, step / np.where(dt > 0, dt, 1), np.nan)
    gap = dt > gap_threshold_s
    df = pd.DataFrame({"x": g.px, "y": g.py, "t": g.t})
    df["dt_prev"] = np.r_[np.nan, dt]
    df["dt_next"] = np.r_[dt, np.nan]
    df["step_prev_m"] = np.r_[np.nan, step]
    df["speed_prev"] = np.r_[np.nan, np.where(gap, np.nan, speed)]
    df["gap_before"] = np.r_[False, gap]
    df["gap_after"] = np.r_[gap, False]
    df["segment_id"] = np.cumsum(df["gap_before"].to_numpy())  # continuous pieces between gaps
    return df
