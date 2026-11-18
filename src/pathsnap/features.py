"""Feature engineering for (GPS observation, candidate arc) pairs.

Leakage rule: every feature uses ONLY the GPS track and the road network. The reference route /
labels are never used here, so the identical code runs at inference time on unseen trips.

Feature groups and the intuition behind each one:
  Geometry       dist_m, dist_rank, dist_minus_min, dist_ratio, n_cands, t_proj, at_endpoint
                 -> the true road is usually (not always) among the closest ones.
  Direction      heading_diff (directed, 0-180), heading_diff_undirected (0-90), heading_valid
                 -> separates parallel roads crossing at angles AND the two directions of a two-way road.
  Context        ctx_mean_dist, ctx_max_dist, smooth_dist
                 -> a single noisy point can sit closer to a wrong road; its neighbours usually don't.
  Kinematics     speed, dt_prev, gap_before, gap_after
                 -> noise behaves differently when stopped / after a gap.
  Topology       arc_len, has_reverse, in_deg_u, out_deg_v
                 -> intersections and short connector arcs are where errors concentrate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff, bearing_xy, point_segment_distance
from .network import TripGeo, track_frame

FEATURE_COLUMNS = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands", "t_proj", "at_endpoint",
    "heading_diff", "heading_diff_undirected", "heading_valid",
    "ctx_mean_dist", "ctx_max_dist", "smooth_dist",
    "speed", "dt_prev", "gap_before", "gap_after",
    "arc_len", "has_reverse", "in_deg_u", "out_deg_v",
]

# v2 direction features (scripts/08_feature_ablation.py measures their effect on train CV only).
# Target: the main error source = choosing between the two directions of a two-way road, which the
# v1 heading cannot do when the car is slow/stopped (heading undefined) or when 1 s heading is noisy.
EXTENDED_COLUMNS = [
    "heading_diff_ff",      # heading carried forward/backward from the nearest moving point (same segment)
    "along_cos_w3",         # cos(angle) between motion over points i-3..i+3 and the arc direction (-1..1)
    "along_speed_w3",       # signed speed along the arc over i-3..i+3 (m/s): + = driving in arc direction
    "twin_hd_adv",          # heading_diff_ff(this arc) - heading_diff_ff(its reverse twin); < 0 = this direction fits better
    "hd_rank_near",         # rank by heading_diff_ff among candidates within 5 m of the closest one
]
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + EXTENDED_COLUMNS


def point_kinematics(g: TripGeo, gap_threshold_s: float, min_speed: float, half_window: int) -> pd.DataFrame:
    tf = track_frame(g, gap_threshold_s)
    n = len(tf)
    seg = tf["segment_id"].to_numpy()
    x, y, t = g.px, g.py, g.t

    # Central-difference heading/speed using the neighbours inside the same gap-free segment.
    ip = np.arange(n) - 1
    inx = np.arange(n) + 1
    ip_ok = (ip >= 0) & (seg[np.clip(ip, 0, n - 1)] == seg)
    in_ok = (inx < n) & (seg[np.clip(inx, 0, n - 1)] == seg)
    a = np.where(ip_ok, ip, np.arange(n))
    b = np.where(in_ok, inx, np.arange(n))
    a, b = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
    dx, dy, dtt = x[b] - x[a], y[b] - y[a], t[b] - t[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dtt > 0, np.hypot(dx, dy) / dtt, np.nan)
    heading = bearing_xy(dx, dy)
    valid = np.nan_to_num(speed) >= min_speed
    tf["speed"] = speed
    tf["heading"] = np.where(valid, heading, np.nan)
    tf["heading_valid"] = valid.astype(np.int8)
    # carry the last/next valid heading through stationary periods (never across a gap)
    tf["heading_ff"] = tf.groupby("segment_id")["heading"].transform(lambda s: s.ffill().bfill())

    # Smoothed position: mean of the +-half_window neighbours in the same segment.
    sx, sy, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    for o in range(-half_window, half_window + 1):
        j = np.arange(n) + o
        ok = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        ok &= seg[jj] == seg
        sx += np.where(ok, x[jj], 0)
        sy += np.where(ok, y[jj], 0)
        cnt += ok
    tf["sx"], tf["sy"] = sx / cnt, sy / cnt
    return tf


def candidate_features(g: TripGeo, cands: pd.DataFrame, kin: pd.DataFrame, half_window: int) -> pd.DataFrame:
    """cands: output of ArcIndex.query + rank_within_point. Returns cands with feature columns."""
    c = cands.copy()
    p = c["point_idx"].to_numpy()
    a = c["arc_id"].to_numpy()
    n = g.n_points
    seg = kin["segment_id"].to_numpy()

    grp = c.groupby("point_idx")["dist_m"]
    c["n_cands"] = grp.transform("size").astype(np.int32)
    dmin = grp.transform("min").to_numpy()
    c["dist_minus_min"] = c["dist_m"] - dmin
    c["dist_ratio"] = c["dist_m"] / (dmin + 1.0)
    c["at_endpoint"] = ((c["t_proj"] <= 0) | (c["t_proj"] >= 1)).astype(np.int8)

    hd = kin["heading"].to_numpy()[p]
    diff = angle_diff(hd, g.arc_bearing[a])
    c["heading_diff"] = diff
    c["heading_diff_undirected"] = np.minimum(diff, 180 - diff)
    c["heading_valid"] = kin["heading_valid"].to_numpy()[p]

    # Context: distance of neighbouring observations (same gap-free segment) to the same arc.
    acc = np.zeros(len(c))
    mx = np.zeros(len(c))
    cnt = np.zeros(len(c))
    for o in range(-half_window, half_window + 1):
        j = p + o
        ok = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        ok &= seg[jj] == seg[p]
        d, _ = point_segment_distance(g.px[jj], g.py[jj], g.ax1[a], g.ay1[a], g.ax2[a], g.ay2[a])
        acc += np.where(ok, d, 0)
        mx = np.maximum(mx, np.where(ok, d, 0))
        cnt += ok
    c["ctx_mean_dist"] = acc / cnt
    c["ctx_max_dist"] = mx
    sd, _ = point_segment_distance(kin["sx"].to_numpy()[p], kin["sy"].to_numpy()[p],
                                   g.ax1[a], g.ay1[a], g.ax2[a], g.ay2[a])
    c["smooth_dist"] = sd

    c["speed"] = kin["speed"].to_numpy()[p]
    c["dt_prev"] = kin["dt_prev"].to_numpy()[p]
    c["gap_before"] = kin["gap_before"].to_numpy()[p].astype(np.int8)
    c["gap_after"] = kin["gap_after"].to_numpy()[p].astype(np.int8)
    c["arc_len"] = g.arc_len[a]
    c["has_reverse"] = g.has_reverse[a].astype(np.int8)
    c["in_deg_u"] = g.in_deg[g.u[a]]
    c["out_deg_v"] = g.out_deg[g.v[a]]
    _extended(g, c, kin, p, a, seg)
    return c


def _extended(g: TripGeo, c: pd.DataFrame, kin: pd.DataFrame, p, a, seg, w: int = 3, near_m: float = 5.0):
    n = g.n_points
    hd = angle_diff(kin["heading_ff"].to_numpy()[p], g.arc_bearing[a])
    c["heading_diff_ff"] = hd

    # motion along the arc over a +-w window inside the same gap-free segment
    def nb(off):
        j = np.clip(p + off, 0, n - 1)
        bad = seg[j] != seg[p]
        # step back towards p until inside the segment
        for k in range(abs(off) - 1, -1, -1):
            if not bad.any():
                break
            jj = np.clip(p + int(np.sign(off)) * k, 0, n - 1)
            j = np.where(bad, jj, j)
            bad = seg[j] != seg[p]
        return j
    jp, jn = nb(-w), nb(w)
    dx, dy = g.ax2[a] - g.ax1[a], g.ay2[a] - g.ay1[a]
    L = np.maximum(g.arc_len[a], 1e-6)
    sp = ((g.px[jp] - g.ax1[a]) * dx + (g.py[jp] - g.ay1[a]) * dy) / L
    sn = ((g.px[jn] - g.ax1[a]) * dx + (g.py[jn] - g.ay1[a]) * dy) / L
    move = np.hypot(g.px[jn] - g.px[jp], g.py[jn] - g.py[jp])
    dt = g.t[jn] - g.t[jp]
    with np.errstate(invalid="ignore", divide="ignore"):
        c["along_cos_w3"] = np.where(move >= 1.0, (sn - sp) / move, np.nan)
        c["along_speed_w3"] = np.where(dt > 0, (sn - sp) / dt, np.nan)

    # pairwise comparison with the reverse twin of the same point
    key = pd.DataFrame({"point_idx": p, "u": g.u[a], "v": g.v[a], "hd": hd})
    twin = key.rename(columns={"u": "v", "v": "u", "hd": "hd_twin"})
    m = key.merge(twin.drop_duplicates(["point_idx", "u", "v"]), on=["point_idx", "u", "v"], how="left")
    c["twin_hd_adv"] = (m["hd"] - m["hd_twin"]).to_numpy()

    near = c["dist_minus_min"].to_numpy() <= near_m
    r = pd.Series(np.where(near, hd, np.nan)).groupby(p).rank(method="min")
    c["hd_rank_near"] = np.where(near, r.to_numpy(), 99.0)
