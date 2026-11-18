"""Candidate generation: all usable arcs within a radius of each GPS observation.

Exactness argument:
  Each arc is sampled every `step` metres (endpoints included). Any point within R of the segment
  is then within sqrt(R^2 + (step/2)^2) of at least one sample, so a KD-tree ball query with that
  enlarged radius returns a superset of the true candidates. Exact point-to-segment distances are
  then computed and filtered to <= R, so no arc is missed and no wrong arc is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import point_segment_distance
from .network import TripGeo


class ArcIndex:
    def __init__(self, g: TripGeo, step_m: float = 10.0):
        self.g = g
        self.step = step_m
        ids = np.flatnonzero(g.usable)
        n_seg = np.maximum(1, np.ceil(g.arc_len[ids] / step_m).astype(int))
        rep = np.repeat(ids, n_seg + 1)
        # fraction along the arc for every sample (0 .. 1 inclusive)
        offs = np.concatenate([np.linspace(0, 1, k + 1) for k in n_seg]) if len(ids) else np.array([])
        sx = g.ax1[rep] + offs * (g.ax2[rep] - g.ax1[rep])
        sy = g.ay1[rep] + offs * (g.ay2[rep] - g.ay1[rep])
        self.sample_arc = rep
        self.tree = cKDTree(np.c_[sx, sy])

    def query(self, px: np.ndarray, py: np.ndarray, radius_m: float) -> pd.DataFrame:
        """Return long table (point_idx, arc_id, dist_m, t_proj) for all arcs within radius."""
        g = self.g
        r_search = np.sqrt(radius_m ** 2 + (self.step / 2) ** 2)
        hits = self.tree.query_ball_point(np.c_[px, py], r=r_search)
        lens = np.fromiter((len(h) for h in hits), int, len(hits))
        pt = np.repeat(np.arange(len(hits)), lens)
        smp = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits]) if lens.sum() else np.array([], int)
        arc = self.sample_arc[smp]
        n_arcs = len(g.u)
        code = np.unique(pt.astype(np.int64) * n_arcs + arc)
        pt, arc = code // n_arcs, code % n_arcs
        d, tp = point_segment_distance(px[pt], py[pt], g.ax1[arc], g.ay1[arc], g.ax2[arc], g.ay2[arc])
        keep = d <= radius_m
        return pd.DataFrame({"point_idx": pt[keep], "arc_id": arc[keep], "dist_m": d[keep], "t_proj": tp[keep]})


def nearest_arcs(index: "ArcIndex", px: np.ndarray, py: np.ndarray, k: int) -> pd.DataFrame:
    """k nearest distinct arcs for each point (used only when nothing lies within the radius)."""
    g = index.g
    n_samples = index.tree.n
    kk = min(n_samples, 64)
    rows = []
    _, S = index.tree.query(np.c_[px, py], k=kk)
    S = S.reshape(len(px), -1)
    for i, (x, y) in enumerate(zip(px, py)):
        arcs = pd.unique(index.sample_arc[S[i]])
        d, tp = point_segment_distance(x, y, g.ax1[arcs], g.ay1[arcs], g.ax2[arcs], g.ay2[arcs])
        o = np.argsort(d, kind="mergesort")[:k]
        rows.append(pd.DataFrame({"point_idx": i, "arc_id": arcs[o], "dist_m": d[o], "t_proj": tp[o]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["point_idx", "arc_id", "dist_m", "t_proj"])


def generate_candidates(g: TripGeo, radius_m: float, step_m: float, fallback_k: int = 2,
                        index: "ArcIndex | None" = None) -> pd.DataFrame:
    """Radius candidates + nearest-k fallback, ranked by distance. Adds `is_fallback`."""
    index = index or ArcIndex(g, step_m)
    c = index.query(g.px, g.py, radius_m)
    c["is_fallback"] = np.int8(0)
    empty = np.setdiff1d(np.arange(g.n_points), c.point_idx.unique())
    if len(empty) and fallback_k > 0:
        f = nearest_arcs(index, g.px[empty], g.py[empty], fallback_k)
        f["point_idx"] = empty[f.point_idx.to_numpy().astype(int)]
        f["is_fallback"] = np.int8(1)
        c = pd.concat([c, f], ignore_index=True)
    return rank_within_point(c)


def rank_within_point(cands: pd.DataFrame) -> pd.DataFrame:
    cands = cands.sort_values(["point_idx", "dist_m"], kind="mergesort").reset_index(drop=True)
    cands["dist_rank"] = cands.groupby("point_idx").cumcount() + 1
    return cands
