"""HMM + Viterbi route reconstruction (docs/DECISIONS.md D14).

Hidden states   : at GPS observation t, the states are that observation's candidate arcs (same
                  candidate sets as the classifier, so every state keeps its arc id).
Emission        : log P(arc is correct | features) from the classifier, scaled by weight w:
                      e_t(b) = w * log(max(p_t(b), 1e-6))
Transition      : Newson & Krumm (2009) idea - a correct move between two consecutive observations
                  has a network driving distance close to the straight-line distance between them:
                      log T(a -> b) = -| d_route(a, b) - d_gps | / beta
                  d_route = distance along the road network from the projection of point t-1 on arc a
                  to the projection of point t on arc b (shortest path, directed graph).
                  * same arc: |t_b - t_a| * L   (small backward jitter allowed when slow/stopped)
                  * different arcs: rest of a + shortest path(v_a -> u_b) + start of b
                  * not reachable within `max_route_m`: log T = -UNREACHABLE (very unlikely, never
                    impossible, so a broken network / route break can't make the decoder fail)
GPS gaps        : across a gap (> gap_threshold_s) the chain is cut: no transition term, every state
                  starts from the best previous score (a new segment).
Missing cands   : the candidate step guarantees >= 1 candidate per point (fallback_k nearest arcs).
Decoding        : Viterbi  delta_t(b) = e_t(b) + max_a [ delta_{t-1}(a) + log T(a, b) ], back-pointers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .network import TripGeo

UNREACHABLE = 50.0


class NetworkDistance:
    """Cached bounded Dijkstra from nodes over the directed, usable-arc graph (lengths in metres)."""

    def __init__(self, g: TripGeo, max_route_m: float = 800.0):
        n = len(g.nx)
        ids = np.flatnonzero(g.usable)
        self.A = csr_matrix((g.arc_len[ids], (g.u[ids], g.v[ids])), shape=(n, n))
        self.limit = max_route_m
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def prefetch(self, nodes):
        todo = [int(x) for x in np.unique(nodes) if int(x) not in self.cache]
        for i in range(0, len(todo), 16):
            chunk = todo[i:i + 16]
            D = dijkstra(self.A, directed=True, indices=chunk, limit=self.limit)
            for src, row in zip(chunk, np.atleast_2d(D)):
                idx = np.flatnonzero(np.isfinite(row))
                self.cache[src] = (idx, row[idx])

    def lookup(self, src: int, targets: np.ndarray) -> np.ndarray:
        idx, dist = self.cache[src]
        pos = np.searchsorted(idx, targets)
        pos = np.clip(pos, 0, max(len(idx) - 1, 0))
        out = np.full(len(targets), np.inf)
        if len(idx):
            hit = idx[pos] == targets
            out[hit] = dist[pos[hit]]
        return out


def prepare_trip(g: TripGeo, cands: pd.DataFrame, gap_threshold_s: float, max_route_m: float = 800.0):
    """Precompute per-step candidate arcs, probabilities and |d_route - d_gps| matrices.

    cands: rows for ONE trip with columns point_idx, arc_id, t_proj, p (classifier probability).
    Returns a list of steps (one per GPS point) + the network distance object.
    """
    c = cands.sort_values(["point_idx", "arc_id"], kind="mergesort")
    groups = {k: d for k, d in c.groupby("point_idx", sort=True)}
    nd = NetworkDistance(g, max_route_m)
    nd.prefetch(g.v[c.arc_id.to_numpy()])
    steps = []
    prev = None
    dt_all = np.r_[np.inf, np.diff(g.t)]
    for t in range(g.n_points):
        d = groups.get(t)
        if d is None:  # no candidate at all (should not happen with fallback): no state, chain is cut
            steps.append(None); prev = None; continue
        arcs = d.arc_id.to_numpy(); tp = d.t_proj.to_numpy(); p = d.p.to_numpy()
        step = {"arcs": arcs, "logp": np.log(np.clip(p, 1e-6, 1.0)), "diff": None}
        if prev is not None and dt_all[t] <= gap_threshold_s:
            pa, ptp = prev["arcs"], prev["tp"]
            gps = float(np.hypot(g.px[t] - g.px[t - 1], g.py[t] - g.py[t - 1]))
            La, Lb = g.arc_len[pa], g.arc_len[arcs]
            diff = np.empty((len(pa), len(arcs)), dtype=np.float32)
            for i, a in enumerate(pa):
                net = nd.lookup(int(g.v[a]), g.u[arcs])                       # v_a -> u_b
                route = (1 - ptp[i]) * La[i] + net + tp * Lb
                same = arcs == a
                route[same] = np.abs(tp[same] - ptp[i]) * La[i]
                diff[i] = np.where(np.isfinite(route), np.abs(route - gps), np.inf)
            step["diff"] = diff
        steps.append(step)
        prev = {"arcs": arcs, "tp": tp}
    return steps


def viterbi(steps, beta: float, w: float) -> np.ndarray:
    """Return the decoded arc id per GPS point (-1 where a point had no candidate)."""
    n = len(steps)
    out = np.full(n, -1, dtype=np.int64)
    seg_start = 0
    delta = None; backs = []

    def backtrack(end, delta_end):
        j = int(np.argmax(delta_end))
        for t in range(end, seg_start - 1, -1):
            out[t] = steps[t]["arcs"][j]
            if t > seg_start:
                j = int(backs[t - seg_start][j])

    for t, s in enumerate(steps):
        if s is None:
            if delta is not None:
                backtrack(t - 1, delta)
            delta = None; backs = []; seg_start = t + 1
            continue
        e = w * s["logp"]
        if delta is None or s["diff"] is None:
            if delta is not None:          # gap: close the previous segment, start a new one
                backtrack(t - 1, delta)
            seg_start = t; backs = [np.zeros(len(e), dtype=np.int32)]
            delta = e.copy()
            continue
        logT = np.where(np.isfinite(s["diff"]), -s["diff"] / beta, -UNREACHABLE)
        score = delta[:, None] + logT
        bp = np.argmax(score, axis=0)
        delta = e + score[bp, np.arange(score.shape[1])]
        backs.append(bp.astype(np.int32))
    if delta is not None:
        backtrack(n - 1, delta)
    return out


def classifier_argmax(steps) -> np.ndarray:
    out = np.full(len(steps), -1, dtype=np.int64)
    for t, s in enumerate(steps):
        if s is not None:
            out[t] = s["arcs"][int(np.argmax(s["logp"]))]
    return out


def continuity(arcs: np.ndarray, g: TripGeo, gap_before: np.ndarray) -> float:
    """Share of consecutive gap-free point pairs whose arcs are equal, adjacent, or 1 arc apart."""
    out_nodes: dict[int, set] = {}
    for a in np.flatnonzero(g.usable):
        out_nodes.setdefault(int(g.u[a]), set()).add(int(g.v[a]))
    ok = n = 0
    for i in range(1, len(arcs)):
        if gap_before[i] or arcs[i] < 0 or arcs[i - 1] < 0:
            continue
        n += 1
        p, q = arcs[i - 1], arcs[i]
        ok += p == q or g.v[p] == g.u[q] or int(g.u[q]) in out_nodes.get(int(g.v[p]), ())
    return ok / max(n, 1)


def route_set_scores(pred_arcs: np.ndarray, ref_route_canon: np.ndarray) -> dict:
    """Sequence-level score: overlap between the SET of predicted arcs and the reference route arcs."""
    P = set(int(a) for a in pred_arcs if a >= 0); R = set(int(a) for a in ref_route_canon)
    tp = len(P & R)
    prec = tp / max(len(P), 1); rec = tp / max(len(R), 1)
    return {"route_precision": prec, "route_recall": rec,
            "route_f1": 2 * prec * rec / max(prec + rec, 1e-9)}
