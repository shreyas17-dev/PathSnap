"""End-to-end prediction pipeline for ONE trip: raw files -> route + metrics.

Used by the FastAPI backend (api/main.py) and by scripts/09_hmm_viterbi.py, so the evaluation and
the live demo run exactly the same code.
Labels (from .route) are computed ONLY for scoring, after prediction, and never returned as geometry.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .features import ALL_FEATURE_COLUMNS, candidate_features, point_kinematics
from .hmm import classifier_argmax, continuity, prepare_trip, route_set_scores, viterbi
from .io import REPO_ROOT, load_trip
from .labels import label_frame
from .network import build_trip_geo


def load_artifacts(models_dir: Path):
    model = joblib.load(models_dir / "xgboost_tuned.joblib")
    hmm_cfg = json.loads((models_dir / "hmm_config.json").read_text(encoding="utf-8")) \
        if (models_dir / "hmm_config.json").exists() else {"beta": 10.0, "w": 1.0, "max_route_m": 800.0}
    try:  # predict on CPU data without the device-mismatch warning
        model.set_params(device="cpu")
    except Exception:
        pass
    return model, hmm_cfg


def candidates_with_features(g, cfg):
    R = float(cfg["candidates"]["radius_m"]); step = float(cfg["candidates"]["densify_step_m"])
    fk = int(cfg["candidates"].get("fallback_k", 2)); gap = float(cfg["time"]["gap_threshold_s"])
    hw = int(cfg["features"]["context_half_window"]); vmin = float(cfg["features"]["min_speed_for_heading_mps"])
    kin = point_kinematics(g, gap, vmin, hw)
    return candidate_features(g, generate_candidates(g, R, step, fk), kin, hw), kin


def arcs_to_segments(g, arcs: np.ndarray, gap_before: np.ndarray):
    """Polyline segments [[lat, lon], ...] following the decoded arc sequence; new segment at gaps."""
    lon, lat = g.trip.nodes.lon.to_numpy(), g.trip.nodes.lat.to_numpy()
    segs, cur, last = [], [], None
    for i, a in enumerate(arcs):
        if gap_before[i] and cur:
            segs.append(cur); cur = []; last = None
        if a < 0 or a == last:
            continue
        for nnode in (g.u[a], g.v[a]):
            pt = [round(float(lat[nnode]), 6), round(float(lon[nnode]), 6)]
            if not cur or cur[-1] != pt:
                cur.append(pt)
        last = a
    if cur:
        segs.append(cur)
    return segs


def run_trip(folder: Path, model, hmm_cfg: dict, cfg: dict, score: bool = True) -> dict:
    timing = {}
    t0 = time.perf_counter()
    trip = load_trip(folder, load_route=score)
    g = build_trip_geo(trip)
    C, kin = candidates_with_features(g, cfg)
    timing["preprocess"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    C["p"] = model.predict_proba(C[ALL_FEATURE_COLUMNS])[:, 1]
    timing["classifier"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    gap = float(cfg["time"]["gap_threshold_s"])
    steps = prepare_trip(g, C[["point_idx", "arc_id", "t_proj", "p"]], gap, float(hmm_cfg.get("max_route_m", 800)))
    hmm_arcs = viterbi(steps, float(hmm_cfg["beta"]), float(hmm_cfg["w"]))
    clf_arcs = classifier_argmax(steps)
    timing["hmm"] = (time.perf_counter() - t2) * 1000

    gap_before = kin["gap_before"].to_numpy().astype(bool)
    res = {"g": g, "hmm_arcs": hmm_arcs, "clf_arcs": clf_arcs, "gap_before": gap_before, "timing_ms": timing,
           "n_points": g.n_points}
    if score and trip.route is not None:
        lf = label_frame(g, float(cfg["labels"]["unreliable_label_dist_m"]))
        ref = lf["ref_arc"]; rel = ~lf["label_unreliable"]
        refset = g.canon[trip.route]
        for name, arcs in [("hmm", hmm_arcs), ("classifier_only", clf_arcs)]:
            ok = arcs == ref
            m = {"top1_accuracy": float(ok.mean()), "top1_reliable": float(ok[rel].mean()) if rel.any() else float("nan"),
                 "n_correct": int(ok.sum()), "route_continuity": continuity(arcs, g, gap_before)}
            m.update(route_set_scores(arcs, refset))
            res[name] = m
    timing["total"] = (time.perf_counter() - t0) * 1000
    return res
