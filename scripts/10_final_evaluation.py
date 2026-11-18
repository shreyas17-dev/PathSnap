"""FINAL evaluation: full pipeline (tuned XGBoost + HMM/Viterbi) on the 10 evaluator_test trips.

Uses src/pathsnap/pipeline.py - exactly the code the API runs - with the frozen artifacts in
models/final/ (xgboost_tuned.joblib, hmm_config.json). Nothing is trained or tuned here; run it once
after scripts 07 and 09, and report the numbers. Do not change anything because of them.

Outputs (results/final_evaluation/):
  final_per_trip.csv          per trip: classifier-only vs HMM Top-1, continuity, route P/R/F1, timings
  final_summary.csv           totals (point-weighted Top-1 over all 19,674 points) + trip means
  final_classifier_vs_hmm.png bar chart per trip
  frozen_artifacts.json       which model / HMM config were evaluated, with timestamps
results/maps/final_eval_<trip>.html   Folium maps: GPS, network, reference route, classifier, HMM

Run:  python scripts/10_final_evaluation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["PATHSNAP_ALLOW_EVAL"] = "1"
warnings.filterwarnings("ignore")

import folium
import numpy as np
import pandas as pd

from pathsnap.io import list_trips, load_config, split_dir
from pathsnap.maps import route_polyline, trip_map
from pathsnap.pipeline import load_artifacts, run_trip
from pathsnap.plotting import setup


def main():
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"]); out = res / "final_evaluation"; out.mkdir(parents=True, exist_ok=True)
    (res / "maps").mkdir(parents=True, exist_ok=True)
    fin = Path(cfg["paths"]["models_dir"]) / "final"
    for f in ["xgboost_tuned.joblib", "hmm_config.json"]:
        if not (fin / f).exists():
            sys.exit(f"models/final/{f} missing - run scripts/07_bayes_opt_xgb.py and scripts/09_hmm_viterbi.py first")
    model, hmm_cfg = load_artifacts(fin)
    (out / "frozen_artifacts.json").write_text(json.dumps({
        "classifier": "models/final/xgboost_tuned.joblib",
        "classifier_mtime": time.ctime((fin / "xgboost_tuned.joblib").stat().st_mtime),
        "hmm_config": hmm_cfg, "hmm_config_mtime": time.ctime((fin / "hmm_config.json").stat().st_mtime),
        "evaluated_at": time.ctime()}, indent=2), encoding="utf-8")
    print(f"Frozen artifacts: tuned XGBoost + HMM {hmm_cfg}")

    rows = []
    gap = float(cfg["time"]["gap_threshold_s"])
    for tid in list_trips(split_dir(cfg, "evaluator_test")):
        r = run_trip(split_dir(cfg, "evaluator_test") / tid, model, hmm_cfg, cfg, score=True)
        c, h = r["classifier_only"], r["hmm"]
        rows.append(dict(trip_id=tid, n_points=r["n_points"],
                         clf_top1=c["top1_accuracy"], hmm_top1=h["top1_accuracy"],
                         clf_correct=c["n_correct"], hmm_correct=h["n_correct"],
                         clf_continuity=c["route_continuity"], hmm_continuity=h["route_continuity"],
                         clf_route_f1=c["route_f1"], hmm_route_f1=h["route_f1"],
                         hmm_route_precision=h["route_precision"], hmm_route_recall=h["route_recall"],
                         total_ms=r["timing_ms"]["total"]))
        print(f"  {tid}: {r['n_points']:5d} pts | classifier Top-1={c['top1_accuracy']:.4f} -> HMM Top-1={h['top1_accuracy']:.4f} "
              f"| continuity {c['route_continuity']:.3f} -> {h['route_continuity']:.3f} | {r['timing_ms']['total']:.0f} ms")
        g = r["g"]
        lc = folium.FeatureGroup(name="Classifier-only prediction", show=False)
        lh = folium.FeatureGroup(name="Classifier + HMM prediction", show=True)
        for arcs, layer, col, wt in [(r["clf_arcs"], lc, "#fd8d3c", 3), (r["hmm_arcs"], lh, "#e7298a", 4)]:
            seq = [a for i, a in enumerate(arcs) if a >= 0 and (i == 0 or a != arcs[i - 1])]
            folium.PolyLine(route_polyline(g, np.array(seq)), color=col, weight=wt, opacity=0.9).add_to(layer)
        m = trip_map(g, gap, f"Trip {tid} (evaluator_test): HMM Top-1 {h['top1_accuracy']:.1%}", [lc, lh],
                     [("Classifier-only prediction", "#fd8d3c"), ("Classifier + HMM prediction", "#e7298a")])
        m.save(str(res / "maps" / f"final_eval_{tid}.html"))

    D = pd.DataFrame(rows); D["hmm_gain"] = D.hmm_top1 - D.clf_top1
    D.to_csv(out / "final_per_trip.csv", index=False)
    N = D.n_points.sum()
    S = pd.DataFrame([
        dict(method="classifier only (tuned XGBoost)", top1_all_points=D.clf_correct.sum() / N, top1_trip_mean=D.clf_top1.mean(),
             route_continuity=np.average(D.clf_continuity, weights=D.n_points), route_f1_trip_mean=D.clf_route_f1.mean()),
        dict(method="classifier + HMM/Viterbi (final)", top1_all_points=D.hmm_correct.sum() / N, top1_trip_mean=D.hmm_top1.mean(),
             route_continuity=np.average(D.hmm_continuity, weights=D.n_points), route_f1_trip_mean=D.hmm_route_f1.mean()),
    ])
    S["n_points"] = N; S["n_trips"] = len(D)
    S.to_csv(out / "final_summary.csv", index=False)

    plt = setup()
    fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(D)); w = 0.38
    ax.bar(x - w / 2, D.clf_top1, w, label="classifier only", color="#9ecae1")
    ax.bar(x + w / 2, D.hmm_top1, w, label="classifier + HMM", color="#1a9850")
    ax.set_xticks(x); ax.set_xticklabels(D.trip_id, rotation=30); ax.set_ylim(max(0, D[["clf_top1", "hmm_top1"]].min().min() - 0.05), 1.0)
    ax.set_ylabel("observation-level Top-1"); ax.legend()
    ax.set_title(f"Evaluator_test: Top-1 {S.top1_all_points.iloc[0]:.1%} (classifier) -> {S.top1_all_points.iloc[1]:.1%} (HMM), all {N:,} points")
    fig.savefig(out / "final_classifier_vs_hmm.png"); plt.close(fig)

    print("\n" + S.round(4).to_string(index=False))
    print(f"\nwrote {out} and results/maps/final_eval_*.html")


if __name__ == "__main__":
    main()
