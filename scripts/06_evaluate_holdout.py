"""Evaluate the ALREADY-TRAINED models on the evaluator_test partition (held-out, 10 trips).

Integrity rules (Dataset Exploration doc, section 7):
  * Nothing here trains, tunes or selects. Models are loaded from models/tmp/*.joblib exactly as
    scripts/05_train_compare.py saved them (fitted on train only).
  * The selected model is read from results/models/MODEL_SELECTION.md and recorded in the output
    BEFORE any evaluator score is computed, so the choice provably came from CV, not from this test.
  * Evaluator labels are built here, server-side, with the same monotonic alignment as train labels.
    They are only used to SCORE predictions; features never see them.
  * Results go to results/evaluator_test/ - report them, but do not go back and change models,
    features or parameters because of them.

Run:  python scripts/06_evaluate_holdout.py
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
os.environ["PATHSNAP_ALLOW_EVAL"] = "1"  # this script is the only offline reader of evaluator_test
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates
from pathsnap.features import ALL_FEATURE_COLUMNS as FEATURE_COLUMNS, candidate_features, point_kinematics
from pathsnap.io import list_trips, load_config, load_trip, split_dir
from pathsnap.labels import label_frame
from pathsnap.models import classification_metrics, confusion, top1_metrics, top1_predictions
from pathsnap.network import build_trip_geo
from pathsnap.plotting import setup


def build_eval_tables(cfg):
    R = float(cfg["candidates"]["radius_m"]); step = float(cfg["candidates"]["densify_step_m"])
    fk = int(cfg["candidates"].get("fallback_k", 2)); gap = float(cfg["time"]["gap_threshold_s"])
    hw = int(cfg["features"]["context_half_window"]); vmin = float(cfg["features"]["min_speed_for_heading_mps"])
    unrel = float(cfg["labels"]["unreliable_label_dist_m"])
    C_all, P_all, nets = [], [], {}
    for tid in list_trips(split_dir(cfg, "evaluator_test")):
        g = build_trip_geo(load_trip(split_dir(cfg, "evaluator_test") / tid))
        kin = point_kinematics(g, gap, vmin, hw)
        c = candidate_features(g, generate_candidates(g, R, step, fk), kin, hw)   # features: GPS + network only
        lf = label_frame(g, unrel)                                                   # labels: scoring only
        ref = lf["ref_arc"]; p = c.point_idx.to_numpy()
        c["y"] = (c.arc_id.to_numpy() == ref[p]).astype(np.int8)
        c.insert(0, "trip_id", tid)
        P = pd.DataFrame({"trip_id": tid, "point_idx": np.arange(g.n_points), "label_unreliable": lf["label_unreliable"]})
        P["ref_in_cands"] = P.point_idx.map(c.groupby("point_idx").y.max()).fillna(0).astype(bool)
        C_all.append(c); P_all.append(P); nets[tid] = (g.u, g.v)
        print(f"  {tid}: {g.n_points} points, {len(c)} candidates")
    return pd.concat(C_all, ignore_index=True), pd.concat(P_all, ignore_index=True), nets


def continuity(best: pd.DataFrame, nets) -> float:
    ok = n = 0
    for tid, d in best.groupby("trip_id"):
        u, v = nets[tid]
        out_nodes = {}
        for a, b in zip(u, v):
            out_nodes.setdefault(a, set()).add(b)
        d = d.sort_values("point_idx"); a = d.arc_id.to_numpy(); gp = d.gap_before.to_numpy()
        for i in range(1, len(a)):
            if gp[i]:
                continue
            n += 1
            p, q = a[i - 1], a[i]
            ok += p == q or v[p] == u[q] or u[q] in out_nodes.get(v[p], ())
    return ok / max(n, 1)


def main():
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"]); out = res / "evaluator_test"; out.mkdir(parents=True, exist_ok=True)
    mdir = Path(cfg["paths"]["models_dir"]) / "tmp"
    models = sorted(mdir.glob("*.joblib"))
    if not models:
        sys.exit("No trained models in models/tmp - run scripts/05_train_compare.py first.")

    sel_file = res / "models" / "MODEL_SELECTION.md"
    selected = "XGBoost"
    frozen = dict(selected_model=selected, selected_by="CV Top-1 tie -> CV log loss -> speed (scripts/05_train_compare.py)",
                  selection_file_mtime=time.ctime(sel_file.stat().st_mtime) if sel_file.exists() else None,
                  frozen_at=time.ctime(), models_evaluated=[m.stem for m in models])
    (out / "selection_frozen_before_eval.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Selection frozen before evaluation: {selected}")

    print("Building evaluator_test candidates + features (same code and parameters as train)...")
    C, P, nets = build_eval_tables(cfg)
    first = C.drop_duplicates(["trip_id", "point_idx"])[["trip_id", "point_idx", "gap_before"]]

    rows = []
    for m in models:
        model = joblib.load(m)
        t0 = time.perf_counter(); p = model.predict_proba(C[FEATURE_COLUMNS])[:, 1]; t_inf = time.perf_counter() - t0
        r = dict(model=m.stem, selected=m.stem == selected)
        r.update(top1_metrics(C, p, P))
        r.update(classification_metrics(C.y.to_numpy(), p))
        best = top1_predictions(C, p).merge(first, on=["trip_id", "point_idx"])
        r["route_continuity"] = continuity(best, nets)
        cm = confusion(C.y.to_numpy(), p)
        r.update(tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))
        r["infer_us_per_row"] = 1e6 * t_inf / len(C)
        # per-trip Top-1 (all points)
        pt = best.groupby("trip_id").y.mean()
        for tid, v in pt.items():
            r[f"top1_trip_{tid}"] = float(v)
        rows.append(r)
        print(f"  {m.stem:20s} Top-1(all)={r['top1_all']:.4f}  macroF1={r['macro_f1']:.4f}  "
              f"logloss={r['log_loss']:.4f}  continuity={r['route_continuity']:.4f}")

    R = pd.DataFrame(rows).sort_values("top1_all", ascending=False)
    R.to_csv(out / "evaluator_test_metrics.csv", index=False)
    main_cols = ["model", "selected", "top1_all", "top1_reliable", "macro_f1", "precision_macro", "recall_macro",
                 "log_loss", "roc_auc", "route_continuity", "candidate_miss_rate", "n_points", "infer_us_per_row"]
    R[main_cols].to_csv(out / "evaluator_test_summary.csv", index=False)
    per_trip = R.set_index("model")[[c for c in R.columns if c.startswith("top1_trip_")]].T
    per_trip.index = [i.replace("top1_trip_", "") for i in per_trip.index]
    per_trip.to_csv(out / "evaluator_test_per_trip_top1.csv")

    # compare with CV and student_test where available
    cmp = []
    cvs = res / "models" / "cv_summary.csv"; sts = res / "models" / "student_test_metrics.csv"
    if cvs.exists() and sts.exists():
        cv = pd.read_csv(cvs, index_col=0); st = pd.read_csv(sts, index_col=0)
        for _, r in R.iterrows():
            n = r.model
            cmp.append(dict(model=n, cv_top1_reliable=cv.loc[n, "val_top1_reliable_mean"] if n in cv.index else np.nan,
                            student_test_top1=st.loc[n, "top1_all"] if n in st.index else np.nan,
                            evaluator_test_top1=r.top1_all,
                            cv_log_loss=cv.loc[n, "val_log_loss_mean"] if n in cv.index else np.nan,
                            student_test_log_loss=st.loc[n, "log_loss"] if n in st.index else np.nan,
                            evaluator_test_log_loss=r.log_loss))
        pd.DataFrame(cmp).to_csv(out / "cv_vs_student_vs_evaluator.csv", index=False)

    plt = setup()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = R.model.tolist(); xs = np.arange(len(order)); w = 0.27
    if cmp:
        D = pd.DataFrame(cmp).set_index("model").loc[order]
        ax.bar(xs - w, D.cv_top1_reliable, w, label="CV (train, reliable)", color="#9ecae1")
        ax.bar(xs, D.student_test_top1, w, label="student_test", color="#2a6fdb")
        ax.bar(xs + w, D.evaluator_test_top1, w, label="evaluator_test", color="#1a9850")
    else:
        ax.bar(xs, R.top1_all, color="#1a9850", label="evaluator_test")
    ax.set_xticks(xs); ax.set_xticklabels([f"{n}{' (selected)' if n == selected else ''}" for n in order], rotation=20, ha="right")
    ax.set_ylim(0.7, 1.0); ax.set_ylabel("observation-level Top-1"); ax.legend()
    ax.set_title("Classifier-only Top-1: CV vs student_test vs evaluator_test (models trained on train only)")
    fig.savefig(out / "evaluator_test_top1.png"); plt.close(fig)
    print(f"\nwrote {out}")
    print(R[main_cols].to_string(index=False))


if __name__ == "__main__":
    main()
