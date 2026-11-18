"""Phase 4 — HMM + Viterbi route reconstruction: tune the decoder, compare with classifier-only.

Data rules: decoder parameters (beta, w) are chosen on TRAIN only, using out-of-fold (OOF) classifier
probabilities, so the HMM never sees probabilities from a model that was trained on the same trip.
student_test is then reported with the final tuned model. evaluator_test is not read here.

Steps
  1. OOF probabilities on all train trips: 5-fold GroupKFold by trip, the tuned XGBoost parameters
     (models/final/xgboost_tuned_config.json), fitted on strided reliable rows of the training trips,
     predicting every candidate row of the held-out trips.
  2. Grid over beta (transition scale, m) x w (emission weight) -> HMM Top-1 on train OOF.
  3. Best (beta, w) -> models/final/hmm_config.json.
  4. Classifier-only vs classifier+HMM on train OOF and student_test: Top-1, continuity, route P/R/F1,
     per-trip gains (where the HMM helped and where it did not), Folium map.

Run:  python scripts/09_hmm_viterbi.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import folium
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from pathsnap.features import ALL_FEATURE_COLUMNS
from pathsnap.hmm import classifier_argmax, continuity, prepare_trip, route_set_scores, viterbi
from pathsnap.io import list_trips, load_config, load_trip, split_dir
from pathsnap.maps import route_polyline, trip_map
from pathsnap.models import cuda_available
from pathsnap.network import build_trip_geo
from pathsnap.plotting import setup

BETAS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
WEIGHTS = [0.5, 1.0, 2.0]


def tuned_params(cfg, device):
    f = Path(cfg["paths"]["models_dir"]) / "final" / "xgboost_tuned_config.json"
    if not f.exists():
        sys.exit("models/final/xgboost_tuned_config.json missing - run scripts/07_bayes_opt_xgb.py first")
    p = json.loads(f.read_text(encoding="utf-8"))["params"]
    p["device"] = device
    return p


def oof_probabilities(cfg, params, stride, log):
    proc = Path(cfg["paths"]["processed_dir"])
    C = pd.read_parquet(proc / "candidates_train.parquet")
    trips = np.array(sorted(C.trip_id.unique()))
    oof = np.zeros(len(C), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for k, (tr_t, va_t) in enumerate(gkf.split(trips, groups=trips)):
        tr_trips, va_trips = set(trips[tr_t]), set(trips[va_t])
        fit_mask = C.trip_id.isin(tr_trips) & (C.point_idx % stride == 0) & ~C.label_unreliable.astype(bool)
        m = xgb.XGBClassifier(**params)
        t0 = time.time(); m.fit(C.loc[fit_mask, ALL_FEATURE_COLUMNS], C.loc[fit_mask, "y"], verbose=False)
        try:
            m.set_params(device="cpu")
        except Exception:
            pass
        va_mask = C.trip_id.isin(va_trips).to_numpy()
        oof[va_mask] = m.predict_proba(C.loc[va_mask, ALL_FEATURE_COLUMNS])[:, 1]
        log(f"  OOF fold {k}: fit on {fit_mask.sum():,} rows, predicted {va_mask.sum():,} rows ({time.time() - t0:.0f}s)")
    C["p"] = oof
    return C


def prepare_split(cfg, split, C, max_trips=None, log=print):
    gap = float(cfg["time"]["gap_threshold_s"])
    proc = Path(cfg["paths"]["processed_dir"])
    P = pd.read_parquet(proc / f"points_{split}.parquet")
    out = {}
    tids = list_trips(split_dir(cfg, split))[: max_trips or None]
    t0 = time.time()
    for tid in tids:
        g = build_trip_geo(load_trip(split_dir(cfg, split) / tid))
        cc = C[C.trip_id == tid]
        steps = prepare_trip(g, cc[["point_idx", "arc_id", "t_proj", "p"]], gap)
        Pt = P[P.trip_id == tid].sort_values("point_idx")
        gb = cc.drop_duplicates("point_idx").set_index("point_idx").gap_before.reindex(range(g.n_points)).fillna(0).to_numpy().astype(bool)
        out[tid] = dict(g=g, steps=steps, ref=Pt.ref_arc.to_numpy(), rel=~Pt.label_unreliable.to_numpy().astype(bool),
                        gap_before=gb, route=g.canon[g.trip.route])
    log(f"  prepared {len(out)} {split} trips in {time.time() - t0:.0f}s")
    return out


def score(prep, arcs_by_trip):
    ok_all = []; ok_rel = []; rows = []
    for tid, d in prep.items():
        a = arcs_by_trip[tid]; ok = a == d["ref"]
        ok_all.append(ok); ok_rel.append(ok[d["rel"]])
        rs = route_set_scores(a, d["route"])
        rows.append(dict(trip_id=tid, n_points=len(a), top1=float(ok.mean()),
                         continuity=continuity(a, d["g"], d["gap_before"]), **rs))
    per = pd.DataFrame(rows)
    return dict(top1_all=float(np.concatenate(ok_all).mean()), top1_reliable=float(np.concatenate(ok_rel).mean()),
                continuity=float(np.average(per.continuity, weights=per.n_points)),
                route_f1=float(per.route_f1.mean()), route_precision=float(per.route_precision.mean()),
                route_recall=float(per.route_recall.mean())), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trips", type=int, default=None, help="limit train trips (quick test)")
    args = ap.parse_args()
    cfg = load_config(); stride = int(cfg.get("models", {}).get("point_stride", 3))
    out = Path(cfg["paths"]["results_dir"]) / "hmm"; out.mkdir(parents=True, exist_ok=True)
    fin = Path(cfg["paths"]["models_dir"]) / "final"
    logf = open(out / "hmm_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()

    device = "cuda" if cuda_available() else "cpu"
    params = tuned_params(cfg, device)
    log(f"=== HMM run {time.ctime()} | device={device} | tuned XGBoost params loaded")

    log("1) out-of-fold probabilities on train")
    C = oof_probabilities(cfg, params, stride, log)
    log("2) precomputing route distances (train)")
    prep_tr = prepare_split(cfg, "train", C, args.max_trips, log)
    clf_tr = {t: classifier_argmax(d["steps"]) for t, d in prep_tr.items()}
    base_tr, per_clf_tr = score(prep_tr, clf_tr)
    log(f"  classifier-only train OOF: Top-1 all={base_tr['top1_all']:.4f} reliable={base_tr['top1_reliable']:.4f}")

    grid = []
    for beta in BETAS:
        for w in WEIGHTS:
            t0 = time.time()
            arcs = {t: viterbi(d["steps"], beta, w) for t, d in prep_tr.items()}
            s, _ = score(prep_tr, arcs)
            grid.append(dict(beta=beta, w=w, **s, seconds=time.time() - t0))
            log(f"  beta={beta:5.1f} w={w:3.1f}: HMM Top-1 reliable={s['top1_reliable']:.4f} all={s['top1_all']:.4f} "
                f"continuity={s['continuity']:.4f}")
    G = pd.DataFrame(grid); G.to_csv(out / "hmm_grid_train_oof.csv", index=False)
    best = G.sort_values(["top1_reliable", "continuity"], ascending=False).iloc[0]
    hmm_cfg = dict(beta=float(best.beta), w=float(best.w), max_route_m=800.0,
                   chosen_by="max train OOF Top-1 (reliable) over grid", train_oof_top1_reliable=float(best.top1_reliable))
    (fin / "hmm_config.json").write_text(json.dumps(hmm_cfg, indent=2), encoding="utf-8")
    log(f"3) best decoder: beta={best.beta} w={best.w} -> models/final/hmm_config.json")

    hmm_tr = {t: viterbi(d["steps"], hmm_cfg["beta"], hmm_cfg["w"]) for t, d in prep_tr.items()}
    s_tr, per_hmm_tr = score(prep_tr, hmm_tr)

    log("4) student_test with the final tuned model")
    model = joblib.load(fin / "xgboost_tuned.joblib")
    try:
        model.set_params(device="cpu")
    except Exception:
        pass
    Cs = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "candidates_student_test.parquet")
    Cs["p"] = model.predict_proba(Cs[ALL_FEATURE_COLUMNS])[:, 1]
    prep_st = prepare_split(cfg, "student_test", Cs, None, log)
    clf_st = {t: classifier_argmax(d["steps"]) for t, d in prep_st.items()}
    hmm_st = {t: viterbi(d["steps"], hmm_cfg["beta"], hmm_cfg["w"]) for t, d in prep_st.items()}
    c_st, per_clf_st = score(prep_st, clf_st); h_st, per_hmm_st = score(prep_st, hmm_st)

    S = pd.DataFrame([
        dict(partition="train (OOF)", method="classifier only", **base_tr),
        dict(partition="train (OOF)", method="classifier + HMM", **s_tr),
        dict(partition="student_test", method="classifier only", **c_st),
        dict(partition="student_test", method="classifier + HMM", **h_st),
    ])
    S.to_csv(out / "classifier_vs_hmm.csv", index=False)
    for nm, pc, ph in [("train_oof", per_clf_tr, per_hmm_tr), ("student_test", per_clf_st, per_hmm_st)]:
        m = pc[["trip_id", "n_points", "top1", "continuity", "route_f1"]].merge(
            ph[["trip_id", "top1", "continuity", "route_f1"]], on="trip_id", suffixes=("_classifier", "_hmm"))
        m["top1_gain"] = m.top1_hmm - m.top1_classifier
        m.sort_values("top1_gain").to_csv(out / f"per_trip_{nm}.csv", index=False)
    log(S.round(4).to_string(index=False))

    # ---- plots + map ----
    plt = setup()
    H = G.pivot(index="w", columns="beta", values="top1_reliable")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(H.values, cmap="Greens", aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(H.columns))); ax.set_xticklabels(H.columns); ax.set_yticks(range(len(H.index))); ax.set_yticklabels(H.index)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("beta (m)"); ax.set_ylabel("emission weight w"); fig.colorbar(im, ax=ax)
    ax.set_title("HMM decoder grid: train OOF Top-1 (reliable)"); fig.savefig(out / "hmm_grid.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, part in zip(axs, ["train (OOF)", "student_test"]):
        d = S[S.partition == part]
        x = np.arange(3); w_ = 0.38
        for k, (meth, col) in enumerate([("classifier only", "#9ecae1"), ("classifier + HMM", "#1a9850")]):
            r = d[d.method == meth].iloc[0]
            vals = [r.top1_all, r.continuity, r.route_f1]
            ax.bar(x + (k - 0.5) * w_, vals, w_, label=meth, color=col)
            for xi, v in zip(x + (k - 0.5) * w_, vals):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["Top-1", "route continuity", "route F1"]); ax.set_ylim(0.6, 1.02)
        ax.set_title(part); ax.legend(fontsize=8)
    fig.suptitle("Classifier-only vs classifier + HMM/Viterbi"); fig.savefig(out / "classifier_vs_hmm.png"); plt.close(fig)

    d = pd.read_csv(out / "per_trip_student_test.csv", dtype={"trip_id": str})
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(d.trip_id, d.top1_gain * 100, color=np.where(d.top1_gain >= 0, "#1a9850", "#d73027"))
    ax.axhline(0, c="k", lw=0.8); ax.set_ylabel("HMM − classifier Top-1 (points)"); plt.xticks(rotation=30)
    ax.set_title("Per-trip effect of the HMM (student_test)"); fig.savefig(out / "per_trip_gain_student_test.png"); plt.close(fig)

    tid = d.sort_values("n_points").trip_id.iloc[len(d) // 2]
    dd = prep_st[tid]; g = dd["g"]
    lc = folium.FeatureGroup(name="Classifier-only prediction", show=False)
    lh = folium.FeatureGroup(name="Classifier + HMM prediction", show=True)
    for arcs, layer, col, wt in [(clf_st[tid], lc, "#fd8d3c", 3), (hmm_st[tid], lh, "#e7298a", 4)]:
        seq = [a for i, a in enumerate(arcs) if a >= 0 and (i == 0 or a != arcs[i - 1])]
        folium.PolyLine(route_polyline(g, np.array(seq)), color=col, weight=wt, opacity=0.9).add_to(layer)
    m = trip_map(g, float(cfg["time"]["gap_threshold_s"]), f"Trip {tid} (student_test): reference vs predicted", [lc, lh],
                 [("Classifier-only prediction", "#fd8d3c"), ("Classifier + HMM prediction", "#e7298a")])
    m.save(str(Path(cfg["paths"]["results_dir"]) / "maps" / f"hmm_student_test_{tid}.html"))
    log(f"map: results/maps/hmm_student_test_{tid}.html")


if __name__ == "__main__":
    main()
