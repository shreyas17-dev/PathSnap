"""Phase 1b-iii — Candidate generation + feature engineering -> model-ready tables.

Partitions: train and student_test are processed with IDENTICAL code and parameters
(nothing is fitted here; scaling/imputation happen inside the model pipelines, fitted on train folds).

Outputs
  data/processed/candidates_<split>.parquet   one row per (GPS point, candidate arc): features + y
  data/processed/points_<split>.parquet       one row per GPS point: ref arc, whether it was a candidate
  results/preprocessing/
    candidate_summary.csv           miss rate, candidates/point, class balance per split
    feature_summary_train.csv       mean/std/NaN-rate per feature, split by class (train)
    feature_single_rank_top1.csv    Top-1 accuracy if candidates are ranked by ONE feature (train)
    feature_univariate_auc.csv      ROC-AUC of each single feature (train, reliable rows)
    feature_correlation.png         redundancy check
    feature_distributions.png       class-conditional distributions of key features
    rule_baselines.csv              non-learned baselines on train and student_test

Run:  python scripts/04_build_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: avoid UnicodeEncodeError on ±, ≤, →
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pathsnap.candidates import generate_candidates
from pathsnap.features import ALL_FEATURE_COLUMNS as FEATURE_COLUMNS, candidate_features, point_kinematics
from pathsnap.io import list_trips, load_config, load_trip, split_dir
from pathsnap.network import build_trip_geo
from pathsnap.plotting import setup


def build_split(cfg, split):
    proc = Path(cfg["paths"]["processed_dir"])
    lab = pd.read_parquet(proc / f"labels_{split}.parquet")
    R = float(cfg["candidates"]["radius_m"]); step = float(cfg["candidates"]["densify_step_m"])
    fk = int(cfg["candidates"].get("fallback_k", 2))
    gap = float(cfg["time"]["gap_threshold_s"]); hw = int(cfg["features"]["context_half_window"])
    vmin = float(cfg["features"]["min_speed_for_heading_mps"])
    cand_frames, pt_frames = [], []
    for tid in list_trips(split_dir(cfg, split)):
        g = build_trip_geo(load_trip(split_dir(cfg, split) / tid))
        L = lab[lab.trip_id == tid].reset_index(drop=True)
        kin = point_kinematics(g, gap, vmin, hw)
        c = candidate_features(g, generate_candidates(g, R, step, fk), kin, hw)
        ref = L.ref_arc.to_numpy()
        p = c.point_idx.to_numpy()
        c["y"] = (c.arc_id.to_numpy() == ref[p]).astype(np.int8)
        c["label_unreliable"] = L.label_unreliable.to_numpy()[p]
        c.insert(0, "trip_id", tid)
        for col in FEATURE_COLUMNS + ["t_proj"]:
            if c[col].dtype == np.float64:
                c[col] = c[col].astype(np.float32)
        cand_frames.append(c)
        hit = c.groupby("point_idx").y.max()
        P = L.copy()
        P["ref_in_cands"] = P.point_idx.map(hit).fillna(0).astype(bool)
        P["n_cands"] = P.point_idx.map(c.groupby("point_idx").size()).fillna(0).astype(int)
        P["is_fallback"] = P.point_idx.map(c.groupby("point_idx").is_fallback.max()).fillna(0).astype(bool)
        pt_frames.append(P)
    C = pd.concat(cand_frames, ignore_index=True)
    P = pd.concat(pt_frames, ignore_index=True)
    C.to_parquet(proc / f"candidates_{split}.parquet", index=False)
    P.to_parquet(proc / f"points_{split}.parquet", index=False)
    return C, P


def top1(C: pd.DataFrame, P: pd.DataFrame, score: np.ndarray, higher_better=True) -> dict:
    """Top-1 over points: candidate with the best score == reference arc. Missed refs count as wrong."""
    s = score if higher_better else -score
    d = pd.DataFrame({"trip_id": C.trip_id.to_numpy(), "point_idx": C.point_idx.to_numpy(), "s": s, "y": C.y.to_numpy()})
    d = d.sort_values(["trip_id", "point_idx", "s"], ascending=[True, True, False], kind="mergesort")
    best = d.drop_duplicates(["trip_id", "point_idx"])
    m = P[["trip_id", "point_idx", "label_unreliable"]].merge(best[["trip_id", "point_idx", "y"]], how="left")
    m["y"] = m.y.fillna(0)
    rel = ~m.label_unreliable
    return {"top1_all": float(m.y.mean()), "top1_reliable": float(m.y[rel].mean())}


def main():
    cfg = load_config()
    out = Path(cfg["paths"]["results_dir"]) / "preprocessing"
    plt = setup()
    data = {}
    for split in ["train", "student_test"]:
        t0 = time.time()
        C, P = build_split(cfg, split)
        data[split] = (C, P)
        print(f"[{split}] {len(P)} points, {len(C)} candidate rows, {time.time() - t0:.1f}s")

    # ---- candidate-set summary ----
    rows = []
    for split, (C, P) in data.items():
        rel = ~P.label_unreliable
        rows.append(dict(split=split, points=len(P), candidate_rows=len(C),
                         cands_per_point_mean=float(P.n_cands.mean()), cands_per_point_p95=float(P.n_cands.quantile(.95)),
                         fallback_points=int(P.is_fallback.sum()),
                         miss_rate_all=float(1 - P.ref_in_cands.mean()),
                         miss_rate_reliable=float(1 - P.ref_in_cands[rel].mean()),
                         positive_rate=float(C.y.mean()),
                         unreliable_points=int((~rel).sum())))
    cs = pd.DataFrame(rows); cs.to_csv(out / "candidate_summary.csv", index=False)
    print(cs.T)

    C, P = data["train"]
    R = C[~C.label_unreliable.astype(bool)]

    # ---- feature summary by class (train) ----
    summ = []
    for f in FEATURE_COLUMNS:
        x = R[f].astype(float)
        summ.append(dict(feature=f, nan_rate=float(x.isna().mean()),
                         mean_pos=float(x[R.y == 1].mean()), mean_neg=float(x[R.y == 0].mean()),
                         std=float(x.std()), min=float(x.min()), max=float(x.max())))
    pd.DataFrame(summ).to_csv(out / "feature_summary_train.csv", index=False)

    # ---- univariate evidence (train only) ----
    uni = []
    for f in FEATURE_COLUMNS:
        x = R[f].astype(float).to_numpy()
        ok = np.isfinite(x)
        auc = roc_auc_score(R.y.to_numpy()[ok], x[ok]) if ok.sum() and R.y[ok].nunique() == 2 else np.nan
        uni.append(dict(feature=f, auc=auc, auc_sign_adjusted=max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
                        direction="higher→true" if auc >= 0.5 else "lower→true"))
    U = pd.DataFrame(uni).sort_values("auc_sign_adjusted", ascending=False)
    U.to_csv(out / "feature_univariate_auc.csv", index=False)
    print(U.to_string(index=False))

    rk = []
    for f, hb in [("dist_m", False), ("smooth_dist", False), ("ctx_mean_dist", False), ("heading_diff", False)]:
        s = C[f].astype(float).fillna(C[f].max() if not hb else C[f].min()).to_numpy()
        rk.append(dict(rank_by=f, **top1(C, P, s, hb)))
    pd.DataFrame(rk).to_csv(out / "feature_single_rank_top1.csv", index=False)

    # ---- rule baselines (no learning): nearest arc, nearest arc + heading tie-break ----
    base = []
    for split, (Cs, Ps) in data.items():
        hd = Cs.heading_diff.astype(float).fillna(90).to_numpy()
        base.append(dict(split=split, baseline="nearest arc", **top1(Cs, Ps, Cs.dist_m.to_numpy(), False)))
        base.append(dict(split=split, baseline="nearest arc, heading tie-break",
                         **top1(Cs, Ps, Cs.dist_m.round(1).to_numpy() * 1000 + hd, False)))
        base.append(dict(split=split, baseline="oracle (upper bound = 1 - miss rate)",
                         **top1(Cs, Ps, Cs.y.to_numpy().astype(float), True)))
    B = pd.DataFrame(base); B.to_csv(out / "rule_baselines.csv", index=False)
    print(B.to_string(index=False))

    # ---- plots ----
    corr = R[FEATURE_COLUMNS].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS))); ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=8)
    ax.set_yticks(range(len(FEATURE_COLUMNS))); ax.set_yticklabels(FEATURE_COLUMNS, fontsize=8)
    ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation between features (train, reliable rows)")
    fig.savefig(out / "feature_correlation.png"); plt.close(fig)
    corr.to_csv(out / "feature_correlation.csv")

    key = ["dist_m", "dist_minus_min", "heading_diff", "ctx_mean_dist", "smooth_dist", "dist_rank"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axs.ravel(), key):
        x = R[f].astype(float)
        hi = np.nanpercentile(x, 99)
        bins = np.linspace(np.nanmin(x), hi, 50)
        ax.hist(x[R.y == 0].clip(upper=hi), bins=bins, density=True, alpha=0.55, color="#d73027", label="wrong arc (y=0)")
        ax.hist(x[R.y == 1].clip(upper=hi), bins=bins, density=True, alpha=0.55, color="#1a9850", label="true arc (y=1)")
        a = U.set_index("feature").auc_sign_adjusted.get(f, np.nan)
        ax.set_title(f"{f}  (AUC {a:.3f})", fontsize=10)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Class-conditional feature distributions (train, reliable rows)")
    fig.savefig(out / "feature_distributions.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(U.feature[::-1], U.auc_sign_adjusted[::-1], color="#2a6fdb")
    ax.axvline(0.5, c="k", lw=0.8); ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("single-feature ROC-AUC (sign-adjusted)"); ax.set_title("Univariate feature strength (train)")
    fig.savefig(out / "feature_univariate_auc.png"); plt.close(fig)


if __name__ == "__main__":
    main()
