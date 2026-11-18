"""Feature ablation: do the v2 direction features improve the classifier? (TRAIN CV ONLY)

Compares the same model (untuned XGBoost from scripts/05) on the same rows and the same GroupKFold
folds, with v1 features (21) vs v1 + v2 direction features (26). student_test is reported alongside.
evaluator_test is NOT used - a feature set must be accepted or rejected on train CV evidence alone.

Outputs: results/ablation/feature_ablation.csv, feature_ablation.png, v2_feature_importance.csv
Run:     python scripts/08_feature_ablation.py
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.features import EXTENDED_COLUMNS, FEATURE_COLUMNS
from pathsnap.io import load_config
from pathsnap.models import classification_metrics, cuda_available, model_specs, top1_metrics
from pathsnap.plotting import setup


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--model", default="XGBoost")
    args = ap.parse_args()
    cfg = load_config(); seed = int(cfg["seed"])
    stride = args.stride or int(cfg.get("models", {}).get("point_stride", 3))
    proc = Path(cfg["paths"]["processed_dir"]); out = Path(cfg["paths"]["results_dir"]) / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    C = pd.read_parquet(proc / "candidates_train.parquet"); P = pd.read_parquet(proc / "points_train.parquet")
    missing = [c for c in EXTENDED_COLUMNS if c not in C.columns]
    if missing:
        sys.exit(f"{missing} not in candidates_train.parquet - re-run scripts/04_build_features.py first")
    C = C[C.point_idx % stride == 0]; P = P[P.point_idx % stride == 0].reset_index(drop=True)
    C = C[~C.label_unreliable.astype(bool)].reset_index(drop=True)
    Cs = pd.read_parquet(proc / "candidates_student_test.parquet"); Ps = pd.read_parquet(proc / "points_student_test.parquet")
    spec = model_specs(seed, cuda_available())[args.model]
    folds = list(GroupKFold(n_splits=5).split(C, C.y, C.trip_id))
    rows = []
    for name, cols in [("v1 (21 features)", FEATURE_COLUMNS), ("v1 + v2 direction (26 features)", FEATURE_COLUMNS + EXTENDED_COLUMNS)]:
        X, y = C[cols], C.y.to_numpy()
        top1, ll, f1 = [], [], []
        for k, (tr, va) in enumerate(folds):
            m = spec["build"](); t0 = time.time(); m.fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            vt = set(C.trip_id.iloc[va].unique())
            top1.append(top1_metrics(C.iloc[va], p, P[P.trip_id.isin(vt)])["top1_reliable"])
            cm = classification_metrics(y[va], p); ll.append(cm["log_loss"]); f1.append(cm["macro_f1"])
            print(f"  {name} fold {k}: Top-1={top1[-1]:.4f} logloss={ll[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        m = spec["build"](); m.fit(X, y); ps = m.predict_proba(Cs[cols])[:, 1]
        st = top1_metrics(Cs, ps, Ps); stc = classification_metrics(Cs.y.to_numpy(), ps)
        rows.append(dict(feature_set=name, model=args.model, cv_top1=np.mean(top1), cv_top1_std=np.std(top1),
                         cv_macro_f1=np.mean(f1), cv_log_loss=np.mean(ll), fold_top1=" ".join(f"{v:.4f}" for v in top1),
                         student_top1=st["top1_all"], student_macro_f1=stc["macro_f1"], student_log_loss=stc["log_loss"]))
        if cols is not FEATURE_COLUMNS and hasattr(m, "feature_importances_"):
            pd.DataFrame({"feature": cols, "importance": m.feature_importances_}).sort_values("importance", ascending=False) \
                .to_csv(out / "v2_feature_importance.csv", index=False)
    R = pd.DataFrame(rows); R.to_csv(out / "feature_ablation.csv", index=False)
    d = R.cv_top1.iloc[1] - R.cv_top1.iloc[0]
    wins = sum(float(b) > float(a) for a, b in zip(R.fold_top1.iloc[0].split(), R.fold_top1.iloc[1].split()))
    print(R.drop(columns="fold_top1").round(4).to_string(index=False))
    print(f"\nCV Top-1 change: {d:+.4f}; v2 better on {wins}/5 folds")
    plt = setup(); fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, lab in [(axs[0], "cv_top1", "CV Top-1 (train, GroupKFold)"), (axs[1], "student_top1", "student_test Top-1")]:
        ax.bar(["v1", "v1 + v2"], R[col], color=["#9ecae1", "#1a9850"])
        for i, v in enumerate(R[col]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
        ax.set_ylim(R[col].min() - 0.03, min(1, R[col].max() + 0.02)); ax.set_title(lab)
    fig.suptitle(f"Feature ablation ({args.model}, same rows and folds): CV change {d:+.4f}, better on {wins}/5 folds")
    fig.savefig(out / "feature_ablation.png"); plt.close(fig)


if __name__ == "__main__":
    main()
