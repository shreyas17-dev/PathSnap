"""Phase 2 — Train and compare >= 5 candidate classifiers (9 models, 7 algorithm families).

Protocol (docs/DECISIONS.md D10-D12):
  * Training data: train partition only, reliable-label rows (unreliable = vehicle parked > 100 m off-road).
  * Group-aware 5-fold CV: GroupKFold by trip_id, so no trip appears in both training and validation folds.
  * Point stride: every `point_stride`-th GPS point (with ALL its candidates) is used for CV and fitting.
    Consecutive 1 Hz points are near-duplicates, so this cuts compute ~3x with little information loss.
    The same rows are used for every model (fair comparison).
  * Each model is then refit on the full strided train set and evaluated on ALL student_test points.
  * Primary selection metric: CV observation-level Top-1 accuracy (reliable points) — it is exactly the
    task objective (pick the right arc per GPS point). Log loss is the tie-breaker because the
    probabilities become HMM emissions. Binary accuracy/precision/recall/F1 are reported as required,
    but they are threshold-dependent and dominated by the 92 % negative class.

Outputs: results/models/  (CSV tables, PNG plots, per-model configs, training histories, train_log.txt)
         models/tmp/<model>.joblib  (git-ignored; the selected model is exported in a later phase)

Run:   python scripts/05_train_compare.py                 # all models (resumes: finished models are skipped)
       python scripts/05_train_compare.py --models XGBoost LightGBM --force
       python scripts/05_train_compare.py --report-only   # rebuild tables/plots from saved runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.features import ALL_FEATURE_COLUMNS as FEATURE_COLUMNS
from pathsnap.io import list_trips, load_config, load_trip, split_dir
from pathsnap.models import (classification_metrics, confusion, cuda_available, fit_model, model_specs,
                             predict_proba, top1_metrics, top1_predictions)
from pathsnap.plotting import setup

N_FOLDS = 5
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n"); self.f.flush()


def load_data(cfg, stride):
    proc = Path(cfg["paths"]["processed_dir"])
    C = pd.read_parquet(proc / "candidates_train.parquet")
    P = pd.read_parquet(proc / "points_train.parquet")
    C = C[(C.point_idx % stride == 0)]
    P = P[(P.point_idx % stride == 0)].reset_index(drop=True)
    C_train = C[~C.label_unreliable.astype(bool)].reset_index(drop=True)
    Cs = pd.read_parquet(proc / "candidates_student_test.parquet")
    Ps = pd.read_parquet(proc / "points_student_test.parquet")
    return C_train, P, Cs, Ps


def continuity(pred: pd.DataFrame, cfg, split: str, arc_col: str) -> dict:
    """Fraction of consecutive (gap-free) points whose predicted arcs are equal, directly connected,
    or connected through one intermediate arc (1 Hz sampling can skip a short arc)."""
    ok_all, n_all = 0, 0
    for tid, d in pred.groupby("trip_id"):
        t = load_trip(split_dir(cfg, split) / tid, load_route=False)
        u, v = t.arcs.u.to_numpy(), t.arcs.v.to_numpy()
        out_nodes = {}
        for a, b in zip(u, v):
            out_nodes.setdefault(a, set()).add(b)
        d = d.sort_values("point_idx")
        a = d[arc_col].to_numpy(); gap = d["gap_before"].to_numpy()
        for i in range(1, len(a)):
            if gap[i]:
                continue
            p, q = a[i - 1], a[i]
            n_all += 1
            if p == q or v[p] == u[q] or u[q] in out_nodes.get(v[p], ()):
                ok_all += 1
    return {"route_continuity": ok_all / max(n_all, 1)}


def run_model(name, spec, C_train, P_train, Cs, Ps, cfg, out, log, seed):
    X, y, groups = C_train[FEATURE_COLUMNS], C_train.y.to_numpy(), C_train.trip_id.to_numpy()
    rng = np.random.default_rng(seed)
    fold_rows, hist_saved = [], None
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups)):
        tr_idx = tr
        if spec["max_train_rows"] and len(tr) > spec["max_train_rows"]:
            tr_idx = np.sort(rng.choice(tr, spec["max_train_rows"], replace=False))
        Xtr, ytr = X.iloc[tr_idx], y[tr_idx]
        Xva, yva = X.iloc[va], y[va]
        model, hist, t_fit = fit_model(name, spec, Xtr, ytr, Xva, yva)
        p_va, t_inf = predict_proba(model, Xva)
        # training-fold score on a sample (overfitting check)
        # sample whole POINTS (all their candidates), never random rows: a row sample drops true candidates
        tr_keys = (C_train.trip_id.iloc[tr_idx].astype(np.int64).to_numpy() * 1_000_000
                   + C_train.point_idx.iloc[tr_idx].to_numpy())
        uk = np.unique(tr_keys)
        keep = rng.choice(uk, min(len(uk), 20_000), replace=False)
        smp = tr_idx[np.isin(tr_keys, keep)]
        p_tr, _ = predict_proba(model, X.iloc[smp])
        va_trips = set(C_train.trip_id.iloc[va].unique())
        pk = P_train.trip_id.astype(np.int64).to_numpy() * 1_000_000 + P_train.point_idx.to_numpy()
        P_smp = P_train[np.isin(pk, keep)]
        row = dict(model=name, fold=k, n_train_rows=len(tr_idx), n_val_rows=len(va), train_seconds=t_fit,
                   infer_seconds=t_inf, infer_us_per_row=1e6 * t_inf / len(va))
        row.update({f"val_{m}": v for m, v in classification_metrics(yva, p_va).items()})
        row.update({f"val_{m}": v for m, v in top1_metrics(C_train.iloc[va], p_va, P_train[P_train.trip_id.isin(va_trips)]).items()})
        row.update({f"train_{m}": v for m, v in classification_metrics(y[smp], p_tr).items()})
        row.update({f"train_{m}": v for m, v in top1_metrics(C_train.iloc[smp], p_tr, P_smp).items()
                    if m.startswith("top1")})
        fold_rows.append(row)
        if hist is not None and hist_saved is None:
            hist.to_csv(out / "histories" / f"{name}_fold0.csv", index=False); hist_saved = True
        log(f"  {name} fold {k}: val Top-1(rel)={row['val_top1_reliable']:.4f} macroF1={row['val_macro_f1']:.4f} "
            f"logloss={row['val_log_loss']:.4f} | train Top-1={row['train_top1_reliable']:.4f} | fit {t_fit:.1f}s")

    # ---- refit on all strided train rows, evaluate on student_test (all points) ----
    idx = np.arange(len(X))
    if spec["max_train_rows"] and len(idx) > spec["max_train_rows"]:
        idx = np.sort(rng.choice(idx, spec["max_train_rows"], replace=False))
    model, hist_full, t_fit = fit_model(name, spec, X.iloc[idx], y[idx], None, None)
    p_st, t_inf = predict_proba(model, Cs[FEATURE_COLUMNS])
    st = dict(model=name, train_seconds_full=t_fit, infer_seconds=t_inf, infer_us_per_row=1e6 * t_inf / len(Cs))
    st.update(classification_metrics(Cs.y.to_numpy(), p_st))
    st.update(top1_metrics(Cs, p_st, Ps))
    best = top1_predictions(Cs, p_st).merge(
        Cs[["trip_id", "point_idx", "gap_before"]].drop_duplicates(["trip_id", "point_idx"]), on=["trip_id", "point_idx"])
    st.update(continuity(best, cfg, "student_test", "arc_id"))
    cm = confusion(Cs.y.to_numpy(), p_st)
    st.update(dict(tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1])))
    np.save(out / "predictions" / f"{name}_student_test_proba.npy", p_st.astype(np.float32))

    # feature importance / coefficients where the model exposes them
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    imp = None
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_.ravel())
    if imp is not None:
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": imp}).sort_values("importance", ascending=False) \
            .to_csv(out / "feature_importance" / f"{name}.csv", index=False)

    Path(cfg["paths"]["models_dir"], "tmp").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(cfg["paths"]["models_dir"]) / "tmp" / f"{name}.joblib")
    cfg_out = dict(model=name, family=spec["family"], params=spec["params"], max_train_rows=spec["max_train_rows"],
                   notes=spec["notes"], seed=seed, features=FEATURE_COLUMNS)
    (out / "configs" / f"{name}.json").write_text(json.dumps(cfg_out, indent=2, default=str), encoding="utf-8")
    (out / "runs" / f"{name}.json").write_text(json.dumps({"folds": fold_rows, "student_test": st}, default=float),
                                               encoding="utf-8")
    log(f"  {name} student_test: Top-1(all)={st['top1_all']:.4f} macroF1={st['macro_f1']:.4f} "
        f"logloss={st['log_loss']:.4f} continuity={st['route_continuity']:.4f}")


def build_report(out: Path, specs: dict, cfg, log):
    plt = setup()
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "runs").glob("*.json"))}
    if not runs:
        return
    folds = pd.DataFrame([r for v in runs.values() for r in v["folds"]])
    st = pd.DataFrame([v["student_test"] for v in runs.values()]).set_index("model")
    folds.to_csv(out / "cv_fold_metrics.csv", index=False)
    st.to_csv(out / "student_test_metrics.csv")
    cols = ["val_top1_reliable", "val_top1_all", "val_macro_f1", "val_log_loss", "val_roc_auc", "val_accuracy",
            "val_precision_macro", "val_recall_macro", "train_top1_reliable", "train_macro_f1", "train_log_loss", "train_seconds",
            "infer_us_per_row"]
    agg = folds.groupby("model")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(out / "cv_summary.csv")

    fam = {n: specs[n]["family"] if n in specs else json.loads((out / "configs" / f"{n}.json").read_text())["family"]
           for n in agg.index}
    keycfg = {n: json.loads((out / "configs" / f"{n}.json").read_text(encoding="utf-8"))["params"] for n in agg.index}
    table = pd.DataFrame({
        "Model": agg.index,
        "Algorithm family": [fam[n] for n in agg.index],
        "Key configuration": [", ".join(f"{k}={v}" for k, v in list(keycfg[n].items())[:4]) for n in agg.index],
        "CV Top-1 (reliable)": [f"{agg.loc[n, 'val_top1_reliable_mean']:.4f} ± {agg.loc[n, 'val_top1_reliable_std']:.4f}" for n in agg.index],
        "CV Macro F1": [f"{agg.loc[n, 'val_macro_f1_mean']:.4f} ± {agg.loc[n, 'val_macro_f1_std']:.4f}" for n in agg.index],
        "CV Log Loss": [f"{agg.loc[n, 'val_log_loss_mean']:.4f}" for n in agg.index],
        "Student-test Top-1": [f"{st.loc[n, 'top1_all']:.4f}" for n in agg.index],
        "Student-test Macro F1": [f"{st.loc[n, 'macro_f1']:.4f}" for n in agg.index],
        "Student-test Precision (macro)": [f"{st.loc[n, 'precision_macro']:.4f}" for n in agg.index],
        "Student-test Recall (macro)": [f"{st.loc[n, 'recall_macro']:.4f}" for n in agg.index],
        "Student-test Log Loss": [f"{st.loc[n, 'log_loss']:.4f}" for n in agg.index],
        "Route continuity (classifier only)": [f"{st.loc[n, 'route_continuity']:.4f}" for n in agg.index],
        "Train time (s, full)": [f"{st.loc[n, 'train_seconds_full']:.1f}" for n in agg.index],
        "Inference (µs/row)": [f"{st.loc[n, 'infer_us_per_row']:.2f}" for n in agg.index],
    }).sort_values("CV Top-1 (reliable)", ascending=False)
    table.to_csv(out / "model_comparison_table.csv", index=False)

    order = agg.sort_values("val_top1_reliable_mean", ascending=False).index.tolist()
    best = order[0]
    # selection record
    lines = ["# Model selection (auto-generated by scripts/05_train_compare.py)", "",
             "Primary metric: mean GroupKFold CV observation-level Top-1 accuracy on reliable points (train partition).",
             "Tie-breaker: CV log loss (probability quality matters for HMM emissions). Student-test is reported, not used to tune.", "",
             "| rank | model | CV Top-1 | std | CV log loss | train-val Macro-F1 gap | student-test Top-1 | µs/row |", "|---|---|---|---|---|---|---|---|"]
    for i, n in enumerate(order):
        gap = agg.loc[n, "train_macro_f1_mean"] - agg.loc[n, "val_macro_f1_mean"]
        lines.append(f"| {i + 1} | {n} | {agg.loc[n, 'val_top1_reliable_mean']:.4f} | {agg.loc[n, 'val_top1_reliable_std']:.4f} | "
                     f"{agg.loc[n, 'val_log_loss_mean']:.4f} | {gap:+.4f} | {st.loc[n, 'top1_all']:.4f} | {st.loc[n, 'infer_us_per_row']:.2f} |")
    if len(order) > 1:
        second = order[1]
        d = agg.loc[best, "val_top1_reliable_mean"] - agg.loc[second, "val_top1_reliable_mean"]
        s = np.hypot(agg.loc[best, "val_top1_reliable_std"], agg.loc[second, "val_top1_reliable_std"])
        lines += ["", f"Leader: **{best}**. Margin over {second}: {d:+.4f} Top-1 (combined fold std {s:.4f}).",
                  "If the margin is within one combined std, prefer the model with lower log loss / faster inference."]
    (out / "MODEL_SELECTION.md").write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines))

    # ---- plots ----
    def bar(metric, title, fname, ylabel, lower_better=False):
        m = agg[f"{metric}_mean"].reindex(order); s = agg[f"{metric}_std"].reindex(order)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(order, m, yerr=s, capsize=4, color=["#1a9850" if n == best else "#2a6fdb" for n in order])
        lo = max(0, (m - s).min() - 0.02) if not lower_better else 0
        ax.set_ylim(lo, (m + s).max() * 1.01 if lower_better else min(1.0, (m + s).max() + 0.01))
        for i, v in enumerate(m):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title); plt.xticks(rotation=25, ha="right")
        fig.savefig(out / fname); plt.close(fig)

    bar("val_top1_reliable", "5-fold GroupKFold CV: observation-level Top-1 accuracy (train, reliable points)",
        "cv_top1_comparison.png", "Top-1 accuracy")
    bar("val_macro_f1", "5-fold GroupKFold CV: Macro F1 (threshold 0.5)", "cv_macro_f1_comparison.png", "Macro F1")
    bar("val_log_loss", "5-fold GroupKFold CV: log loss (lower is better)", "cv_logloss_comparison.png", "log loss", True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.38; xs = np.arange(len(order))
    ax.bar(xs - w / 2, st.loc[order, "top1_all"], w, label="Top-1 accuracy", color="#1a9850")
    ax.bar(xs + w / 2, st.loc[order, "macro_f1"], w, label="Macro F1", color="#2a6fdb")
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=25, ha="right"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Student-test (10 trips, all points): Top-1 accuracy and Macro F1"); ax.legend()
    fig.savefig(out / "student_test_top1_f1.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, met, lab in [(axs[0], "macro_f1", "Macro F1 (higher = better)"), (axs[1], "log_loss", "log loss (lower = better)")]:
        ax.bar(xs - w / 2, agg.loc[order, f"train_{met}_mean"], w, label="training folds", color="#f4a582")
        ax.bar(xs + w / 2, agg.loc[order, f"val_{met}_mean"], w, label="validation folds", color="#2a6fdb")
        ax.set_xticks(xs); ax.set_xticklabels(order, rotation=25, ha="right"); ax.set_title(lab); ax.legend()
        if met == "log_loss":
            ax.set_yscale("log")
    fig.suptitle("Over/under-fitting check: training vs validation folds (5-fold GroupKFold)")
    fig.savefig(out / "overfitting_train_vs_val.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for n in order:
        ax.scatter(st.loc[n, "infer_us_per_row"], agg.loc[n, "val_top1_reliable_mean"], s=60,
                   c="#1a9850" if n == best else "#2a6fdb")
        ax.annotate(n, (st.loc[n, "infer_us_per_row"], agg.loc[n, "val_top1_reliable_mean"]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("inference time (µs per candidate row, log)"); ax.set_ylabel("CV Top-1")
    ax.set_title("Accuracy vs inference cost"); fig.savefig(out / "accuracy_vs_speed.png"); plt.close(fig)

    n = len(order); cols_ = 3; rows_ = int(np.ceil(n / cols_))
    fig, axs = plt.subplots(rows_, cols_, figsize=(12, 3.6 * rows_))
    for ax, m in zip(np.ravel(axs), order):
        cm = np.array([[st.loc[m, "tn"], st.loc[m, "fp"]], [st.loc[m, "fn"], st.loc[m, "tp"]]])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(m, fontsize=10)
    for ax in np.ravel(axs)[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on student-test candidate rows (threshold 0.5, row-normalised)")
    fig.savefig(out / "confusion_matrices.png"); plt.close(fig)

    hists = sorted((out / "histories").glob("*_fold0.csv"))
    if hists:
        fig, axs = plt.subplots(1, len(hists), figsize=(5 * len(hists), 4))
        for ax, h in zip(np.atleast_1d(axs), hists):
            d = pd.read_csv(h)
            ax.plot(d.iteration, d.train_logloss, label="training fold"); ax.plot(d.iteration, d.val_logloss, label="validation fold")
            ax.set_xlabel("boosting round" if "MLP" not in h.stem else "epoch"); ax.set_ylabel("log loss")
            ax.set_title(h.stem.replace("_fold0", "") + " (CV fold 0)"); ax.legend()
        fig.savefig(out / "training_curves.png"); plt.close(fig)

    imps = sorted((out / "feature_importance").glob("*.csv"))
    if imps:
        fig, axs = plt.subplots(1, len(imps), figsize=(4.2 * len(imps), 6), sharey=False)
        for ax, f in zip(np.atleast_1d(axs), imps):
            d = pd.read_csv(f).set_index("feature").reindex(FEATURE_COLUMNS)
            v = d.importance / d.importance.sum()
            ax.barh(FEATURE_COLUMNS[::-1], v.values[::-1], color="#2a6fdb"); ax.set_title(f.stem, fontsize=10)
            ax.tick_params(axis="y", labelsize=7)
        fig.suptitle("Normalised feature importance (trees) / |coefficient| (logistic regression)")
        fig.savefig(out / "feature_importance.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="re-run models that already have results")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--stride", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed = int(cfg["seed"]); np.random.seed(seed)
    stride = args.stride or int(cfg.get("models", {}).get("point_stride", 3))
    out = Path(cfg["paths"]["results_dir"]) / "models"
    for sub in ["runs", "configs", "histories", "feature_importance", "predictions"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    log = Logger(out / "train_log.txt")
    use_gpu = cuda_available()
    specs = model_specs(seed, use_gpu)

    if not args.report_only:
        log(f"=== run start | stride={stride} folds={N_FOLDS} gpu={use_gpu} models={list(specs)}")
        C_train, P_train, Cs, Ps = load_data(cfg, stride)
        log(f"train rows={len(C_train):,} (reliable, strided) points={len(P_train):,} trips={C_train.trip_id.nunique()} "
            f"| student_test rows={len(Cs):,} points={len(Ps):,}")
        for name in (args.models or list(specs)):
            if name not in specs:
                log(f"skip unknown model {name}"); continue
            if (out / "runs" / f"{name}.json").exists() and not args.force:
                log(f"{name}: already done (use --force to re-run)"); continue
            log(f"--- {name} [{specs[name]['family']}] params={specs[name]['params']}")
            t0 = time.time()
            run_model(name, specs[name], C_train, P_train, Cs, Ps, cfg, out, log, seed)
            log(f"--- {name} done in {time.time() - t0:.0f}s")
    build_report(out, specs, cfg, log)


if __name__ == "__main__":
    main()
