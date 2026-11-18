"""Phase 3 — Bayesian hyperparameter optimisation of the selected model (XGBoost).

Protocol (Models & Training Guidelines section 7, docs/DECISIONS.md D13):
  * Search strategy: Optuna TPE (Tree-structured Parzen Estimator) = Bayesian optimisation; seed 42.
  * Objective: MAXIMISE mean 5-fold GroupKFold (by trip) observation-level Top-1 on reliable train points
    = the same primary metric used for model selection. Log loss is recorded for every trial.
  * Data: TRAIN partition only, same strided rows and same folds as scripts/05_train_compare.py,
    so tuned vs untuned is an apples-to-apples comparison. student_test is used only after the search,
    to report tuned vs untuned. evaluator_test is never read.
  * Pruning: a trial whose running fold mean is clearly below the median of earlier trials is stopped
    early (MedianPruner) to save compute; pruned trials are kept in the record.
  * Every trial is stored (results/optuna/xgb_study.db + xgb_trials.csv), so the search is resumable
    and fully auditable.

Search space and why (from the untuned run: validation log loss bottomed out near ~100 rounds at
lr=0.1 and then rose -> mild overfitting; train-val Macro-F1 gap +0.040):
  n_estimators      100-1500 (log)   rounds and learning rate trade off; let the search find the pair
  learning_rate     0.01-0.3 (log)   smaller steps + more rounds usually generalise better
  max_depth         3-10             tree complexity, main overfitting control
  min_child_weight  1-50 (log)       minimum evidence per leaf; larger = smoother
  subsample         0.5-1.0          row sampling per tree (variance reduction)
  colsample_bytree  0.5-1.0          feature sampling per tree (21 correlated features)
  reg_lambda        1e-3-10 (log)    L2 leaf-weight penalty
  reg_alpha         1e-3-10 (log)    L1 leaf-weight penalty
  gamma             0-5              minimum loss reduction to split

Run:   python scripts/07_bayes_opt_xgb.py                 # 40 trials (resumes if interrupted)
       python scripts/07_bayes_opt_xgb.py --trials 25     # fewer trials if short on time
       python scripts/07_bayes_opt_xgb.py --report-only   # rebuild tables/plots from the saved study
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

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from pathsnap.features import ALL_FEATURE_COLUMNS as FEATURE_COLUMNS
from pathsnap.io import load_config
from pathsnap.models import classification_metrics, cuda_available, top1_metrics
from pathsnap.plotting import setup

N_FOLDS = 5


def load_data(cfg, stride):
    proc = Path(cfg["paths"]["processed_dir"])
    C = pd.read_parquet(proc / "candidates_train.parquet")
    P = pd.read_parquet(proc / "points_train.parquet")
    C = C[(C.point_idx % stride == 0)]
    P = P[(P.point_idx % stride == 0)].reset_index(drop=True)
    C = C[~C.label_unreliable.astype(bool)].reset_index(drop=True)
    Cs = pd.read_parquet(proc / "candidates_student_test.parquet")
    Ps = pd.read_parquet(proc / "points_student_test.parquet")
    return C, P, Cs, Ps


def make_params(trial, device, seed):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        tree_method="hist", device=device, eval_metric="logloss", random_state=seed, n_jobs=-1,
    )


def cv_score(params, C, P, folds, trial=None):
    X, y = C[FEATURE_COLUMNS], C.y.to_numpy()
    top1s, lls, f1s, fit_s = [], [], [], []
    for k, (tr, va) in enumerate(folds):
        m = xgb.XGBClassifier(**params)
        t0 = time.perf_counter(); m.fit(X.iloc[tr], y[tr], verbose=False); fit_s.append(time.perf_counter() - t0)
        p = m.predict_proba(X.iloc[va])[:, 1]
        va_trips = set(C.trip_id.iloc[va].unique())
        top1s.append(top1_metrics(C.iloc[va], p, P[P.trip_id.isin(va_trips)])["top1_reliable"])
        cm = classification_metrics(y[va], p); lls.append(cm["log_loss"]); f1s.append(cm["macro_f1"])
        if trial is not None:
            trial.report(float(np.mean(top1s)), k)
            if trial.should_prune():
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
    return dict(top1=float(np.mean(top1s)), top1_std=float(np.std(top1s)), log_loss=float(np.mean(lls)),
                macro_f1=float(np.mean(f1s)), fit_seconds=float(np.sum(fit_s)), fold_top1=top1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout-min", type=float, default=None, help="stop the search after this many minutes")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--stride", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(); seed = int(cfg["seed"])
    stride = args.stride or int(cfg.get("models", {}).get("point_stride", 3))
    out = Path(cfg["paths"]["results_dir"]) / "optuna"; out.mkdir(parents=True, exist_ok=True)
    mdir = Path(cfg["paths"]["models_dir"]); (mdir / "tmp").mkdir(parents=True, exist_ok=True)
    device = "cuda" if cuda_available() else "cpu"
    storage = f"sqlite:///{(out / 'xgb_study.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(study_name="xgb_top1", direction="maximize", storage=storage,
                                load_if_exists=True, sampler=sampler, pruner=pruner)

    C, P, Cs, Ps = load_data(cfg, stride)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(C, C.y, C.trip_id))
    print(f"train rows={len(C):,}  trips={C.trip_id.nunique()}  device={device}  stride={stride}")

    if not args.report_only:
        done = len([t for t in study.trials if t.state.is_finished()])
        todo = max(0, args.trials - done)
        print(f"study has {done} finished trials; running {todo} more")

        def objective(trial):
            params = make_params(trial, device, seed)
            r = cv_score(params, C, P, folds, trial)
            for k in ["top1_std", "log_loss", "macro_f1", "fit_seconds"]:
                trial.set_user_attr(k, r[k])
            print(f"  trial {trial.number:3d}: CV Top-1={r['top1']:.4f} ±{r['top1_std']:.4f} "
                  f"logloss={r['log_loss']:.4f} fit={r['fit_seconds']:.0f}s  {trial.params}", flush=True)
            return r["top1"]

        if todo:
            t0 = time.time()
            study.optimize(objective, n_trials=todo, timeout=args.timeout_min * 60 if args.timeout_min else None,
                           gc_after_trial=True)
            print(f"search time: {(time.time() - t0) / 60:.1f} min")

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials.to_csv(out / "xgb_trials.csv", index=False)
    best = study.best_trial
    print(f"best trial #{best.number}: CV Top-1={best.value:.4f}  params={best.params}")

    # ---- refit tuned model on all strided train rows; compare with the untuned run from script 05 ----
    tuned_params = make_params(optuna.trial.FixedTrial(best.params), device, seed)
    X, y = C[FEATURE_COLUMNS], C.y.to_numpy()
    cv_tuned = cv_score(tuned_params, C, P, folds)  # re-run CV once for full fold-level numbers
    m = xgb.XGBClassifier(**tuned_params)
    t0 = time.perf_counter(); m.fit(X, y, verbose=False); fit_full = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict_proba(Cs[FEATURE_COLUMNS])[:, 1]; t_inf = time.perf_counter() - t0
    st = classification_metrics(Cs.y.to_numpy(), p); st.update(top1_metrics(Cs, p, Ps))
    joblib.dump(m, mdir / "tmp" / "XGBoost_tuned.joblib")
    (mdir / "final").mkdir(exist_ok=True)
    joblib.dump(m, mdir / "final" / "xgboost_tuned.joblib")
    (mdir / "final" / "xgboost_tuned_config.json").write_text(json.dumps(
        {"model": "XGBoost (Bayesian-tuned)", "best_trial": best.number, "cv_top1": best.value,
         "params": {k: v for k, v in tuned_params.items()}, "features": FEATURE_COLUMNS,
         "trained_on": f"train partition, reliable labels, point stride {stride}"}, indent=2, default=str), encoding="utf-8")

    rows = [dict(model="XGBoost tuned", cv_top1=cv_tuned["top1"], cv_top1_std=cv_tuned["top1_std"],
                 cv_macro_f1=cv_tuned["macro_f1"], cv_log_loss=cv_tuned["log_loss"],
                 student_top1=st["top1_all"], student_macro_f1=st["macro_f1"], student_log_loss=st["log_loss"],
                 infer_us_per_row=1e6 * t_inf / len(Cs), train_seconds_full=fit_full)]
    run05 = Path(cfg["paths"]["results_dir"]) / "models" / "runs" / "XGBoost.json"
    if run05.exists():
        r = json.loads(run05.read_text(encoding="utf-8")); f = pd.DataFrame(r["folds"]); s = r["student_test"]
        rows.insert(0, dict(model="XGBoost untuned", cv_top1=f.val_top1_reliable.mean(), cv_top1_std=f.val_top1_reliable.std(ddof=0),
                            cv_macro_f1=f.val_macro_f1.mean(), cv_log_loss=f.val_log_loss.mean(),
                            student_top1=s["top1_all"], student_macro_f1=s["macro_f1"], student_log_loss=s["log_loss"],
                            infer_us_per_row=s["infer_us_per_row"], train_seconds_full=s["train_seconds_full"]))
    cmp = pd.DataFrame(rows); cmp.to_csv(out / "tuned_vs_untuned.csv", index=False)
    pd.DataFrame([best.params]).to_csv(out / "best_params.csv", index=False)

    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    total_min = trials["duration"].dt.total_seconds().sum() / 60 if "duration" in trials else float("nan")
    lines = ["# Bayesian optimisation of XGBoost (auto-generated by scripts/07_bayes_opt_xgb.py)", "",
             "Sampler: Optuna TPE (seed 42, 10 random start-up trials, multivariate). Pruner: MedianPruner.",
             "Objective: maximise mean 5-fold GroupKFold CV observation-level Top-1 (reliable train points).", "",
             f"Trials: {len(study.trials)} total = {n_done} complete + {n_pruned} pruned; total search compute ≈ {total_min:.1f} min.",
             f"Best trial #{best.number}: CV Top-1 = {best.value:.4f}", "", "Best parameters:"]
    lines += [f"- {k} = {v:.5g}" if isinstance(v, float) else f"- {k} = {v}" for k, v in best.params.items()]
    lines += ["", "Tuned vs untuned (same rows, same folds):", "", cmp.round(4).to_markdown(index=False)]
    (out / "BAYES_OPT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---- plots ----
    plt = setup()
    comp = trials[trials.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(comp.number, comp.value, s=18, c="#2a6fdb", label="completed trial")
    ax.plot(comp.number, comp.value.cummax(), c="#1a9850", lw=2, label="best so far")
    if len(cmp) == 2:
        ax.axhline(cmp.cv_top1.iloc[0], c="#d73027", ls="--", label="untuned XGBoost")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 (reliable)"); ax.legend()
    ax.set_title("Bayesian optimisation history (Optuna TPE, GroupKFold objective)")
    fig.savefig(out / "optimisation_history.png"); plt.close(fig)

    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#2a6fdb")
        ax.set_xlabel("relative importance (fANOVA)"); ax.set_title("Which hyperparameters mattered")
        fig.savefig(out / "param_importance.png"); plt.close(fig)
        pd.DataFrame({"param": list(imp), "importance": list(imp.values())}).to_csv(out / "param_importance.csv", index=False)
    except Exception as e:  # needs >1 complete trial
        print("param importance skipped:", e)

    if len(cmp) == 2:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (col, lab) in zip(axs, [("cv_top1", "CV Top-1"), ("student_top1", "student_test Top-1"),
                                        ("student_log_loss", "student_test log loss (lower better)")]):
            ax.bar(cmp.model, cmp[col], color=["#9ecae1", "#1a9850"])
            for i, v in enumerate(cmp[col]):
                ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
            lo = cmp[col].min(); hi = cmp[col].max()
            ax.set_ylim(lo - (hi - lo) * 3 - 1e-3, hi + (hi - lo) * 2 + 1e-3); ax.set_title(lab)
        fig.suptitle("XGBoost: untuned vs Bayesian-tuned")
        fig.savefig(out / "tuned_vs_untuned.png"); plt.close(fig)


if __name__ == "__main__":
    main()
