"""Model zoo + evaluation metrics for the candidate classifier.

Design choices (see docs/DECISIONS.md, D10-D12):
  * Every model outputs P(candidate is the true arc). Probabilities are needed later as HMM emissions,
    so no class re-weighting is applied (it would distort probabilities); the per-point ranking that
    drives Top-1 accuracy is unaffected by the 8 % positive rate.
  * Models that cannot handle NaN or need scaled inputs get SimpleImputer(median) + StandardScaler
    INSIDE their Pipeline, so these transforms are fitted on the training fold only (no leakage).
  * Boosting libraries (HistGB, XGBoost, LightGBM) handle NaN natively, no imputation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _scaled(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", est)])


def _imputed(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", est)])


def cuda_available() -> bool:
    try:
        import torch  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import subprocess

        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


def model_specs(seed: int, use_gpu: bool) -> dict:
    """name -> dict(family, build(), params, max_train_rows, notes). Initial (untuned) configurations."""
    specs = {}

    p = dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)
    specs["LogisticRegression"] = dict(family="Linear", params=p, max_train_rows=None,
                                       build=lambda: _scaled(LogisticRegression(**p, random_state=seed)),
                                       notes="linear decision boundary on standardised features")

    p2 = dict(var_smoothing=1e-9)
    specs["GaussianNB"] = dict(family="Probabilistic", params=p2, max_train_rows=None,
                               build=lambda: _scaled(GaussianNB(**p2)),
                               notes="assumes conditionally independent Gaussian features")

    p3 = dict(n_neighbors=50, weights="distance", algorithm="auto", n_jobs=-1)
    specs["KNN"] = dict(family="Instance-based", params=p3, max_train_rows=100_000,
                        build=lambda: _scaled(KNeighborsClassifier(**p3)),
                        notes="inference cost grows with training size -> trained on a 100k-row random subsample")

    p4 = dict(n_estimators=200, max_depth=None, min_samples_leaf=5, max_features="sqrt", n_jobs=-1)
    specs["RandomForest"] = dict(family="Ensemble (bagging)", params=p4, max_train_rows=None,
                                 build=lambda: _imputed(RandomForestClassifier(**p4, random_state=seed)),
                                 notes="bootstrap + random feature subsets, averaged trees")

    p5 = dict(n_estimators=200, max_depth=None, min_samples_leaf=5, max_features="sqrt", n_jobs=-1)
    specs["ExtraTrees"] = dict(family="Ensemble (randomised trees)", params=p5, max_train_rows=None,
                               build=lambda: _imputed(ExtraTreesClassifier(**p5, random_state=seed)),
                               notes="random split thresholds -> lower variance than RF")

    p6 = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0,
              early_stopping=False)
    specs["HistGradientBoosting"] = dict(family="Boosting", params=p6, max_train_rows=None,
                                         build=lambda: HistGradientBoostingClassifier(**p6, random_state=seed),
                                         notes="sklearn histogram boosting, native NaN handling")

    try:
        import xgboost as xgb

        p7 = dict(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=1.0, reg_lambda=1.0, tree_method="hist",
                  device="cuda" if use_gpu else "cpu", eval_metric="logloss")
        specs["XGBoost"] = dict(family="Boosting", params=p7, max_train_rows=None,
                                build=lambda: xgb.XGBClassifier(**p7, random_state=seed, n_jobs=-1),
                                notes=f"gradient boosting, {'GPU (CUDA)' if use_gpu else 'CPU'} hist trees",
                                history="xgb")
    except ImportError:
        pass

    try:
        import lightgbm as lgb

        p8 = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0)
        specs["LightGBM"] = dict(family="Boosting", params=p8, max_train_rows=None,
                                 build=lambda: lgb.LGBMClassifier(**p8, random_state=seed, n_jobs=-1, verbose=-1),
                                 notes="leaf-wise gradient boosting", history="lgb")
    except ImportError:
        pass

    p9 = dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, batch_size=1024,
              learning_rate_init=1e-3, epochs=25)
    specs["MLP"] = dict(family="Neural network", params=p9, max_train_rows=None,
                        build=lambda: _scaled(MLPClassifier(hidden_layer_sizes=p9["hidden_layer_sizes"],
                                                            activation="relu", alpha=p9["alpha"],
                                                            batch_size=p9["batch_size"],
                                                            learning_rate_init=p9["learning_rate_init"],
                                                            random_state=seed)),
                        notes="2 hidden layers, Adam, trained epoch by epoch to record train/val loss",
                        history="mlp")
    return specs


# ------------------------------------------------------------------ fitting with histories
def fit_model(name: str, spec: dict, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None):
    """Fit and return (model, history_df or None, train_seconds)."""
    model = spec["build"]()
    hist = None
    t0 = time.perf_counter()
    kind = spec.get("history")
    if kind == "xgb" and X_val is not None:
        model.fit(X, y, eval_set=[(X, y), (X_val, y_val)], verbose=False)
        r = model.evals_result()
        hist = pd.DataFrame({"iteration": np.arange(1, len(r["validation_0"]["logloss"]) + 1),
                             "train_logloss": r["validation_0"]["logloss"], "val_logloss": r["validation_1"]["logloss"]})
    elif kind == "lgb" and X_val is not None:
        model.fit(X, y, eval_set=[(X, y), (X_val, y_val)], eval_names=["train", "val"], eval_metric="binary_logloss")
        r = model.evals_result_
        hist = pd.DataFrame({"iteration": np.arange(1, len(r["train"]["binary_logloss"]) + 1),
                             "train_logloss": r["train"]["binary_logloss"], "val_logloss": r["val"]["binary_logloss"]})
    elif kind == "mlp":
        # Manual epochs: impute+scale fitted once on the training fold, then partial_fit per epoch.
        prep = Pipeline(model.steps[:-1]).fit(X)
        Xt = prep.transform(X)
        Xv = prep.transform(X_val) if X_val is not None else None
        mlp = model.steps[-1][1]
        rng = np.random.default_rng(0)
        rows = []
        for ep in range(spec["params"]["epochs"]):
            idx = rng.permutation(len(Xt))
            mlp.partial_fit(Xt[idx], y[idx], classes=np.array([0, 1]))
            row = {"iteration": ep + 1, "train_logloss": log_loss(y, mlp.predict_proba(Xt)[:, 1], labels=[0, 1])}
            if Xv is not None:
                row["val_logloss"] = log_loss(y_val, mlp.predict_proba(Xv)[:, 1], labels=[0, 1])
            rows.append(row)
        hist = pd.DataFrame(rows)
    else:
        model.fit(X, y)
    return model, hist, time.perf_counter() - t0


def predict_proba(model, X) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0


# ------------------------------------------------------------------ metrics
def point_key(df: pd.DataFrame) -> np.ndarray:
    return df["trip_id"].astype(np.int64).to_numpy() * 1_000_000 + df["point_idx"].to_numpy().astype(np.int64)


def top1_predictions(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Per point: the candidate with the highest probability. Returns key, trip_id, point_idx, arc_id, y, p."""
    d = pd.DataFrame({"key": point_key(df), "trip_id": df["trip_id"].to_numpy(), "point_idx": df["point_idx"].to_numpy(),
                      "arc_id": df["arc_id"].to_numpy(), "y": df["y"].to_numpy(), "p": p})
    d = d.sort_values(["key", "p"], ascending=[True, False], kind="mergesort")
    return d.drop_duplicates("key")


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(
        accuracy=accuracy_score(y, yhat),
        precision_pos=precision_score(y, yhat, zero_division=0),
        recall_pos=recall_score(y, yhat, zero_division=0),
        f1_pos=f1_score(y, yhat, zero_division=0),
        precision_macro=precision_score(y, yhat, average="macro", zero_division=0),
        recall_macro=recall_score(y, yhat, average="macro", zero_division=0),
        macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
        log_loss=log_loss(y, pc, labels=[0, 1]),
        roc_auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
    )


def top1_metrics(df: pd.DataFrame, p: np.ndarray, points: pd.DataFrame) -> dict:
    """Observation-level Top-1 over ALL points in `points` (points with no candidate / missed ref = wrong).

    top1_reliable excludes points whose reference label is unreliable (> 100 m from road, parked off-network).
    """
    best = top1_predictions(df, p)
    m = points[["trip_id", "point_idx", "label_unreliable", "ref_in_cands"]].copy()
    m["key"] = point_key(m)
    m = m.merge(best[["key", "y"]], on="key", how="left")
    m["y"] = m["y"].fillna(0)
    rel = ~m["label_unreliable"].astype(bool)
    return dict(top1_all=float(m["y"].mean()), top1_reliable=float(m["y"][rel].mean()),
                candidate_miss_rate=float(1 - m["ref_in_cands"].mean()), n_points=int(len(m)))


def confusion(y, p, thr=0.5):
    return confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
