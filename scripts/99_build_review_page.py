"""Builds results/REVIEW.html: one offline page showing every evidence table, plot and map.

Open it in a browser during the review / viva. Re-run after any script to refresh it.
Sections only appear when their result files exist, so the page grows with each project phase.

Run:  python scripts/99_build_review_page.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: avoid UnicodeEncodeError on ±, ≤, →
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from pathsnap.io import load_config

SECTIONS = [
    ("1. Dataset inventory & quality (train + student_test)", "eda", [
        ("EDA_SUMMARY.md", "Auto-generated findings"),
        ("split_summary.csv", "Partition summary"),
        ("trip_locations.png", "Trips span 25 UTM zones → per-trip projection"),
        ("projection_error.png", "Projection check vs exact geodesic"),
        ("points_per_trip.png", "Trajectory length"),
        ("sampling_interval_hist.png", "Sampling intervals → gap threshold"),
        ("sampling_intervals.csv", "Steps above each threshold"),
        ("speed_hist.png", "Implied speed / GPS jumps"),
        ("trajectory_grid.png", "GPS trajectory plots"),
        ("network_trip_00000010.png", "Road network + route + GPS"),
        ("network_size.png", "Network size"),
        ("network_connectivity.png", "Network connectivity"),
    ]),
    ("2. Reference labels per GPS point (train + student_test)", "preprocessing", [
        ("label_alignment_summary.csv", "Monotonic vs naive labelling"),
        ("label_naive_vs_monotonic.png", "Why order-preserving alignment"),
        ("label_distance_hist.png", "GPS-to-reference distance (train)"),
        ("label_distance_percentiles.csv", "Distance percentiles (train)"),
    ]),
    ("3. Candidate generation (radius chosen on train only)", "preprocessing", [
        ("radius_decision.txt", "Decision"),
        ("radius_study.png", "Miss rate vs candidate-set size"),
        ("radius_study.csv", "Radius grid"),
        ("rank_study.csv", "Rank of the true arc among candidates"),
        ("twin_tie_study.csv", "Two-way roads: distance ties"),
        ("candidate_summary.csv", "Final candidate sets per split"),
    ]),
    ("4. Feature engineering (train)", "preprocessing", [
        ("rule_baselines.csv", "Non-learned baselines to beat"),
        ("feature_univariate_auc.png", "Single-feature strength"),
        ("feature_distributions.png", "Class-conditional distributions"),
        ("feature_correlation.png", "Redundancy check"),
        ("feature_summary_train.csv", "Feature statistics"),
    ]),
    ("5. Model experimentation — 9 models, GroupKFold(5) by trip on train; student_test for comparison", "models", [
        ("MODEL_SELECTION.md", "Ranking by the primary metric"),
        ("model_comparison_table.csv", "Report table 3.1"),
        ("cv_top1_comparison.png", "CV Top-1 accuracy (primary metric)"),
        ("cv_macro_f1_comparison.png", "CV Macro F1"),
        ("cv_logloss_comparison.png", "CV log loss (probability quality)"),
        ("student_test_top1_f1.png", "Student-test Top-1 and Macro F1"),
        ("overfitting_train_vs_val.png", "Over/under-fitting check"),
        ("training_curves.png", "Epoch-wise training histories (boosting rounds / MLP epochs)"),
        ("accuracy_vs_speed.png", "Accuracy vs inference cost"),
        ("confusion_matrices.png", "Confusion matrices (student-test)"),
        ("feature_importance.png", "What the models learned from the features"),
        ("cv_summary.csv", "CV mean/std of every metric"),
        ("student_test_metrics.csv", "Student-test metrics"),
    ]),
    ("4b. Feature ablation: v2 direction features (train CV only)", "ablation", [
        ("feature_ablation.csv", "v1 (21) vs v1+v2 (26) features, same model, rows and folds"),
        ("feature_ablation.png", "CV and student_test Top-1"),
        ("v2_feature_importance.csv", "Importance with the new features"),
    ]),
    ("6. Bayesian hyperparameter optimisation of XGBoost (Optuna TPE, GroupKFold on train)", "optuna", [
        ("BAYES_OPT_SUMMARY.md", "Search record, best parameters, tuned vs untuned"),
        ("optimisation_history.png", "Optimisation history"),
        ("param_importance.png", "Which hyperparameters mattered"),
        ("tuned_vs_untuned.png", "Tuned vs untuned"),
        ("tuned_vs_untuned.csv", "Tuned vs untuned (table)"),
        ("xgb_trials.csv", "Every trial (first 40 rows)"),
    ]),
    ("7. HMM + Viterbi route reconstruction (decoder tuned on train OOF; student_test reported)", "hmm", [
        ("classifier_vs_hmm.csv", "Classifier-only vs classifier + HMM"),
        ("classifier_vs_hmm.png", "Top-1, route continuity, route F1"),
        ("hmm_grid.png", "Decoder grid (beta x emission weight) on train OOF"),
        ("hmm_grid_train_oof.csv", "Decoder grid (table)"),
        ("per_trip_gain_student_test.png", "Where the HMM helped / did not help"),
        ("per_trip_student_test.csv", "Per-trip comparison (student_test)"),
        ("per_trip_train_oof.csv", "Per-trip comparison (train OOF)"),
    ]),
    ("9. FINAL: full pipeline (tuned XGBoost + HMM) on evaluator_test", "final_evaluation", [
        ("frozen_artifacts.json", "Artifacts evaluated (frozen)"),
        ("final_summary.csv", "Final accuracy over all evaluator points"),
        ("final_classifier_vs_hmm.png", "Per trip: classifier vs HMM"),
        ("final_per_trip.csv", "Per-trip details"),
    ]),
    ("8. Held-out evaluator_test (models trained on train only; selection frozen before this test)", "evaluator_test", [
        ("selection_frozen_before_eval.json", "Model choice recorded before evaluation"),
        ("evaluator_test_summary.csv", "All models on evaluator_test"),
        ("evaluator_test_top1.png", "CV vs student_test vs evaluator_test"),
        ("cv_vs_student_vs_evaluator.csv", "Generalisation across partitions"),
        ("evaluator_test_per_trip_top1.csv", "Per-trip Top-1"),
    ]),
]

CSS = """body{font-family:system-ui,Segoe UI,Arial;margin:0;background:#f6f7f9;color:#1d2330}
header{background:#1d2330;color:#fff;padding:18px 32px}main{max-width:1200px;margin:auto;padding:20px}
section{background:#fff;border-radius:10px;padding:18px 24px;margin:18px 0;box-shadow:0 1px 3px #0002}
h2{margin-top:0}h3{font-size:15px;color:#3a4a6b;margin:22px 0 8px}img{max-width:100%;border:1px solid #e3e6ea;border-radius:6px}
table{border-collapse:collapse;font-size:12.5px;margin:6px 0;display:block;overflow-x:auto}
td,th{border:1px solid #dde1e6;padding:4px 8px;text-align:right}th{background:#eef1f5}
pre{background:#f1f3f6;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:12.5px}
a{color:#2a6fdb}"""


def render(path: Path) -> str:
    if path.suffix == ".png":
        return f'<img src="{path.parent.name}/{path.name}" loading="lazy">'
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if len(df) > 40:
            df = df.head(40)
        return df.to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
    return f"<pre>{html.escape(path.read_text(encoding='utf-8'))}</pre>"


def main():
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"])
    parts = []
    for title, sub, items in SECTIONS:
        body = [f"<h3>{cap} — <code>{sub}/{fn}</code></h3>{render(res / sub / fn)}"
                for fn, cap in items if (res / sub / fn).exists()]
        if body:
            parts.append(f"<section><h2>{title}</h2>{''.join(body)}</section>")
    maps = sorted((res / "maps").glob("*.html")) if (res / "maps").exists() else []
    if maps:
        links = "".join(f'<li><a href="maps/{m.name}" target="_blank">{m.name}</a></li>' for m in maps)
        parts.append(f"<section><h2>Interactive Folium maps</h2><ul>{links}</ul></section>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap — review evidence</title>"
            f"<style>{CSS}</style></head><body><header><h1>PathSnap — review evidence</h1>"
            f"<div>Every table and figure below is produced by the scripts in <code>scripts/</code>. "
            f"Decisions and reasoning: <code>docs/DECISIONS.md</code></div></header><main>{''.join(parts)}</main></body></html>")
    (res / "REVIEW.html").write_text(page, encoding="utf-8")
    print(f"wrote {res / 'REVIEW.html'}")


if __name__ == "__main__":
    main()
