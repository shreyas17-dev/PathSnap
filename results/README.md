# Results (committed evidence)

Every file here is written by a numbered script, so re-running the script reproduces it (seed 42).
Open `REVIEW.html` in a browser to see everything on one page.

| Folder / file | Written by | What it shows |
|---|---|---|
| `eda/` | `scripts/01_eda.py` | dataset inventory, data quality, sampling, speed, projection check |
| `preprocessing/` | `scripts/02`–`04` | label alignment, candidate-radius study, rule baselines, feature analysis |
| `models/` | `scripts/05_train_compare.py` | model comparison: GroupKFold CV, student_test metrics, curves, confusion matrices |
| `ablation/` | `scripts/08_feature_ablation.py` | v1 vs v1+v2 features |
| `evaluator_test/` | `scripts/06_evaluate_holdout.py` | trained models on evaluator_test, selection frozen first |
| `optuna/` | `scripts/07_bayes_opt_xgb.py` | Bayesian tuning of XGBoost: trials, history, tuned vs untuned |
| `hmm/` | `scripts/09_hmm_viterbi.py` | HMM grid on train OOF, classifier vs HMM |
| `final_evaluation/` | `scripts/10_final_evaluation.py` | final pipeline on evaluator_test, per trip and in total |
| `maps/` | `scripts/01`, `09`, `10` | interactive Folium maps: `eda_*`, `hmm_student_test_*`, `final_eval_*` |
| `REVIEW.html` | `scripts/99_build_review_page.py` | all of the above on one page |
