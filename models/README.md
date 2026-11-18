# Trained models

| File | Made by | Used by |
|---|---|---|
| `final/xgboost_tuned.joblib` | `scripts/07_bayes_opt_xgb.py` (Optuna-tuned XGBoost) | the API and `scripts/10_final_evaluation.py`, through `src/pathsnap/pipeline.py` → `load_artifacts()` |
| `final/xgboost_tuned_config.json` | `scripts/07_bayes_opt_xgb.py` (best trial, CV Top-1, hyperparameters) | `scripts/09_hmm_viterbi.py` |
| `final/hmm_config.json` | `scripts/09_hmm_viterbi.py` (HMM `beta`, `w`, `max_route_m`, tuned on train OOF) | the API and `scripts/10_final_evaluation.py` |
| `old_v1/LightGBM.joblib` | an earlier LightGBM model from `scripts/05_train_compare.py` | nothing; kept for reference only |

`models/tmp/` and `models/checkpoints/` hold per-run intermediate models and are git-ignored.
The files in `final/` are frozen: they were fixed before `evaluator_test` was evaluated
(see `results/final_evaluation/frozen_artifacts.json`).
