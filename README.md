# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.
We reconstruct the road segments a vehicle actually drove from noisy GPS, using a candidate classifier
followed by HMM/Viterbi decoding. Deployment: two laptops (model + FastAPI on Laptop 1, UI on Laptop 2).

**Design decisions and their evidence:** [`docs/DECISIONS.md`](docs/DECISIONS.md) ·
**Team roles:** [`docs/TEAM_PLAN.md`](docs/TEAM_PLAN.md) ·
**All results on one page:** `results/REVIEW.html` (open in a browser) ·
**Live demo on two laptops:** [`docs/DEMO.md`](docs/DEMO.md)

## Pipeline
```
raw trip files ──► per-trip UTM projection ──► monotonic label alignment (train/student_test only)
       │
       └─► candidate arcs within R=60 m ──► 21 features ──► classifier P(arc is correct)
                                                                   │
                                     HMM (states = candidates) + Viterbi ◄─┘ ──► route + accuracy
```

## Repository layout
```
configs/config.yaml        every parameter + the evidence file behind it
src/pathsnap/              library code (io, geo, network, labels, candidates, features, models,
                           plotting, hmm, pipeline, maps)
scripts/                   numbered, runnable steps, each writing evidence to results/
  01_eda.py                dataset inventory, quality checks, plots, Folium maps
  02_build_labels.py       per-point reference arcs (monotonic alignment vs naive)
  03_candidate_radius_study.py   chooses candidate radius on TRAIN only
  04_build_features.py     candidates + features -> data/processed/*.parquet
  05_train_compare.py      models, GroupKFold CV + student_test, all metrics/plots -> results/models/
  06_evaluate_holdout.py   trained models on evaluator_test once, selection frozen first -> results/evaluator_test/
  07_bayes_opt_xgb.py      Optuna TPE tuning of XGBoost -> results/optuna/, models/final/
  08_feature_ablation.py   v1 vs v1+v2 features on train CV -> results/ablation/
  09_hmm_viterbi.py        HMM decoder tuned on train OOF; classifier vs HMM -> results/hmm/, models/final/hmm_config.json
  10_final_evaluation.py   frozen model + HMM on evaluator_test, run once -> results/final_evaluation/, results/maps/
  99_build_review_page.py  builds results/REVIEW.html
  export_csv.py            preprocessed data as CSV -> data/processed/csv/
results/                   committed evidence: CSV, PNG, HTML maps
models/                    final trained artifacts (committed)
api/main.py                FastAPI backend, Laptop 1 (contract: docs/API_CONTRACT.md), start with run_api.bat
ui/                        evaluator UI, Laptop 2 (plain HTML/JS + Leaflet, see ui/README.md)
data/raw, data/processed   git-ignored
docs/                      decision log, team plan
```

## Setup — Laptop 1 (Windows, RTX 4060)
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
The dataset is read from `../Student folder` (the organiser folder next to this repo, see `raw_dir` in
`configs/config.yaml`). Then:
```bat
run_all.bat
```
Steps 01–04 (EDA + preprocessing) take ~3 min. Step 05 (9 models × 5 folds) takes ~15–30 min and is
resumable: finished models are skipped on re-run. Then open `results\REVIEW.html`.

Run a single step: `python scripts\01_eda.py`. Every script reads `configs/config.yaml`.

## Full pipeline order (Laptop 1)
```bat
python scripts\01_eda.py
python scripts\02_build_labels.py
python scripts\03_candidate_radius_study.py
python scripts\04_build_features.py
python scripts\08_feature_ablation.py
python scripts\05_train_compare.py --models LogisticRegression GaussianNB RandomForest XGBoost MLP
python scripts\07_bayes_opt_xgb.py --trials 40 --timeout-min 45
python scripts\09_hmm_viterbi.py
python scripts\06_evaluate_holdout.py
python scripts\10_final_evaluation.py
python scripts\99_build_review_page.py
run_api.bat
```

## Data partitions and leakage rules
| Partition | Trips | Used for |
|---|---|---|
| train | 80 | fitting, GroupKFold CV (grouped by trip), Bayesian optimisation, radius and threshold choices |
| student_test | 10 | development-stage comparison only |
| evaluator_test | 10 | final evaluation only. The loader refuses to read it unless `PATHSNAP_ALLOW_EVAL=1` (set only by the API) |

## Reproducibility
Seed 42 everywhere. All scripts are deterministic, so re-running reproduces every table and figure in `results/`.
