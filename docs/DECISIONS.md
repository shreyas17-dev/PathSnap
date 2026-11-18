# PathSnap — Design decision log (viva sheet)

Every decision has: **what** we chose, **why**, **what we rejected**, and the **evidence file** that proves it.
Numbers below come from the scripts in `scripts/` (deterministic, seed 42). Re-run to verify.
Partitions: *train* = 80 trips, *student_test* = 10 trips. *evaluator_test* is never read by development code
(`src/pathsnap/io.py` raises an error unless the API process sets `PATHSNAP_ALLOW_EVAL=1`).

---

## Phase 1 — Data understanding & preprocessing

### D1. How the raw data fits together
* Each trip folder has its **own** road network. Node and arc ids only make sense inside that trip.
* `.track` row i is a GPS observation. It has **no** direct link to any `.nodes` or `.arcs` row.
* `.route` is the **ordered list of arcs driven**, not one label per GPS point (median 280 arcs vs 1529 points per trip).
* Arcs are directed. ~95% (median) have a reverse twin, i.e. both directions of a two-way road.
* Evidence: `results/eda/inventory.csv`, `results/eda/EDA_SUMMARY.md`

### D2. Data-quality checks: findings and actions
| Check | Finding | Action |
|---|---|---|
| Missing values | 0 NaN in every file | nothing to impute in the raw data |
| Coordinate order | 2nd column is always within ±90; GPS lies inside the network bbox | confirmed `lon, lat` |
| Id validity | every arc→node and route→arc id is in range | none |
| Timestamp order | 0 non-increasing steps | no re-sorting needed |
| Sampling | 96.7% of steps are exactly 1 s, 98.65% ≤ 2 s; 617 steps > 10 s in 42 trips; max gap 153,064 s (42 h) | gap handling (D5) |
| Implausible speed (>70 m/s) | 152 steps | **kept**, not deleted (see below) |
| Self-loop arcs | 24 | never used as candidates (zero length, no direction) |
| Duplicate arcs (same from→to) | 1,903 | mapped to one canonical arc id (D6) |
| Route breaks | 14 breaks in 11 trips (99.94% of consecutive route arcs are connected) | route treated as reference, not perfect truth |
| Parked off-network | trips 54 and 33 have long stationary periods ~85–140 m from any road (car parks) | flagged as unreliable labels (D4) |

**Why outlier GPS points are not deleted:** the task needs a prediction for *every* observation. A GPS jump is
exactly the kind of noise the model and the HMM have to cope with. Deleting points would also change what gets
evaluated. Instead the points stay in, and their effect is limited by gap-aware kinematics and context features.
Evidence: `results/eda/data_quality.csv`, `speed_hist.png`, `results/maps/eda_trip_00000054*`

### D3. Coordinate system: per-trip UTM
* **Problem:** lon/lat are angles, so Euclidean distance in degrees is wrong. On our trips it gives a **29.9% median
  error, up to 67.9%**, and the error grows with latitude.
* **Why not one projection:** the trips are spread over **25 UTM zones** on several continents.
* **Choice:** project each trip to the UTM zone of its own GPS median point (pyproj, WGS84).
  Measured error vs exact ellipsoidal geodesic: **worst trip median 0.086%** (≈ 9 cm per 100 m), far below GPS noise.
* **Rejected:** haversine everywhere. It is accurate, but point-to-segment projection, KD-trees and headings all
  need planar coordinates, and UTM gives those at negligible error.
* Evidence: `results/eda/projection_check.csv`, `projection_error.png`, `trip_locations.png`

### D4. Per-observation reference labels: monotonic alignment
* **Problem:** we need "GPS point i → arc" labels, but `.route` only gives the arc order.
* **Choice:** dynamic-programming alignment. Pick j(0) ≤ j(1) ≤ … (positions in the route) that minimise the total
  point-to-arc distance. This is the exact optimum, O(N·M), and it never lets the car drive the route backwards.
* **Rejected:** naive "nearest route arc". It disagrees on **6.6% of train points** (10.3% of student_test) and
  produces **3,847 backward jumps**. In trip 11 the car drives back along the same road, and naive labels give the
  *opposite-direction* arcs to **43%** of that trip.
* **Unreliable labels:** points more than 100 m from their aligned arc (9.9% of train points, mostly the two parked
  trips; 17 points in student_test) are flagged. The distance distribution has a clear second mode at 120–140 m
  that matches off-road parking, not road noise. These points are **excluded from training** (they would teach
  "a far arc is correct"). They are **always included** when reporting Top-1, and we report both numbers.
* Evidence: `results/preprocessing/label_alignment_*.csv`, `label_naive_vs_monotonic.png`, `label_distance_hist.png`

### D5. GPS gap threshold = 10 s
* Only 0.27% of steps exceed 10 s, so normal 1–2 s sampling and short jitter are never treated as gaps.
* At the median moving speed (11.4 m/s), 10 s ≈ 114 m ≈ 2 × candidate radius. Past this, the candidate circles
  of two consecutive points stop overlapping, so local continuity can't be assumed.
* **Effect:** speed, heading and context features are never computed across a gap (`segment_id`). Later, the HMM
  transition model relaxes across gaps instead of forcing a direct connection.
* Evidence: `results/eda/sampling_intervals.csv`, `sampling_interval_hist.png`

### D6. Network cleaning
* Duplicate arcs have identical geometry, so no model could tell them apart. Each group maps to the smallest
  arc id. This affects 3 route entries.
* Self-loops are excluded (zero length means no bearing and no meaningful distance).
* No isolated nodes exist (every node is used by some arc).
* Evidence: `results/eda/data_quality.csv`

### D7. Candidate generation: radius R = 60 m, no top-K cap
* **Method:** every arc is sampled every 10 m into a KD-tree. A ball query of radius √(R² + 5²) returns a
  *guaranteed superset* of the arcs within R, then exact point-to-segment distances filter down to R.
  No candidate is missed because of the indexing.
* **Rule declared before looking at results:** the smallest R on the grid whose reliable-point miss rate on
  **train** is ≤ 1%.
* **Result:** R = 60 m → 0.95% miss, 11.1 candidates per point (p95 = 36). R = 50 gives 1.43% miss;
  R = 100 doubles the candidate count (22.4) to cut the miss rate further.
* **Fallback:** points with nothing within 60 m (8.4%, almost all parked off-road) get their 2 nearest arcs, so
  every observation still gets a prediction.
* **No top-K cap:** a top-30 cap would still lose 0.19% of true arcs, and the set sizes are already manageable.
* Evidence: `results/preprocessing/radius_study.*`, `rank_study.csv`, `candidate_summary.csv`

### D8. Problem formulation: binary candidate classification + per-point ranking
* Each (point, candidate arc) row has label y = 1 if the arc is the reference arc. Train positive rate ≈ 8%.
* A model's probability ranks the candidates of each point. Top-1 = the highest-probability candidate.
  The same probabilities become **HMM emission scores**, which is why the models must give calibrated
  probabilities (log loss is reported).
* **Why not multi-class over arc ids:** arc ids are local to each trip (a different network every time), so a
  fixed class set doesn't exist.

### D9. Features (21): why each group exists
* **Rule baselines (no learning):** nearest arc = **55.4%** Top-1 (reliable train points). Nearest arc + heading
  tie-break = **79.4%**. Oracle ceiling = 99.5%.
* **Why direction matters:** for **69.9%** of points the true arc's reverse twin is a candidate at *exactly the
  same distance*, so distance alone is a coin flip on two-way roads.
* **Groups:** geometry (distance, rank, gap to nearest, ratio), direction (directed and undirected heading
  difference), context (neighbours' mean and max distance, smoothed-position distance), kinematics, topology.
* **Univariate ROC-AUC (train):** dist_minus_min 0.945, dist_rank 0.941, smooth_dist 0.918, ctx_mean_dist 0.915,
  dist_m 0.911, heading_diff 0.804. Gap flags are ≈0.50 on their own. They are kept for interactions and for the
  decoder; the model comparison's feature importance will confirm or reject that.
* **Leakage guard:** features use only GPS + network, never the route. The identical code runs at inference.
* **Missing heading** (speed < 1 m/s, i.e. stationary): left as NaN plus a `heading_valid` flag.
  Tree/boosting models handle NaN natively. For linear/kNN/SVM/MLP, imputation and standard scaling sit
  **inside the sklearn Pipeline**, fitted only on the training folds.
* Evidence: `results/preprocessing/rule_baselines.csv`, `twin_tie_study.csv`, `feature_univariate_auc.*`,
  `feature_distributions.png`, `feature_correlation.png`

---

### D9b. v2 direction features (+5 features, 26 total) — accepted on train CV evidence
* Error analysis target: two-way-road direction choice (69.9 % of points have an equidistant twin) is
  weak when the car is slow (heading undefined) or 1 s heading is noisy.
* Added: `heading_diff_ff` (heading carried through stops), `along_cos_w3` / `along_speed_w3`
  (motion along the arc over points i-3..i+3), `twin_hd_adv` (heading fit vs the reverse twin),
  `hd_rank_near` (heading rank among candidates within 5 m of the closest).
* Ablation (scripts/08_feature_ablation.py, untuned XGBoost, same rows and folds): decided on train
  GroupKFold CV only, student_test reported alongside, evaluator_test NOT used.
* Evidence: results/ablation/feature_ablation.csv (CV Top-1 change and folds improved).

## Phase 2 — Model experimentation (scripts/05_train_compare.py)

### D10. Validation protocol: GroupKFold(5) by trip
* Rows of the same trip are strongly correlated (consecutive 1 Hz points, same road network). A random
  row split would leak near-duplicates into validation and inflate scores. GroupKFold keeps every trip
  wholly in one fold. This is what "group-aware CV" in the guidelines asks for.
* Train partition only, reliable-label rows only (see D4). student_test (10 trips, all points) is used
  only for development-stage comparison. evaluator_test is never touched.
* Point stride 3: every 3rd GPS point with **all** its candidates is used for CV and fitting
  (~780k rows). Neighbouring 1 Hz points are near-duplicates, so this cuts compute ~3x. **The same
  rows go to every model**, so the comparison stays fair. student_test is always scored on all points.

### D11. Primary selection metric: CV observation-level Top-1 accuracy (reliable points)
* It *is* the task: for each GPS point, is the highest-scoring candidate the reference arc?
* Binary accuracy/precision/recall/F1 are reported (they are required), but they depend on the 0.5
  threshold and are dominated by the 92% negative class. A model can have a high F1 and still rank
  the wrong arc first.
* **Tie-breaker: log loss.** The probabilities become HMM emission scores, so their calibration matters.
* Also recorded: fold std (stability), train-vs-validation gap (overfitting), training time, µs per row
  (inference), route continuity of classifier-only predictions on student_test.

### D12. Model set: 9 models, 7 families, untuned initial configurations
| Family | Models | Why it is in the comparison |
|---|---|---|
| Linear | Logistic Regression | strong, calibrated baseline; tests whether the features are roughly linearly separable |
| Probabilistic | Gaussian NB | tests the independence assumption (our features are strongly correlated, see the correlation plot) |
| Instance-based | KNN (k=50, distance-weighted) | similarity in standardised feature space; trained on 100k rows because inference cost is O(n) |
| Bagging ensemble | Random Forest | non-linear interactions (e.g. heading only matters when distances tie), low variance |
| Randomised ensemble | Extra Trees | same as RF with random thresholds; checks the variance/bias trade-off |
| Boosting | HistGradientBoosting, XGBoost (CUDA on the RTX 4060), LightGBM | sequential error correction, usually strongest on tabular data, native NaN handling |
| Neural network | MLP (64-32, Adam, 25 epochs) | non-linear representation; gives epoch-wise train/validation loss curves |

* **No class re-weighting:** it distorts the probabilities (bad for the HMM), and Top-1 is a per-point
  ranking, which is unaffected by imbalance.
* Imputation and scaling sit inside each Pipeline, so they are fitted on the training fold only.
* Clustering (k-means etc.) is not used as a classifier. Cluster ids are not labels, and the guidelines
  warn against treating them as such.

## Phase 3 — Bayesian optimisation (scripts/07_bayes_opt_xgb.py)

### D13. Optuna TPE on XGBoost, objective = CV Top-1
* Selected model: XGBoost. The top models were tied on CV Top-1 (gap << fold std); the pre-declared
  tie-breakers (log loss, then speed) chose XGBoost.
* Search: TPE sampler (seed 42, 10 random start-up trials, multivariate), MedianPruner, up to 40 trials.
  Space: n_estimators 100-1500 (log), learning_rate 0.01-0.3 (log), max_depth 3-10,
  min_child_weight 1-50 (log), subsample 0.5-1, colsample_bytree 0.5-1, reg_lambda/alpha 1e-3-10 (log),
  gamma 0-5, motivated by the untuned training curve (validation log loss rose after ~100 rounds).
* Objective: mean 5-fold GroupKFold Top-1 on train (same rows/folds as the comparison). Every trial is stored
  (results/optuna/xgb_study.db, xgb_trials.csv). Tuned vs untuned: results/optuna/tuned_vs_untuned.csv.

## Phase 4 — HMM + Viterbi (scripts/09_hmm_viterbi.py, src/pathsnap/hmm.py)

### D14. Decoder design
* Hidden states: the candidate arcs of each GPS observation (the classifier's candidate sets, so each
  state keeps its arc id).
* Emission: w * log P(correct) from the tuned XGBoost (clipped at 1e-6).
* Transition (Newson & Krumm 2009): log T(a->b) = -|d_route - d_gps| / beta, d_route = directed
  shortest-path driving distance between the two projections (bounded Dijkstra, 800 m), d_gps = straight
  distance between the two GPS points. Same arc: |delta t_proj| x length (small backward jitter allowed).
  Not reachable: large finite penalty (-50), never impossible, so network/route breaks can't stall decoding.
* GPS gaps (> 10 s): the chain is cut and a new segment starts; no transition is forced across a gap.
* Missing candidates: impossible by construction (2-nearest fallback); a point with none is skipped.
* Viterbi: delta_t(b) = e_t(b) + max_a [delta_{t-1}(a) + log T(a,b)], back-pointers per segment.
* Decoder parameters chosen on TRAIN out-of-fold probabilities (5-fold GroupKFold, tuned params) over
  beta {1,2,5,10,20,50} x w {0.5,1,2}; student_test reported. Evidence: results/hmm/.
* Sequence-level metrics: route continuity, and route precision/recall/F1 of the predicted arc set vs the
  reference route arcs.

### D15. API & UI
* FastAPI on Laptop 1 (api/main.py), contract in docs/API_CONTRACT.md. UI on Laptop 2 sends only a trip id.
* The API runs the full pipeline live for every request (src/pathsnap/pipeline.py, the same code as
  the scripts), scores it server-side against labels derived from .route, and returns only aggregate
  metrics + predicted geometry. The console logs every request (request id, trip, timings, accuracy).
