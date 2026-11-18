# Team plan — 4 members, 2 laptops, 2 reviews

## Who owns what
Ownership = you build it, you present it, you answer the first viva question on it.
**Everyone** must still be able to explain every decision in `docs/DECISIONS.md` (the viva can pick anyone).

| Member | Owns | Code | Viva topics |
|---|---|---|---|
| **M1: Data lead** (Laptop 1 owner) | EDA, projection, label alignment, candidate generation, features | `src/pathsnap/{io,geo,network,labels,candidates,features}.py`, `scripts/01–04` | dataset structure, preprocessing, feature engineering |
| **M2: Modelling lead** | ≥5 models, GroupKFold CV, metrics, comparison plots, final selection | `scripts/05_*`, `src/pathsnap/models*.py`, `results/models/` | model comparison, why the chosen model won, over/underfitting |
| **M3: Optimisation & decoding lead** | Optuna Bayesian search, HMM + Viterbi, classifier-vs-HMM comparison | `scripts/06_*`, `scripts/07_*`, `src/pathsnap/hmm.py` | search space, objective, emission/transition, gaps |
| **M4: Deployment & report lead** (Laptop 2 owner) | FastAPI, UI, Folium result maps, integration test, PDF report | `api/`, `ui/`, `src/pathsnap/maps.py`, `report/` | API design, UI, end-to-end demo |

## Review 1 (before lunch): what to show, ~2 min each
1. **M1**: problem statement → dataset structure → EDA → preprocessing decisions (open `results/REVIEW.html`).
2. **M2**: problem formulation (candidate classification), rule baselines to beat (55% → 79%), model plan + metric choice.
3. **M3**: HMM design on paper (states = candidates, emission = classifier probability, transition = network distance), gap handling.
4. **M4**: two-laptop architecture diagram, API contract (`POST /predict {trip_id}`), UI wireframe, repo structure.

## Final review
Same order, now with results: M2 comparison table + plots, M3 tuned-vs-untuned and classifier-vs-HMM,
M4 live demo (evaluator picks any of the 10 trips; show API logs on Laptop 1).

## Git workflow (the repo history is examined)
* `main` is protected by convention: work on branches `m1-data`, `m2-models`, `m3-hmm`, `m4-app`, then merge.
* **Each member commits from their own GitHub account**, in small commits with clear messages.
  The history then honestly shows who did what.
* Never commit `data/raw/` or `data/processed/` (git-ignored). Do commit `results/` and the final `models/` artifacts.
* Pull before you push: `git pull --rebase origin main`.

## Laptop roles
* **Laptop 1** (RTX 4060): training, all `scripts/`, `models/`, FastAPI server, holds evaluator data + labels.
* **Laptop 2** (no GPU): runs only `ui/`. Talks to Laptop 1 over the LAN (same Wi-Fi / hotspot). Never needs the dataset.
