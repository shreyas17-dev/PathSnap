# Raw data (not committed)

The dataset is never committed. Where the code looks for it is set by `raw_dir` in `configs/config.yaml`.

**Option 1 — default, no config change.** Keep the organiser-provided `Student folder` next to the repo folder:

```
<parent folder>/
  PathSnap/                 this repo
  Student folder/
    train/00000001/00000001.{track,nodes,arcs,route}   (80 trips)
    student_test/00000000/...                           (10 trips)
    evaluator_test/...                                  (10 trips, Laptop 1 only, read ONLY by the API)
```

**Option 2 — data inside the repo.** Copy `train/` and `student_test/` (and `evaluator_test/` on Laptop 1)
into this folder, `data/raw/`, then change the config line to `raw_dir: data/raw`.
Everything in `data/raw/` except this README is git-ignored.

Laptop 2 (UI only) needs no data at all.
