"""Export the preprocessed data (data/processed/*.parquet) to CSV for viewing / submission.

Writes to data/processed/csv/:
  points_<split>.csv                 one row per GPS point: lon, lat, t, reference arc, label distance, flags
  candidates_<split>.csv             one row per (GPS point, candidate arc): all 21 features + label y
  candidates_train_sample_excel.csv  first 5 train trips only (fits in Excel's 1,048,576-row limit)
  README_columns.csv                 meaning of every column

Note: candidates_train.csv is ~2.35M rows (several hundred MB) - open it with pandas, not Excel.

Run:  python scripts/export_csv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from pathsnap.io import load_config

COLUMNS = {
    "trip_id": "trip folder name",
    "point_idx": "row number of the GPS point inside its .track file (0-based)",
    "lon / lat / t": "raw GPS longitude, latitude (degrees) and time (s)",
    "ref_arc": "reference arc for this GPS point from the monotonic route alignment (canonical arc id)",
    "ref_dist": "distance in metres from the GPS point to its reference arc",
    "route_pos": "position of ref_arc inside the .route sequence",
    "label_unreliable": "True if ref_dist > 100 m (vehicle parked off the road network) - excluded from training",
    "ref_in_cands": "True if the reference arc is among this point's candidates",
    "n_cands / is_fallback": "number of candidates; True if no arc was within 60 m and the 2 nearest arcs were used",
    "arc_id": "candidate arc id (row of the .arcs file)",
    "y": "label: 1 = this candidate is the reference arc, 0 = not",
    "dist_m": "perpendicular distance GPS point -> candidate arc (m, per-trip UTM)",
    "dist_rank": "1 = closest candidate of this point, 2 = second closest, ...",
    "dist_minus_min": "dist_m minus the closest candidate's distance (m)",
    "dist_ratio": "dist_m / (closest distance + 1)",
    "t_proj": "where the point projects along the arc: 0 = start node, 1 = end node",
    "at_endpoint": "1 if the projection falls beyond the arc's ends",
    "heading_diff": "angle between GPS travel direction and arc direction, 0-180 deg (NaN if stationary)",
    "heading_diff_undirected": "same but ignoring direction, 0-90 deg",
    "heading_valid": "1 if speed >= 1 m/s so heading is meaningful",
    "ctx_mean_dist / ctx_max_dist": "mean / max distance of points i-2..i+2 (same gap-free segment) to this arc (m)",
    "smooth_dist": "distance of the 5-point averaged position to this arc (m)",
    "speed": "speed around this point (m/s), never computed across a gap",
    "dt_prev": "seconds since previous GPS point",
    "gap_before / gap_after": "1 if a gap > 10 s is just before / after this point",
    "arc_len": "arc length (m)",
    "has_reverse": "1 if the opposite-direction arc exists (two-way road)",
    "in_deg_u / out_deg_v": "number of arcs entering the arc's start node / leaving its end node",
}


def main():
    cfg = load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    out = proc / "csv"
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "student_test"]:
        for kind in ["points", "candidates"]:
            src = proc / f"{kind}_{split}.parquet"
            if not src.exists():
                print(f"missing {src} - run scripts/04_build_features.py first"); continue
            df = pd.read_parquet(src)
            df.to_csv(out / f"{kind}_{split}.csv", index=False, float_format="%.4f")
            print(f"wrote {kind}_{split}.csv  ({len(df):,} rows, {df.shape[1]} columns)")
            if kind == "candidates" and split == "train":
                trips = sorted(df.trip_id.unique())[:5]
                s = df[df.trip_id.isin(trips)]
                s.to_csv(out / "candidates_train_sample_excel.csv", index=False, float_format="%.4f")
                print(f"wrote candidates_train_sample_excel.csv  ({len(s):,} rows, trips {trips})")
    pd.DataFrame({"column": list(COLUMNS), "meaning": list(COLUMNS.values())}).to_csv(out / "README_columns.csv", index=False)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
