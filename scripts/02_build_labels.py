"""Phase 1b-i — Derive one reference arc per GPS observation from the ordered .route.

Partitions: train + student_test (both have .route). evaluator_test is never read here; its labels
are computed only inside the API process at evaluation time, with the same function.

Decision evidence produced (results/preprocessing/):
  label_alignment_per_trip.csv   monotonic vs naive labels per trip
  label_alignment_summary.csv    totals per split
  label_distance_hist.png        GPS-to-reference-arc distance (train) -> GPS noise level
  label_naive_vs_monotonic.png   example where the naive rule is wrong
Data (data/processed/, git-ignored):
  labels_<split>.parquet         trip_id, point_idx, lon, lat, t, ref_arc, ref_dist, route_pos, label_unreliable

Run:  python scripts/02_build_labels.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: avoid UnicodeEncodeError on ±, ≤, →
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from pathsnap.io import list_trips, load_config, load_trip, split_dir
from pathsnap.labels import label_frame
from pathsnap.network import build_trip_geo
from pathsnap.plotting import setup


def main():
    cfg = load_config()
    unrel = float(cfg["labels"]["unreliable_label_dist_m"])
    out = Path(cfg["paths"]["results_dir"]) / "preprocessing"
    proc = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True); proc.mkdir(parents=True, exist_ok=True)
    plt = setup()

    per_trip, example = [], None
    for split in ["train", "student_test"]:
        frames = []
        t0 = time.time()
        for tid in list_trips(split_dir(cfg, split)):
            g = build_trip_geo(load_trip(split_dir(cfg, split) / tid))
            lf = label_frame(g, unrel)
            tr = g.trip.track
            df = pd.DataFrame({
                "trip_id": tid, "point_idx": np.arange(g.n_points), "lon": tr.lon.to_numpy(), "lat": tr.lat.to_numpy(),
                "t": tr.t.to_numpy(), "ref_arc": lf["ref_arc"], "ref_dist": lf["ref_dist"],
                "route_pos": lf["route_pos"], "label_unreliable": lf["label_unreliable"],
            })
            frames.append(df)
            disagree = lf["ref_arc"] != lf["naive_arc"]
            per_trip.append(dict(
                split=split, trip_id=tid, n_points=g.n_points, route_len=len(g.trip.route),
                ref_dist_median=float(np.median(lf["ref_dist"])), ref_dist_p95=float(np.percentile(lf["ref_dist"], 95)),
                ref_dist_max=float(lf["ref_dist"].max()),
                unreliable_points=int(lf["label_unreliable"].sum()),
                naive_disagree_points=int(disagree.sum()), naive_disagree_frac=float(disagree.mean()),
                naive_order_violations=lf["naive_order_violations"],
                route_arcs_never_assigned=int(len(g.trip.route) - len(np.unique(lf["route_pos"]))),
            ))
            if split == "train" and disagree.any():
                score = float(disagree.mean())  # trip where naive labelling is wrong most often
                if example is None or score > example[1]:
                    example = (g, score, lf)
        lab = pd.concat(frames, ignore_index=True)
        lab.to_parquet(proc / f"labels_{split}.parquet", index=False)
        print(f"[{split}] {lab.trip_id.nunique()} trips, {len(lab)} points labelled in {time.time() - t0:.1f}s")

    pt = pd.DataFrame(per_trip)
    pt.to_csv(out / "label_alignment_per_trip.csv", index=False)
    summ = pt.groupby("split").agg(points=("n_points", "sum"), unreliable=("unreliable_points", "sum"),
                                   naive_disagree=("naive_disagree_points", "sum"),
                                   naive_order_violations=("naive_order_violations", "sum"),
                                   trips_with_disagreement=("naive_disagree_points", lambda s: int((s > 0).sum())))
    summ["naive_disagree_frac"] = summ.naive_disagree / summ.points
    summ["unreliable_frac"] = summ.unreliable / summ.points
    summ.to_csv(out / "label_alignment_summary.csv")
    print(summ)

    # Distance of GPS to its reference arc = empirical GPS error (+ label error). Train only.
    tr_lab = pd.read_parquet(proc / "labels_train.parquet")
    d = tr_lab.ref_dist.to_numpy()
    pct = {p: float(np.percentile(d, p)) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}
    pd.DataFrame({"percentile": list(pct), "ref_dist_m": list(pct.values())}).to_csv(out / "label_distance_percentiles.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(np.clip(d, 0, 60), bins=120, color="#2a6fdb")
    axs[0].set_xlabel("distance GPS → reference arc (m, clipped 60)"); axs[0].set_ylabel("points")
    axs[0].set_title("GPS error w.r.t. reference arc (train)")
    xs = np.sort(d); ys = 1 - np.arange(1, len(xs) + 1) / len(xs)
    axs[1].loglog(np.maximum(xs, 0.01), np.maximum(ys, 1e-7), color="#2a6fdb")
    for p in [95, 99]:
        axs[1].axvline(pct[p], ls="--", c="#888"); axs[1].text(pct[p], 0.3, f" p{p}={pct[p]:.1f}m", fontsize=8)
    axs[1].set_xlabel("distance (m)"); axs[1].set_ylabel("P(dist > x)"); axs[1].set_title("Tail of the distance distribution")
    fig.savefig(out / "label_distance_hist.png"); plt.close(fig)

    # Example: route position assigned to each GPS point over time. A correct labelling must be a
    # non-decreasing staircase (the vehicle cannot drive the route backwards).
    if example:
        g, _, lf = example
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        # zoom on the 200-point window with the most backward jumps
        back = np.r_[False, np.diff(lf["naive_route_pos"]) < 0].astype(int)
        w = 200
        c0 = int(np.argmax(np.convolve(back, np.ones(w), "valid"))) if len(back) > w else 0
        sl = slice(c0, c0 + w)
        tt = (g.t - g.t[0])[sl]
        lf = {**lf, "naive_route_pos": lf["naive_route_pos"][sl], "route_pos": lf["route_pos"][sl]}
        axs[0].plot(tt, lf["naive_route_pos"], ".-", ms=3, lw=0.4, c="#d73027")
        axs[0].set_title(f"Naive nearest-route-arc: {lf['naive_order_violations']} backward jumps")
        axs[1].plot(tt, lf["route_pos"], ".-", ms=3, lw=0.4, c="#1a9850")
        axs[1].set_title("Monotonic alignment: order preserved by construction")
        for ax in axs:
            ax.set_xlabel("time since start (s)")
        axs[0].set_ylabel("index in .route sequence")
        fig.suptitle(f"Trip {g.trip.trip_id} (train): {int((lf['ref_arc'] != lf['naive_arc']).sum())} of "
                     f"{g.n_points} points get a different label — 200-point window shown")
        fig.savefig(out / "label_naive_vs_monotonic.png"); plt.close(fig)


if __name__ == "__main__":
    main()
