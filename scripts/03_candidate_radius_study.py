"""Phase 1b-ii — Choose the candidate search radius (and check whether a top-K cap is needed).

Partition: TRAIN ONLY (80 trips). student_test / evaluator_test are not used to pick R.

Trade-off being measured:
  small R -> the true arc is often missing (a candidate "miss" can never be predicted correctly)
  large R -> many candidates per point -> heavier class imbalance, slower training/Viterbi
Decision rule (stated before looking at results): pick the smallest R on the grid whose miss rate on
reliable-label points is <= 1 %.

Outputs (results/preprocessing/):
  radius_study.csv / radius_study.png   miss rate and candidate-set size vs R
  rank_study.csv                        how often the reference arc is the k-th nearest candidate
  radius_decision.txt                   chosen R with the numbers behind it
  twin_tie_study.csv                    how often the reference arc ties with its reverse twin on distance

Run:  python scripts/03_candidate_radius_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: avoid UnicodeEncodeError on ±, ≤, →
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from pathsnap.candidates import ArcIndex, rank_within_point
from pathsnap.io import list_trips, load_config, load_trip, split_dir
from pathsnap.network import build_trip_geo
from pathsnap.plotting import setup

GRID = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200]
SUBSAMPLE = 5            # every 5th point for candidate-count estimation (keeps memory small)
TARGET_MISS = 0.01


def main():
    cfg = load_config()
    rmax = float(cfg["candidates"]["study_max_radius_m"])
    step = float(cfg["candidates"]["densify_step_m"])
    out = Path(cfg["paths"]["results_dir"]) / "preprocessing"
    proc = Path(cfg["paths"]["processed_dir"])
    plt = setup()

    lab = pd.read_parquet(proc / "labels_train.parquet")
    rows = []
    for tid in list_trips(split_dir(cfg, "train")):
        g = build_trip_geo(load_trip(split_dir(cfg, "train") / tid))
        L = lab[lab.trip_id == tid].reset_index(drop=True)
        sel = np.arange(0, g.n_points, SUBSAMPLE)
        c = rank_within_point(ArcIndex(g, step).query(g.px[sel], g.py[sel], rmax))
        c["point_idx"] = sel[c.point_idx.to_numpy()]
        ref = L.ref_arc.to_numpy()
        c["is_ref"] = c.arc_id.to_numpy() == ref[c.point_idx.to_numpy()]
        # reverse twin of the reference arc with the same distance -> distance alone cannot decide
        twin_of_ref = np.zeros(len(c), bool)
        ru, rv = g.u[ref], g.v[ref]
        pu, pv = ru[c.point_idx.to_numpy()], rv[c.point_idx.to_numpy()]
        twin_of_ref = (g.u[c.arc_id.to_numpy()] == pv) & (g.v[c.arc_id.to_numpy()] == pu)
        c["is_twin"] = twin_of_ref
        per_pt = pd.DataFrame({"point_idx": sel})
        per_pt["trip_id"] = tid
        per_pt["reliable"] = ~L.label_unreliable.to_numpy()[sel]
        per_pt["ref_dist"] = L.ref_dist.to_numpy()[sel]
        hit = c[c.is_ref].set_index("point_idx")
        per_pt["ref_rank"] = per_pt.point_idx.map(hit.dist_rank)
        twin = c[c.is_twin].set_index("point_idx").dist_m
        per_pt["twin_dist_gap"] = (per_pt.point_idx.map(twin) - per_pt.ref_dist).abs()
        for R in GRID:
            per_pt[f"n_cands_{R}"] = per_pt.point_idx.map(c[c.dist_m <= R].groupby("point_idx").size()).fillna(0)
        rows.append(per_pt)
        print(tid, end=" ", flush=True)
    print()
    P = pd.concat(rows, ignore_index=True)

    rel = P[P.reliable]
    res = []
    for R in GRID:
        miss_rel_trip = rel.groupby("trip_id").apply(lambda d: (d.ref_dist > R).mean(), include_groups=False)
        nc = P[f"n_cands_{R}"]
        res.append(dict(
            radius_m=R,
            miss_rate_all=float((P.ref_dist > R).mean()),
            miss_rate_reliable=float((rel.ref_dist > R).mean()),
            miss_rate_reliable_trip_mean=float(miss_rel_trip.mean()),
            miss_rate_reliable_trip_max=float(miss_rel_trip.max()),
            cands_mean=float(nc.mean()), cands_median=float(nc.median()), cands_p95=float(nc.quantile(0.95)),
            cands_max=int(nc.max()), points_with_no_candidate=float((nc == 0).mean()),
            positive_rate=float(((P.ref_dist <= R).sum()) / max(nc.sum(), 1)),
        ))
    rs = pd.DataFrame(res)
    rs.to_csv(out / "radius_study.csv", index=False)
    ok = rs[rs.miss_rate_reliable <= TARGET_MISS]
    chosen = int(ok.radius_m.iloc[0]) if len(ok) else int(rs.radius_m.iloc[-1])
    print(rs.round(4).to_string(index=False))

    # rank study (reliable points whose ref arc is within the chosen R)
    within = rel[rel.ref_dist <= chosen]
    ks = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    rk = pd.DataFrame({"k": ks, "frac_ref_within_top_k": [float((within.ref_rank <= k).mean()) for k in ks]})
    rk.to_csv(out / "rank_study.csv", index=False)
    print(rk.to_string(index=False))

    tw = pd.DataFrame([dict(
        reliable_points=len(rel),
        frac_ref_has_twin_candidate=float(rel.twin_dist_gap.notna().mean()),
        frac_twin_equidistant_lt_0p5m=float((rel.twin_dist_gap < 0.5).mean()),
        nearest_arc_top1_reliable=float((rel.ref_rank == 1).mean()),
    )])
    tw.to_csv(out / "twin_tie_study.csv", index=False)
    print(tw.T)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(rs.radius_m, rs.miss_rate_reliable * 100, "o-", c="#d73027", label="miss rate, reliable points")
    ax1.plot(rs.radius_m, rs.miss_rate_all * 100, "s--", c="#f4a582", label="miss rate, all points (incl. parked off-road)")
    ax1.axhline(TARGET_MISS * 100, c="#d73027", ls=":", lw=1)
    ax1.axvline(chosen, c="k", ls=":", lw=1)
    ax1.set_xlabel("candidate radius R (m)"); ax1.set_ylabel("reference arc NOT in candidates (%)")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(rs.radius_m, rs.cands_mean, "^-", c="#2a6fdb", label="mean candidates / point")
    ax2.set_ylabel("candidates per point", color="#2a6fdb"); ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    ax1.set_title(f"Candidate radius study (train) — chosen R = {chosen} m (rule: reliable miss ≤ {TARGET_MISS:.0%})")
    fig.savefig(out / "radius_study.png"); plt.close(fig)

    row = rs[rs.radius_m == chosen].iloc[0]
    (out / "radius_decision.txt").write_text(
        f"Rule: smallest R in {GRID} with reliable-point miss rate <= {TARGET_MISS:.0%} (train only).\n"
        f"Chosen R = {chosen} m\n"
        f"  miss rate reliable = {row.miss_rate_reliable:.4%}, all points = {row.miss_rate_all:.4%}\n"
        f"  mean candidates/point = {row.cands_mean:.1f}, p95 = {row.cands_p95:.0f}, positive rate = {row.positive_rate:.4f}\n"
        f"Top-K cap check: reference arc within top-{rk.k.iloc[-1]} nearest = {rk.frac_ref_within_top_k.iloc[-1]:.4%}\n"
        f"Set candidates.radius_m in configs/config.yaml to this value.\n", encoding="utf-8")
    print((out / "radius_decision.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
