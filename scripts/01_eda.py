"""Phase 1a — Exploratory data analysis of the raw PathSnap dataset.

Partitions used: train (80 trips) and student_test (10 trips). evaluator_test is never read.

Outputs (results/eda/):
  inventory.csv            per-trip file sizes, row counts, duration, extent, UTM zone
  data_quality.csv         NaNs, duplicates, ordering, gaps, speed outliers, network & route checks
  projection_check.csv     distance error of per-trip UTM vs naive degree-Euclidean vs geodesic
  sampling_intervals.csv   distribution of time steps (basis for the gap threshold)
  split_summary.csv        aggregated stats per split
  *.png                    plots for the report
  EDA_SUMMARY.md           auto-generated plain-text record of findings
results/maps/
  eda_world_overview.html  where the trips are
  eda_trip_<id>.html       GPS + network + reference route for selected trips

Run:  python scripts/01_eda.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: avoid UnicodeEncodeError on ±, ≤, →
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import folium
import numpy as np
import pandas as pd

from pathsnap.geo import geodesic_dist
from pathsnap.io import list_trips, load_config, load_trip, split_dir
from pathsnap.maps import trip_map
from pathsnap.network import build_trip_geo, network_stats, route_stats
from pathsnap.plotting import SPLIT_COLORS, setup

SPEED_OUTLIER_MPS = 70.0  # 252 km/h: physically implausible for road vehicles -> GPS jump


def analyse_trip(split: str, folder: Path, gap_thr: float):
    trip = load_trip(folder)
    tr = trip.track
    g = build_trip_geo(trip)
    tid = trip.trip_id
    lon, lat, t = tr.lon.to_numpy(), tr.lat.to_numpy(), tr.t.to_numpy()
    dt = np.diff(t)
    step_utm = np.hypot(np.diff(g.px), np.diff(g.py))
    step_geo = geodesic_dist(lon[:-1], lat[:-1], lon[1:], lat[1:])
    step_deg = np.hypot(np.diff(lon), np.diff(lat)) * 111_320.0  # what naive Euclidean-in-degrees gives
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, step_geo / dt, np.nan)
    moving = step_geo > 1.0  # ignore near-zero steps when computing relative errors

    inv = dict(
        split=split, trip_id=tid,
        track_bytes=(folder / f"{tid}.track").stat().st_size,
        nodes_bytes=(folder / f"{tid}.nodes").stat().st_size,
        arcs_bytes=(folder / f"{tid}.arcs").stat().st_size,
        route_bytes=(folder / f"{tid}.route").stat().st_size if (folder / f"{tid}.route").exists() else 0,
        n_points=len(tr), n_nodes=len(trip.nodes), n_arcs=len(trip.arcs),
        n_route_arcs=0 if trip.route is None else len(trip.route),
        duration_s=float(t[-1] - t[0]), t_start=float(t[0]),
        path_km=float(step_geo.sum() / 1000),
        lon_min=lon.min(), lon_max=lon.max(), lat_min=lat.min(), lat_max=lat.max(),
        extent_x_km=float((g.px.max() - g.px.min()) / 1000), extent_y_km=float((g.py.max() - g.py.min()) / 1000),
        centroid_lon=float(np.median(lon)), centroid_lat=float(np.median(lat)), utm_epsg=g.proj.epsg,
    )

    nodes = trip.nodes
    in_net_bbox = ((lon >= nodes.lon.min()) & (lon <= nodes.lon.max()) &
                   (lat >= nodes.lat.min()) & (lat <= nodes.lat.max())).mean()
    q = dict(
        split=split, trip_id=tid,
        nan_track=int(tr.isna().sum().sum()), nan_nodes=int(nodes.isna().sum().sum()),
        nan_arcs=int(trip.arcs.isna().sum().sum()),
        lat_in_range=bool((np.abs(lat) <= 90).all() and (np.abs(nodes.lat) <= 90).all()),
        gps_inside_network_bbox=float(in_net_bbox),
        arc_node_ids_valid=bool(max(trip.arcs.u.max(), trip.arcs.v.max()) < len(nodes) and trip.arcs.min().min() >= 0),
        route_ids_valid=bool(trip.route is None or (trip.route.max() < len(trip.arcs) and trip.route.min() >= 0)),
        non_increasing_t=int((dt <= 0).sum()),
        duplicate_timestamps=int((dt == 0).sum()),
        duplicate_consecutive_positions=int((step_geo == 0).sum()),
        frac_dt_1s=float((dt == 1).mean()), frac_dt_le2s=float((dt <= 2).mean()),
        median_dt=float(np.median(dt)), max_dt=float(dt.max()),
        n_gaps=int((dt > gap_thr).sum()),
        missing_1hz_samples=int(np.clip(dt - 1, 0, None).sum()),
        speed_median_mps=float(np.nanmedian(speed)), speed_p99_mps=float(np.nanpercentile(speed, 99)),
        speed_outliers=int((speed > SPEED_OUTLIER_MPS).sum()),
    )
    q.update(network_stats(g))
    q.update(route_stats(g))

    proj = dict(
        split=split, trip_id=tid, utm_epsg=g.proj.epsg, centroid_lat=inv["centroid_lat"],
        utm_rel_err_median=float(np.median(np.abs(step_utm[moving] - step_geo[moving]) / step_geo[moving])),
        utm_rel_err_max=float(np.max(np.abs(step_utm[moving] - step_geo[moving]) / step_geo[moving])),
        deg_rel_err_median=float(np.median(np.abs(step_deg[moving] - step_geo[moving]) / step_geo[moving])),
        deg_rel_err_max=float(np.max(np.abs(step_deg[moving] - step_geo[moving]) / step_geo[moving])),
    )
    return inv, q, proj, dt, speed[np.isfinite(speed)], g


def main():
    cfg = load_config()
    gap_thr = float(cfg["time"]["gap_threshold_s"])
    out = Path(cfg["paths"]["results_dir"]) / "eda"
    maps_dir = Path(cfg["paths"]["results_dir"]) / "maps"
    out.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    plt = setup()

    inv_rows, q_rows, p_rows, dts, speeds, geos = [], [], [], {}, {}, {}
    t0 = time.time()
    for split in ["train", "student_test"]:
        trips = list_trips(split_dir(cfg, split))
        print(f"[{split}] {len(trips)} trips")
        dts[split], speeds[split] = [], []
        for tid in trips:
            inv, q, p, dt, sp, g = analyse_trip(split, split_dir(cfg, split) / tid, gap_thr)
            inv_rows.append(inv); q_rows.append(q); p_rows.append(p)
            dts[split].append(dt); speeds[split].append(sp)
            geos[(split, tid)] = g
    print(f"analysed in {time.time() - t0:.1f}s")

    inv = pd.DataFrame(inv_rows); q = pd.DataFrame(q_rows); proj = pd.DataFrame(p_rows)
    inv.to_csv(out / "inventory.csv", index=False)
    q.to_csv(out / "data_quality.csv", index=False)
    proj.to_csv(out / "projection_check.csv", index=False)

    # ---- sampling intervals (basis for the gap threshold) ----
    all_dt = np.concatenate([np.concatenate(v) for v in dts.values()])
    thr_list = [1, 2, 3, 5, 10, 15, 30, 60, 300, 3600]
    si = pd.DataFrame({"dt_threshold_s": thr_list,
                       "frac_steps_greater": [(all_dt > x).mean() for x in thr_list],
                       "n_steps_greater": [(all_dt > x).sum() for x in thr_list],
                       "n_trips_with_step_greater": [int((q.max_dt > x).sum()) for x in thr_list]})
    si.to_csv(out / "sampling_intervals.csv", index=False)
    all_speed = np.concatenate([np.concatenate(v) for v in speeds.values()])
    med_speed_moving = float(np.median(all_speed[all_speed > 1.0]))

    # ---- split summary ----
    ss = inv.groupby("split").agg(trips=("trip_id", "count"), points=("n_points", "sum"),
                                  points_median=("n_points", "median"), points_min=("n_points", "min"),
                                  points_max=("n_points", "max"), duration_h=("duration_s", lambda s: s.sum() / 3600),
                                  path_km=("path_km", "sum"), arcs_median=("n_arcs", "median"),
                                  route_arcs_median=("n_route_arcs", "median"))
    ss.to_csv(out / "split_summary.csv")

    # ================= PLOTS =================
    # 1. Trip locations (why one projection is not enough)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for s, d in inv.groupby("split"):
        ax.scatter(d.centroid_lon, d.centroid_lat, s=28, c=SPLIT_COLORS[s], label=f"{s} ({len(d)})", alpha=0.85)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 75)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Trip centroids — {inv.utm_epsg.nunique()} distinct UTM zones (train + student_test)")
    ax.legend(); fig.savefig(out / "trip_locations.png"); plt.close(fig)

    # 2. Observations per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(inv.n_points.min()), np.log10(inv.n_points.max()), 25)
    for s, d in inv.groupby("split"):
        ax.hist(d.n_points, bins=bins, color=SPLIT_COLORS[s], alpha=0.7, label=s)
    ax.set_xscale("log"); ax.set_xlabel("GPS observations per trip (log)"); ax.set_ylabel("trips")
    ax.set_title("Trajectory length distribution"); ax.legend(); fig.savefig(out / "points_per_trip.png"); plt.close(fig)

    # 3. Sampling intervals
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    vals, cnts = np.unique(np.clip(all_dt, 0, 31), return_counts=True)
    axs[0].bar(vals, cnts, color="#2a6fdb"); axs[0].set_yscale("log")
    axs[0].axvline(gap_thr + 0.5, color="#d73027", ls="--", label=f"gap threshold = {gap_thr:g}s")
    axs[0].set_xlabel("time step Δt (s), clipped at 31"); axs[0].set_ylabel("count (log)")
    axs[0].set_title("Sampling intervals, all steps"); axs[0].legend()
    xs = np.sort(all_dt); ys = 1 - np.arange(1, len(xs) + 1) / len(xs)
    axs[1].loglog(xs, np.maximum(ys, 1e-7), color="#2a6fdb")
    axs[1].axvline(gap_thr, color="#d73027", ls="--")
    axs[1].set_xlabel("Δt (s)"); axs[1].set_ylabel("P(step > Δt)"); axs[1].set_title("Survival function of Δt")
    fig.savefig(out / "sampling_interval_hist.png"); plt.close(fig)

    # 4. Speed distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(all_speed, 0, 80), bins=160, color="#2a6fdb")
    ax.axvline(SPEED_OUTLIER_MPS, color="#d73027", ls="--", label=f"implausible > {SPEED_OUTLIER_MPS:g} m/s")
    ax.set_yscale("log"); ax.set_xlabel("implied speed between consecutive points (m/s, clipped 80)")
    ax.set_ylabel("count (log)"); ax.set_title("Implied speed (geodesic step / Δt)"); ax.legend()
    fig.savefig(out / "speed_hist.png"); plt.close(fig)

    # 5. Projection error
    fig, ax = plt.subplots(figsize=(8, 4))
    o = proj.sort_values("centroid_lat")
    ax.semilogy(o.centroid_lat, o.deg_rel_err_median * 100, "o", c="#d73027", label="Euclidean in degrees ×111 km")
    ax.semilogy(o.centroid_lat, o.utm_rel_err_median * 100, "o", c="#1a9850", label="per-trip UTM")
    ax.set_xlabel("trip latitude"); ax.set_ylabel("median relative step-length error (%)")
    ax.set_title("Distance error vs exact geodesic (WGS84)"); ax.legend()
    fig.savefig(out / "projection_error.png"); plt.close(fig)

    # 6. Network size vs trip size
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, d in inv.groupby("split"):
        ax.scatter(d.n_points, d.n_arcs, c=SPLIT_COLORS[s], label=s, s=22)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("GPS observations"); ax.set_ylabel("arcs in supplied network")
    ax.set_title("Road-network size per trip"); ax.legend(); fig.savefig(out / "network_size.png"); plt.close(fig)

    # 7. GPS trajectory grid (projected metres)
    pick = inv[inv.split == "train"].sort_values("n_points").trip_id.to_numpy()
    pick = pick[np.linspace(0, len(pick) - 1, 12).astype(int)]
    fig, axs = plt.subplots(3, 4, figsize=(14, 10))
    for ax, tid in zip(axs.ravel(), pick):
        g = geos[("train", tid)]
        ax.plot(g.px - g.px.mean(), g.py - g.py.mean(), lw=0.6, c="#2166ac")
        ax.scatter(g.px[0] - g.px.mean(), g.py[0] - g.py.mean(), c="green", s=20, zorder=3)
        ax.set_title(f"{tid}  n={g.n_points}", fontsize=9); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    fig.suptitle("Sample GPS trajectories (train, projected metres; green = start)")
    fig.savefig(out / "trajectory_grid.png"); plt.close(fig)

    # 8. Road network + GPS + route for one trip
    tid = "00000010" if ("train", "00000010") in geos else pick[len(pick) // 2]
    g = geos[("train", tid)]
    fig, ax = plt.subplots(figsize=(8, 8))
    pad = 300
    box = ((g.ax1 > g.px.min() - pad) & (g.ax1 < g.px.max() + pad) & (g.ay1 > g.py.min() - pad) & (g.ay1 < g.py.max() + pad))
    for i in np.flatnonzero(box & g.usable):
        ax.plot([g.ax1[i], g.ax2[i]], [g.ay1[i], g.ay2[i]], c="#bbbbbb", lw=0.6)
    for a in g.trip.route:
        ax.plot([g.ax1[a], g.ax2[a]], [g.ay1[a], g.ay2[a]], c="#1a9850", lw=2.5)
    ax.scatter(g.px, g.py, s=2, c="#2166ac", zorder=3)
    ax.set_aspect("equal"); ax.set_title(f"Trip {tid}: road network (grey), reference route (green), GPS (blue)")
    fig.savefig(out / f"network_trip_{tid}.png"); plt.close(fig)

    # 9. Network connectivity summary
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(q.weak_components, bins=30, color="#2a6fdb"); axs[0].set_xlabel("weakly connected components")
    axs[0].set_ylabel("trips"); axs[0].set_title("Fragmentation of supplied networks")
    axs[1].hist(q.largest_strong_frac, bins=30, color="#e8833a"); axs[1].set_xlabel("fraction of nodes in largest SCC")
    axs[1].set_title("Strong connectivity (directed)")
    fig.savefig(out / "network_connectivity.png"); plt.close(fig)

    # ================= FOLIUM MAPS =================
    wm = folium.Map(location=[30, 10], zoom_start=2, tiles="openstreetmap")
    for _, r in inv.iterrows():
        folium.CircleMarker((r.centroid_lat, r.centroid_lon), radius=6, color=SPLIT_COLORS[r.split], fill=True,
                            fill_opacity=0.85, tooltip=f"{r.trip_id} ({r.split}) n={r.n_points} UTM {r.utm_epsg}").add_to(wm)
    wm.save(str(maps_dir / "eda_world_overview.html"))

    gap_trip = q[q.split == "train"].sort_values("n_gaps", ascending=False).trip_id.iat[0]
    loop_trip = q[q.split == "train"].sort_values("route_repeated_arcs", ascending=False).trip_id.iat[0]
    for split, tid, why in [("train", tid, "typical urban trip"), ("train", gap_trip, "most GPS gaps"),
                            ("train", loop_trip, "most repeated route arcs"),
                            ("student_test", list_trips(split_dir(cfg, "student_test"))[0], "student_test example")]:
        m = trip_map(geos[(split, tid)], gap_thr, f"Trip {tid} ({split}) — {why}")
        m.save(str(maps_dir / f"eda_trip_{tid}.html"))
        print(f"map: eda_trip_{tid}.html ({why})")

    # ================= TEXT SUMMARY =================
    tq = q
    lines = [
        "# EDA summary (auto-generated by scripts/01_eda.py)",
        "", "Partitions analysed: train, student_test. evaluator_test not read.", "",
        "## Inventory",
        ss.round(1).to_markdown(),
        "", "## Data quality",
        f"- Missing values: track={int(tq.nan_track.sum())}, nodes={int(tq.nan_nodes.sum())}, arcs={int(tq.nan_arcs.sum())}",
        f"- Coordinate order lon,lat confirmed: latitude column within ±90 for all trips = {bool(tq.lat_in_range.all())}; "
        f"median share of GPS points inside network bbox = {tq.gps_inside_network_bbox.median():.3f}",
        f"- Arc/route ids within valid range: arcs={bool(tq.arc_node_ids_valid.all())}, route={bool(tq.route_ids_valid.all())}",
        f"- Non-increasing timestamps: {int(tq.non_increasing_t.sum())}",
        f"- Consecutive identical positions (stationary/jitter): {int(tq.duplicate_consecutive_positions.sum())}",
        f"- Steps exactly 1 s: {(all_dt == 1).mean():.4f}; steps ≤ 2 s: {(all_dt <= 2).mean():.4f}",
        f"- Gaps > {gap_thr:g}s: {int(tq.n_gaps.sum())} in {int((tq.n_gaps > 0).sum())} trips; max gap {all_dt.max():.0f}s",
        f"- Missing 1 Hz samples implied by gaps: {int(tq.missing_1hz_samples.sum())}",
        f"- Median moving speed: {med_speed_moving:.1f} m/s; implausible speeds (> {SPEED_OUTLIER_MPS:g} m/s): {int(tq.speed_outliers.sum())}",
        "", "## Road network",
        f"- Nodes per trip: median {tq.n_nodes.median():.0f}, arcs per trip: median {tq.n_arcs.median():.0f}",
        f"- Isolated nodes (in no arc): {int(tq.isolated_nodes.sum())}",
        f"- Self-loop arcs: {int(tq.self_loops.sum())}; duplicate arcs: {int(tq.duplicate_arcs.sum())}",
        f"- Two-way share (arc has reverse twin): median {tq.two_way_frac.median():.3f}",
        f"- Weak components per trip: median {tq.weak_components.median():.0f}, max {tq.weak_components.max()}; "
        f"largest SCC share: median {tq.largest_strong_frac.median():.3f}",
        f"- Arc length: median of trip medians {tq.arc_len_median_m.median():.1f} m, max {tq.arc_len_max_m.max():.0f} m",
        "", "## Reference routes",
        f"- Route breaks (consecutive arcs not connected): {int(tq.route_breaks.sum())} in {int((tq.route_breaks > 0).sum())} trips",
        f"- Repeated arcs in route (loops / revisits): {int(tq.route_repeated_arcs.sum())} in {int((tq.route_repeated_arcs > 0).sum())} trips",
        f"- Route uses a non-canonical duplicate arc: {int(tq.route_uses_duplicate_arc.sum())}; uses self-loop: {int(tq.route_uses_self_loop.sum())}",
        "", "## Projection",
        f"- Distinct UTM zones: {proj.utm_epsg.nunique()}",
        f"- Per-trip UTM median relative error: max over trips {proj.utm_rel_err_median.max() * 100:.3f}%",
        f"- Degree-Euclidean median relative error: median over trips {proj.deg_rel_err_median.median() * 100:.1f}%, worst {proj.deg_rel_err_median.max() * 100:.1f}%",
    ]
    (out / "EDA_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
