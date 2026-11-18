"""Folium map builders (EDA now; predicted routes are added in later phases)."""
from __future__ import annotations

import folium
import numpy as np
from folium.plugins import Fullscreen, MeasureControl

from .network import TripGeo

LEGEND_HTML = """
<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;
border:1px solid #999;border-radius:6px;font-size:13px;line-height:1.6">
<b>{title}</b><br>{rows}</div>"""


def _legend(m: folium.Map, title: str, items: list[tuple[str, str]]):
    rows = "".join(f'<span style="color:{c};font-size:16px">&#9632;</span> {lbl}<br>' for lbl, c in items)
    m.get_root().html.add_child(folium.Element(LEGEND_HTML.format(title=title, rows=rows)))


def nearby_arcs(g: TripGeo, buffer_m: float = 150.0) -> np.ndarray:
    """Usable arc ids whose midpoint lies within buffer of any GPS point (keeps HTML small)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.c_[g.px, g.py])
    mx = (g.ax1 + g.ax2) / 2
    my = (g.ay1 + g.ay2) / 2
    d, _ = tree.query(np.c_[mx, my], k=1)
    return np.flatnonzero(g.usable & (d <= buffer_m + g.arc_len / 2))


def _arc_lines(g: TripGeo, arc_ids: np.ndarray):
    lon_u, lat_u = g.trip.nodes.lon.to_numpy()[g.u[arc_ids]], g.trip.nodes.lat.to_numpy()[g.u[arc_ids]]
    lon_v, lat_v = g.trip.nodes.lon.to_numpy()[g.v[arc_ids]], g.trip.nodes.lat.to_numpy()[g.v[arc_ids]]
    return [[(a, b), (c, d)] for a, b, c, d in zip(lat_u, lon_u, lat_v, lon_v)]


def route_polyline(g: TripGeo, arc_seq: np.ndarray):
    lon, lat = g.trip.nodes.lon.to_numpy(), g.trip.nodes.lat.to_numpy()
    pts = []
    for a in arc_seq:
        for n in (g.u[a], g.v[a]):
            p = (lat[n], lon[n])
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts


def trip_map(g: TripGeo, gap_threshold_s: float, title: str, extra_layers=None, extra_legend=None) -> folium.Map:
    tr = g.trip.track
    m = folium.Map(location=[tr.lat.median(), tr.lon.median()], zoom_start=14, tiles=None,
                   control_scale=True)
    folium.TileLayer("openstreetmap", name="OpenStreetMap").add_to(m)

    net = folium.FeatureGroup(name="Road network (arcs near track)", show=True)
    folium.PolyLine(_arc_lines(g, nearby_arcs(g)), color="#9a9a9a", weight=1.5, opacity=0.7).add_to(net)
    net.add_to(m)

    if g.trip.route is not None:
        ref = folium.FeatureGroup(name="Reference route (.route)", show=True)
        folium.PolyLine(route_polyline(g, g.trip.route), color="#1a9850", weight=6, opacity=0.75,
                        tooltip="reference route").add_to(ref)
        ref.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS observations", show=True)
    step = max(1, len(tr) // 1500)  # cap markers for very long trips; every point kept in the polyline
    for i in range(0, len(tr), step):
        folium.CircleMarker((tr.lat.iat[i], tr.lon.iat[i]), radius=2, color="#2166ac", fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"#{i} t={tr.t.iat[i]:.0f}s").add_to(gps)
    gps.add_to(m)

    dt = np.diff(tr.t.to_numpy())
    gaps = np.flatnonzero(dt > gap_threshold_s)
    if len(gaps):
        gl = folium.FeatureGroup(name=f"GPS gaps (> {gap_threshold_s:g}s)", show=True)
        for i in gaps:
            folium.PolyLine([(tr.lat.iat[i], tr.lon.iat[i]), (tr.lat.iat[i + 1], tr.lon.iat[i + 1])],
                            color="#d73027", weight=3, dash_array="6", tooltip=f"gap {dt[i]:.0f}s").add_to(gl)
        gl.add_to(m)

    folium.Marker((tr.lat.iat[0], tr.lon.iat[0]), tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((tr.lat.iat[-1], tr.lon.iat[-1]), tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    for layer in extra_layers or []:
        layer.add_to(m)

    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    MeasureControl(primary_length_unit="meters").add_to(m)
    items = [("Road network", "#9a9a9a"), ("Reference route", "#1a9850"), ("Raw GPS", "#2166ac"),
             ("GPS gap", "#d73027")] + list(extra_legend or [])
    _legend(m, title, items)
    return m
