"""Raw-data loading for PathSnap.

File formats (verified on all 90 student trips, see scripts/01_eda.py):
  <trip>.track : lon  lat  t        (whitespace separated, no header, t in seconds)
  <trip>.nodes : lon  lat           (row index = node_id, 0-based)
  <trip>.arcs  : from_node to_node  (row index = arc_id, 0-based, directed)
  <trip>.route : arc_id             (chronological ground-truth arc sequence)

The road network is supplied *per trip*: node/arc ids are only meaningful inside one trip folder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the repo root so scripts work from any working directory.
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((REPO_ROOT / v).resolve())
    return cfg


def split_dir(cfg: dict, split: str) -> Path:
    if split == "evaluator_test" and not _allow_evaluator():
        raise PermissionError(
            "evaluator_test must not be read by development scripts (leakage rule). "
            "Only the API process may load it (set PATHSNAP_ALLOW_EVAL=1 there)."
        )
    return Path(cfg["paths"]["raw_dir"]) / cfg["splits"][split]


def _allow_evaluator() -> bool:
    import os

    return os.environ.get("PATHSNAP_ALLOW_EVAL") == "1"


def list_trips(directory: str | Path) -> list[str]:
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame          # columns lon, lat, t
    nodes: pd.DataFrame          # columns lon, lat
    arcs: pd.DataFrame           # columns u, v
    route: np.ndarray | None     # int arc ids, or None if not supplied
    folder: Path

    @property
    def n_points(self) -> int:
        return len(self.track)


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")
    if df.shape[1] != len(names):
        raise ValueError(f"{path.name}: expected {len(names)} columns, got {df.shape[1]}")
    return df


def load_trip(folder: str | Path, load_route: bool = True) -> Trip:
    folder = Path(folder)
    tid = folder.name
    track = _read(folder / f"{tid}.track", ["lon", "lat", "t"], np.float64)
    nodes = _read(folder / f"{tid}.nodes", ["lon", "lat"], np.float64)
    arcs = _read(folder / f"{tid}.arcs", ["u", "v"], np.int64)
    route = None
    rp = folder / f"{tid}.route"
    if load_route and rp.exists():
        route = pd.read_csv(rp, header=None, names=["arc"], dtype=np.int64)["arc"].to_numpy()
    return Trip(tid, track, nodes, arcs, route, folder)
