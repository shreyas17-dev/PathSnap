"""PathSnap FastAPI backend (Laptop 1). Contract: docs/API_CONTRACT.md

Start (from the repo root, venv active):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

* Loads models/final/xgboost_tuned.joblib + models/final/hmm_config.json once at startup.
* Serves the evaluator_test trips (set PATHSNAP_API_SPLIT=student_test to rehearse on student_test).
* Every /predict runs the full pipeline live (no cached or hardcoded results) and logs to this console:
  request id, trip, timings and accuracy - show this terminal to the evaluators.
* Ground truth never leaves this process: only aggregate metrics are returned.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PATHSNAP_ALLOW_EVAL", "1")   # only the API process may read evaluator_test

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.io import list_trips, load_config, load_trip, split_dir
from pathsnap.pipeline import arcs_to_segments, load_artifacts, run_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

SPLIT = os.environ.get("PATHSNAP_API_SPLIT", "evaluator_test")
CFG = load_config()
MODELS_DIR = Path(CFG["paths"]["models_dir"]) / "final"
STATE: dict = {"model": None, "hmm": None, "trips": {}}

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.on_event("startup")
def startup():
    try:
        STATE["model"], STATE["hmm"] = load_artifacts(MODELS_DIR)
        log.info(f"model loaded from {MODELS_DIR} | HMM config {STATE['hmm']}")
    except Exception as e:
        log.error(f"MODEL NOT LOADED: {e}")
    folder = split_dir(CFG, SPLIT)
    for tid in list_trips(folder):
        tr = load_trip(folder / tid, load_route=False).track
        STATE["trips"][tid] = {
            "trip_id": tid, "n_points": int(len(tr)), "duration_s": float(tr.t.iloc[-1] - tr.t.iloc[0]),
            "center": [round(float(tr.lat.median()), 6), round(float(tr.lon.median()), 6)],
            "bbox": [[float(tr.lat.min()), float(tr.lon.min())], [float(tr.lat.max()), float(tr.lon.max())]],
        }
    log.info(f"serving {len(STATE['trips'])} trips from '{SPLIT}': {list(STATE['trips'])}")


@app.get("/health")
def health():
    return {"status": "ok" if STATE["model"] is not None else "model not loaded",
            "model": "XGBoost (Bayesian-tuned)", "hmm": STATE["hmm"] is not None,
            "n_trips": len(STATE["trips"]), "split": SPLIT, "version": "1.0"}


@app.get("/trips")
def trips():
    return {"trips": list(STATE["trips"].values())}


@app.post("/predict")
def predict(req: PredictRequest):
    rid = uuid.uuid4().hex[:6]
    tid = req.trip_id.strip()
    log.info(f"[{rid}] POST /predict trip_id={tid}")
    if tid not in STATE["trips"]:
        log.warning(f"[{rid}] unknown trip_id {tid}")
        raise HTTPException(status_code=404, detail="unknown trip_id")
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        r = run_trip(split_dir(CFG, SPLIT) / tid, STATE["model"], STATE["hmm"], CFG, score=True)
    except Exception as e:
        log.exception(f"[{rid}] pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))
    g = r["g"]; tr = g.trip.track
    metrics = {"n_points": r["n_points"]}
    for k in ["hmm", "classifier_only"]:
        if k in r:
            metrics[k] = {"top1_accuracy": round(r[k]["top1_accuracy"], 4),
                          "route_continuity": round(r[k]["route_continuity"], 4),
                          "n_correct": r[k]["n_correct"],
                          "route_f1": round(r[k]["route_f1"], 4)}
    timing = {k: round(v) for k, v in r["timing_ms"].items()}
    h = metrics.get("hmm", {}); c = metrics.get("classifier_only", {})
    log.info(f"[{rid}] trip {tid}: {r['n_points']} points | classifier Top-1={c.get('top1_accuracy')} | "
             f"HMM Top-1={h.get('top1_accuracy')} continuity={h.get('route_continuity')} | timing_ms={timing}")
    return {
        "trip_id": tid,
        "n_points": r["n_points"],
        "bbox": STATE["trips"][tid]["bbox"],
        "gps": np.round(np.c_[tr.lat.to_numpy(), tr.lon.to_numpy()], 6).tolist(),
        "predicted_route": arcs_to_segments(g, r["hmm_arcs"], r["gap_before"]),
        "classifier_route": arcs_to_segments(g, r["clf_arcs"], r["gap_before"]),
        "metrics": metrics,
        "timing_ms": timing,
        "model": "XGBoost (Bayesian-tuned) + HMM/Viterbi",
        "request_id": rid,
    }
