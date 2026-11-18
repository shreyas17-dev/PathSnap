# PathSnap API contract (Laptop 1 API  <->  Laptop 2 UI)

Single source of truth for both sides. The API (Laptop 1, `api/`) implements exactly this; the UI
(Laptop 2, `ui/`) consumes exactly this. Change it only by agreement, and update this file first.

## Network
* API base URL: `http://<LAPTOP1_LAN_IP>:8000` (both laptops on the same Wi-Fi / hotspot).
* API runs `uvicorn api.main:app --host 0.0.0.0 --port 8000`; CORS allows all origins.
* JSON everywhere, UTF-8. Coordinates are WGS84 `[lat, lon]` pairs (Leaflet order).

## Rules (from the hackathon guidelines)
* The UI sends ONLY a trip id. All prediction and accuracy computation happens in the API.
* Ground-truth labels / reference route are NEVER returned: no reference polyline, no per-point
  correctness flags. Only aggregate metrics.
* No hardcoded or cached outputs on the UI side: every result shown comes from a live API call.

## Endpoints

### GET /health
```json
{"status": "ok", "model": "XGBoost (Bayesian-tuned)", "hmm": true, "n_trips": 10, "version": "1.0"}
```

### GET /trips
The 10 evaluator trips (ids and non-label metadata only).
```json
{"trips": [
  {"trip_id": "00000015", "n_points": 6854, "duration_s": 7020.0,
   "center": [52.51, 13.40], "bbox": [[52.48, 13.35], [52.54, 13.45]]}
]}
```

### POST /predict
Request:
```json
{"trip_id": "00000015"}
```
Response 200:
```json
{
  "trip_id": "00000015",
  "n_points": 6854,
  "bbox": [[52.48, 13.35], [52.54, 13.45]],
  "gps": [[52.5012, 13.4011], [52.5013, 13.4013]],
  "predicted_route": [[[52.5011, 13.4009], [52.5016, 13.4020]], [[52.5100, 13.4100], [52.5105, 13.4110]]],
  "classifier_route": [[[52.5011, 13.4009], [52.5016, 13.4020]]],
  "metrics": {
    "hmm":            {"top1_accuracy": 0.912, "route_continuity": 0.998, "n_correct": 6251, "route_f1": 0.95},
    "classifier_only":{"top1_accuracy": 0.871, "route_continuity": 0.974, "n_correct": 5970, "route_f1": 0.90},
    "n_points": 6854
  },
  "timing_ms": {"preprocess": 850, "classifier": 120, "hmm": 400, "total": 1400},
  "model": "XGBoost (Bayesian-tuned) + HMM/Viterbi",
  "request_id": "a1b2c3"
}
```
* `gps`: raw GPS observations in time order (draw as small dots).
* `predicted_route`: final route (classifier + HMM/Viterbi) as a list of polyline SEGMENTS
  (list of lists of [lat, lon]); a new segment starts after a GPS gap. Draw each as a polyline.
* `classifier_route`: same format, classifier-only predictions (optional comparison layer).
* `metrics.*.top1_accuracy`: fraction of GPS points whose predicted arc equals the API's internal
  reference arc (0-1). Show as a percentage.

Errors: `404 {"detail": "unknown trip_id"}`, `503 {"detail": "model not loaded"}`,
`500 {"detail": "<message>"}`. A request can take ~1-10 s.
