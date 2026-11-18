# Evaluator UI (Laptop 2) — Phase 4

Lists the 10 evaluation trips, sends the chosen trip id to the API on Laptop 1, and draws the returned
route and metrics on an interactive map. Needs no dataset and no GPU — just Python (for a static file
server) and a browser. Leaflet loads from the CDN, falling back to the bundled copy in `vendor/leaflet/`;
the street-map background (OpenStreetMap tiles) needs internet.

Plain HTML/JS, no build step:

```
ui/
├── index.html
├── app.js
├── style.css
├── vendor/leaflet/       # bundled Leaflet 1.9.4 — used automatically if the CDN is blocked
├── mock/                 # sample data for developing without the API (Mock mode only)
│   ├── health.json
│   ├── trips.json
│   └── predict_sample.json
└── README.md
```

The UI talks **only** to the API: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`.
Every result shown comes from a live request (no caching); it never receives or shows ground truth.

## How to run

```bat
cd ui
python -m http.server 5500
```

Then open **http://localhost:5500** in a browser on Laptop 2.
(Don't double-click `index.html` — opened as a file, the browser blocks some requests.)

## Setting the API URL

1. At the top of the page, type Laptop 1's address into **API base URL**, e.g. `http://192.168.1.10:8000`
   (default). It is saved in the browser, so you only do this once.
2. Click **Test connection** (or press Enter). Green = connected, and it shows the model name.
   Red = not connected; the panel on the right lists what to check.
3. Pick a trip in the list → **Run** (or double-click the trip). A spinner shows while Laptop 1 works (1–10 s).

## Finding Laptop 1's IP address

On **Laptop 1** (Windows): open Command Prompt and run

```bat
ipconfig
```

Under the adapter you are using (usually **Wireless LAN adapter Wi-Fi**), read **IPv4 Address**, e.g.
`192.168.1.10`. The API URL is then `http://192.168.1.10:8000`.
The IP can change when you switch networks (e.g. to a phone hotspot) — check again after switching.

## Troubleshooting checklist

Work through these in order if **Test connection** turns red:

- [ ] **Same network.** Both laptops on the same Wi-Fi / hotspot. College Wi-Fi often blocks
      device-to-device traffic → use a **phone hotspot** for both laptops.
- [ ] **API listening on the network.** On Laptop 1 the API was started with
      `--host 0.0.0.0 --port 8000` (with `127.0.0.1` other laptops can't reach it).
- [ ] **Firewall.** Windows Firewall on Laptop 1 allows inbound TCP 8000 — or click **Allow** on the
      Windows popup for Python the first time the API starts (tick Private networks).
- [ ] **Health check in the browser.** From Laptop 2's browser open `http://<IP>:8000/health`.
      It must show JSON like `{"status":"ok",...}` before the UI can work.
- [ ] **Right IP in the UI.** The API base URL matches Laptop 1's current IPv4 address and port 8000.

What the error messages mean:

| Message in the UI | Meaning |
|---|---|
| Cannot reach the API | Network problem — wrong IP, different Wi-Fi, firewall, or API not running. Use the checklist. |
| Unknown trip (404) | The API doesn't know that trip_id — click **Reload** on the trips list. |
| Model not loaded (503) | The API is up but the model isn't ready — wait, or restart the API on Laptop 1. |
| Server error (500) | Something failed inside the API — check the API terminal on Laptop 1. |
| No response after 60 s | Laptop 1 is stuck or unreachable — check its terminal. |

Map background is plain grey but routes and metrics show → Laptop 2 has no internet, so OpenStreetMap
tiles can't load (a note appears on the map). Everything else still works; turn on mobile data on the hotspot phone
to get the street map back.

## Mock mode (development only)

Tick **Mock mode** to read `mock/*.json` instead of calling the API (useful before the API exists).
A striped **MOCK DATA** banner is shown while it's on, and it returns the same sample result for every trip.
It is **off by default** on every page load and is never saved — the real demo must use the live API.
