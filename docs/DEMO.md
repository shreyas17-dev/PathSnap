# Running the live demo (two laptops)

Laptop 1 runs the model and the FastAPI backend. Laptop 2 runs only the evaluator UI and talks to
Laptop 1 over the local network. Do a full dry run with both laptops on the hotspot before the review.

## 1. Same network
- Connect **both** laptops to the same **phone hotspot**. College Wi-Fi often blocks device-to-device traffic.
- On Laptop 1: Settings → Network & internet → Wi-Fi → (the hotspot) → Network profile type → **Private**.

## 2. Laptop 1 — start the API
```bat
cd PathSnap
run_api.bat
```
`run_api.bat` prints this laptop's IPv4 address(es) and starts `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
The address under **Wireless LAN adapter Wi-Fi** (e.g. `10.134.135.154`) is the one Laptop 2 needs.
The IP can change every time you reconnect to the hotspot, so check it again on the day.

First time only, allow port 8000 through Windows Firewall (Command Prompt **as Administrator**):
```bat
netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
```
or click **Allow** (private networks) on the Windows popup for Python.

Check on Laptop 1: `http://localhost:8000/health` shows `{"status":"ok", ...}`.
Keep this terminal visible during the demo: every request is logged with its id, trip, timings and accuracy.

## 3. Laptop 2 — start the UI
```bat
cd PathSnap\ui
python -m http.server 5500
```
1. Open `http://<Laptop 1 IP>:8000/health` in the browser: it must show JSON.
2. Open **http://localhost:5500**, type `http://<Laptop 1 IP>:8000` into **API base URL**, click **Test connection** → green.
3. Pick a trip → **Run**. Mock mode must stay **off**.

## 4. If it doesn't connect

| Symptom | Fix |
|---|---|
| `/health` works on Laptop 1 but not from Laptop 2 | Firewall rule (step 2) or network profile set to Public (step 1) |
| `Test-NetConnection <IP> -Port 8000` on Laptop 2 says `False` | Not on the same hotspot, or the phone isolates devices: try another phone |
| UI says "Cannot reach the API" but `/health` opens in the browser | Wrong API base URL in the UI, or CORS settings changed in `api/main.py` |
| Worked before, fails now | The IP changed: run `ipconfig` on Laptop 1 again |
| Map background is grey, routes still show | Laptop 2 has no internet for map tiles: turn on mobile data on the hotspot |

Don't use `ping` to test: Windows blocks it by default even when everything works.
More UI details: [`ui/README.md`](../ui/README.md).
