# FastAPI backend (Laptop 1)

Contract: ../docs/API_CONTRACT.md. Needs models/final/xgboost_tuned.joblib (scripts/07) and
models/final/hmm_config.json (scripts/09).

```bat
cd "C:\Users\abhay\Desktop\ML 1811\codes"
.venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
* Test locally: open http://localhost:8000/health and http://localhost:8000/docs (interactive Swagger page).
* Laptop 2 uses http://<this laptop's IPv4 from `ipconfig`>:8000
* Rehearse on student_test instead of evaluator_test:  `set PATHSNAP_API_SPLIT=student_test` before uvicorn.
* Allow Python through Windows Firewall when asked (private network). Use a phone hotspot for both laptops.
* Keep this terminal visible during the demo: every request is logged with id, trip, timings and accuracy.
