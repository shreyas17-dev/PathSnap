@echo off
REM Start the PathSnap API for the final demo (Laptop 1)
call .venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
