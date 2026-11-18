@echo off
REM Start the PathSnap API for the final demo (Laptop 1). Run from the repo root.
if not exist .venv\Scripts\activate.bat (
  echo No virtual environment found in .venv - create it first, see "Setup" in README.md.
  exit /b 1
)
if not exist models\final\xgboost_tuned.joblib (
  echo models\final\xgboost_tuned.joblib is missing - the API would start without a model.
  exit /b 1
)
call .venv\Scripts\activate
echo.
echo Laptop 2 must use  http://YOUR-IPv4:8000  - this laptop's IPv4 address(es):
ipconfig | findstr /c:"IPv4"
echo.
uvicorn api.main:app --host 0.0.0.0 --port 8000
