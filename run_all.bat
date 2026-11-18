@echo off
REM Phase 1 pipeline on Windows (run from the repo root with the venv activated)
python scripts\01_eda.py || exit /b 1
python scripts\02_build_labels.py || exit /b 1
python scripts\03_candidate_radius_study.py || exit /b 1
python scripts\04_build_features.py || exit /b 1
python scripts\05_train_compare.py || exit /b 1
python scripts\99_build_review_page.py || exit /b 1
echo Done. Open results\REVIEW.html
