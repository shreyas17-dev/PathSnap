#!/usr/bin/env bash
set -e
python scripts/01_eda.py
python scripts/02_build_labels.py
python scripts/03_candidate_radius_study.py
python scripts/04_build_features.py
python scripts/05_train_compare.py
python scripts/99_build_review_page.py
echo "Done. Open results/REVIEW.html"
