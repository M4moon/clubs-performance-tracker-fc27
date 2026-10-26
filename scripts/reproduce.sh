#!/usr/bin/env bash
# Rebuild every table, chart, model output and the static dashboard from data/raw.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/model
python src/build_tables.py data/raw data/processed
python model/train_model.py data/raw results/model/model_latest.json    # about 30 seconds
python model/review_model.py data/raw results/model/review.json         # about 10 seconds
python src/analyze.py data/raw data/processed results
python dashboard/build_static.py data/raw results/model docs/index.html
python -m unittest discover -s tests
