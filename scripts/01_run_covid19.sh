#!/usr/bin/env bash
set -euo pipefail

# Runs the full COVID-19 pipeline: data preparation -> grid search ->
# test forecasting -> feature extraction -> baselines -> itsop.
#
# DATASET = "covid19" is already the default in every baseline/itsop
# script, so no manual edits are required to run this dataset.
#
# Run from anywhere; this script cd's into the repository root itself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python data_preparation/covid19/create_dataset.py
python grid_search/covid19_grid_search.py
python test_forecasting/covid19_test_forecasting.py
python feature_extraction/covid19_feature_extraction.py
python baselines/fforms.py
python baselines/autoforecast.py
python baselines/random_k.py
python baselines/validation_oracle.py
python itsop/mlp_onehot.py
python itsop/mlp_features.py
python itsop/m_estimator.py
