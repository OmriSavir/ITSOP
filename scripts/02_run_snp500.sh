#!/usr/bin/env bash
set -euo pipefail

# Runs the full S&P 500 pipeline: grid search -> test forecasting ->
# feature extraction -> baselines -> itsop.
#
# grid_search/, test_forecasting/, and feature_extraction/ are
# dataset-specific by file name and need no editing.
#
# baselines/*.py and itsop/*.py each hardcode DATASET near the top of the
# file. Before running the baselines/itsop stage, this script verifies that
# DATASET = "snp500" is actually set in all seven of them, and aborts with
# an explicit error if not -- forgetting this edit would otherwise silently
# rerun everything on covid19 instead of raising an error.
#
# Run from anywhere; this script cd's into the repository root itself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python grid_search/snp500_grid_search.py
python test_forecasting/snp500_test_forecasting.py
python feature_extraction/snp500_feature_extraction.py

REQUIRED_FILES=(
    baselines/autoforecast.py
    baselines/fforms.py
    baselines/random_k.py
    baselines/validation_oracle.py
    itsop/mlp_onehot.py
    itsop/mlp_features.py
    itsop/m_estimator.py
)

for f in "${REQUIRED_FILES[@]}"; do
    if ! grep -q '^DATASET = "snp500"' "$f"; then
        echo "ERROR: $f does not have DATASET = \"snp500\" set." >&2
        echo "Edit the DATASET line near the top of $f before running this script." >&2
        exit 1
    fi
done

python baselines/fforms.py
python baselines/autoforecast.py
python baselines/random_k.py
python baselines/validation_oracle.py
python itsop/mlp_onehot.py
python itsop/mlp_features.py
python itsop/m_estimator.py
