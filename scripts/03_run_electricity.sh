#!/usr/bin/env bash
set -euo pipefail

# Runs the full electricity pipeline: grid search -> test forecasting ->
# feature extraction -> baselines -> itsop.
#
# The electricity dataset is proprietary and not included in this
# repository. consumption_dataset.pkl and
# electricity_feature_extraction_catch22.pkl (with a city_Electricity
# column) must be placed in data/electricity/ before running this script --
# this script checks for both and aborts with an explicit error if either
# is missing.
#
# As with 02_run_snp500.sh, this script also verifies that DATASET =
# "electricity" is set in all seven baselines/itsop scripts before running
# them, since forgetting that edit would otherwise silently rerun on
# covid19 instead of raising an error.
#
# Run from anywhere; this script cd's into the repository root itself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="${1:-data/electricity}"

REQUIRED_DATA_FILES=(
    "$DATA_DIR/consumption_dataset.pkl"
    "$DATA_DIR/electricity_feature_extraction_catch22.pkl"
)

for f in "${REQUIRED_DATA_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f" >&2
        echo "Place the electricity dataset files in $DATA_DIR before running this script." >&2
        exit 1
    fi
done

python grid_search/electricity_grid_search.py --data-dir "$DATA_DIR"
python test_forecasting/electricity_test_forecasting.py --data-dir "$DATA_DIR"
python feature_extraction/electricity_feature_extraction.py --data-dir "$DATA_DIR"

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
    if ! grep -q '^DATASET = "electricity"' "$f"; then
        echo "ERROR: $f does not have DATASET = \"electricity\" set." >&2
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
