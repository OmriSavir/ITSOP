#!/usr/bin/env bash
set -euo pipefail

# Installs the Python dependencies used across the pipeline.
# cvxpy is used with the SCS solver for the M-estimator (itsop/m_estimator.py).

pip install pandas numpy torch prophet statsmodels scikit-learn scipy cvxpy tsfresh pycatch22 tsfeatures
