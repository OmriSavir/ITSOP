#!/usr/bin/env bash
set -euo pipefail

# Rank / incoherence analysis across all three datasets. Requires
# grid_search/*.py to have already been run for covid19, snp500, and
# electricity (this script does not run them itself).
#
# Run from anywhere; this script cd's into the repository root itself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python analysis/rank_incoherence_estimation.py
