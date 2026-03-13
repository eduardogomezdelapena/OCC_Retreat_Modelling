#!/usr/bin/env bash

# helper script to rebuild the big all_merged.gpkg dataset
# usage: ./scripts/regenerate_all_merged.sh

set -euo pipefail

echo "running merge_slr_sat.py to recreate outputs/all_merged.gpkg..."
PYTHONPATH="$(pwd)" python merge_slr_sat.py

echo "done.  the regenerated file will be placed in outputs/all_merged.gpkg"
