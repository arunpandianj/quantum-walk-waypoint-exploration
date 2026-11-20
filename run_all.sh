#!/usr/bin/env bash
# Run the experiments and the analysis from the repository root.
#   bash run_all.sh quick   -> smoke test of experiments 01-03 only (a few minutes); the other steps are skipped
#   bash run_all.sh         -> full reproduction of the reported results (several hours)
set -euo pipefail
if [ "${1:-}" = "quick" ]; then
  export QUICK=1
  python experiments/01_glued_trees_hypercubes.py
  python experiments/02_corridors_grids_rooms.py
  python experiments/03_exact_theory_full_graphs.py
  exit 0
fi
python experiments/01_glued_trees_hypercubes.py
python experiments/02_corridors_grids_rooms.py
python experiments/03_exact_theory_full_graphs.py
python experiments/04_exact_reduced_chain.py
python experiments/05_spreading_law.py
python analysis/make_figures.py
python analysis/make_tables.py
