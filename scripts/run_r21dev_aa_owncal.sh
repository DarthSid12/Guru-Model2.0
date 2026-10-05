#!/usr/bin/env bash
# Score model 2's best checkpoint on the two face sets with separate UU fits.
set -euo pipefail
cd /home/siagrawal/combined_lpnet

PYTHON_BIN=${PYTHON_BIN:-/home/siagrawal/miniconda3/envs/themodel2/bin/python}
"$PYTHON_BIN" -c 'import antialiased_cnns, torch'

"$PYTHON_BIN" run_sim_seeds.py \
  --model r21dev_aa_s42 --gpu 3 --seeds 101-150 \
  --experiment rfwWM64 \
  --out-dir runs/sim_seeds_r21dev_aa_rfwWM64_owncal \
  --calib-on-eval-seeds

"$PYTHON_BIN" run_sim_seeds.py \
  --model r21dev_aa_s42 --gpu 3 --seeds 101-150 \
  --experiment cfdWM64 \
  --out-dir runs/sim_seeds_r21dev_aa_cfdWM64_owncal \
  --calib-on-eval-seeds
