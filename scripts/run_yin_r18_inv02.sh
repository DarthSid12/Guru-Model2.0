#!/usr/bin/env bash
# Yin (1969) battery for r18_inv02 -- the 2% inverted-exposure arm.
#
# Completes the inverted-exposure ladder: r16_rfw is the matched 0% control
# (same recipe, --invert-p 0 only), r18_inv05 and r18_inv20 already scored on
# 2026-09-12. Those two abolished the effect outright (cfdWM64 gap +8.21 at 0%
# -> -3.37 at 5% -> -0.58 at 20%), so 2% is the only point that can locate
# where the inversion effect actually dies.
#
# wm64_rfwcal calibrates noise per model to match upright performance before
# scoring, so the gaps are read across models, not the raw accuracies.
# Scored on the final epoch (run_sim_seeds._final_ckpt), as r16/r17/r18 all are.
set -u
MODEL=r18_inv02
GPU=${1:-6}
PY=/home/siagrawal/miniconda3/envs/lpnet/bin/python
cd /home/siagrawal/combined_lpnet
echo "=== START yin ${MODEL} gpu${GPU} $(date -Iseconds) ==="
"$PY" run_sim_seeds.py --model "$MODEL" --gpu "$GPU" \
  --experiment wm64_rfwcal --sims yin \
  > "logs/wm64rfwcal_${MODEL}.log" 2>&1
rc=$?
echo "=== END yin ${MODEL} rc=${rc} $(date -Iseconds) ==="
