#!/usr/bin/env bash
# Score the finished r19_rfwh seed replicates on the rfwWM64-calibrated Yin
# battery -- the same experiment, store set, seed range and calibration
# procedure as every r15/r16/r17 row in paper/results_rfwcal_battery.md, so the
# r19 rows drop straight into that table.
#
# Deliberately NOT passing --calib-on-eval-seeds. It is the better procedure
# (the existing rows fit p on 8 probe seeds at base seed 42 but report on
# 101-150, so the fit is out-of-sample and drifts), but every model already in
# the battery was fitted the old way. Matching them matters more here than
# being right in isolation; switching it on would mean rescoring all seven
# existing models to keep the table internally consistent.
#
# Sequential on one GPU: three concurrent sims on a box where other people are
# training is how r17_houses lost seeds 118-124 to a co-tenant OOM.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-2}"
shift || true
MODELS=("${@:-r19_rfwh_s42 r19_rfwh_s43 r19_rfwh_s44}")
[ "$#" -eq 0 ] && MODELS=(r19_rfwh_s42 r19_rfwh_s43 r19_rfwh_s44)

echo "=================================================================="
echo "[$(date -Is)] r19 batch-1 sims on GPU $GPU: ${MODELS[*]} (pid $$)"
echo "=================================================================="

for M in "${MODELS[@]}"; do
  echo "[$(date -Is)] === $M : wm64_rfwcal, seeds 101-150 ==="
  python run_sim_seeds.py --model "$M" --gpu "$GPU" \
    --experiment wm64_rfwcal --seeds 101-150 --threads 8
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date -Is)] $M FAILED rc=$rc -- continuing to next model"
  else
    echo "[$(date -Is)] $M done."
  fi
done

echo "[$(date -Is)] ALL r19 batch-1 sims finished."
