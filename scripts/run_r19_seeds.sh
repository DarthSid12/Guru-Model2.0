#!/usr/bin/env bash
# Launch the three r19_rfwh replicates -- same recipe, three different weight
# initialisations -- detached, one per free GPU.
#
#   scripts/run_r19_seeds.sh
#
# Seeds 42/43/44. 42 matches the seed every r15/r16/r17 run used, so
# r19_rfwh_s42 is the arm that is directly comparable to the existing battery;
# 43 and 44 measure how much of any difference is just initialisation. This is
# the variance control the repo has never had: every cross-model number in
# paper/results_rfwcal_battery.md is a single training run per model, so a
# 5-point gap between two rounds has never been checked against the spread you
# get from re-rolling the same recipe.
#
# --curriculum-seed stays at 0 in all three, so the curriculum content is
# identical and initialisation is the only variable.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 43 44)
GPUS=(0 6 7)

# Pre-warm the NFS page cache ONCE, not three times concurrently. A cold cache
# drops training to ~45 img/s -- the difference between a ~18 h run and a
# multi-day one. All three arms read the same packed arrays.
echo "[$(date -Is)] pre-warming packed arrays..."
for c in faces faces_vgg faces_rfwW faces_rfwO objects houses_zubud137 houses; do
  cat fixation_data/"$c"/*/*.npy > /dev/null 2>&1
done
echo "[$(date -Is)] cache warm."

for i in "${!SEEDS[@]}"; do
  SEED="${SEEDS[$i]}"; GPU="${GPUS[$i]}"
  LOG="runs/logs_r19_rfwh_s${SEED}.log"
  echo "[$(date -Is)] launching seed $SEED on cuda:$GPU -> $LOG"
  setsid nohup bash scripts/train_r19_rfwh.sh "cuda:${GPU}" "$SEED" \
    >>"$LOG" 2>&1 < /dev/null &
  sleep 5   # stagger, so three processes do not build the label map at once
done

echo
echo "[$(date -Is)] all three launched. Progress:"
echo "  scripts/status_r19.sh"
