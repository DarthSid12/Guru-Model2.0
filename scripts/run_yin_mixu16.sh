#!/usr/bin/env bash
# Follow-on Yin arm for the 2026-08-23 round, on faces_mixu16 instead of
# faces_mixed64.
#
# Why: on mixed64 all three models top out at 88-90% upright at p=0, below Yin's
# 96.29 anchor, so the joint fit pins at the floor and reports a ceiling rather
# than a calibration. faces_mixu16 (48 known + 16 unknown) is the rung of the
# Step 10 ladder where faces actually reach the anchor, so p has somewhere to go
# and the faces and houses anchors can be met at once.
#
# This WAITS for the in-flight r10/r11/r12 screens to finish first. The sims are
# CPU-bound and run_sim_seeds.py's own docstring records the cost of
# oversubscribing this box: 324s vs 12s for the same single condition with three
# extra sweeps running. Six at once would be worse than useless.
#
#   screen -dmS mixu16 scripts/run_yin_mixu16.sh

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=runs/logs_yin_mixu16.log
exec >>"$LOG" 2>&1

echo "=================================================================="
echo "[$(date -Is)] joint_zubud_mixu16 follow-on queued"
echo "=================================================================="

# --- wait for the headline round to clear -------------------------------
# The r10/r11/r12 screens exit only after their Kanwisher sweep returns.
while screen -ls 2>/dev/null | grep -qE '\.(r10|r11|r12)[[:space:]]'; do
  echo "[$(date -Is)] waiting: $(screen -ls 2>/dev/null | grep -cE '\.(r10|r11|r12)[[:space:]]') sweep(s) still running"
  sleep 300
done
echo "[$(date -Is)] box is clear -- starting mixu16 sweeps"

# --- three models, one GPU each, in parallel ----------------------------
# --calib-on-eval-seeds fits p on the same 100 seeds the sweep reports, which is
# the Step 7 fix; it is why calibration costs 100 samples per grid point rather
# than 8. --calib-range trims the coarse grid: p is known to sit low for these
# models (the mixed64 fits pinned at 0.00-0.05), so probing up to 0.50 would
# spend most of the budget on noise levels that collapse the task.
pids=()
gpu=0
for m in r10_dev_h80 r11_dev_h80_generic r12_dev_h80_enbloc; do
  echo "[$(date -Is)] launching $m on gpu $gpu"
  python run_sim_seeds.py --model "$m" --gpu "$gpu" \
    --experiment joint_zubud_mixu16 --seeds 101-200 \
    --calib-on-eval-seeds --calib-range 0.00 0.35 0.05 --threads 8 &
  pids+=($!)
  gpu=$((gpu + 1))
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "[warn] sweep pid $p returned nonzero"; rc=1; }
done

echo "[$(date -Is)] mixu16 sweeps finished (rc=$rc)"
echo "[$(date -Is)] results in runs/sim_seeds_joint_zubud_mixu16/"
