#!/usr/bin/env bash
# Validate --noise-phase on a real model: sweep p under both/study and compare.
# Launched as a SCRIPT under setsid (not as a bare backgrounded command), which
# is what actually survives the parent session going away.
set -uo pipefail
cd "$(dirname "$0")/.."

# Single-instance lock. An earlier launch of this script ran twice, so five
# (p, phase) combos had two processes writing the SAME log concurrently --
# interleaved output and wasted GPU. mkdir is atomic, so a second copy exits.
LOCK=runs/phasecheck/.lock
mkdir -p runs/phasecheck
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[$(date -Is)] another phasecheck is running (lock $LOCK); exiting."
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
D=$(ls -d runs/*_r19_rfwh_s42 | head -1); CKPT=$(ls "$D"/final_model_*.pth | head -1)
GPUS=(0 6 7); i=0
# One thread per process. torch defaults to 32 here on a 64-core box, so even a
# handful of concurrent sims oversubscribe the machine and fight each other.
# The sim is CPU-bound (the model forward is 5% of runtime), so threads are the
# scarce resource, not GPUs.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for p in 0.0 0.1 0.3 0.5 0.7 0.9 0.99; do
  for phase in both study; do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g python -u yin_tests/simulate_yin1969.py \
      --category faces_cfdWM64 --run-dir "$D" --checkpoint "$CKPT" \
      --device cuda:0 --shuffle-items --seed 101 \
      --noise "$p" --noise-phase "$phase" \
      > "runs/phasecheck/${phase}_p${p}.log" 2>&1 &
    i=$((i+1))
  done
done
wait
echo "[$(date -Is)] phasecheck complete"
