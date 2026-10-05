#!/usr/bin/env bash
# Two-noise Yin battery, ONE noise pair per (model, seed), shared across all
# categories -- the same structure the single-noise battery uses, where p is
# fitted on faces_rfwWM64 alone and the other categories are measured at that p
# without a vote.
#
#   scripts/run_twonoise.sh <GPU>     # start a worker; run many
#
# Work unit = (model, seed). Within a unit:
#   1. calibrate on faces_rfwWM64 -> p_up (UU=0.96), p_inv (II=0.82).
#      That run also produces faces_rfwWM64's own four condition rows.
#   2. run faces_cfdWM64, houses_yin64, objects at those SAME two noises,
#      with calibration skipped (--noise / --noise-inv).
# So calibration is per seed, and all four categories share one noise pair.
#
# Units are claimed atomically with mkdir, so workers need no coordination and
# can be added or killed at will.
set -uo pipefail
cd "$(dirname "$0")/.."

# One thread per process: torch grabs 32 by default on this 64-core box, and the
# sim is CPU-bound (compute_familiarity_score is 61% of runtime, the model
# forward 5%), so cores are the scarce resource, not GPUs.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

GPU="${1:?usage: run_twonoise.sh GPU}"
PHASE="${NOISE_PHASE:-study}"
OUT="runs/cyl_${PHASE}"; CLAIM="$OUT/.claims"; mkdir -p "$OUT" "$CLAIM"

MODELS=(r20_cyl_s42)
# Each face store gets its OWN fitted noise pair. houses_yin64 and objects are
# deliberately out for now and will be run later.
CATS=(faces_rfwWM64 faces_cfdWM64)
SEEDS=$(seq 101 150)   # 50, matching the single-noise battery's eval seeds

# Face stores need --shuffle-items for the same reason the battery sets
# yin_shuffle on them; houses/objects keep their packed order.
shuffle_for() { case "$1" in faces_*) echo "--shuffle-items";; *) echo "";; esac; }

for M in "${MODELS[@]}"; do
  D=$(ls -d runs/*_"$M" 2>/dev/null | head -1); [ -z "$D" ] && continue
  CKPT=$(ls "$D"/final_model_*.pth 2>/dev/null | head -1); [ -z "$CKPT" ] && continue
  for C in "${CATS[@]}"; do
    for S in $SEEDS; do
      U="${M}__${C}__seed${S}"
      mkdir "$CLAIM/$U" 2>/dev/null || continue
      LOG="$OUT/${U}.log"
      CUDA_VISIBLE_DEVICES="$GPU" python -u simulate_yin1969_bothnoise.py \
        --category "$C" --run-dir "$D" --checkpoint "$CKPT" --device cuda:0 \
        --seed "$S" --noise-phase "$PHASE" $(shuffle_for "$C") > "$LOG" 2>&1
      if grep -q "upright p=" "$LOG"; then
        echo "[$(date -Is)] gpu$GPU  done $U  $(grep -o 'upright p=[0-9.]*  *inverted p=[0-9.]*' "$LOG" | head -1)"
      elif grep -q "UNREACHABLE" "$LOG"; then
        echo "[$(date -Is)] gpu$GPU  UNREACHABLE $U (kept, no retry)"
      else
        echo "[$(date -Is)] gpu$GPU  FAILED $U"
        rmdir "$CLAIM/$U" 2>/dev/null
      fi
    done
  done
done
echo "[$(date -Is)] gpu$GPU: no work left."
