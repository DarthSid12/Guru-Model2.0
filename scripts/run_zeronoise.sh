#!/usr/bin/env bash
# Zero-noise Yin battery: all four conditions at p=0, calibration skipped.
# Answers "what is UI when retrieval noise is removed entirely" -- i.e. how much
# of the UI deficit is representational rather than noise-induced.
#   scripts/run_zeronoise.sh <GPU>
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GPU="${1:?usage: run_zeronoise.sh GPU}"
OUT="runs/zeronoise"; CLAIM="$OUT/.claims"; mkdir -p "$OUT" "$CLAIM"
MODELS=(r19_rfwh_s42 r19_rfwh_s43 r19_rfwh_s44 r19_rfwh_s45 r19_rfwh_s46 r19_rfwh_s47)
CATS=(faces_rfwWM64 faces_cfdWM64)
SEEDS=$(seq 101 125)
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
        --seed "$S" --noise 0 --noise-inv 0 --noise-phase study \
        --shuffle-items > "$LOG" 2>&1 \
        || { echo "FAILED $U"; rmdir "$CLAIM/$U" 2>/dev/null; }
    done
  done
done
echo "gpu$GPU: no work left."
