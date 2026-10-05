#!/usr/bin/env bash
# UI/IU only, noise on BOTH phases, perception noise keyed to the TEST
# orientation: UI = study p_up / test p_inv, IU = study p_inv / test p_up.
# Calibration still fits p_up on UU and p_inv on II per (model, store, seed).
#   yin_tests/run_percept.sh <GPU>
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GPU="${1:?usage: run_percept.sh GPU}"
OUT="runs/percept_bytest"; CLAIM="$OUT/.claims"; mkdir -p "$OUT" "$CLAIM"
MODELS=(r19_rfwh_s42 r19_rfwh_s43 r19_rfwh_s44 r19_rfwh_s45 r19_rfwh_s46 r19_rfwh_s47)
CATS=(faces_rfwWM64 faces_cfdWM64)
SEEDS=$(seq 101 150)
for M in "${MODELS[@]}"; do
  D=$(ls -d runs/*_"$M" 2>/dev/null | head -1); [ -z "$D" ] && continue
  CKPT=$(ls "$D"/final_model_*.pth 2>/dev/null | head -1); [ -z "$CKPT" ] && continue
  for C in "${CATS[@]}"; do
    for S in $SEEDS; do
      U="${M}__${C}__seed${S}"
      mkdir "$CLAIM/$U" 2>/dev/null || continue
      LOG="$OUT/${U}.log"
      CUDA_VISIBLE_DEVICES="$GPU" python -u yin_tests/simulate_yin1969_bothnoise.py \
        --category "$C" --run-dir "$D" --checkpoint "$CKPT" --device cuda:0 \
        --seed "$S" --noise-phase both --perception-by-test-orient \
        --only-mismatch --shuffle-items > "$LOG" 2>&1
      grep -q "upright p=\|UNREACHABLE" "$LOG" || { echo "FAILED $U"; rmdir "$CLAIM/$U" 2>/dev/null; }
    done
  done
done
echo "gpu$GPU: no work left."
