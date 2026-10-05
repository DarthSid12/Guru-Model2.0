#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/siagrawal/combined_lpnet
OUT="$ROOT/runs/yin_latest_vgg_20260927"
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

case "${1:?worker required}" in
    noaa) model=r21vgg2k_vgg16_bn_s42; gpu=1; predecessor_key=noaa_single;
          faces=(faces_rfwWM64 faces_cfdWM64) ;;
    aa5_rfw) model=r21vgg2k_vgg16_bn_aa5_s42; gpu=2; predecessor_key=aa5_rfw;
             faces=(faces_rfwWM64) ;;
    aa5_cfd) model=r21vgg2k_vgg16_bn_aa5_s42; gpu=7; predecessor_key=aa5_cfd;
             faces=(faces_cfdWM64) ;;
    *) echo "Unknown worker $1" >&2; exit 2 ;;
esac

predecessor=""
while read -r pair _; do
    if [[ "${pair%%=*}" == "$predecessor_key" ]]; then
        predecessor="${pair#*=}"
        break
    fi
done < "$OUT/workers.txt"
[[ -n "$predecessor" ]] || { echo "Missing predecessor PID" >&2; exit 1; }
while kill -0 "$predecessor" 2>/dev/null; do sleep 30; done
for face in "${faces[@]}"; do
    if [[ ! -f "$OUT/single_noise/$model/$face/DONE.json" ]]; then
        echo "Missing completed single-noise run for $model $face" >&2
        exit 1
    fi
    echo "[$(date -Is)] transferring two_cross $model $face on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        "$OUT/source/run_latest_vgg_two_cross_transfer.py" \
        --model "$model" --face "$face" --device cuda:0 \
        --out-root "$OUT/transfer"
done
