#!/usr/bin/env bash
set -euo pipefail
cd /home/siagrawal/combined_lpnet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
case "$1" in
  noaa_mixed) export CUDA_VISIBLE_DEVICES=1; models=(noaa mixed) ;;
  aa4) export CUDA_VISIBLE_DEVICES=2; models=(aa4) ;;
  bin5) export CUDA_VISIBLE_DEVICES=7; models=(bin5) ;;
  *) exit 2 ;;
esac
for model in "${models[@]}"; do
  "$PYTHON" -u scripts/evaluate_vgg_training_checkpoints.py --model "$model"
done
