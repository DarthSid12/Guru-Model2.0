#!/usr/bin/env bash
set -euo pipefail
cd /home/siagrawal/combined_lpnet
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
OUT=runs/kanw_latest_vgg_20260927
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

case "${1:?worker 0 or 3 required}" in
  0) gpu=0; cells=(noaa:rfw noaa:cfd noaa:setA bin5:rfw bin5:cfd) ;;
  3) gpu=3; cells=(aa1331:rfw aa1331:cfd aa1331:setA bin5:setA) ;;
  *) exit 2 ;;
esac

for cell in "${cells[@]}"; do
  model=${cell%%:*}
  dataset=${cell#*:}
  log="$OUT/logs/${model}_${dataset}.log"
  printf '[%s] starting %s %s on GPU %s\n' "$(date -Is)" "$model" "$dataset" "$gpu"
  "$PYTHON" -u scripts/run_latest_vgg_kanwisher.py \
    --model "$model" --dataset "$dataset" --gpu "$gpu" > "$log" 2>&1
  printf '[%s] finished %s %s\n' "$(date -Is)" "$model" "$dataset"
done
