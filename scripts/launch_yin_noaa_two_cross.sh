#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/siagrawal/combined_lpnet
OUT="$ROOT/runs/yin_noaa_two_cross_20260927"
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"

worker() {
    local face="$1" gpu="$2" attempt result
    result=1
    for attempt in 1 2 3; do
        printf '[%s] %s gpu%s attempt %s\n' "$(date -Is)" "$face" "$gpu" "$attempt"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$OUT/source/run_yin_orientation.py" \
            --run-dir "$OUT/checkpoint" --checkpoint "$OUT/checkpoint/weights.pth" \
            --out-dir "$OUT/results/$face" --category "$face" \
            --model r21vgg2k_vgg16_bn_s42 --epoch 124 --stage final \
            --device cuda:0 --seeds 101-150
        result=$?
        if (( result == 0 )); then break; fi
        printf '[%s] retry after exit %s\n' "$(date -Is)" "$result"
        sleep 15
    done
    return "$result"
}

worker faces_rfwWM64 7 > "$OUT/logs/rfw.log" 2>&1 &
rfw_pid=$!
worker faces_cfdWM64 2 > "$OUT/logs/cfd.log" 2>&1 &
cfd_pid=$!
printf '%s\n' "rfw_pid=$rfw_pid gpu=7" "cfd_pid=$cfd_pid gpu=2" > "$OUT/workers.txt"

wait "$rfw_pid"; rfw_status=$?
wait "$cfd_pid"; cfd_status=$?
"$PYTHON" "$ROOT/scripts/summarize_yin_noaa_two_cross.py" > "$OUT/logs/aggregate.log" 2>&1
aggregate_status=$?
if (( rfw_status == 0 && cfd_status == 0 && aggregate_status == 0 )); then
    printf 'complete\n' > "$OUT/STATUS"
else
    printf 'failed rfw=%s cfd=%s aggregate=%s\n' "$rfw_status" "$cfd_status" "$aggregate_status" > "$OUT/STATUS"
fi
cat "$OUT/STATUS"
