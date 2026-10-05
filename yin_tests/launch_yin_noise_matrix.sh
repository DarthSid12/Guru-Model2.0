#!/usr/bin/env bash
# One model per GPU; each worker runs the two face calibrations in sequence.
set -uo pipefail

ROOT=/home/siagrawal/combined_lpnet
OUT="$ROOT/runs/yin_noise_matrix_20260926"
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"

worker() {
    local model="$1" gpu="$2" face attempt result
    for face in faces_rfwWM64 faces_cfdWM64; do
        result=1
        for attempt in 1 2 3; do
            printf '[%s] %s %s gpu%s attempt %s\n' "$(date -Is)" "$model" "$face" "$gpu" "$attempt"
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$OUT/source/run_yin_noise_matrix.py" \
                --model "$model" --face "$face" --device cuda:0 \
                --out-root "$OUT"
            result=$?
            if (( result == 0 )); then break; fi
            printf '[%s] retrying after exit %s\n' "$(date -Is)" "$result"
            sleep 15
        done
        if (( result != 0 )); then
            printf '[%s] FAILED %s %s after three attempts\n' "$(date -Is)" "$model" "$face"
            return "$result"
        fi
    done
}

worker r21dev_vgg16_bn_aa_s42 7 > "$OUT/logs/developmental_vgg.log" 2>&1 &
dev_pid=$!
worker r21vgg2k_vgg16_bn_aa_s42 0 > "$OUT/logs/vgg2k.log" 2>&1 &
vgg_pid=$!
printf '%s\n' "dev_vgg_pid=$dev_pid gpu=7" "vgg2k_pid=$vgg_pid gpu=0" > "$OUT/workers.txt"

wait "$dev_pid"; dev_status=$?
wait "$vgg_pid"; vgg_status=$?
"$PYTHON" "$ROOT/yin_tests/summarize_yin_noise_matrix.py" --out-root "$OUT" \
    > "$OUT/logs/aggregate.log" 2>&1
aggregate_status=$?
if (( dev_status == 0 && vgg_status == 0 && aggregate_status == 0 )); then
    printf 'complete\n' > "$OUT/STATUS"
else
    printf 'failed dev=%s vgg=%s aggregate=%s\n' "$dev_status" "$vgg_status" "$aggregate_status" > "$OUT/STATUS"
fi
cat "$OUT/STATUS"
