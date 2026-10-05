#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/siagrawal/combined_lpnet
OUT="$ROOT/runs/yin_category_uu_20260926"
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"

worker() {
    local gpu="$1" model="$2" category="$3" attempt result
    result=1
    for attempt in 1 2 3; do
        printf '[%s] %s %s gpu%s attempt %s\n' "$(date -Is)" "$model" "$category" "$gpu" "$attempt"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$OUT/source/run_yin_category_uu.py" \
            --model "$model" --category "$category" --device cuda:0 \
            --out-root "$OUT"
        result=$?
        if (( result == 0 )); then break; fi
        printf '[%s] retrying after exit %s\n' "$(date -Is)" "$result"
        sleep 15
    done
    if (( result != 0 )); then
        printf '[%s] FAILED %s %s after three attempts\n' "$(date -Is)" "$model" "$category"
    fi
    return "$result"
}

(
    worker 7 r21dev_vgg16_bn_aa_s42 objects || exit $?
    worker 7 r21vgg2k_vgg16_bn_aa_s42 houses_yin64
) > "$OUT/logs/gpu7.log" 2>&1 &
pid7=$!
worker 0 r21dev_vgg16_bn_aa_s42 houses_yin64 > "$OUT/logs/gpu0.log" 2>&1 &
pid0=$!
worker 3 r21vgg2k_vgg16_bn_aa_s42 objects > "$OUT/logs/gpu3.log" 2>&1 &
pid3=$!
printf '%s\n' "gpu7_pid=$pid7" "gpu0_pid=$pid0" "gpu3_pid=$pid3" > "$OUT/workers.txt"

wait "$pid7"; status7=$?
wait "$pid0"; status0=$?
wait "$pid3"; status3=$?
"$PYTHON" "$ROOT/scripts/summarize_yin_category_uu.py" > "$OUT/logs/aggregate.log" 2>&1
aggregate_status=$?
"$PYTHON" "$ROOT/scripts/summarize_yin_noise_matrix.py" \
    --out-root "$ROOT/runs/yin_noise_matrix_20260926" \
    > "$OUT/logs/main_report.log" 2>&1
main_status=$?
if (( status7 == 0 && status0 == 0 && status3 == 0 && aggregate_status == 0 && main_status == 0 )); then
    printf 'complete\n' > "$OUT/STATUS"
else
    printf 'failed gpu7=%s gpu0=%s gpu3=%s aggregate=%s main=%s\n' \
        "$status7" "$status0" "$status3" "$aggregate_status" "$main_status" > "$OUT/STATUS"
fi
cat "$OUT/STATUS"
