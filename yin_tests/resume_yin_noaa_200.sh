#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/siagrawal/combined_lpnet
OUT="$ROOT/runs/yin_noaa_200_20260928"
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Wait for the initial launch to finish or exit. The runner saves every cell.
while pgrep -f '[r]un_yin_noaa_200.py --face' >/dev/null; do
    sleep 30
done
if [[ "$(cat "$OUT/STATUS" 2>/dev/null)" == complete ]]; then
    exit 0
fi
printf 'running\n' > "$OUT/STATUS"

run_one() {
    local face="$1" category="$2" gpu="$3" label="$4"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$OUT/source/run_yin_noaa_200.py" \
        --face "$face" --category "$category" --device cuda:0 \
        > "$OUT/logs/resume_$label.log" 2>&1
}

run_one faces_rfwWM64 face 1 rfw_face & a=$!
run_one faces_cfdWM64 face 2 cfd_face & b=$!
run_one faces_cfdWM64 objects 7 cfd_objects & c=$!
wait "$a"; sa=$?
wait "$b"; sb=$?
wait "$c"; sc=$?
if (( sa == 0 && sb == 0 && sc == 0 )); then
    run_one faces_rfwWM64 objects 1 rfw_objects
    sd=$?
else
    sd=1
fi
if (( sa == 0 && sb == 0 && sc == 0 && sd == 0 )); then
    "$PYTHON" "$ROOT/yin_tests/summarize_yin_noaa_200.py" \
        > "$OUT/logs/resume_aggregate.log" 2>&1
    se=$?
else
    se=1
fi
if (( se != 0 )); then
    printf 'failed resume: %s %s %s %s aggregate=%s\n' \
        "$sa" "$sb" "$sc" "$sd" "$se" > "$OUT/STATUS"
fi
