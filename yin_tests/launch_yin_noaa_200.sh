#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/siagrawal/combined_lpnet
OUT="$ROOT/runs/yin_noaa_200_20260928"
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"
printf 'running\n' > "$OUT/STATUS"

worker() {
    local face="$1" category="$2" gpu="$3" label="$4"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$OUT/source/run_yin_noaa_200.py" \
        --face "$face" --category "$category" --device cuda:0 \
        > "$OUT/logs/$label.log" 2>&1
}

worker faces_rfwWM64 face 1 rfw_face & rfw_face=$!
worker faces_cfdWM64 face 2 cfd_face & cfd_face=$!
worker faces_cfdWM64 objects 7 cfd_objects & cfd_objects=$!
printf 'rfw_face=%s gpu=1\ncfd_face=%s gpu=2\ncfd_objects=%s gpu=7\n' \
    "$rfw_face" "$cfd_face" "$cfd_objects" > "$OUT/workers.txt"

wait "$rfw_face"; rfw_face_status=$?
if (( rfw_face_status == 0 )); then
    worker faces_rfwWM64 objects 1 rfw_objects & rfw_objects=$!
    printf 'rfw_objects=%s gpu=1\n' "$rfw_objects" >> "$OUT/workers.txt"
else
    rfw_objects_status=1
fi
wait "$cfd_face"; cfd_face_status=$?
wait "$cfd_objects"; cfd_objects_status=$?
if (( rfw_face_status == 0 )); then
    wait "$rfw_objects"; rfw_objects_status=$?
fi

if (( rfw_face_status == 0 && rfw_objects_status == 0 && cfd_face_status == 0 && cfd_objects_status == 0 )); then
    "$PYTHON" "$ROOT/yin_tests/summarize_yin_noaa_200.py" > "$OUT/logs/aggregate.log" 2>&1
    aggregate_status=$?
else
    aggregate_status=1
fi
if (( aggregate_status != 0 )); then
    printf 'failed rfw_face=%s rfw_objects=%s cfd_face=%s cfd_objects=%s aggregate=%s\n' \
        "$rfw_face_status" "$rfw_objects_status" "$cfd_face_status" \
        "$cfd_objects_status" "$aggregate_status" > "$OUT/STATUS"
fi
cat "$OUT/STATUS"
