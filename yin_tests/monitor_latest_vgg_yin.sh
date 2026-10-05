#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/siagrawal/combined_lpnet
OUT="$ROOT/runs/yin_latest_vgg_20260927"
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
while read -r pair _; do
    [[ "$pair" == monitor_pid=* ]] && continue
    pid="${pair#*=}"
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done < "$OUT/transfer_workers.txt"
if "$PYTHON" "$ROOT/yin_tests/summarize_latest_vgg_yin.py" > "$OUT/logs/aggregate.log" 2>&1; then
    cat "$OUT/STATUS"
else
    printf 'failed; see logs/aggregate.log and worker logs\n' > "$OUT/STATUS"
    cat "$OUT/STATUS"
    exit 1
fi
