#!/usr/bin/env bash
# Run inside tmux. Workers and the queue resume saved progress after a restart.
set -uo pipefail
YIN_OUT_DIR="${1:?usage: supervise_yin_orientation.sh OUTPUT_DIR}"
YIN_PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec >> "$YIN_OUT_DIR/queue.log" 2>&1
while true; do
  "$YIN_PYTHON" -u "$YIN_OUT_DIR/source/queue_yin_orientation.py" run --out-dir "$YIN_OUT_DIR"
  yin_queue_rc=$?
  if [ "$yin_queue_rc" -eq 0 ]; then
    exit 0
  fi
  printf '[%s] queue exited with rc=%s; restarting in 30 seconds\n' "$(date -Is)" "$yin_queue_rc"
  sleep 30
done
