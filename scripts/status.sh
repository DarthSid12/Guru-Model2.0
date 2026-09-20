#!/usr/bin/env bash
# Compact progress for the 2026-08-23 round. The training logs carry tqdm
# progress bars, which makes `tail` useless; this pulls out just the per-epoch
# summary lines train.py prints.
#
#   scripts/status.sh          one-line-per-model summary + last 2 epochs each
#   scripts/status.sh -n 10    last 10 epochs each
cd "$(dirname "$0")/.."

N=2
[ "${1:-}" = "-n" ] && N="${2:-2}"

echo "screens:  $(screen -ls 2>/dev/null | grep -cE '\b(r10|r11|r12)\b') of 3 alive"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  | awk -F, '$1+0<=2 {printf "gpu%s:%s util%s  ", $1, $2, $3}'; echo
echo

for f in runs/logs_r1*.log; do
  name=$(basename "$f" .log | sed 's/^logs_//')
  # tqdm writes \r-separated bars into the same stream; -a treats the log as
  # text even once a binary-looking byte sneaks in.
  last=$(grep -a -o -- '-> Epoch [0-9]*:.*' "$f" | tail -n "$N")
  stage=$(grep -a -o -- '=== Stage [0-9]*/6:[^=]*' "$f" | tail -1)
  echo "### $name"
  [ -n "$stage" ] && echo "  $stage"
  if [ -n "$last" ]; then
    echo "$last" | sed 's/^/  /'
  else
    echo "  (no epoch completed yet)"
  fi
  for phase in "Yin (joint_zubud" "Kanwisher (kanw3" "ALL DONE" "TRAINING FAILED"; do
    grep -aq "$phase" "$f" && echo "  >> $(grep -a "$phase" "$f" | tail -1)"
  done
  echo
done
