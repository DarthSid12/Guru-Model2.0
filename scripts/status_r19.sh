#!/usr/bin/env bash
# Compact progress for the three r19_rfwh seed replicates.
#   scripts/status_r19.sh          last 2 epochs each
#   scripts/status_r19.sh -n 10    last 10 epochs each
cd "$(dirname "$0")/.."

N=2
[ "${1:-}" = "-n" ] && N="${2:-2}"

# Count MAIN processes only. --num-workers 8 means each run also spawns 8
# dataloader children that inherit the parent's argv verbatim, so a naive
# `pgrep -fc` reports ~27 alive, not 3. Keep only PIDs whose parent is not
# itself a train.py process.
alive=0
mains=""
for pid in $(pgrep -f 'train.py.*r19_rfwh' 2>/dev/null); do
  # pgrep -f also matches this script's own shell, whose command line contains
  # the pattern text. Require the process itself to actually be python.
  case "$(ps -o comm= -p "$pid" 2>/dev/null)" in python*) ;; *) continue ;; esac
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  case "$(ps -o args= -p "$ppid" 2>/dev/null)" in
    *train.py*) ;;                       # a dataloader worker; skip
    *) alive=$((alive+1)); mains="$mains $pid" ;;
  esac
done
echo "alive:  $alive trainer(s)  (pids:${mains:-  none})"
if [ -f runs/logs_r19_queue.log ]; then
  if pgrep -f 'queue_r19_seeds\.sh' >/dev/null 2>&1; then q="ARMED"; else q="not running"; fi
  echo "queue:  $q  |  $(tail -1 runs/logs_r19_queue.log)"
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  | awk -F, '$1+0==0 || $1+0==6 || $1+0==7 {printf "gpu%s:%s util%s  ", $1, $2, $3}'; echo
echo

for f in runs/logs_r19_rfwh_s*.log; do
  [ -e "$f" ] || { echo "(no logs yet)"; break; }
  name=$(basename "$f" .log | sed 's/^logs_//')
  # tqdm writes \r-separated bars into the same stream; -a keeps grep in text
  # mode once a binary-looking byte sneaks in.
  last=$(grep -a -o -- '-> Epoch [0-9]*:.*' "$f" | tail -n "$N")
  stage=$(grep -a -o -- '=== Stage [0-9]*/6:[^=]*' "$f" | tail -1)
  echo "### $name"
  [ -n "$stage" ] && echo "  $stage"
  if [ -n "$last" ]; then echo "$last" | sed 's/^/  /'
  else echo "  (no epoch completed yet)"; fi
  echo
done
