#!/usr/bin/env bash
# Wait for the in-flight r19_rfwh batch to clear, then launch the next batch of
# seed replicates on the same three GPUs.
#
#   setsid nohup bash scripts/queue_r19_seeds.sh 45 46 47 \
#     >> runs/logs_r19_queue.log 2>&1 < /dev/null &
#
# Detached with setsid so it survives the shell, the ssh session and the Claude
# session that started it: it is reparented to init and holds no tty. Nothing
# about it depends on a live terminal.
#
# Why poll instead of `wait`: the running trainers are not children of this
# script, so `wait` cannot see them. We poll on the RUN TAG rather than on PIDs
# captured up front, which is robust to PID recycling over a multi-hour wait.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=("${@:-45 46 47}")
[ "$#" -eq 0 ] && SEEDS=(45 46 47)
GPUS=(0 6 7)
POLL=60
MAX_WAIT=$((48 * 3600))   # give up rather than poll forever if something hangs

# Main trainer processes only: --num-workers 8 means each run also spawns 8
# dataloader children with the parent's argv verbatim, and pgrep -f additionally
# matches any shell whose command line merely contains the pattern text.
count_trainers() {
  local n=0 pid ppid
  for pid in $(pgrep -f 'train.py.*r19_rfwh' 2>/dev/null); do
    case "$(ps -o comm= -p "$pid" 2>/dev/null)" in python*) ;; *) continue ;; esac
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    case "$(ps -o args= -p "$ppid" 2>/dev/null)" in
      *train.py*) ;;                     # dataloader worker
      *) n=$((n + 1)) ;;
    esac
  done
  echo "$n"
}

echo "=================================================================="
echo "[$(date -Is)] queue armed for seeds ${SEEDS[*]} on GPUs ${GPUS[*]} (pid $$)"
echo "=================================================================="

waited=0
while :; do
  n=$(count_trainers)
  [ "$n" -eq 0 ] && break
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "[$(date -Is)] ABORT: $n trainer(s) still alive after ${MAX_WAIT}s. Not launching."
    exit 1
  fi
  [ $((waited % 1800)) -eq 0 ] && echo "[$(date -Is)] waiting; $n trainer(s) still alive (${waited}s)"
  sleep "$POLL"; waited=$((waited + POLL))
done
echo "[$(date -Is)] in-flight batch clear after ${waited}s."

# Report how the previous batch ended, so a crash is visible here and not only
# buried in a per-seed log.
for f in runs/logs_r19_rfwh_s*.log; do
  [ -e "$f" ] || continue
  tag=$(basename "$f" .log | sed 's/^logs_//')
  if grep -aq 'Saved:' "$f" 2>/dev/null || ls runs/*"${tag}"/final_model_*.pth >/dev/null 2>&1; then
    echo "[$(date -Is)] previous: $tag finished (final_model present)"
  else
    echo "[$(date -Is)] previous: $tag has NO final_model -- it crashed or was killed"
  fi
done

# Cheap after an 18 h run (the arrays are already in page cache), but correct if
# something evicted them.
echo "[$(date -Is)] pre-warming packed arrays..."
for c in faces faces_vgg faces_rfwW faces_rfwO objects houses_zubud137 houses; do
  cat fixation_data/"$c"/*/*.npy > /dev/null 2>&1
done
echo "[$(date -Is)] cache warm."

for i in "${!SEEDS[@]}"; do
  SEED="${SEEDS[$i]}"; GPU="${GPUS[$i]}"
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 0)
  if [ "${used:-0}" -gt 2000 ]; then
    echo "[$(date -Is)] SKIP seed $SEED: cuda:$GPU has ${used} MiB in use by someone else."
    continue
  fi
  LOG="runs/logs_r19_rfwh_s${SEED}.log"
  echo "[$(date -Is)] launching seed $SEED on cuda:$GPU -> $LOG"
  setsid nohup bash scripts/train_r19_rfwh.sh "cuda:${GPU}" "$SEED" \
    >>"$LOG" 2>&1 < /dev/null &
  sleep 5
done

echo "[$(date -Is)] queued batch launched."
