#!/usr/bin/env bash
# Emit one line whenever any of the four background streams changes phase.
# Written for the Monitor tool: stdout lines become notifications, and the
# script exits once every stream has reached a terminal state.
#
# Coverage matters more than tidiness here -- a watcher that greps only for the
# success marker stays silent through a crash, and silence looks exactly like
# "still running". The alternation below therefore includes the failure
# signatures as well as the happy path.

cd "$(dirname "$0")/.."

LOGS=(runs/logs_r10_dev_h80.log
      runs/logs_r11_dev_h80_generic.log
      runs/logs_r12_dev_h80_enbloc.log
      runs/logs_yin_mixu16.log)

# Phase markers (happy path) + crash signatures (must never be silent).
MARKERS='Yin \(joint_zubud|Kanwisher \(kanw3|ALL DONE|TRAINING FAILED|starting mixu16 sweeps|mixu16 sweeps finished|Traceback|CUDA out of memory|No space left on device|Killed|MemoryError|\[warn\]'

declare -A seen
while true; do
  terminal=0
  for f in "${LOGS[@]}"; do
    name=$(basename "$f" .log | sed 's/^logs_//')
    last=$(grep -a -oE "$MARKERS" "$f" 2>/dev/null | tail -1)
    [ -z "$last" ] && last="(waiting)"

    if [ "${seen[$name]:-unset}" != "$last" ]; then
      echo "[$(date +%H:%M)] $name -> $last"
      seen[$name]="$last"
    fi

    case "$last" in
      "ALL DONE"|"TRAINING FAILED"|"mixu16 sweeps finished") terminal=$((terminal + 1)) ;;
    esac
  done

  if [ "$terminal" -ge ${#LOGS[@]} ]; then
    echo "[$(date +%H:%M)] all ${#LOGS[@]} streams terminal -- watch ending"
    break
  fi
  sleep 120
done
