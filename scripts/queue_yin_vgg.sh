#!/usr/bin/env bash
# Run the wm64_rfwcal Yin battery on a VGG run the moment its training ends.
#   scripts/queue_yin_vgg.sh MODEL_KEY TRAIN_LOG TRAIN_PID GPU
# Waits while TRAIN_PID is alive. Runs the sims only if the log says "Done."
# and best_model.pth exists, so a crashed run is never scored.
set -uo pipefail
cd "$(dirname "$0")/.."
M="$1"; LOG="$2"; PID="$3"; GPU="$4"
D=$(python -c "import run_sim_seeds as r; print(r.MODELS['$M'])")
echo "[$(date -Is)] waiting on pid $PID ($M)"
while kill -0 "$PID" 2>/dev/null; do sleep 300; done
if tr '\r' '\n' < "$LOG" | grep -q "^Done. Best valid acc" && [ -f "$D/best_model.pth" ]; then
  echo "[$(date -Is)] training done; Yin battery on GPU $GPU"
  python run_sim_seeds.py --model "$M" --gpu "$GPU" \
    --experiment wm64_rfwcal --seeds 101-150 --threads 8
  echo "[$(date -Is)] Yin rc=$?"
else
  echo "[$(date -Is)] training did NOT finish cleanly -- no sims run."
fi
