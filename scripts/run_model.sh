#!/usr/bin/env bash
# Train one of the 2026-08-23 developmental models, plot it, then run the Yin
# and Kanwisher simulations on it. Detached-safe: everything is logged to
# runs/logs_<tag>.log, and each phase is gated on the previous one succeeding,
# so a training crash does not leave sims scoring a half-written checkpoint.
#
#   scripts/run_model.sh r10 0
#
# The three models share faces (128), objects (64), 80 individuated ZuBuD
# buildings, 80 epochs, and houses held at 20% of every batch. They differ only
# in the house arm -- see paper/experiment_story.md Step 15 for why that is the
# variable worth moving.

set -uo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: run_model.sh TAG GPU  (TAG is r10, r11 or r12)}"
GPU="${2:?usage: run_model.sh TAG GPU  (TAG is r10, r11 or r12)}"

EPOCHS=(6 6 8 8 12 40)

case "$TAG" in
  r10)  # gradual house ladder: 0 -> 0 -> 8 -> 16 -> 40 -> 80
    NAME=r10_dev_h80
    CATS=(faces objects houses_zubud)
    STAGES=("faces=4,objects=4,houses_zubud=0"
            "faces=8,objects=16,houses_zubud=0"
            "faces=16,objects=32,houses_zubud=8"
            "faces=32,objects=64,houses_zubud=16"
            "faces=64,objects=64,houses_zubud=40"
            "faces=128,objects=64,houses_zubud=80")
    WEIGHTS=(faces=0.40 objects=0.40 houses_zubud=0.20)
    ;;
  r11)  # same ladder + the generic one-class 'houses' category from epoch 1
    NAME=r11_dev_h80_generic
    CATS=(faces objects houses_zubud houses)
    STAGES=("faces=4,objects=4,houses_zubud=0,houses=1"
            "faces=8,objects=16,houses_zubud=0,houses=1"
            "faces=16,objects=32,houses_zubud=8,houses=1"
            "faces=32,objects=64,houses_zubud=16,houses=1"
            "faces=64,objects=64,houses_zubud=40,houses=1"
            "faces=128,objects=64,houses_zubud=80,houses=1")
    WEIGHTS=(faces=0.35 objects=0.35 houses_zubud=0.20 houses=0.10)
    ;;
  r12)  # houses withheld entirely, then all 80 arrive in the final stage
    NAME=r12_dev_h80_enbloc
    CATS=(faces objects houses_zubud)
    STAGES=("faces=4,objects=4,houses_zubud=0"
            "faces=8,objects=16,houses_zubud=0"
            "faces=16,objects=32,houses_zubud=0"
            "faces=32,objects=64,houses_zubud=0"
            "faces=64,objects=64,houses_zubud=0"
            "faces=128,objects=64,houses_zubud=80")
    WEIGHTS=(faces=0.40 objects=0.40 houses_zubud=0.20)
    ;;
  *) echo "unknown tag $TAG" >&2; exit 2 ;;
esac

OUT="runs/$NAME"
LOG="runs/logs_${NAME}.log"
MODEL_KEY="$NAME"   # matches the MODELS entry registered in run_sim_seeds.py

exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date -Is)] $NAME on GPU $GPU (host $(hostname))"
echo "=================================================================="

# --- 0. warm the NFS page cache ------------------------------------------
# A cold cache drops training to ~45 img/s; a sequential read of the packed
# arrays first is the difference between a 10-hour run and a multi-day one.
echo "[$(date -Is)] pre-warming packed arrays..."
for c in "${CATS[@]}"; do
  cat fixation_data/"$c"/*/*.npy > /dev/null 2>&1
done
echo "[$(date -Is)] cache warm."

# --- 1. train -------------------------------------------------------------
# --patience 40 >= the longest stage, i.e. early stopping is off: the whole
# point of this round is to give the house arm its full 40-epoch final stage.
# --dataloader-sharing file_system because other users' jobs have filled the
# shared /dev/shm and killed a run here before.
echo "[$(date -Is)] training -> $OUT"
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  --categories "${CATS[@]}" \
  --variant lp --backbone resnet18 --num-fixations 16 \
  --lr 1e-3 --lr-schedule cosine \
  --max-classes-per-category houses_zubud=80 \
  --curriculum \
  --curriculum-stages "${STAGES[@]}" \
  --curriculum-epochs "${EPOCHS[@]}" \
  --category-weights "${WEIGHTS[@]}" \
  --batch-size 256 --num-workers 8 \
  --dataloader-sharing file_system \
  --patience 40 --seed 42 \
  --device cuda:0 \
  --output-dir "$OUT"
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[$(date -Is)] TRAINING FAILED rc=$TRAIN_RC -- not running sims."
  exit $TRAIN_RC
fi
echo "[$(date -Is)] training done."

# --- 2. training figures --------------------------------------------------
python paper/plot_training.py "$OUT" || echo "[warn] plotting failed; continuing"

# --- 3. simulations -------------------------------------------------------
# joint_zubud = the Step 13 Yin arm: faces + objects + houses on ONE noise
# level fitted to the faces and houses anchors together, so the three rows of
# the table are comparable. Houses are auto-restricted to the 121 ZuBuD
# buildings this model never trained on (EXCLUDE_SEEN in run_sim_seeds.py).
echo "[$(date -Is)] Yin (joint_zubud, 100 seeds)..."
python run_sim_seeds.py --model "$MODEL_KEY" --gpu "$GPU" \
  --experiment joint_zubud --seeds 101-200 \
  --calib-on-eval-seeds --threads 8 \
  || echo "[warn] Yin sweep returned nonzero"

# kanw3 = the matching task. 20 seeds, not 100: one Kanwisher seed scores
# ~100k triplets, so its SEMs are already +-0.1-0.2.
echo "[$(date -Is)] Kanwisher (kanw3, 20 seeds)..."
python run_sim_seeds.py --model "$MODEL_KEY" --gpu "$GPU" \
  --experiment kanw3 --seeds 101-120 \
  --calib-on-eval-seeds --threads 8 \
  || echo "[warn] Kanwisher sweep returned nonzero"

echo "[$(date -Is)] ALL DONE for $NAME"
