#!/usr/bin/env bash
# Two-noise Yin battery over the finished r19 seeds.
#
#   yin_tests/run_bothnoise.sh <GPU>        # start one worker; run several
#
# Work unit = (model, category, sim-seed). Workers claim units atomically with
# mkdir, so you can add or kill workers at any time without coordination and
# without redoing finished units -- which matters here because GPUs 0/6/7 are
# training until ~06:00 and can join the pool afterwards.
#
# Ten sim seeds per unit, NOT one. The smoke test fitted p_up=0.088 by landing
# on a 100.00% reading: with 24 test pairs a single seed quantises accuracy to
# 4.17%, so it cannot resolve a 96.29% target at all. And the single-noise
# battery just measured seed SD of 2.96 on rfw UU-II and 6.62 on houses, so a
# one-seed number here would be noise. Each seed refits BOTH noises, so the
# per-seed fitted p's are themselves the distribution of interest.
set -uo pipefail
cd "$(dirname "$0")/.."

# One thread per process. torch defaults to 32 on this 64-core box, so a few
# concurrent sims already oversubscribe it. Profiling showed the sim is
# CPU-bound -- compute_familiarity_score is 61% of runtime in a Python loop
# doing 245,760 tiny tensor ops, while the model forward is 5% -- so cores are
# the scarce resource and GPUs are nearly idle at 656 MiB / ~5% each.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

GPU="${1:?usage: run_bothnoise.sh GPU}"
NOISE_PHASE="${NOISE_PHASE:-study}"
OUT="runs/bothnoise_${NOISE_PHASE}"; CLAIM="$OUT/.claims"; mkdir -p "$OUT" "$CLAIM"
MODELS=(r19_rfwh_s42 r19_rfwh_s43 r19_rfwh_s44 r19_rfwh_s45 r19_rfwh_s46 r19_rfwh_s47)
CATS=(faces_cfdWM64 faces_rfwWM64 houses_yin64 objects)
SEEDS=$(seq 101 110)
# houses_yin64 and objects are packed in class order; the face stores need
# --shuffle-items for the same reason the single-noise battery sets yin_shuffle.
shuffle_for() { case "$1" in faces_*) echo "--shuffle-items";; *) echo "";; esac; }

for M in "${MODELS[@]}"; do
  D=$(ls -d runs/*_"$M" 2>/dev/null | head -1)
  [ -z "$D" ] && { echo "[$(date -Is)] no run dir for $M, skipping"; continue; }
  CKPT=$(ls "$D"/final_model_*.pth 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "[$(date -Is)] $M has no final_model yet, skipping"; continue; }
  for C in "${CATS[@]}"; do
    for S in $SEEDS; do
      U="${M}__${C}__${S}"
      mkdir "$CLAIM/$U" 2>/dev/null || continue   # already claimed by someone
      LOG="$OUT/${U}.log"
      echo "[$(date -Is)] gpu$GPU -> $U"
      CUDA_VISIBLE_DEVICES="$GPU" python -u yin_tests/simulate_yin1969_bothnoise.py \
        --category "$C" --run-dir "$D" --checkpoint "$CKPT" \
        --device cuda:0 --seed "$S" --noise-phase "$NOISE_PHASE" \
        $(shuffle_for "$C") > "$LOG" 2>&1
      if grep -q "upright p=" "$LOG"; then
        echo "[$(date -Is)] gpu$GPU    done $U"
      elif grep -q "UNREACHABLE" "$LOG"; then
        # A target that no noise level can hit is a property of the model and
        # category, not a transient failure. Keep the claim so nobody retries it.
        echo "[$(date -Is)] gpu$GPU    UNREACHABLE $U (kept, will not retry)"
      else
        echo "[$(date -Is)] gpu$GPU    FAILED $U (see $LOG)"
        rmdir "$CLAIM/$U" 2>/dev/null   # transient: release for one retry
      fi
    done
  done
done
echo "[$(date -Is)] gpu$GPU: no work left."
