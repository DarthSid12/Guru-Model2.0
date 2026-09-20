#!/usr/bin/env bash
# r19_rfwh -- r16_rfw's face diet with the house arm fixed FROM SCRATCH, so no
# fine-tune is needed to get a working house control.
#
# r16_rfw is the best face model in the repo (+16.25 UU-II on faces_cfdWM64 at
# its anchored p=0.33, against a human +14.41) and the one whose house control
# is dead: houses_yin64 at 42.17% upright, below the 50% 2AFC floor, with
# UU-II -8.58 (t=-7.27) -- inverted houses read BETTER than upright. r16rfwftH
# repairs that with a 20-epoch fine-tune. This run asks whether the same result
# is reachable in one pass, by never introducing the damage in the first place.
#
# Two changes from scripts/train_r16_rfw.sh, both taken from the r17 round:
#
#   houses_gen -> houses      The r17_base single-variable test. houses_gen is
#                             the generic Houses-dataset class PLUS 291 crops
#                             from the 97 leftover ZuBuD buildings. Training
#                             ZuBuD exemplars under a label the model is taught
#                             never to individuate collapses ZuBuD identity
#                             structure. r17_base changed this one token and
#                             took houses_yin64 from 33.92% to 85.17%.
#
#   houses_zubud (40, w0.08)  The r17_houses scale-up. 40 ids / 120 images at
#     -> houses_zubud137      8% of every batch is a memorisation regime: each
#        (137, w0.16)         unique crop is drawn ~100x/epoch against 60 over
#                             the whole run for faces_vgg. 137 ids / 411 images
#                             is what made r17_houses read 96.50%. 137 = every
#                             ZuBuD building not in houses_yin64 (201 - 64),
#                             verified zero overlap, so the Yin house test set
#                             stays fully held out. The 97 ex-houses_gen
#                             buildings are now individuated instead of being
#                             trained both ways at once.
#
# Where the house weight comes from. The house arm goes 0.13 -> 0.21. Six of
# those eight points are paid by objects (0.25 -> 0.19), which is the cheapest
# donor: at r16_rfw's anchored p objects read 95.92% upright against a human
# anchor of 84.79, i.e. far above where the control needs to sit. The remaining
# two come from faces_vgg (0.28 -> 0.26), leaving the face arm at 0.60 against
# r16_rfw's 0.62 -- a 3% relative cut, deliberately small, because the face
# effect is the thing being preserved. r17_houses by contrast paid 0.07 out of
# faces_vgg and objects both, and its object rows are not comparable to the
# rest of the battery as a result; the same caveat applies here but smaller.
#
# Everything else is byte-identical to train_r16_rfw.sh: same RFW white-first
# ladder, same 6-stage/80-epoch curriculum, same cosine lr 1e-3, same wd, same
# patience, same seed.
#
# Held out: houses_yin64, faces_rfwHO (500 ids, 125/race -- the only
# race-balanced identity-disjoint store, the ORE instrument, do not train on
# it), faces_vggHO, faces_cfdWM64, faces_rfwWM64, faces_setA.
#
# KNOWN RISK. r17 is the only precedent for fixing houses from scratch, and both
# r17 arms developed a negative cfd UU-II late in the final 40-epoch stage
# (r17_base was +6.75 at epoch ~60 and -7.92 at epoch 80). The cause is open.
# Stage checkpoints only fire at stage boundaries, so there is nothing saved
# between epoch 40 and 80 to diagnose it with -- score the final weights, and if
# cfd UU-II comes out negative, that is this effect and not the house change.
#
# Usage: scripts/train_r19_rfwh.sh [DEVICE] [SEED]
#   scripts/train_r19_rfwh.sh cuda:0 42
#
# SEED sets torch.manual_seed + np.random.seed, i.e. backbone init and batch
# order. --curriculum-seed is left at its default 0 in every arm, so which
# identities enter at which stage is IDENTICAL across seeds and the only thing
# that varies is the initialisation. The run tag carries the seed because
# out_dir is built from categories/variant/lr/backbone/run-tag and does NOT
# include the seed -- three runs sharing a tag would overwrite each other's
# config.json, label_map.json and best_model.pth.
set -euo pipefail
cd /mnt/sphere/home/siagrawal/combined_lpnet

DEVICE="${1:-cuda:0}"
SEED="${2:-42}"

python train.py \
  --categories faces faces_vgg faces_rfwW faces_rfwO objects houses_zubud137 houses \
  --variant lp --num-fixations 16 --device "$DEVICE" \
  --max-classes-per-category faces_rfwW=1200 faces_rfwO=300 \
  --lr 1e-3 --lr-schedule cosine --weight-decay 0.05 \
  --epochs 80 --batch-size 256 --num-workers 8 \
  --amp --channels-last --dataloader-sharing file_system --patience 20 \
  --curriculum \
  --curriculum-stages \
    "faces=4,faces_vgg=16,faces_rfwW=50,faces_rfwO=0,objects=4,houses_zubud137=0,houses=1" \
    "faces=8,faces_vgg=32,faces_rfwW=150,faces_rfwO=0,objects=16,houses_zubud137=0,houses=1" \
    "faces=16,faces_vgg=64,faces_rfwW=400,faces_rfwO=0,objects=32,houses_zubud137=24,houses=1" \
    "faces=32,faces_vgg=160,faces_rfwW=700,faces_rfwO=75,objects=64,houses_zubud137=48,houses=1" \
    "faces=64,faces_vgg=320,faces_rfwW=1000,faces_rfwO=180,objects=64,houses_zubud137=88,houses=1" \
    "faces=128,faces_vgg=480,faces_rfwW=1200,faces_rfwO=300,objects=64,houses_zubud137=137,houses=1" \
  --curriculum-epochs 6 6 8 8 12 40 \
  --category-weights \
    faces=0.14 faces_vgg=0.26 faces_rfwW=0.16 faces_rfwO=0.04 \
    objects=0.19 houses_zubud137=0.16 houses=0.05 \
  --seed "$SEED" \
  --run-tag "r19_rfwh_s${SEED}"
