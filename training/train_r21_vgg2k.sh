#!/usr/bin/env bash
# r21_vgg2k -- model 4. Identical to r21_dev (models 2 and 3: same curriculum,
# epochs, sqrt sampling, 4/1 houses, edge-margin fixations, random fixations,
# per-sample rotation) with ONE change: the face domain is a single store,
# faces_vgg2k -- 2048 identities from the full VGGFace2, ~200 photos each --
# instead of faces_vgg + faces + faces_rfwW + faces_rfwO.
#
# faces_vgg2k (data_preparation/make_faces_vgg2k.py) is taken in a FIXED order, passed as
# a pin file so every stage takes a prefix of it:
#   identities 1-512    all Caucasian (stages 1-8); the first 8 are the same
#                       hand-picked pins as r21_dev
#   identities 513-2048 988 Caucasian + 548 other-race, randomly interleaved
#                       -> 823/201 at stage 9, 1500/548 at stage 10
# Race is perceived race from FairFace (Karkkainen & Joo 2021) collapsed to RFW's
# four groups (Caucasian / African / Asian / Indian) so it matches the
# faces_rfwHO ORE test set; per-identity labels are in
# data/vggface2_raw/faces_vgg2k_order_race.csv. Unlike the RFW-based r21_dev,
# every face identity here -- own- and other-race -- has ~200 photos, so the
# other-race manipulation no longer confounds race with photos per identity.
#
# All 2048 are VGGFace2 TRAIN-split identities, so faces_vggHO (VGGFace2 test
# split) is identity-disjoint by construction.
#
# Usage: training/train_r21_vgg2k.sh [DEVICE] [SEED] [BACKBONE]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEVICE="${1:-cuda:3}"
SEED="${2:-42}"
BACKBONE="${3:-vgg16_bn_aa5}"

python training/train.py \
  --categories faces_vgg2k objects houses_zubud137_41 houses \
  --variant lp --num-fixations 16 --device "$DEVICE" \
  --backbone "$BACKBONE" \
  --lr 1e-3 --lr-schedule cosine --weight-decay 0.05 \
  --batch-size 256 --num-workers 8 \
  --amp --channels-last --dataloader-sharing file_system --patience 20 \
  --random-fixations \
  --steps-per-epoch 2000 \
  --curriculum \
  --curriculum-stages \
    "faces_vgg2k=4,objects=0,houses_zubud137_41=0,houses=0" \
    "faces_vgg2k=8,objects=0,houses_zubud137_41=0,houses=0" \
    "faces_vgg2k=16,objects=4,houses_zubud137_41=0,houses=0" \
    "faces_vgg2k=32,objects=8,houses_zubud137_41=4,houses=0" \
    "faces_vgg2k=64,objects=16,houses_zubud137_41=8,houses=1" \
    "faces_vgg2k=128,objects=32,houses_zubud137_41=16,houses=1" \
    "faces_vgg2k=256,objects=64,houses_zubud137_41=32,houses=1" \
    "faces_vgg2k=512,objects=64,houses_zubud137_41=64,houses=1" \
    "faces_vgg2k=1024,objects=64,houses_zubud137_41=137,houses=1" \
    "faces_vgg2k=2048,objects=64,houses_zubud137_41=137,houses=1" \
  --curriculum-epochs 2 2 4 4 8 8 16 16 32 32 \
  --curriculum-image-caps \
    none none none none "houses=4" "houses=8" "houses=16" "houses=32" "houses=64" "houses=128" \
  --curriculum-pin faces_vgg2k=@data/vggface2_raw/faces_vgg2k_order.txt \
  --curriculum-warmup-steps 500 \
  --category-weighting sqrt \
  --seed "$SEED" \
  --run-tag "r21vgg2k_${BACKBONE#resnet18_}_s${SEED}"
