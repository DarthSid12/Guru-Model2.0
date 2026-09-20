#!/usr/bin/env bash
# r17_base -- r16_base with ONE variable changed: the generic house class is
# `houses` (the 435-building Houses-dataset train split, exactly what r15 used)
# instead of `houses_gen` (those 435 PLUS 97 leftover ZuBuD buildings).
#
# Why: all three r16 models score houses_yin64 at 33.9-43.1% upright -- at or
# below the 50% 2AFC chance floor -- against r15_vgg's 90.58% on the identical
# store. Folding 97 ZuBuD buildings into a single generic `house` label appears
# to collapse ZuBuD representations and destroy held-out ZuBuD individuation.
# r8 (84.83), r9 (92.00) and r15 (90.58) all generalise to held-out ZuBuD and
# none of them have ZuBuD in a generic class; the three r16 models are the only
# ones that do. This run is the single-variable test of that.
#
# Everything else is byte-identical to scripts/train_r16_base.sh: same
# categories otherwise, same 6-stage/80-epoch curriculum, same weights, same lr.
# Held out: houses_yin64, faces_vggHO, faces_cfdWM64, faces_rfwWM64, faces_setA.
set -euo pipefail
cd /mnt/sphere/home/siagrawal/combined_lpnet

python train.py \
  --categories faces faces_vgg objects houses_zubud houses \
  --variant lp --num-fixations 16 --device cuda:4 \
  --max-classes-per-category houses_zubud=40 \
  --lr 1e-3 --lr-schedule cosine --weight-decay 0.05 \
  --epochs 80 --batch-size 256 --num-workers 8 \
  --amp --channels-last --dataloader-sharing file_system --patience 20 \
  --curriculum \
  --curriculum-stages \
    "faces=4,faces_vgg=16,objects=4,houses_zubud=0,houses=1" \
    "faces=8,faces_vgg=32,objects=16,houses_zubud=0,houses=1" \
    "faces=16,faces_vgg=64,objects=32,houses_zubud=8,houses=1" \
    "faces=32,faces_vgg=160,objects=64,houses_zubud=16,houses=1" \
    "faces=64,faces_vgg=320,objects=64,houses_zubud=24,houses=1" \
    "faces=128,faces_vgg=480,objects=64,houses_zubud=40,houses=1" \
  --curriculum-epochs 6 6 8 8 12 40 \
  --category-weights \
    faces=0.20 faces_vgg=0.40 objects=0.25 houses_zubud=0.10 houses=0.05 \
  --run-tag r17_base
