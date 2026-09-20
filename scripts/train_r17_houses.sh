#!/usr/bin/env bash
# r17_houses -- r17_base with the house arm scaled up on BOTH axes.
#
#   identities  houses_zubud137 (137 ids, 411 imgs) instead of houses_zubud
#               capped to 40 (120 imgs). 137 = every ZuBuD building that is not
#               in houses_yin64, so the Yin test set stays fully held out
#               (asserted in the store builder, zero overlap).
#   weight      houses_zubud137 0.10 -> 0.20, paid for by faces_vgg 0.40 -> 0.33
#               and objects 0.25 -> 0.22.
#
# Why both: in r16/r17 the house arm supplies 120 images (1,920 unique crops,
# 0.067% of unique training content) but takes 10% of every batch, so each
# unique house crop is drawn ~149x/epoch and ~11,909x over the run -- against 60
# for faces_vgg. That is a memorisation regime, and raising the weight alone
# would only deepen it. Identity count is what makes the face arm generalise to
# novel identities (608 face ids); 40 house ids cannot support a transferable
# individuation metric. This run tests whether 137 can.
#
# Note this moves the objects weight, so object rows are NOT directly comparable
# to r16/r17. Read houses against r17_base, which is otherwise identical.
# Held out: houses_yin64, faces_vggHO, faces_cfdWM64, faces_rfwWM64, faces_setA.
set -euo pipefail
cd /mnt/sphere/home/siagrawal/combined_lpnet

python train.py \
  --categories faces faces_vgg objects houses_zubud137 houses \
  --variant lp --num-fixations 16 --device cuda:5 \
  --lr 1e-3 --lr-schedule cosine --weight-decay 0.05 \
  --epochs 80 --batch-size 256 --num-workers 8 \
  --amp --channels-last --dataloader-sharing file_system --patience 20 \
  --curriculum \
  --curriculum-stages \
    "faces=4,faces_vgg=16,objects=4,houses_zubud137=0,houses=1" \
    "faces=8,faces_vgg=32,objects=16,houses_zubud137=0,houses=1" \
    "faces=16,faces_vgg=64,objects=32,houses_zubud137=24,houses=1" \
    "faces=32,faces_vgg=160,objects=64,houses_zubud137=48,houses=1" \
    "faces=64,faces_vgg=320,objects=64,houses_zubud137=88,houses=1" \
    "faces=128,faces_vgg=480,objects=64,houses_zubud137=137,houses=1" \
  --curriculum-epochs 6 6 8 8 12 40 \
  --category-weights \
    faces=0.20 faces_vgg=0.33 objects=0.22 houses_zubud137=0.20 houses=0.05 \
  --run-tag r17_houses
