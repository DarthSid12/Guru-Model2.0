#!/usr/bin/env bash
# r18_inv05 -- r16_rfw's recipe, byte-identical apart from --invert-p 0.05.
#
# WHY. The rfwWM64-calibrated battery (paper/results_rfwcal_battery.md) shows the
# models reproduce Yin's orientation MAIN effects by the bottleneck -- r16rfwftH
# gives upright-study +7.17 against a human +9.98 and upright-test +4.75 against
# +4.43 -- but carry a study/test CONGRUENCE interaction of +16.50 against the
# human +7.73, roughly double. The layer sweep localises that congruence to the
# input encoding: it is already present at layer1 (+4.9) where NO orientation
# main effect exists at all, so it is architectural, not learned.
#
# Hypothesis: a small amount of inverted experience buys partially
# orientation-tolerant features, lifting the mismatched (UI/IU) cells and
# collapsing congruence, while upright still being 95% of the diet preserves
# the upright advantage. Success = congruence falls toward +7.7 WHILE study stays
# near +7 and test near +4.5. If all three shrink together, inverted exposure is
# just degrading training and the hypothesis is wrong.
#
# CAVEAT. This deliberately breaks the upright-only training invariant that
# paper/yin_replication.pdf S3.1 calls out ("a vertical flip is never applied
# during training, since that would leak inverted views into the network and
# contaminate the very inversion manipulation the simulation is designed to
# test"). Every model up to r17 honoured it. These arms therefore test a
# DIFFERENT claim -- how much inverted experience abolishes the effect -- and
# their Yin rows are only interpretable against each other and against r16_rfw,
# which is the matched 0% control (same recipe, already trained).
#
# 0% arm = runs/...r16_rfw (existing). Inversion is per-sample and applied
# BEFORE the +-15 deg jitter, so inverted crops are augmented like upright ones.
set -euo pipefail
cd /mnt/sphere/home/siagrawal/combined_lpnet

python train.py \
  --categories faces faces_vgg faces_rfwW faces_rfwO objects houses_zubud houses_gen \
  --variant lp --num-fixations 16 --device cuda:2 \
  --max-classes-per-category houses_zubud=40 faces_rfwW=1200 faces_rfwO=300 \
  --lr 1e-3 --lr-schedule cosine --weight-decay 0.05 \
  --epochs 80 --batch-size 256 --num-workers 8 \
  --amp --channels-last --dataloader-sharing file_system --patience 20 \
  --curriculum \
  --curriculum-stages \
    "faces=4,faces_vgg=16,faces_rfwW=50,faces_rfwO=0,objects=4,houses_zubud=0,houses_gen=1" \
    "faces=8,faces_vgg=32,faces_rfwW=150,faces_rfwO=0,objects=16,houses_zubud=0,houses_gen=1" \
    "faces=16,faces_vgg=64,faces_rfwW=400,faces_rfwO=0,objects=32,houses_zubud=8,houses_gen=1" \
    "faces=32,faces_vgg=160,faces_rfwW=700,faces_rfwO=75,objects=64,houses_zubud=16,houses_gen=1" \
    "faces=64,faces_vgg=320,faces_rfwW=1000,faces_rfwO=180,objects=64,houses_zubud=24,houses_gen=1" \
    "faces=128,faces_vgg=480,faces_rfwW=1200,faces_rfwO=300,objects=64,houses_zubud=40,houses_gen=1" \
  --curriculum-epochs 6 6 8 8 12 40 \
  --category-weights \
    faces=0.14 faces_vgg=0.28 faces_rfwW=0.16 faces_rfwO=0.04 \
    objects=0.25 houses_zubud=0.08 houses_gen=0.05 \
  --invert-p 0.05 \
  --run-tag r18_inv05
