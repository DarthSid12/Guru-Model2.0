#!/usr/bin/env bash
# r16_rfw -- r16_base's diet plus 1500 RFW identities, introduced white-first to
# model the other-race effect: 1200 Caucasian (faces_rfwW) from stage 1, 300
# African/Asian/Indian (faces_rfwO, 100 each) only from stage 4. Weights pin the
# RFW arm at 20% of the diet split exactly 80/20 white:other -- natural image
# counts would give 72/28, because the top-100-per-race "other" ids are deeper
# (5.03 photos/id) than the top-1200 white ones (3.23).
# Held out: faces_rfwHO (500 ids, 125/race, 4 photos) -- the only race-balanced
# identity-disjoint store, i.e. the only instrument for measuring the ORE on
# novel identities. Do not train on it.
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
  --run-tag r16_rfw
