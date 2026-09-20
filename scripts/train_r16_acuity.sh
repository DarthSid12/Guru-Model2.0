#!/usr/bin/env bash
# r16_acuity -- r16_base's diet and schedule EXACTLY, plus a blur->sharp acuity
# schedule (Vogelsang et al. 2018 PNAS): sigma 8/6/4/2/1/0 px on the raw fixation
# crop, train-time only. valid/test always run at full acuity.
# One-variable comparison against r16_base: does initial low acuity produce
# configural face processing and a larger Yin inversion effect?
set -euo pipefail
cd /mnt/sphere/home/siagrawal/combined_lpnet

python train.py \
  --categories faces faces_vgg objects houses_zubud houses_gen \
  --variant lp --num-fixations 16 --device cuda:6 \
  --max-classes-per-category houses_zubud=40 \
  --lr 1e-3 --lr-schedule cosine --weight-decay 0.05 \
  --epochs 80 --batch-size 256 --num-workers 8 \
  --amp --channels-last --dataloader-sharing file_system --patience 20 \
  --curriculum \
  --curriculum-stages \
    "faces=4,faces_vgg=16,objects=4,houses_zubud=0,houses_gen=1" \
    "faces=8,faces_vgg=32,objects=16,houses_zubud=0,houses_gen=1" \
    "faces=16,faces_vgg=64,objects=32,houses_zubud=8,houses_gen=1" \
    "faces=32,faces_vgg=160,objects=64,houses_zubud=16,houses_gen=1" \
    "faces=64,faces_vgg=320,objects=64,houses_zubud=24,houses_gen=1" \
    "faces=128,faces_vgg=480,objects=64,houses_zubud=40,houses_gen=1" \
  --curriculum-epochs 6 6 8 8 12 40 \
  --acuity-sigmas 8 6 4 2 1 0 \
  --category-weights \
    faces=0.20 faces_vgg=0.40 objects=0.25 houses_zubud=0.10 houses_gen=0.05 \
  --run-tag r16_acuity
