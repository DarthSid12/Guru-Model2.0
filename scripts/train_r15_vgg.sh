#!/usr/bin/env bash
# r15_vgg -- from-scratch log-polar model with VGGFace2 added as a fourth face
# source. 727 face identities (128 CelebA + 480 VGGFace2 + 119 CFD) against
# r13's 247. Held out: 60 VGGFace2 identities (faces_vggHO), 64 CFD males
# (faces_cfdWM64), 64 ZuBuD buildings (houses_yin64), all 40 Set_A identities.
set -euo pipefail
cd /mnt/sphere/home/siagrawal/combined_lpnet

python train.py \
  --categories faces faces_vgg faces_cfd objects houses_zubud houses \
  --variant lp --num-fixations 16 \
  --max-classes-per-category houses_zubud=40 \
  --lr 1e-3 --lr-schedule cosine --weight-decay 0.05 \
  --epochs 80 --batch-size 256 --num-workers 8 \
  --amp --channels-last \
  --dataloader-sharing file_system \
  --patience 20 \
  --curriculum \
  --curriculum-stages \
    "faces=4,faces_vgg=16,faces_cfd=4,objects=4,houses_zubud=0,houses=1" \
    "faces=8,faces_vgg=32,faces_cfd=8,objects=16,houses_zubud=0,houses=1" \
    "faces=16,faces_vgg=64,faces_cfd=16,objects=32,houses_zubud=8,houses=1" \
    "faces=32,faces_vgg=160,faces_cfd=32,objects=64,houses_zubud=16,houses=1" \
    "faces=64,faces_vgg=320,faces_cfd=64,objects=64,houses_zubud=24,houses=1" \
    "faces=128,faces_vgg=480,faces_cfd=119,objects=64,houses_zubud=40,houses=1" \
  --curriculum-epochs 6 6 8 8 12 40 \
  --category-weights \
    faces=0.15 faces_vgg=0.30 faces_cfd=0.10 \
    objects=0.20 houses_zubud=0.17 houses=0.08 \
  --run-tag r15_vgg
