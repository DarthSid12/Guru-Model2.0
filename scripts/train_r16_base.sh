#!/usr/bin/env bash
# r16_base -- from-scratch log-polar developmental model.
# Diet: 128 CelebA + 480 VGGFace2 faces, 64 objects, 40 ZuBuD house identities,
# and one generic `house` class (97 leftover ZuBuD + 635 Houses-dataset) that
# the model sees but never individuates.
# No CFD. Held out: 64 ZuBuD buildings (houses_yin64), faces_vggHO, faces_setA.
# Schedule is r15-matched so this run reads directly against r15_vgg.
set -euo pipefail
cd /mnt/sphere/home/siagrawal/combined_lpnet

python train.py \
  --categories faces faces_vgg objects houses_zubud houses_gen \
  --variant lp --num-fixations 16 --device cuda:1 \
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
  --category-weights \
    faces=0.20 faces_vgg=0.40 objects=0.25 houses_zubud=0.10 houses_gen=0.05 \
  --run-tag r16_base
