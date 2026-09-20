#!/usr/bin/env bash
# r16rfwftH -- house fine-tune of r16_rfw, the round's best FACE model
# (+14.25 UU-II on faces_cfdWM64 against a human +14.41, the only cell in the
# wm64 battery reproducing the Yin signature) and the one whose house arm is
# broken (houses_yin64 UU 43.1% against r15_vgg's 90.58%).
#
# Question: can a working house arm be bolted onto the best face model without
# a full 18 h retrain? This is the cheap alternative to r17_base / r17_houses.
#
# Changes from the r16_rfw diet:
#   houses_zubud (40 ids)  -> houses_zubud137 (137 ids, every ZuBuD building not
#                             in houses_yin64), so houses are individuated at
#                             3.4x the identity count.
#   houses_gen             -> houses. houses_gen holds 97 leftover ZuBuD under a
#                             single generic label, which is what collapsed ZuBuD
#                             individuation in the first place -- and those same
#                             97 are now individuated in houses_zubud137, so
#                             keeping houses_gen would train them both ways.
#
# Face/object categories are capped at 40 img/class, following the r7ft/r8ft
# fine-tunes, to keep the run short without dropping the arms entirely.
# houses take 45% of every batch since they are the target.
#
# The two renamed categories (41 rows) cannot carry over by NAME and will
# re-initialise; the other 2,172 face/object rows do carry over. train.py warns
# that re-initialised rows in thin categories can fail to relearn before early
# stopping fires on aggregate valid accuracy -- that is how r8ft lost its house
# arm (90.0% -> 17.5%). Mitigated here by giving houses 45% of the diet and
# setting patience to the full run so early stopping cannot fire at all.
# Score this on its FINAL epoch, not best_model.pth.
set -euo pipefail
cd /mnt/sphere/home/siagrawal/combined_lpnet

python train.py \
  --categories faces faces_vgg faces_rfwW faces_rfwO objects houses_zubud137 houses \
  --variant lp --num-fixations 16 --device cuda:6 \
  --pretrained-path runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud_houses_gen_lp_16fix_lr0.001_resnet18_r16_rfw/final_model_20260902_141925.pth \
  --pretrained-label-map runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud_houses_gen_lp_16fix_lr0.001_resnet18_r16_rfw/label_map.json \
  --max-classes-per-category faces_rfwW=1200 faces_rfwO=300 \
  --max-images-per-class faces=40 faces_vgg=40 objects=40 \
  --lr 1e-4 --lr-schedule cosine --weight-decay 0.05 \
  --epochs 20 --batch-size 256 --num-workers 8 \
  --amp --channels-last --dataloader-sharing file_system --patience 999 \
  --category-weights \
    faces=0.10 faces_vgg=0.15 faces_rfwW=0.08 faces_rfwO=0.02 \
    objects=0.15 houses_zubud137=0.45 houses=0.05 \
  --run-tag r16rfwftH
