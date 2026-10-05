#!/usr/bin/env bash
# r21_dev -- the developmental diet rebuilt around a strict power-of-two ladder.
#
# Five changes from scripts/train_r20.sh, all of them about the DIET rather than
# the architecture, which is held at r20's resnet18_cyl so the diet is the only
# variable.
#
# 1. TOTAL face identities double every stage: 4 8 16 32 64 128 256 512 1024 2048.
#    r20 grew each face store on its own ad-hoc ladder and the total was whatever
#    fell out (16 -> 2108 over six stages). Here the total is the quantity that
#    doubles and the four stores are just how each rung is filled. Which store
#    supplies a rung is the developmental claim: stages 1-6 are 100% DEEP
#    identities (faces_vgg at 199 photos/id, faces at 130), RFW breadth enters at
#    stage 7 and other-race at stage 9, so the deep fraction slides 100% -> 30%.
#    Objects (4..64), house identities (4..137) and generic house photos (4..128)
#    each double on their own ladder, one store apiece.
#
# 2. Staggered onsets: faces stage 1, objects stage 3, house identities stage 4,
#    generic house photos stage 5. r20 started faces, objects and generic houses
#    together at stage 1.
#
# 3. Stage 1 and 2 are PINNED, not sampled. --curriculum-pin fixes the first
#    eight identities to specific VGGFace2 ids rather than letting
#    --curriculum-seed draw them, so "the first four faces" is a property of the
#    experiment instead of a property of the seed. These eight are white, which
#    is what makes the later other-race arm a manipulation rather than a
#    confound. A pinned id that is missing from the label map is a hard error.
#
# 4. --steps-per-epoch 2000 makes an epoch a fixed compute budget instead of one
#    pass over the active subset. Without it stage 1 gets 49 optimizer steps
#    against stage 10's 457,535, i.e. the entire early "developmental" phase is a
#    rounding error next to the final stage and the curriculum cannot possibly be
#    doing what it claims. With it stage 1 gets 4,000.
#
# 5. --random-fixations. The packed stores hold 32 fixations per image and
#    training used the first 16, always the same 16, so half of every store has
#    never been seen by any model in this repo. Slot j now draws from
#    {j, j+16}, which keeps 16 DISTINCT fixations per image per epoch while
#    making all 32 reachable. The starved house arm gains the most: its effective
#    pool goes from 8,768 crops to 17,536.
#
# Sampling: --category-weighting sqrt, no hand-set shares. At every level of the
# hierarchy an item's weight is proportional to the square root of the number of
# items beneath it: the four face stores form one domain, and each domain's
# share of a batch is sqrt(active classes) over the stage total (the generic
# house category, a single class, is counted by its active photos); within a
# domain each identity is weighted by sqrt(its photos), which matters only for
# faces, where photos per identity range from ~3 (RFW) to ~200 (VGG). The same
# sqrt rule sets the epoch schedule. It is the geometric midpoint between equal
# shares per domain (r20's hand-set weights renormalised, which gave four
# buildings a quarter of every stage-4 batch) and equal shares per class (which
# hands half the final batch to RFW and cuts objects to 3%). It ends at ~55%
# faces over the run, next to r20's hand-tuned 60%, without tuning.
#
# Epoch schedule: 2 2 4 4 8 8 16 16 32 32. The rule is "epochs double every two
# stages", equivalently epochs ~ sqrt(classes), and it is chosen because it needs
# no per-stage justification. It has a useful emergent property: from stage 4 on,
# passes over each stage's data stay flat at 4-9x without being tuned for.
#
# Data changes this run depends on, both already applied to fixation_data:
#   edge_margin 0.15   Every fixation now falls in the central 70% of the image.
#                      Fixations used to land anywhere in 0..223, and a fixation
#                      at the border yields a 180px crop that is ~50% zero
#                      padding -- a full forward pass and a label attached to
#                      almost no stimulus. Rejection happens inside the sampler,
#                      not as a post-filter: only 10-48% of images had 32 of
#                      their old points inside the central region, so filtering
#                      afterwards would have left most images short.
#                      ALL 146 packed splits were recomputed; the originals are
#                      beside them as coords_edge0.npy. Every checkpoint before
#                      r21 was trained on the old coordinates, so re-scoring an
#                      older model against these stores is a mismatch -- restore
#                      the backup first.
#   houses_zubud137_41 ZuBuD re-split 4 train / 1 valid, no test (was 3/1/1).
#                      411 -> 548 house training images, +33% in the most
#                      data-starved arm. The old test split was never a
#                      generalisation set: transforms["test"] rotates 180deg, so
#                      it was the inverted-presentation monitor, and for ZuBuD it
#                      compared upright view04 against inverted view05 -- an
#                      orientation/image-set confound. Every reported inversion
#                      result comes from simulate_yin1969.py, which takes both
#                      orientations from the same valid images and is unaffected.
#
# Held out, do not train on: faces_rfwHO (500 ids, the ORE instrument; verified
# disjoint from faces_rfwW and faces_rfwO), faces_vggHO, faces_cfdWM64,
# faces_setA, houses_ho64, houses_yin64, objects_all128_backup's pruned half.
#
# Usage: scripts/train_r21_dev.sh [DEVICE] [SEED] [BACKBONE]
#   scripts/train_r21_dev.sh cuda:3 42
#
# SEED sets torch/numpy init and batch order only. --curriculum-seed stays at its
# default 0 in every arm, so which identities enter at which stage is IDENTICAL
# across seeds and initialisation is the only thing that varies. The run tag
# carries the seed because out_dir is built from categories/variant/lr/backbone/
# run-tag and does NOT include it -- two seeds sharing a tag would overwrite each
# other's config.json, label_map.json and best_model.pth.
set -euo pipefail
cd /home/siagrawal/combined_lpnet

DEVICE="${1:-cuda:3}"
SEED="${2:-42}"
BACKBONE="${3:-resnet18_cyl}"

python train.py \
  --categories faces_vgg faces faces_rfwW faces_rfwO objects houses_zubud137_41 houses \
  --variant lp --num-fixations 16 --device "$DEVICE" \
  --backbone "$BACKBONE" \
  --max-classes-per-category faces_rfwW=1152 faces_rfwO=288 \
  --lr 1e-3 --lr-schedule cosine --weight-decay 0.05 \
  --batch-size 256 --num-workers 8 \
  --amp --channels-last --dataloader-sharing file_system --patience 20 \
  --random-fixations \
  --steps-per-epoch 2000 \
  --curriculum \
  --curriculum-stages \
    "faces_vgg=4,faces=0,faces_rfwW=0,faces_rfwO=0,objects=0,houses_zubud137_41=0,houses=0" \
    "faces_vgg=8,faces=0,faces_rfwW=0,faces_rfwO=0,objects=0,houses_zubud137_41=0,houses=0" \
    "faces_vgg=16,faces=0,faces_rfwW=0,faces_rfwO=0,objects=4,houses_zubud137_41=0,houses=0" \
    "faces_vgg=32,faces=0,faces_rfwW=0,faces_rfwO=0,objects=8,houses_zubud137_41=4,houses=0" \
    "faces_vgg=48,faces=16,faces_rfwW=0,faces_rfwO=0,objects=16,houses_zubud137_41=8,houses=1" \
    "faces_vgg=96,faces=32,faces_rfwW=0,faces_rfwO=0,objects=32,houses_zubud137_41=16,houses=1" \
    "faces_vgg=160,faces=64,faces_rfwW=32,faces_rfwO=0,objects=64,houses_zubud137_41=32,houses=1" \
    "faces_vgg=288,faces=96,faces_rfwW=128,faces_rfwO=0,objects=64,houses_zubud137_41=64,houses=1" \
    "faces_vgg=400,faces=128,faces_rfwW=448,faces_rfwO=48,objects=64,houses_zubud137_41=137,houses=1" \
    "faces_vgg=480,faces=128,faces_rfwW=1152,faces_rfwO=288,objects=64,houses_zubud137_41=137,houses=1" \
  --curriculum-epochs 2 2 4 4 8 8 16 16 32 32 \
  --curriculum-image-caps \
    none none none none "houses=4" "houses=8" "houses=16" "houses=32" "houses=64" "houses=128" \
  --curriculum-pin \
    faces_vgg=vgg_n000002,vgg_n000003,vgg_n000004,vgg_n000005,vgg_n000012,vgg_n000016,vgg_n000018,vgg_n000019 \
  --curriculum-warmup-steps 500 \
  --category-weighting sqrt \
  --weight-domains faces=faces_vgg,faces,faces_rfwW,faces_rfwO \
  --seed "$SEED" \
  --run-tag "r21dev_${BACKBONE#resnet18_}_s${SEED}"
