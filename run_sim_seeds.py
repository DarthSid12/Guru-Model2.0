"""
run_sim_seeds.py

Multi-seed driver for the Yin (1969) and Dobs/Kanwisher (2023) simulations.

Single-seed sims are noisy -- Yin scores 24 test pairs per condition, so one pair
is 4.2 points and two seeds of the same checkpoint can differ by 8 points. This
runs both simulations over many seeds and writes one tidy CSV, so conditions can
be reported as mean +/- SEM instead of a single draw.

Two phases:
  1. calibrate: ONE retrieval-noise level p per (model, experiment), shared by
     every category. It is fitted jointly -- a coarse then fine grid of p is
     measured on all categories, and the p minimising the summed |accuracy -
     human target| wins. A single p cannot hit all three human targets at once,
     so the per-category accuracy at the chosen p is recorded alongside it in
     noise_<model>.json under "<sim>_upright_at_p" and should be reported.
     Calibrating per seed would fold calibration jitter into the seed variance,
     which is exactly what we are trying to measure.
  2. seeds: every seed is then run at that fixed p, varying only the item sample.

Results are appended to the CSV as they land, so a partial run is still usable.

    python run_sim_seeds.py --model r8_developmental --gpu 1 --seeds 101-120
    python run_sim_seeds.py --model r8_developmental --gpu 1 --experiment ratio

--experiment picks which sweep to run; each writes its own out-dir (see
EXPERIMENTS):

  mix64    (default) Yin on the 64-identity celeb+Set_A store with its
           objects/houses_zubud controls, plus the Kanwisher identity-COUNT
           sweep from 24 (celeb only) to 49 (celeb + 25 Set_A).
  celeb24  Yin on the 24 trained-on celeb identities alone.
  mixE64   Yin on 24 celeb + 40 Set_E identities.
  ratio    Kanwisher at a fixed pool of 40 identities, sweeping the composition
           from 24 celeb + 16 Set_A to 0 celeb + 40 Set_A.
  cfdWM    Yin on the 93 white-male Chicago Face Database identities, one
           neutral photo each, novel to every model.
  cfdWM64  The same store cut to 64 identities, matching the pool size the
           mix64 / Set_E anchors were established at.

All face stores are built by make_faces_mix64.py.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

_GURU = "/home/d1deutsch/Guru-Model2.0"

MODELS = {
    "r7_curriculum": "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r7_curriculum",
    "r5b_allatonce": "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r5b_houses_zubud_30ep",
    "r8_developmental": "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r8_developmental",
    "r9_developmental_weighted": "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r9_developmental_weighted",
    "house_control_r1": f"{_GURU}/runs/faces_objects_houses_lp_16fix_lr0.001_resnet18_house_control_r1",
    # ---- 2026-08-23 round: 80 individuated ZuBuD buildings, 80 epochs, houses
    # held at 20% of every batch (house_control_r1's proportion, not r9's 10%).
    # r10 vs r12 is the single-variable contrast the house arm needs: the house
    # class ladder grows gradually (r10) or all 80 arrive at once in the final
    # stage (r12), with everything else identical. r11 adds the generic 'houses'
    # category from epoch 1 -- basic-level houseness before any individuation.
    # All three train on houses_zubud under that exact category name, so
    # EXCLUDE_SEEN holds out the 121 buildings each never saw, no
    # LABELMAP_CATEGORY entry needed.
    # 2026-08-27 fine-tunes: r7 / r8 warm-started, then trained on 159 new face
    # classes -- 90 CFD white females, the 29 CFD white males OUTSIDE the
    # held-out 64, and 40 Set_A identities -- alongside their original
    # categories (faces/objects capped at 40 img/class) at lr 1e-4 for 20
    # epochs. faces_cfdWM64 is still fully held out for both, so cfdWM64 Yin
    # numbers are comparable to the pre-fine-tune r7/r8 runs; faces_cfdWM (93)
    # is NOT, since 29 of its identities are now trained on -- score that store
    # only with --exclude-seen-from, which drops exactly those 29. Set_A is now
    # FAMILIAR to both, so mix64 / growA / familiarity4 do not apply to them.
    "r7ftA": "runs/faces_faces_new_objects_houses_zubud_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r7ftA",
    "r7ftB": "runs/faces_faces_new_objects_houses_zubud_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r7ftB",
    "r8ftA": "runs/faces_faces_new_objects_houses_zubud_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r8ftA",
    "r8ftB": "runs/faces_faces_new_objects_houses_zubud_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r8ftB",
    "r11ftA": "runs/faces_faces_new_objects_houses_zubud_houses_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r11ftA",
    "r11ftB": "runs/faces_faces_new_objects_houses_zubud_houses_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r11ftB",
    "r13_cfd_h40": "runs/faces_faces_cfd_objects_houses_zubud_houses_lp_16fix_lr0.001_resnet18_r13_cfd_h40",
    "r14_cfd_lo": "runs/faces_faces_cfd_objects_houses_zubud_houses_lp_16fix_lr0.001_resnet18_r14_cfd_lo",
    "r10_dev_h80": "runs/r10_dev_h80",
    "r11_dev_h80_generic": "runs/r11_dev_h80_generic",
    "r12_dev_h80_enbloc": "runs/r12_dev_h80_enbloc",
    # 2026-08-31 from-scratch run with VGGFace2 added as a fourth face source:
    # 727 face identities (128 CelebA + 480 VGGFace2 + 119 CFD) against r13/r14's
    # 247, 832 classes over the same six-stage curriculum. Aimed at the one gap
    # CFD training did not close -- Kanwisher matching under NATURAL photo
    # variation, where CFD's expression-only within-identity variation gave the
    # models nothing (Set_A: r8 +2.89, r11 +2.30 vs human +10.70).
    # Held out and therefore scorable: faces_vggHO (60 VGGFace2 identities that
    # appear nowhere in faces_vgg -- verified zero shared classes and zero
    # pixel-identical images), faces_cfdWM64, houses_yin64, and -- unlike every
    # fine-tune -- faces_setA, which r15 never trained on.
    "r15_vgg": "runs/faces_faces_vgg_faces_cfd_objects_houses_zubud_houses_lp_16fix_lr0.001_resnet18_r15_vgg",
    # ---- 2026-09-01 round (r16). Same 6-stage / 80-epoch r15-matched curriculum
    # for all three, no CFD in any of them, and a generic `house` class
    # (houses_gen = 97 leftover ZuBuD + 635 Houses-dataset buildings) that is
    # seen but never individuated alongside 40 ZuBuD identities.
    #   r16_base    128 CelebA + 480 VGGFace2 faces, 64 objects.
    #   r16_acuity  r16_base EXACTLY, plus a train-time blur->sharp schedule
    #               (sigma 8/6/4/2/1/0 per stage, Vogelsang et al. 2018).
    #               Single-variable contrast against r16_base.
    #   r16_rfw     r16_base plus 1500 RFW identities introduced white-first
    #               (1200 Caucasian from stage 1, 300 other-race from stage 4).
    # Held out and therefore scorable for all three: faces_cfdWM64 (a fully
    # NOVEL domain here, unlike r15 which trained on 119 CFD identities),
    # faces_rfwWM64, faces_rfwHO, faces_vggHO, faces_setA, houses_yin64.
    "r16_base": "runs/faces_faces_vgg_objects_houses_zubud_houses_gen_lp_16fix_lr0.001_resnet18_r16_base",
    "r16_acuity": "runs/faces_faces_vgg_objects_houses_zubud_houses_gen_lp_16fix_lr0.001_resnet18_r16_acuity",
    "r16_rfw": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud_houses_gen_lp_16fix_lr0.001_resnet18_r16_rfw",
    # 2026-09-03 house fine-tune of r16_rfw: warm-started from its final epoch,
    # then 20 epochs at lr 1e-4 with houses_zubud137 (137 ZuBuD identities, every
    # building NOT in houses_yin64) at 45% of the diet and the generic class
    # switched from houses_gen back to `houses`. Face/object categories capped at
    # 40 img/class. houses_yin64 remains fully held out (zero overlap asserted in
    # the store builder), so its Yin row is directly comparable to r16_rfw's.
    # Scored on the FINAL epoch: best_model.pth is from epoch 1, because
    # aggregate valid accuracy is dominated by the face categories and peaks
    # before the 138 re-initialised house rows have learned anything.
    "r16rfwftH": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud137_houses_lp_16fix_lr0.0001_resnet18_faces40img_faces_vgg40img_objects40img_r16rfwftH",
    # 2026-09-04, scored MID-RUN at ~epoch 63/80 on best_model.pth (both were in
    # the final curriculum stage, so best_model.pth is a valid stage-6 weight
    # set). Not final models -- LR was still ~1e-4 and decaying. Re-score both
    # when the runs land.
    #   r17_base    r16_base with the generic house class back to `houses`
    #               (Houses-dataset only, no ZuBuD) -- the single-variable test
    #               of whether houses_gen is what broke held-out ZuBuD.
    #   r17_houses  that, plus houses_zubud137 (137 ids) at weight 0.20, paid for
    #               by faces_vgg 0.40->0.33 and objects 0.25->0.22. NOTE the
    #               objects weight moved, so its object rows are not comparable
    #               to r16/r17_base.
    # 2026-09-10: inverted-exposure arms. r16_rfw's recipe, byte-identical apart
    # from --invert-p, so r16_rfw itself is the matched 0% control. These
    # DELIBERATELY break the upright-only training invariant (see
    # scripts/train_r18_inv*.sh), so their Yin rows are interpretable only
    # against each other and r16_rfw -- never against the r15/r16/r17 battery.
    # Classification inversion gap at the final epoch: 0% 17.95, 5% 2.18,
    # 20% 1.54 -- i.e. 5% already abolishes it and 20% adds nothing, so the
    # informative exposures are BELOW 5%.
    "r18_inv02": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud_houses_gen_lp_16fix_lr0.001_resnet18_r18_inv02",
    "r18_inv05": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud_houses_gen_lp_16fix_lr0.001_resnet18_r18_inv05",
    "r18_inv20": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud_houses_gen_lp_16fix_lr0.001_resnet18_r18_inv20",
    "r17_base": "runs/faces_faces_vgg_objects_houses_zubud_houses_lp_16fix_lr0.001_resnet18_r17_base",
    "r17_houses": "runs/faces_faces_vgg_objects_houses_zubud137_houses_lp_16fix_lr0.001_resnet18_r17_houses",
    # 2026-09-18/19: r19_rfwh -- r16_rfw's face diet (128 CelebA + 480 VGGFace2
    # + 1200 RFW-W + 300 RFW-O, white-first ladder) with the r17 house fix
    # applied FROM SCRATCH instead of by fine-tune: generic class houses_gen ->
    # houses, and houses_zubud (40 ids @0.08) -> houses_zubud137 (137 ids
    # @0.16), paid for out of objects 0.25->0.19 and faces_vgg 0.28->0.26. So
    # r16_rfw is the matched broken-house control and r16rfwftH the matched
    # repaired-by-fine-tune arm. houses_yin64 stays fully held out (zero overlap
    # with houses_zubud137, verified against both stores' meta.json).
    #
    # SIX SEED REPLICATES of ONE recipe, differing only in --seed (torch/numpy
    # init and batch order). --curriculum-seed is 0 in all six, so which
    # identity enters at which stage is identical and initialisation is the only
    # variable. This is the repo's first variance control: every other row in
    # paper/results_rfwcal_battery.md is a single training run, so no
    # cross-model gap there has ever been checked against seed noise. Seed 42
    # matches what every r15/r16/r17 run used, so s42 is the arm directly
    # comparable to the existing battery.
    "r19_rfwh_s42": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud137_houses_lp_16fix_lr0.001_resnet18_r19_rfwh_s42",
    "r19_rfwh_s43": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud137_houses_lp_16fix_lr0.001_resnet18_r19_rfwh_s43",
    "r19_rfwh_s44": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud137_houses_lp_16fix_lr0.001_resnet18_r19_rfwh_s44",
    "r19_rfwh_s45": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud137_houses_lp_16fix_lr0.001_resnet18_r19_rfwh_s45",
    "r19_rfwh_s46": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud137_houses_lp_16fix_lr0.001_resnet18_r19_rfwh_s46",
    "r19_rfwh_s47": "runs/faces_faces_vgg_faces_rfwW_faces_rfwO_objects_houses_zubud137_houses_lp_16fix_lr0.001_resnet18_r19_rfwh_s47",
}
# Checkpoint override, per model. `--run-dir` otherwise resolves to
# best_model.pth (simulate_yin1969.resolve_from_run_dir), which is selected on
# AGGREGATE valid accuracy -- and for the fine-tunes that metric is dominated by
# the 2110 face and 8208 object valid images, so it peaks at epoch 1-3 while the
# 77 new-face images keep improving for another seven epochs. r8ftA's best
# checkpoint has faces_new at 28.6% against 54.5% at epoch 20, at no cost to the
# known categories (houses 92.5 vs 90.0, faces 88.3 vs 88.4). Scoring the best
# checkpoint would test a model with half the CFD adaptation the run produced,
# so the fine-tunes are scored on their FINAL epoch instead.
_FINAL = {
    "r7ftA": "runs/faces_faces_new_objects_houses_zubud_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r7ftA/final_model_20260829_113414.pth",
    "r7ftB": "runs/faces_faces_new_objects_houses_zubud_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r7ftB/final_model_20260829_113434.pth",
    "r8ftA": "runs/faces_faces_new_objects_houses_zubud_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r8ftA/final_model_20260829_113343.pth",
    "r8ftB": "runs/faces_faces_new_objects_houses_zubud_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r8ftB/final_model_20260829_113336.pth",
    "r11ftA": "runs/faces_faces_new_objects_houses_zubud_houses_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r11ftA/final_model_20260830_145544.pth",
    "r11ftB": "runs/faces_faces_new_objects_houses_zubud_houses_lp_16fix_lr0.0001_resnet18_faces40img_objects40img_r11ftB/final_model_20260830_145528.pth",
    "r13_cfd_h40": "runs/faces_faces_cfd_objects_houses_zubud_houses_lp_16fix_lr0.001_resnet18_r13_cfd_h40/final_model_20260831_020730.pth",
    "r14_cfd_lo": "runs/faces_faces_cfd_objects_houses_zubud_houses_lp_16fix_lr0.001_resnet18_r14_cfd_lo/final_model_20260831_085136.pth",
}
CHECKPOINT = dict(_FINAL)


def _final_ckpt(run_dir):
    import glob as _glob
    c = sorted(_glob.glob(os.path.join(run_dir, "final_model_*.pth")))
    return c[-1] if c else None


# r15_vgg is scored on its final epoch for the same reason r13/r14 are: with a
# curriculum, best_model.pth is picked on aggregate valid accuracy, which peaks
# inside an early stage while only a fraction of the classes are active. Its
# timestamp is not known until the run ends, so resolve it at import time rather
# than hard-coding one into _FINAL.
_r15 = _final_ckpt(MODELS["r15_vgg"])
if _r15:
    CHECKPOINT["r15_vgg"] = _r15

# The r16 round is scored on its final epoch too, so it is read against r15 on
# the same footing (their best_model.pth do land in the final stage, but mixing
# best- and final-epoch checkpoints across a comparison is not worth the risk).
# r17 joins them for that same reason: it is read directly against r16, and its
# 2026-09-04 rows were scored mid-run at ~epoch 63/80 because the runs had not
# landed yet. Those CSVs are archived as results_*_interim_ep63.csv.
# r19 joins them: same reason, and its best_model.pth is selected on aggregate
# valid accuracy, which here is dominated by the 1200-identity RFW arm sitting
# near its floor -- not a metric that should choose the weights the Yin sim
# reads. Seeds still training simply have no final_model_*.pth yet and are
# skipped by the `if _c` below until they land.
for _m in ("r16_base", "r16_acuity", "r16_rfw", "r16rfwftH",
           "r17_base", "r17_houses", "r18_inv02", "r18_inv05", "r18_inv20",
           "r19_rfwh_s42", "r19_rfwh_s43", "r19_rfwh_s44",
           "r19_rfwh_s45", "r19_rfwh_s46", "r19_rfwh_s47"):
    _c = _final_ckpt(MODELS[_m])
    if _c:
        CHECKPOINT[_m] = _c


# Every model is scored on the same three packed stores in this repo, so the
# numbers are comparable across models. house_control_r1 lives in the
# Guru-Model2.0 tree but is evaluated here, not on its own copies.
DEFAULT_CATEGORIES = ["faces_setA", "objects", "houses_zubud"]
# Yin uses Set_E, which has 64 identities -- exactly the 40 study + 24 new the
# design calls for, so both sides of the 2AFC come from ONE store. Set_A could
# not do this (only 40 identities) and drawing its distractors from the old
# faces store made the task solvable by dataset appearance rather than memory:
# studied and UNSTUDIED Set_A items both scored 90% against old-store
# distractors, while the honest same-store test sat at chance. Never let the
# study and distractor pools come from different stimulus sets.
# Both sims use Set_E. Kanwisher on Set_A scored 69.3% against an 87.5% human
# anchor; on Set_E, same model/task/p and a matched 40 identities x 5 photos, it
# scores 88.0%. Set_A photos of one person barely resemble each other
# (within-identity pixel corr +0.078 vs Set_E +0.113) while every identity is a
# young female celebrity, so between-identity variability is low too -- both
# directions hurt a matching task. Not a resolution effect: Set_E has MORE
# sub-224px images. Yin survived on Set_A only because its "old" item is the
# identical photo re-read, so it never has to generalise across images.
#
# faces_mix64 (see make_faces_mix64.py) replaces Set_E as the face store: 24
# CelebA identities every model TRAINED on (`celeb_*`, familiar) + all 40 Set_A
# identities none of them saw (`setA_*`, unfamiliar), 5 photos each. The celeb
# photos still come from the held-out 'valid' split, so no exact image was seen
# -- only the identity was.
#
# Yin keeps the objects and houses_zubud controls; Kanwisher runs on the faces
# store alone, because there its design is the identity-count sweep below rather
# than a category comparison.

# ---- Kanwisher identity sets ------------------------------------------------
# Each point is (label, [(class-name prefix, n identities), ...]). --group-sizes
# selections NEST as n grows, so at a fixed seed a point that asks for more of a
# prefix keeps everything the smaller point used.
#
# Identity-COUNT sweep: hold the 24 celeb identities fixed and add Set_A ones 5
# at a time, so the pool grows from 24 to 49.
KANW_COUNT_SWEEP = [(24 + k, [("celeb_", 24), ("setA_", k)])
                    for k in (0, 5, 10, 15, 20, 25)]
# Identity-RATIO sweep: the pool is ALWAYS 40 -- Dobs et al.'s Exp. 1 size -- and
# only its composition changes, from 24 trained-on celebs + 16 novel Set_A
# identities through to 40 novel ones. Holding the count fixed separates "how
# familiar are these identities" from "how many identities is the task
# discriminating among", which the count sweep above confounds.
KANW_RATIO_SWEEP = [(f"{c}celeb_{40 - c}setA", [("celeb_", c), ("setA_", 40 - c)])
                    for c in (24, 20, 16, 12, 8, 4, 0)]
# The two most-familiar points of that sweep, run on their own so p can be fitted
# where the models can actually REACH the 87.5% human anchor. Fitting at the
# all-unfamiliar end pins p at 0 (no model exceeds 87.5% there), which leaves the
# familiar points running several points ABOVE the anchor and makes their
# inversion effects unanchored. At a 40-identity pool with 24 celebs every model
# sits at 89-91% upright at p=0, so a positive p exists that lands on 87.5.
KANW_RATIO_HI = KANW_RATIO_SWEEP[:2]

# ---- experiment presets -----------------------------------------------------
# One preset = one self-contained sweep: which sims run, on which packed stores,
# how the Kanwisher identity set is composed at each point, and where results
# land. --experiment picks one. Presets share nothing but MODELS and the human
# targets, and each writes its own out-dir, so adding one can never disturb an
# existing CSV.
#
#   fields
#     categories        {sim: [packed store, ...]} -- what that sim is run on
#     calib_categories  {sim: [...]} -- which of those get a vote in fitting p
#     yin_shuffle       stores whose item list Yin must shuffle before splitting
#                       it into study / never-studied pools. Every mixed store
#                       needs this: they pack their celeb identities before the
#                       novel ones, so without a shuffle every celeb would land
#                       in study and the distractor pool would be purely novel,
#                       letting "old vs new" be decided on sub-population rather
#                       than memory. faces_celeb24 needs it for a different
#                       reason: with only 24 items the 15/9 split consumes the
#                       whole store, so without a reshuffle every seed would
#                       score the identical 9 pairs and show zero variance.
#     kanw_units        {store: [(label, groups), ...]} -- the sweep points
#     kanw_calib_label  the ONE point p is fitted at, then held fixed across all
#                       of them; a per-point p would absorb exactly the accuracy
#                       differences the sweep exists to measure
#     label_col         name of the CSV column the point label goes in
EXPERIMENTS = {
    # The original sweep: Yin on the 64-identity mix plus its object/house
    # controls, and the Kanwisher identity-count sweep.
    "mix64": dict(
        out_dir="runs/sim_seeds_mix64",
        sims=["yin", "kanwisher"],
        categories={"yin": ["faces_mix64", "objects", "houses_zubud"],
                    "kanwisher": ["faces_mix64"]},
        calib_categories={"yin": ["faces_mix64"], "kanwisher": ["faces_mix64"]},
        yin_shuffle={"faces_mix64"},
        kanw_units={"faces_mix64": KANW_COUNT_SWEEP},
        kanw_calib_label=49,
        label_col="n_identities",
    ),
    # Yin on the 24 trained-on celeb identities ALONE -- every item familiar,
    # both the studied ones and the distractors. 24 items cannot fill the 40/24
    # design, so simulate_yin1969.py scales it to study=15 / test=9, keeping the
    # 5:3 ratio. Nine pairs means 11.1 points per pair, so this condition is only
    # readable across seeds, never from one.
    "celeb24": dict(
        out_dir="runs/sim_seeds_celeb24",
        sims=["yin"],
        categories={"yin": ["faces_celeb24"]},
        calib_categories={"yin": ["faces_celeb24"]},
        yin_shuffle={"faces_celeb24"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Yin on 24 celeb + 40 Set_E: the same familiar/unfamiliar design as mix64,
    # with Set_E as the unfamiliar half instead of Set_A. Set_E's photos of one
    # identity resemble each other more than Set_A's do, so this is the cleaner
    # of the two unfamiliar sets and a check on whether mix64's Yin numbers are
    # a Set_A artefact.
    "mixE64": dict(
        out_dir="runs/sim_seeds_mixE64",
        sims=["yin"],
        categories={"yin": ["faces_mixE64"]},
        calib_categories={"yin": ["faces_mixE64"]},
        yin_shuffle={"faces_mixE64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Kanwisher at a fixed pool of 40, sweeping the familiar/unfamiliar ratio.
    # p is fitted at the all-unfamiliar end, because that is the condition the
    # 87.5% human anchor was measured in -- Dobs et al.'s participants matched
    # faces they had never seen before.
    "ratio": dict(
        out_dir="runs/sim_seeds_ratio",
        sims=["kanwisher"],
        categories={"kanwisher": ["faces_mix64"]},
        calib_categories={"kanwisher": ["faces_mix64"]},
        yin_shuffle=set(),
        kanw_units={"faces_mix64": KANW_RATIO_SWEEP},
        kanw_calib_label="0celeb_40setA",
        label_col="composition",
    ),
    # The 24- and 20-celeb points of that sweep, calibrated where the anchor is
    # reachable. --calib-label moves the fit between the two points, so the same
    # preset gives both readings: fit at 24 (the 20 point then floats below the
    # anchor) or fit at 20 (the 24 point floats above it). Use a separate
    # --out-dir per fit; the noise JSON is per-directory.
    "ratio_hi": dict(
        out_dir="runs/sim_seeds_ratio_hi",
        sims=["kanwisher"],
        categories={"kanwisher": ["faces_mix64"]},
        calib_categories={"kanwisher": ["faces_mix64"]},
        yin_shuffle=set(),
        kanw_units={"faces_mix64": KANW_RATIO_HI},
        kanw_calib_label="24celeb_16setA",
        label_col="composition",
    ),
    # ---- 2026-08-18 round --------------------------------------------------
    # Experiment 1 re-run: the same 24 familiar celebs, but with the objects and
    # houses controls scored at the SAME p that faces calibrate to, so all three
    # Yin categories are readable off one noise level (the earlier celeb24 run
    # scored faces alone, and §4 of results_celeb_experiments.md had to borrow
    # its house row from the mix64 sweep, where p was fitted on a different
    # store). Pair with --calib-seeds 40 --calib-on-eval-seeds: with 9 pairs per
    # condition the default 8-sample fit is only good to ~1.4 points, which is
    # exactly the overshoot being corrected.
    "celeb24_ctrl": dict(
        out_dir="runs/sim_seeds_celeb24_ctrl",
        sims=["yin"],
        categories={"yin": ["faces_celeb24", "objects", "houses_zubud"]},
        calib_categories={"yin": ["faces_celeb24"]},
        yin_shuffle={"faces_celeb24"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Houses calibrated on their OWN human anchor (90.71) instead of inheriting
    # the faces p. Experiment 4 as reported was faces-calibrated, so its house
    # rows sat wherever that p happened to put them; this fits p on houses so the
    # house inversion cost is measured at human-matched house difficulty. If a
    # model cannot reach 90.71 upright at any p (r8 held-out ZuBuD tops out near
    # 84%), the fit pins at p=0 and the residual in noise_<model>.json is the
    # honest ceiling -- report it as such rather than as a calibrated row.
    "houses_cal": dict(
        out_dir="runs/sim_seeds_houses_cal",
        sims=["yin"],
        categories={"yin": ["houses_zubud"]},
        calib_categories={"yin": ["houses_zubud"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Growth sweep, arm A: start at the 24 familiar celebs and add 4 novel Set_A identities (the requested arm; Set_A is near-chance under Yin, so expect the drop to be steep and to reflect stimulus quality as much as familiarity)
    # at a time, up to 64 identities. p is fitted ONCE on faces_celeb24 and held
    # fixed across every point -- the sweep exists to find where upright
    # accuracy falls to Yin's 96.29, and refitting per point would pin it there
    # by construction. Every store is shuffled before the 5:3 study/test split,
    # and the split scales with the store (24 items -> 15/9, 64 -> 40/24), so
    # pool size and study-list length grow together exactly as in Yin's design.
    "growthA": dict(
        out_dir="runs/sim_seeds_growthA",
        sims=["yin"],
        categories={"yin": ['faces_celeb24', 'faces_growA1', 'faces_growA2', 'faces_growA3', 'faces_growA4', 'faces_growA6', 'faces_growA8', 'faces_growA12', 'faces_growA16', 'faces_growA24', 'faces_growA32', 'faces_growA40']},
        calib_categories={"yin": ["faces_celeb24"]},
        yin_shuffle={'faces_growA12', 'faces_growA8', 'faces_growA32', 'faces_growA2', 'faces_growA24', 'faces_growA4', 'faces_growA16', 'faces_growA3', 'faces_growA40', 'faces_growA6', 'faces_celeb24', 'faces_growA1'},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Growth sweep, arm E: start at the 24 familiar celebs and add 4 novel Set_E identities -- the clean unfamiliar set
    # at a time, up to 64 identities. p is fitted ONCE on faces_celeb24 and held
    # fixed across every point -- the sweep exists to find where upright
    # accuracy falls to Yin's 96.29, and refitting per point would pin it there
    # by construction. Every store is shuffled before the 5:3 study/test split,
    # and the split scales with the store (24 items -> 15/9, 64 -> 40/24), so
    # pool size and study-list length grow together exactly as in Yin's design.
    "growthE": dict(
        out_dir="runs/sim_seeds_growthE",
        sims=["yin"],
        categories={"yin": ['faces_celeb24', 'faces_growE1', 'faces_growE2', 'faces_growE3', 'faces_growE4', 'faces_growE6', 'faces_growE8', 'faces_growE12', 'faces_growE16', 'faces_growE24', 'faces_growE32', 'faces_growE40']},
        calib_categories={"yin": ["faces_celeb24"]},
        yin_shuffle={'faces_growE6', 'faces_growE24', 'faces_growE2', 'faces_growE1', 'faces_growE8', 'faces_growE32', 'faces_growE4', 'faces_growE3', 'faces_growE40', 'faces_growE12', 'faces_growE16', 'faces_celeb24'},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Growth sweep, arm F: start at the 24 familiar celebs and add 4 further TRAINED celeb identities -- the familiarity control: same pool growth, no novelty
    # at a time, up to 64 identities. p is fitted ONCE on faces_celeb24 and held
    # fixed across every point -- the sweep exists to find where upright
    # accuracy falls to Yin's 96.29, and refitting per point would pin it there
    # by construction. Every store is shuffled before the 5:3 study/test split,
    # and the split scales with the store (24 items -> 15/9, 64 -> 40/24), so
    # pool size and study-list length grow together exactly as in Yin's design.
    "growthF": dict(
        out_dir="runs/sim_seeds_growthF",
        sims=["yin"],
        categories={"yin": ['faces_celeb24', 'faces_growF1', 'faces_growF2', 'faces_growF3', 'faces_growF4', 'faces_growF6', 'faces_growF8', 'faces_growF12', 'faces_growF16', 'faces_growF24', 'faces_growF32', 'faces_growF40']},
        calib_categories={"yin": ["faces_celeb24"]},
        yin_shuffle={'faces_growF16', 'faces_growF6', 'faces_growF8', 'faces_growF2', 'faces_growF1', 'faces_growF24', 'faces_growF40', 'faces_growF12', 'faces_growF4', 'faces_celeb24', 'faces_growF32', 'faces_growF3'},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Per-store recalibration of the Set_A growth sweep. The fixed-p sweep
    # (growthA) holds p at the 24-celeb value, so upright accuracy falls as
    # identities are added and the inversion cost is read at a DIFFERENT task
    # difficulty at every point -- which is the confound that made the fixed-p
    # cost curve hard to interpret. These presets instead fit p separately for
    # each store, so every point is read where the model sits on Yin's 96.29
    # upright anchor and only the stimulus composition differs. p necessarily
    # DECREASES as identities are added (the store gets harder, so less
    # retrieval noise is needed to stay on the anchor); where a store cannot
    # reach 96.29 even at p=0 the fit pins at the floor and the point is a
    # ceiling, not a calibration -- check yin_upright_at_p before reporting it.
    "growA_cal1": dict(
        out_dir="runs/sim_seeds_growA_cal1",
        sims=["yin"],
        categories={"yin": ["faces_growA1"]},
        calib_categories={"yin": ["faces_growA1"]},
        yin_shuffle={"faces_growA1"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "growA_cal2": dict(
        out_dir="runs/sim_seeds_growA_cal2",
        sims=["yin"],
        categories={"yin": ["faces_growA2"]},
        calib_categories={"yin": ["faces_growA2"]},
        yin_shuffle={"faces_growA2"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "growA_cal3": dict(
        out_dir="runs/sim_seeds_growA_cal3",
        sims=["yin"],
        categories={"yin": ["faces_growA3"]},
        calib_categories={"yin": ["faces_growA3"]},
        yin_shuffle={"faces_growA3"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "growA_cal4": dict(
        out_dir="runs/sim_seeds_growA_cal4",
        sims=["yin"],
        categories={"yin": ["faces_growA4"]},
        calib_categories={"yin": ["faces_growA4"]},
        yin_shuffle={"faces_growA4"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "growA_cal6": dict(
        out_dir="runs/sim_seeds_growA_cal6",
        sims=["yin"],
        categories={"yin": ["faces_growA6"]},
        calib_categories={"yin": ["faces_growA6"]},
        yin_shuffle={"faces_growA6"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "growA_cal8": dict(
        out_dir="runs/sim_seeds_growA_cal8",
        sims=["yin"],
        categories={"yin": ["faces_growA8"]},
        calib_categories={"yin": ["faces_growA8"]},
        yin_shuffle={"faces_growA8"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "growA_cal12": dict(
        out_dir="runs/sim_seeds_growA_cal12",
        sims=["yin"],
        categories={"yin": ["faces_growA12"]},
        calib_categories={"yin": ["faces_growA12"]},
        yin_shuffle={"faces_growA12"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "growA_cal16": dict(
        out_dir="runs/sim_seeds_growA_cal16",
        sims=["yin"],
        categories={"yin": ["faces_growA16"]},
        calib_categories={"yin": ["faces_growA16"]},
        yin_shuffle={"faces_growA16"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # ---- 2026-08-20 round: familiarity composition at fixed noise ----------
    # Four 64-identity face stores that differ ONLY in how many of the
    # identities the model trained on, plus the 64 trained object classes as a
    # control. Run at a FIXED p=0.30 for every store and every model rather than
    # calibrated per store: p=0.30 is the value the 24-known store calibrates to
    # on the Yin anchor, so holding it fixed makes the four compositions
    # comparable to each other and across models at matched retrieval noise.
    # The trade-off is the one the growth sweep made explicit -- upright
    # accuracy will differ between stores, so a cost difference here is not read
    # at matched task difficulty. Report the upright accuracies alongside.
    #
    # Every face store is shuffled before the 40/24 study/test split: they pack
    # their known identities before their unknown ones, so without a shuffle
    # every known identity would land in the study pool and the distractors
    # would be purely unknown, letting "old vs new" be decided on familiarity
    # rather than memory.
    "familiarity4": dict(
        out_dir="runs/sim_seeds_familiarity4",
        sims=["yin"],
        categories={"yin": ["faces_k64", "faces_k32u32", "faces_u64",
                            "faces_mixed64", "objects"]},
        calib_categories={"yin": ["faces_k64"]},
        yin_shuffle={"faces_k64", "faces_k32u32", "faces_u64", "faces_mixed64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Yin's 40 study / 24 test design split along FAMILIARITY: 40 identities the
    # model trained on as the study pool, 24 novel ones as the never-studied
    # distractors. p is fitted so upright-upright lands on Yin's 96.29 human
    # anchor for that store. Deliberately NOT shuffled -- the packed order is
    # the design (see make_faces_mix64.py).
    #
    # faces_u40k24 runs alongside as the validity control, not as a result: same
    # two populations, reversed roles, so familiarity opposes the correct answer
    # instead of aligning with it. If k40u24 scores far above u40k24, the 2AFC is
    # being won on sub-population appearance rather than on memory for the study
    # episode, and the k40u24 number is not a Yin result. Both are scored at the
    # p fitted on k40u24 so the comparison is at one noise level.
    "yin4024": dict(
        out_dir="runs/sim_seeds_yin4024",
        sims=["yin"],
        categories={"yin": ["faces_k40u24", "faces_u40k24"]},
        calib_categories={"yin": ["faces_k40u24"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # The mixed familiarity store (24 named celebs + 16 other trained + 12 Set_E
    # + 12 Set_A) with its object and house controls, all at one fixed noise
    # level. Separate from familiarity4 so the two noise levels never share a
    # results CSV; --fixed-noise sets p, nothing here is calibrated.
    "mixed64_ctrl": dict(
        out_dir="runs/sim_seeds_mixed64_ctrl",
        sims=["yin"],
        categories={"yin": ["faces_mixed64", "objects", "houses_zubud"]},
        calib_categories={"yin": ["faces_mixed64"]},
        yin_shuffle={"faces_mixed64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # ---- headline round: ONE noise level fitted jointly on faces AND houses --
    # p is fitted to minimise the summed deviation from BOTH human anchors at
    # once (faces 96.29, houses 90.71) and then held fixed across all three
    # categories, so the inversion costs are read at one noise level rather than
    # one per category. Objects ride along without a vote -- they cannot reach
    # their 84.79 anchor at any usable p (every object class was trained on).
    #
    # houses_common is the house store: 94 Houses-dataset building exteriors
    # unseen by r7, r8 AND house_control_r1, so this is the first house row the
    # three models can be compared on. ZuBuD arm below is the cross-check.
    "joint_common": dict(
        out_dir="runs/sim_seeds_joint_common",
        sims=["yin"],
        categories={"yin": ["faces_mixed64", "objects", "houses_common"]},
        calib_categories={"yin": ["faces_mixed64", "houses_common"]},
        yin_shuffle={"faces_mixed64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Same joint fit on ZuBuD instead. r8 is scored on its 161 held-out classes,
    # house_control_r1 never saw ZuBuD, r7 trained on all 201 -- so r7's row here
    # is TRAINED-ON and must be labelled. Kanwisher houses can only use this
    # store (houses_ident's valid and test splits are bit-identical).
    "joint_zubud": dict(
        out_dir="runs/sim_seeds_joint_zubud",
        sims=["yin"],
        categories={"yin": ["faces_mixed64", "objects", "houses_zubud"]},
        calib_categories={"yin": ["faces_mixed64", "houses_zubud"]},
        yin_shuffle={"faces_mixed64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Same joint fit as joint_zubud, but on faces_mixu16 (48 known + 16 unknown)
    # instead of faces_mixed64 (40/24). Reason: on mixed64 every model in the
    # 2026-08-23 round tops out at 88-90% upright at p=0, BELOW Yin's 96.29
    # anchor, so the joint fit pins at the floor and the "calibration" is a
    # ceiling report rather than a fit. mixu16 is the rung of the Step 10 ladder
    # where faces actually reach the anchor (r8 read 95.75 there), so p has
    # somewhere to go and the faces and houses anchors can genuinely be met at
    # once. Houses are unchanged and still auto-held-out.
    #
    # mixu16 packs its 48 known identities before its 16 unknown ones, so
    # yin_shuffle is mandatory: without it every known face lands in the study
    # pool and the distractors are purely novel, which lets "old vs new" be
    # decided on sub-population rather than on memory (Step 5).
    "joint_zubud_mixu16": dict(
        out_dir="runs/sim_seeds_joint_zubud_mixu16",
        sims=["yin"],
        categories={"yin": ["faces_mixu16", "objects", "houses_zubud"]},
        calib_categories={"yin": ["faces_mixu16", "houses_zubud"]},
        yin_shuffle={"faces_mixu16"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Kanwisher arm of the headline round. p is fitted separately from Yin's --
    # the task and its human anchor are different (87.5% upright matching) -- and
    # on faces alone, because objects and houses sit near their targets across a
    # wide range of p and would drag faces far below its anchor.
    #
    # Houses are ZuBuD and NOT houses_common: houses_ident's valid and test
    # splits are bit-identical, so a triplet match there would compare an image
    # with itself. ZuBuD borrows both splits (KANW_SPLITS) to get two genuinely
    # different views per building.
    #
    # 20 seeds, not 100: one Kanwisher seed scores ~100k triplets, so its SEMs
    # run +-0.1-0.2 where a 100-seed Yin run is still at +-0.6.
    "kanw3": dict(
        out_dir="runs/sim_seeds_kanw3",
        sims=["kanwisher"],
        categories={"kanwisher": ["faces_mixed64", "objects", "houses_zubud"]},
        calib_categories={"kanwisher": ["faces_mixed64"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Familiarity ladder, one preset per rung so the six can run as six parallel
    # processes instead of one 600-unit serial sweep -- the sims are CPU-bound
    # and each saturates only ~2 cores, so width is the only way to use the box.
    # Every rung is a 64-identity store; only the known/unknown ratio moves.
    "ladder_u8": dict(
        out_dir="runs/sim_seeds_ladder/u8",
        sims=["yin"],
        categories={"yin": ["faces_mixu8"]},
        calib_categories={"yin": ["faces_mixu8"]},
        yin_shuffle={"faces_mixu8"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "ladder_u16": dict(
        out_dir="runs/sim_seeds_ladder/u16",
        sims=["yin"],
        categories={"yin": ["faces_mixu16"]},
        calib_categories={"yin": ["faces_mixu16"]},
        yin_shuffle={"faces_mixu16"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "ladder_u32": dict(
        out_dir="runs/sim_seeds_ladder/u32",
        sims=["yin"],
        categories={"yin": ["faces_mixu32"]},
        calib_categories={"yin": ["faces_mixu32"]},
        yin_shuffle={"faces_mixu32"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "ladder_u40": dict(
        out_dir="runs/sim_seeds_ladder/u40",
        sims=["yin"],
        categories={"yin": ["faces_mixu40"]},
        calib_categories={"yin": ["faces_mixu40"]},
        yin_shuffle={"faces_mixu40"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "ladder_u48": dict(
        out_dir="runs/sim_seeds_ladder/u48",
        sims=["yin"],
        categories={"yin": ["faces_mixu48"]},
        calib_categories={"yin": ["faces_mixu48"]},
        yin_shuffle={"faces_mixu48"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # The u=24 rung is the existing baseline store, kept under its own name.
    "ladder_u24": dict(
        out_dir="runs/sim_seeds_ladder/u24",
        sims=["yin"],
        categories={"yin": ["faces_mixed64"]},
        calib_categories={"yin": ["faces_mixed64"]},
        yin_shuffle={"faces_mixed64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Full Yin (all four study/test orientation cells) at a FIXED p=0.24, one
    # preset per category so the three can run as three parallel processes on a
    # single card -- the sims are CPU-bound, so splitting by category is the
    # cheapest way to cut wall clock. Nothing is calibrated here: p is imposed.
    "p024_faces": dict(
        out_dir="runs/sim_seeds_p024/faces",
        sims=["yin"], categories={"yin": ["faces_mixed64"]},
        calib_categories={"yin": ["faces_mixed64"]},
        yin_shuffle={"faces_mixed64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "p024_houses": dict(
        out_dir="runs/sim_seeds_p024/houses",
        sims=["yin"], categories={"yin": ["houses_zubud"]},
        calib_categories={"yin": ["houses_zubud"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "p024_objects": dict(
        out_dir="runs/sim_seeds_p024/objects",
        sims=["yin"], categories={"yin": ["objects"]},
        calib_categories={"yin": ["objects"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # House familiarity ladder for house_control_r1, one preset per rung so the
    # eleven can run as eleven parallel processes. Each store is 40 ZuBuD
    # buildings; 4 trained buildings are swapped for 4 never-trained ones at each
    # step, from 40 known / 0 unknown to 0 known / 40 unknown. Built by
    # make_houses_ladder.py, which reads which buildings the model actually
    # trained on from its label_map (r8 and house_control_r1 trained on DIFFERENT
    # 40s, overlapping in only 12, so the ladder is per-model).
    #
    # Shuffled, and it matters: the stores pack their known buildings before
    # their unknown ones, so without a shuffle every known building would land in
    # the study pool and every distractor would be unknown -- letting "old vs
    # new" be decided by familiarity rather than memory. That is exactly the
    # artefact that invalidated faces_k40u24.
    #
    # With 40 items simulate_yin1969.py scales its 40/24 split to 25 study /
    # 15 test, so a rung scores 15 pairs per condition; read across seeds only.
    "hlad_k40": dict(
        out_dir="runs/sim_seeds_hladder/k40",
        sims=["yin"], categories={"yin": ["houses_dvlad_k40"]},
        calib_categories={"yin": ["houses_dvlad_k40"]},
        yin_shuffle={"houses_dvlad_k40"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k36": dict(
        out_dir="runs/sim_seeds_hladder/k36",
        sims=["yin"], categories={"yin": ["houses_dvlad_k36"]},
        calib_categories={"yin": ["houses_dvlad_k36"]},
        yin_shuffle={"houses_dvlad_k36"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k32": dict(
        out_dir="runs/sim_seeds_hladder/k32",
        sims=["yin"], categories={"yin": ["houses_dvlad_k32"]},
        calib_categories={"yin": ["houses_dvlad_k32"]},
        yin_shuffle={"houses_dvlad_k32"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k28": dict(
        out_dir="runs/sim_seeds_hladder/k28",
        sims=["yin"], categories={"yin": ["houses_dvlad_k28"]},
        calib_categories={"yin": ["houses_dvlad_k28"]},
        yin_shuffle={"houses_dvlad_k28"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k24": dict(
        out_dir="runs/sim_seeds_hladder/k24",
        sims=["yin"], categories={"yin": ["houses_dvlad_k24"]},
        calib_categories={"yin": ["houses_dvlad_k24"]},
        yin_shuffle={"houses_dvlad_k24"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k20": dict(
        out_dir="runs/sim_seeds_hladder/k20",
        sims=["yin"], categories={"yin": ["houses_dvlad_k20"]},
        calib_categories={"yin": ["houses_dvlad_k20"]},
        yin_shuffle={"houses_dvlad_k20"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k16": dict(
        out_dir="runs/sim_seeds_hladder/k16",
        sims=["yin"], categories={"yin": ["houses_dvlad_k16"]},
        calib_categories={"yin": ["houses_dvlad_k16"]},
        yin_shuffle={"houses_dvlad_k16"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k12": dict(
        out_dir="runs/sim_seeds_hladder/k12",
        sims=["yin"], categories={"yin": ["houses_dvlad_k12"]},
        calib_categories={"yin": ["houses_dvlad_k12"]},
        yin_shuffle={"houses_dvlad_k12"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k8": dict(
        out_dir="runs/sim_seeds_hladder/k8",
        sims=["yin"], categories={"yin": ["houses_dvlad_k8"]},
        calib_categories={"yin": ["houses_dvlad_k8"]},
        yin_shuffle={"houses_dvlad_k8"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k4": dict(
        out_dir="runs/sim_seeds_hladder/k4",
        sims=["yin"], categories={"yin": ["houses_dvlad_k4"]},
        calib_categories={"yin": ["houses_dvlad_k4"]},
        yin_shuffle={"houses_dvlad_k4"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "hlad_k0": dict(
        out_dir="runs/sim_seeds_hladder/k0",
        sims=["yin"], categories={"yin": ["houses_dvlad_k0"]},
        calib_categories={"yin": ["houses_dvlad_k0"]},
        yin_shuffle={"houses_dvlad_k0"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Object familiarity ladder for r8, one preset per rung. 64 ImageNet classes
    # per store, 8 trained classes swapped for 8 never-trained ones per step.
    # The unknown half is the 64 classes pruned from the packed store on
    # 2026-07-16 and preserved in objects_all128_backup -- the only held-out
    # object set that exists. house_control_r1 trained on all 128 and cannot run
    # this. Shuffled, for the same reason the house ladder is: known classes are
    # packed before unknown ones and would otherwise fill the study pool.
    "olad_k64": dict(
        out_dir="runs/sim_seeds_oladder/k64",
        sims=["yin"], categories={"yin": ["objects_r8lad_k64"]},
        calib_categories={"yin": ["objects_r8lad_k64"]},
        yin_shuffle={"objects_r8lad_k64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "olad_k56": dict(
        out_dir="runs/sim_seeds_oladder/k56",
        sims=["yin"], categories={"yin": ["objects_r8lad_k56"]},
        calib_categories={"yin": ["objects_r8lad_k56"]},
        yin_shuffle={"objects_r8lad_k56"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "olad_k48": dict(
        out_dir="runs/sim_seeds_oladder/k48",
        sims=["yin"], categories={"yin": ["objects_r8lad_k48"]},
        calib_categories={"yin": ["objects_r8lad_k48"]},
        yin_shuffle={"objects_r8lad_k48"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "olad_k40": dict(
        out_dir="runs/sim_seeds_oladder/k40",
        sims=["yin"], categories={"yin": ["objects_r8lad_k40"]},
        calib_categories={"yin": ["objects_r8lad_k40"]},
        yin_shuffle={"objects_r8lad_k40"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "olad_k32": dict(
        out_dir="runs/sim_seeds_oladder/k32",
        sims=["yin"], categories={"yin": ["objects_r8lad_k32"]},
        calib_categories={"yin": ["objects_r8lad_k32"]},
        yin_shuffle={"objects_r8lad_k32"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "olad_k24": dict(
        out_dir="runs/sim_seeds_oladder/k24",
        sims=["yin"], categories={"yin": ["objects_r8lad_k24"]},
        calib_categories={"yin": ["objects_r8lad_k24"]},
        yin_shuffle={"objects_r8lad_k24"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "olad_k16": dict(
        out_dir="runs/sim_seeds_oladder/k16",
        sims=["yin"], categories={"yin": ["objects_r8lad_k16"]},
        calib_categories={"yin": ["objects_r8lad_k16"]},
        yin_shuffle={"objects_r8lad_k16"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "olad_k8": dict(
        out_dir="runs/sim_seeds_oladder/k8",
        sims=["yin"], categories={"yin": ["objects_r8lad_k8"]},
        calib_categories={"yin": ["objects_r8lad_k8"]},
        yin_shuffle={"objects_r8lad_k8"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "olad_k0": dict(
        out_dir="runs/sim_seeds_oladder/k0",
        sims=["yin"], categories={"yin": ["objects_r8lad_k0"]},
        calib_categories={"yin": ["objects_r8lad_k0"]},
        yin_shuffle={"objects_r8lad_k0"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Yin on the 93 white-male Chicago Face Database identities, one neutral
    # photo each. CFD gives only 29 of the 93 targets a full expression set, but
    # the Yin design needs exactly one representative image per item anyway
    # (build_items takes the first photo of each class), so the single neutral
    # shot loses nothing here. Novel to every model -- CFD identities appear in
    # no label_map, hence no EXCLUDE_SEEN entry. Shuffled because the store packs
    # in CFD id order (WM-001 ...): without it every seed would study the same 40
    # items and probe the same 24 pairs, leaving only sampling noise to vary.
    # 93 items comfortably fills the 40 study / 24 old / 24 new design.
    # The same store cut to 64 identities -- the pool size the mix64 / Set_E
    # anchors were established at -- so the CFD numbers can be read against
    # those rather than only against each other. Built by SLICING the packed
    # 93-identity store, not by re-running preprocess_fixations: the Gabor
    # fixation sampler is stochastic and unseeded, so a re-pack of the same
    # photos yields a different coord sequence (98.6% of coords differed when
    # checked) and would change pool size and fixation draw at once.
    # The cfdWM64 faces run plus its object and house category controls, all
    # read at the one p fitted on faces_cfdWM64 -- the mix64 design. Objects are
    # scored on all 64 packed classes, every one of them trained-on, which is
    # the existing convention (objects is not in EXCLUDE_SEEN). Houses are held
    # out for the r8 family (40 trained of 201) but NOT for the r7 family, which
    # trained on all 201, so r7* is waived below and its house numbers are on
    # TRAINED-ON buildings -- a different condition from r8's, not comparable
    # across the two families.
    # Set_A read on the ONE photo per identity that the fine-tunes never saw.
    # The packed faces_setA store cannot be used for these models: the Yin sim
    # takes the first packed image of each class, and all 40 of those are in the
    # fine-tune training set, so it would score memorised photos. faces_setA_ho
    # is the held-out 5th photo of each identity, sliced out of faces_new/valid.
    # NOTE this is a FAMILIARITY contrast, not a like-for-like one: the 40
    # identities are novel to r7/r8 but TRAINED-ON (4 photos each) for the
    # fine-tunes, so a difference confounds better representation with identity
    # familiarity. 40 items scales Yin's 40/24 design down to 25 study / 15 test.
    "setA_ho": dict(
        out_dir="runs/sim_seeds_setA_ho",
        sims=["yin"],
        categories={"yin": ["faces_setA_ho"]},
        calib_categories={"yin": ["faces_setA_ho"]},
        yin_shuffle={"faces_setA_ho"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # r13 / r14 (from-scratch, CFD faces in training, 40 individuated buildings).
    # Houses are scored on houses_yin64 -- house0041..house0104, the 64 buildings
    # immediately after the trained 40 -- so faces and houses both get a 64-item
    # held-out store and the two categories are matched in pool size for the
    # first time. No EXCLUDE_SEEN entry: the store is disjoint by construction.
    # Plain Dobs/Kanwisher matching on Set_A: 40 identities x 5 photos, no
    # sub-population split (kanw_units empty -> one run on the whole store), p
    # fitted to the 87.5% human upright anchor. Valid ONLY for models that never
    # trained on Set_A -- the baselines r7/r8/r11 and the from-scratch r13/r14.
    # The fine-tunes (r7ft*, r8ft*, r11ft*) trained on 4 of the 5 photos of every
    # Set_A identity, so this preset must never be pointed at them.
    # Dobs/Kanwisher matching on the HELD-OUT CFD males: the 36 of the 64 that
    # have >= 2 photos (29 with five, 5 with four, 2 with three; the other 28 are
    # neutral-only and cannot support a matching task). Never trained on by any
    # model. Note the within-identity variation here is EXPRESSION ONLY -- same
    # session, lighting, pose and background -- so this is an easier match than
    # Dobs' natural photo variation, and accuracy is not directly comparable to
    # the 87.5% human anchor the fit targets.
    "kanw_cfdWM": dict(
        out_dir="runs/sim_seeds_kanw_cfdWM",
        sims=["kanwisher"],
        categories={"kanwisher": ["faces_cfdWMk"]},
        calib_categories={"kanwisher": ["faces_cfdWMk"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "kanw_setA": dict(
        out_dir="runs/sim_seeds_kanw_setA",
        sims=["kanwisher"],
        categories={"kanwisher": ["faces_setA"]},
        calib_categories={"kanwisher": ["faces_setA"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "cfd_h40_ctrl": dict(
        out_dir="runs/sim_seeds_cfd_h40_ctrl",
        sims=["yin"],
        categories={"yin": ["faces_cfdWM64", "objects", "houses_yin64"]},
        calib_categories={"yin": ["faces_cfdWM64"]},
        yin_shuffle={"faces_cfdWM64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # ---- 2026-08-31, for r15_vgg. faces_vggHO is 60 VGGFace2 identities held
    # out of training (the mirror's val/ folder, disjoint from its train/ folder
    # by construction and verified so). build_items takes one photo per class, so
    # 60 items scale Yin's 40/24 design to study=37 / test=23 -- 4.35 points per
    # pair, close enough to the usual 24-pair resolution to read the same way.
    # This is the first held-out face store with NATURAL photo variation that no
    # model has trained on: faces_setA is burnt for the fine-tunes and
    # faces_cfdWM64 varies only by expression.
    "vggHO": dict(
        out_dir="runs/sim_seeds_vggHO",
        sims=["yin"],
        categories={"yin": ["faces_vggHO"]},
        calib_categories={"yin": ["faces_vggHO"]},
        yin_shuffle={"faces_vggHO"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Three-category Yin for r15: houses on houses_yin64 because r15 individuates
    # only 40 buildings, same as r13/r14, so this is directly comparable to
    # cfd_h40_ctrl with the face store swapped.
    "vggHO_ctrl": dict(
        out_dir="runs/sim_seeds_vggHO_ctrl",
        sims=["yin"],
        categories={"yin": ["faces_vggHO", "objects", "houses_yin64"]},
        calib_categories={"yin": ["faces_vggHO"]},
        yin_shuffle={"faces_vggHO"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # The test r15 was built for: Dobs/Kanwisher matching under natural photo
    # variation on identities never trained on. Six photos per identity, so the
    # matching task has real within-identity variation to bridge -- unlike
    # kanw_cfdWM, where the two photos differ only in expression and every model
    # already hits the human anchor.
    "kanw_vggHO": dict(
        out_dir="runs/sim_seeds_kanw_vggHO",
        sims=["kanwisher"],
        categories={"kanwisher": ["faces_vggHO"]},
        calib_categories={"kanwisher": ["faces_vggHO"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # ---- 2026-09-01, RFW (Racial Faces in-the-Wild). 500 identities held out of
    # the faces_rfw training store, 125 per demographic subset, 4 photos each.
    # For r15 this is the cleanest face test available: never trained on, and
    # unlike faces_vggHO it does not share a source pipeline with training
    # (RFW crops MS-Celeb-1M, VGGFace2 crops image search), so a result here is
    # not confounded with domain familiarity the way the vggHO rows are.
    # Demographically balanced, so the per-class CSV also reads out as an
    # other-race contrast. 500 items scale Yin's 40/24 design well past its
    # usual pool -- read the four cells, not the item count.
    "rfwHO": dict(
        out_dir="runs/sim_seeds_rfwHO",
        sims=["yin"],
        categories={"yin": ["faces_rfwHO"]},
        calib_categories={"yin": ["faces_rfwHO"]},
        yin_shuffle={"faces_rfwHO"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Four photos per identity, so matching has real within-identity variation
    # to bridge -- the same test kanw_vggHO runs, on a store that is cross-domain
    # rather than same-source.
    "kanw_rfwHO": dict(
        out_dir="runs/sim_seeds_kanw_rfwHO",
        sims=["kanwisher"],
        categories={"kanwisher": ["faces_rfwHO"]},
        calib_categories={"kanwisher": ["faces_rfwHO"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "cfdWM64_ctrl": dict(
        out_dir="runs/sim_seeds_cfdWM64_ctrl",
        sims=["yin"],
        categories={"yin": ["faces_cfdWM64", "objects", "houses_zubud"]},
        calib_categories={"yin": ["faces_cfdWM64"]},
        yin_shuffle={"faces_cfdWM64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # ---- 2026-09-02: the matched white-male Yin battery. faces_cfdWM64 (studio
    # -controlled, one neutral shot per identity) against faces_rfwWM64 (the same
    # 64 count, race and sex, but in-the-wild photos), with houses_yin64 and
    # objects as the non-face controls. Every store yields exactly 64 items, so
    # all four run the standard 40 study / 24 test design with no scaling.
    #
    # Calibration is fit JOINTLY on the two face stores so both -- and the two
    # control stores that inherit p -- are scored at ONE noise level. That is the
    # whole point: vggHO_ctrl and cfd_h40_ctrl calibrated separately (p=0.27 vs
    # p=0.00) and their shared houses_yin64 store came out +11.4 vs +18.9, so
    # cross-experiment comparison of the house arm was meaningless.
    #
    # faces_rfwWM64 is built by scripts/make_rfw_wm64.py from the Caucasian half
    # of faces_rfwHO. RFW ships no sex labels, so its 64 males were read off
    # contact sheets by eye -- visual judgements, not ground-truth metadata.
    "wm64": dict(
        out_dir="runs/sim_seeds_wm64",
        sims=["yin"],
        categories={"yin": ["faces_cfdWM64", "faces_rfwWM64", "houses_yin64", "objects"]},
        calib_categories={"yin": ["faces_cfdWM64", "faces_rfwWM64"]},
        yin_shuffle={"faces_cfdWM64", "faces_rfwWM64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # ---- 2026-09-08: wm64 with p fitted on faces_rfwWM64 ALONE.
    # `wm64` is the only experiment here that fits p on two stores jointly, and
    # on the r17 final weights that broke: faces_cfdWM64 reads 83.85% upright at
    # p=0 against the 96.29 human anchor, so it can never reach its target, and
    # the summed-error fit is dragged to the p=0 grid floor no matter what
    # faces_rfwWM64 does (r17_base fitted p=0.0, total err 0.136 -- 0.124 of it
    # cfd's irreducible residual). A railed fit is not a calibration.
    # faces_rfwWM64 brackets the anchor (97.40% at p=0, 95.80% at p=0.1), so it
    # can be fitted honestly; cfd is still run and still reported, it just gets
    # no vote. This is what every other experiment in this dict already does.
    # Separate out-dir, so the joint-fit wm64 rows are left untouched.
    "wm64_rfwcal": dict(
        out_dir="runs/sim_seeds_wm64_rfwcal",
        sims=["yin"],
        categories={"yin": ["faces_cfdWM64", "faces_rfwWM64", "houses_yin64", "objects"]},
        calib_categories={"yin": ["faces_rfwWM64"]},
        yin_shuffle={"faces_cfdWM64", "faces_rfwWM64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # ---- 2026-09-03: houses_yin64 replacement. All three r16 models score
    # houses_yin64 at 33.9-43.1% upright -- at or below the 50% 2AFC chance floor
    # -- against r15_vgg's 90.6% on the identical store, because r16 trains
    # `houses_gen` with 97 leftover ZuBuD buildings under one generic `house`
    # label, which collapses ZuBuD representations. houses_ho64 is 64
    # Houses-dataset buildings verified disjoint from BOTH houses/train (r15's
    # generic exposure) and houses_gen/train (r16's), so no model has seen them
    # even generically, and it is a different dataset from ZuBuD entirely.
    #
    # Run with --fixed-noise yin=<the model's wm64 p> so these rows slot into the
    # wm64 tables without recalibration. Unlike wm64, the house store IS
    # shuffled here: a single-sim diagnostic showed the fixed 40/24 split adds a
    # systematic below-chance bias (UU 33.92% unshuffled vs 50.00% shuffled).
    "ho64": dict(
        out_dir="runs/sim_seeds_ho64",
        sims=["yin"],
        categories={"yin": ["houses_ho64"]},
        calib_categories={"yin": ["houses_ho64"]},
        yin_shuffle={"houses_ho64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "cfdWM64": dict(
        out_dir="runs/sim_seeds_cfdWM64",
        sims=["yin"],
        categories={"yin": ["faces_cfdWM64"]},
        calib_categories={"yin": ["faces_cfdWM64"]},
        yin_shuffle={"faces_cfdWM64"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    "cfdWM": dict(
        out_dir="runs/sim_seeds_cfdWM",
        sims=["yin"],
        categories={"yin": ["faces_cfdWM"]},
        calib_categories={"yin": ["faces_cfdWM"]},
        yin_shuffle={"faces_cfdWM"},
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
    # Kanwisher on the white-female half of CFD. Unlike faces_cfdWM -- one
    # neutral shot per identity, so Yin-only -- this store keeps every identity
    # that has >= 2 photos: 37 of the 90 WF identities, 35 with all
    # five CFD expressions (N/HC/HO/A/F) and 2 with four. 183 images in 'valid'
    # alone, so no --splits borrowing is needed.
    #
    # Read this as a HARDER same-identity match than faces_mixed64 or Set_E: the
    # target and its match differ by EXPRESSION, not by photo session, so the
    # task demands identity generalisation across a large deformation. The 87.5%
    # anchor was not established on such a set; treat a shortfall as a property
    # of the stimulus set, the way Set_A's was, until the upright number says
    # otherwise.
    #
    # 37 identities is close to the canonical 40, and pool size drives the
    # distractor count -- do not compare its absolute accuracy to a 64-identity
    # store without noting that. Novel to every model (CFD is in no label_map),
    # hence no EXCLUDE_SEEN entry.
    "cfdWF": dict(
        out_dir="runs/sim_seeds_cfdWF",
        sims=["kanwisher"],
        categories={"kanwisher": ["faces_cfdWF"]},
        calib_categories={"kanwisher": ["faces_cfdWF"]},
        yin_shuffle=set(),
        kanw_units={}, kanw_calib_label=None, label_col="n_identities",
    ),
}
# Set by main() from --experiment, and read by the helpers below.
EXP = EXPERIMENTS["mix64"]


def cats_for(sim):
    return EXP["categories"].get(sim, DEFAULT_CATEGORIES)
# Yin (1969) human upright-upright accuracy, per category (Tables 1 & 2)
YIN_TARGET = {"faces": 0.9629, "objects": 0.8479, "houses": 0.9071}


def target_for(sim, category):
    """Human upright accuracy for a store, by category family.

    Keyed on the store's prefix rather than its full name so the growth-sweep
    stores (faces_growA4, faces_growE8, ...) need no entry of their own -- every
    faces_* store is scored against the same human anchor.
    """
    if sim == "kanwisher":
        return 0.875
    for prefix, t in YIN_TARGET.items():
        if category.startswith(prefix):
            return t
    raise SystemExit(f"no human target for category {category!r}")
# Dobs et al. (2023) human upright matching accuracy. One value for every
# category, matching the --calib-target default the sim already used.

# Which categories the noise level is FITTED on. Everything else is still run and
# still has its accuracy at the chosen p recorded -- it just gets no vote in
# picking p. BOTH sims are fitted on faces alone. Kanwisher: objects and houses
# sat near the human target across a wide range of p and dragged faces 25+
# points below its anchor. Yin: objects cannot reach their 84.79 human target
# at any usable p (they read 95.8-100% for every p up to 0.35 and only fall to
# ~85% at p>=0.40, where faces have collapsed to 62%), so the objects term is a
# large constant error whose +-1-pair jitter outvoted the faces term -- that is
# why r8 landed on p=0.00.
# (the per-experiment `calib_categories` field above)
# The Kanwisher triplet task needs two photos of the same identity. ZuBuD has
# only one house photo per class in valid, so it borrows test as well (the same
# splits the earlier runs/kanwisher logs used). Set_A has 5 photos per identity
# in 'valid' alone.
KANW_SPLITS = {"houses_zubud": ["valid", "test"], "houses": ["valid", "test"]}
KANW_IMAGES_PER_ID = {"houses": 2}

# ---- held-out stimuli -------------------------------------------------------
# Houses: EVERY model trained on some subset of the 201 packed ZuBuD classes, so
# each is tested only on the classes its own label_map does not list.
# house_control_r1 included -- its houses are a 40-building ZuBuD subset, not a
# separate dataset (train stems 'object0101.view02'), so 161 are held out for it
# exactly as for r8/r9. It was scored on all 201 until 2026-08-21 because the
# filter matched raw class names and its label_map spells them differently; see
# LABELMAP_CATEGORY below and canon_class() in simulate_yin1969.py.
EXCLUDE_SEEN = {"houses_zubud"}
# ...except where a model has no held-out classes left. r5b_allatonce trained on
# ALL 201 packed ZuBuD classes, so the held-out rule leaves it zero items and the
# condition scores a structural 0.0%. It is run on its own TRAINED-ON houses
# instead, which is the only way to get a number out of it -- that number is NOT
# comparable to r8/house_control_r1, whose house scores are held-out, and must be
# labelled as trained-on wherever it is reported. r8_developmental saw 40 of 201
# and house_control_r1 saw none, so both keep the held-out rule.
# The label_map prefix a model files its houses under, where it is not the packed
# category name. house_control_r1 trained on a 40-building SUBSET OF ZUBUD but
# recorded it as category 'houses', so without this the held-out filter matched
# nothing and scored it on all 201 buildings -- 40 of them trained-on.
LABELMAP_CATEGORY = {("house_control_r1", "houses_zubud"): "houses"}
EXCLUDE_SEEN_WAIVED = {"r7_curriculum": {"houses_zubud"},
                       "r7ftA": {"houses_zubud"},
                       "r7ftB": {"houses_zubud"},
                       "r5b_allatonce": {"houses_zubud"},
                       # r7 trained on ALL 201 packed ZuBuD classes too, so the
                       # held-out rule leaves it zero items and the condition
                       # scores a structural 0.0% rather than erroring. Its ZuBuD
                       # numbers are TRAINED-ON and must be labelled as such;
                       # houses_common is the store where r7 has a genuine
                       # held-out house test.
                       "r7_curriculum": {"houses_zubud"}}
# Cross-store distractors: EMPTY on purpose. Yin now draws study and distractor
# items from a single store (see SIM_CATEGORIES). The mechanism is kept because
# the sim still supports it, but using it re-creates the dataset-discrimination
# artefact described above.
YIN_UNKNOWN_FROM = {}
# Objects are deliberately unchanged from the earlier sweeps: the packed objects
# store holds 64 classes and every model saw all of them, so no held-out object
# set exists to switch to.

# Retrieval noise is a BIT-FLIP probability on the binary code, so p and 1-p are
# mirror images: past 0.5 the code is systematically inverted and discrimination
# comes back as an artefact (measured: faces 54% at p=0.50 climbing back to 96%
# at p=0.65). Only p <= 0.5 is meaningful. The old calibration never hit this
# because it stopped at the first p crossing the target; a full grid search must
# exclude the mirror half explicitly or it will happily "fit" the human data there.
MAX_NOISE = 0.5

# Seeds averaged per calibration grid point (Yin). One seed scores 24 pairs, so
# the curve moves in 4.17-point steps and is decided by coin flips.
CALIB_SEEDS = 8
# Base seed for the calibration probe. Set to the FIRST EVALUATION SEED by
# --calib-on-eval-seeds, so p is fitted on exactly the item samples that will be
# reported. With the default 42 the fit and the reported seeds draw different
# items, and on a small store that alone moves upright accuracy by a point or
# more: celeb24 fitted p=0.33 at a measured 95.8%, then read 97.2% over seeds
# 101-120. Fitting in-sample removes that gap.
CALIB_BASE_SEED = 42
# Overrides the default coarse grid when --calib-range is given.
CALIB_RANGE = None
# Which network representation the Yin memory model reads. "h" is the 256-bit
# bottleneck code every published run used; anything else probes an earlier
# layer (see simulate_yin1969.--layer) and needs its OWN out-dir, since a layer
# is a different measurement of the same model, not a different model.
LAYER = "h"

RE_NOISE = re.compile(r"Using noise p=([\d.]+)")
RE_GRID = re.compile(r"noise ([\d.]+) -> ([\d.]+)%")
RE_YIN = re.compile(r"^\s*(Upright|Inverted)\s+(Upright|Inverted)\s+([\d.]+)%", re.M)
RE_KANW = re.compile(r"^\s*(Upright|Inverted)\s+([\d.]+)%\s+(\d+)\s*$", re.M)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seeds", default="101-150",
                    help="inclusive range 'a-b' or a comma list")
    ap.add_argument("--experiment", default="mix64", choices=sorted(EXPERIMENTS),
                    help="which sweep to run (see EXPERIMENTS); sets the sims, "
                         "the stores, the sweep points and the out-dir")
    # Each experiment writes its own directory. Never point two at one: the
    # 9-column schema is shared but the label column means something different
    # per experiment, and runs/sim_seeds (the oldest Set_E sweeps) is an 8-column
    # schema that appending to would corrupt.
    ap.add_argument("--out-dir", default=None,
                    help="defaults to the experiment's own out-dir")
    ap.add_argument("--sims", nargs="+", default=None,
                    help="defaults to the experiment's sims")
    ap.add_argument("--calib-label", default=None,
                    help="override which sweep point the noise is fitted at "
                         "(must be one of the experiment's labels); pair it with "
                         "--out-dir so the two fits do not share a noise JSON")
    ap.add_argument("--fixed-noise", nargs="+", default=[], metavar="SIM=P",
                    help="skip calibration and use this noise level, e.g. "
                         "--fixed-noise yin=0.3 kanwisher=0.4. The upright "
                         "accuracy each category reaches at P is still measured "
                         "and recorded, so the residuals stay reportable.")
    ap.add_argument("--calib-seeds", type=int, default=CALIB_SEEDS,
                    help="item samples averaged per Yin calibration grid point. "
                         "One sample scores 9-24 pairs, so the fitted accuracy "
                         "is quantised to 4-11 points / this many seeds; raise "
                         "it when the fit must land within ~1 point of the "
                         "human anchor (celeb24: use 40)")
    ap.add_argument("--calib-base-seed", type=int, default=None,
                    help="base seed for the Yin calibration probe (default 42). "
                         "Set it to the first evaluation seed to fit on the same "
                         "item draws the sweep reports, without being forced to "
                         "use as many samples as --calib-on-eval-seeds requires")
    ap.add_argument("--calib-range", nargs=3, type=float, default=None,
                    metavar=("LO", "HI", "STEP"),
                    help="restrict the COARSE calibration grid to this range "
                         "instead of the default 0.00-0.50 in steps of 0.05. The "
                         "fine pass around the winner is unchanged. Use it when "
                         "the plausible range of p is already known -- a full "
                         "coarse grid is 11 probe points per store, and most of "
                         "them are wasted once the curve's shape is known")
    ap.add_argument("--calib-on-eval-seeds", action="store_true",
                    help="fit p on the evaluation seeds themselves (base seed = "
                         "first --seeds entry) instead of the default base 42, "
                         "so the calibrated accuracy is the one the seed sweep "
                         "will report")
    ap.add_argument("--layer", default="h",
                    choices=["h", "probs", "layer1", "layer2", "layer3", "layer4"],
                    help="representation the Yin memory model reads; non-default "
                         "values REQUIRE --out-dir so the layer sweep does not "
                         "overwrite the bottleneck-code results")
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads per sim subprocess (see run()); keep the "
                         "total across concurrent sweeps well under the core count")
    return ap.parse_args()


def expand_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def run(cmd, gpu, threads=8):
    """Run one sim subprocess pinned to `gpu` and to `threads` CPU threads.

    The thread cap matters far more than it looks: the work is CPU-bound (crop
    assembly, not the GPU forward), and torch otherwise opens one thread per
    core. Several sweeps in parallel on a shared box then oversubscribe the CPU
    and thrash -- measured 324s vs 12s for the *same* single condition with
    three sweeps running. Keep this set whenever sweeps run concurrently.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def group_args(groups):
    """--group-sizes for one point of a Kanwisher sweep."""
    return ["--group-sizes"] + [f"{prefix}={n}" for prefix, n in groups]


def work_units(sim):
    """The (category, extra flags, label) units one seed runs for `sim`.

    Everything is one unit per category, except the Kanwisher sweep categories,
    which contribute one unit per sweep point.
    """
    for cat in cats_for(sim):
        if sim == "kanwisher" and cat in EXP["kanw_units"]:
            for label, groups in EXP["kanw_units"][cat]:
                yield cat, group_args(groups), label
        else:
            yield cat, [], ""


def calib_extra(sim, category):
    """Extra flags the calibration probe must match the seed runs on. A sweep is
    fitted at its kanw_calib_label point; every other unit has nothing to add."""
    if sim == "kanwisher" and category in EXP["kanw_units"]:
        for label, groups in EXP["kanw_units"][category]:
            if label == EXP["kanw_calib_label"]:
                return group_args(groups)
        raise SystemExit(f"kanw_calib_label {EXP['kanw_calib_label']!r} is not a "
                         f"sweep point of {category}")
    return []


def base_cmd(model, category, sim, extra=()):
    """Common invocation for both phases: script, category, checkpoint, data."""
    script = "simulate_yin1969.py" if sim == "yin" else "simulate_kanwisher2023.py"
    cmd = [sys.executable, script, "--category", category,
           "--run-dir", MODELS[model], "--device", "cuda:0"]
    if model in CHECKPOINT:
        cmd += ["--checkpoint", CHECKPOINT[model]]
    if category in EXCLUDE_SEEN and category not in EXCLUDE_SEEN_WAIVED.get(model, ()):
        cmd += ["--exclude-seen-from",
                os.path.join(MODELS[model], "label_map.json")]
        lm_cat = LABELMAP_CATEGORY.get((model, category))
        if lm_cat:
            cmd += ["--exclude-seen-category", lm_cat]
    if sim == "yin":
        if LAYER != "h":
            cmd += ["--layer", LAYER]
        if category in YIN_UNKNOWN_FROM:
            cmd += ["--unknown-category", YIN_UNKNOWN_FROM[category]]
        if category in EXP["yin_shuffle"]:
            cmd += ["--shuffle-items"]
    if sim == "kanwisher":
        if category in KANW_SPLITS:
            cmd += ["--splits", *KANW_SPLITS[category]]
        if category in KANW_IMAGES_PER_ID:
            cmd += ["--images-per-identity", str(KANW_IMAGES_PER_ID[category])]
    return cmd + list(extra)


def probe_grid(model, category, sim, grid, gpu, log_dir, tag, threads):
    """Upright(-upright) accuracy at each p in `grid`, as {p: accuracy}."""
    cmd = (base_cmd(model, category, sim, calib_extra(sim, category))
           + ["--calib-grid"] + [f"{p:.2f}" for p in grid])
    if sim == "yin":
        cmd += ["--calib-seeds", str(CALIB_SEEDS),
                "--seed", str(CALIB_BASE_SEED)]
    rc, out = run(cmd, gpu, threads)
    with open(os.path.join(log_dir, f"calib_{sim}_{model}_{category}_{tag}.log"), "w") as f:
        f.write(out)
    curve = {float(p): float(a) / 100.0 for p, a in RE_GRID.findall(out)}
    if not curve:
        print(f"[warn] empty {tag} grid for {sim}/{model}/{category} (rc={rc})", flush=True)
    return curve


def joint_fit(curves, targets):
    """Pick the single p minimising total |accuracy - human target| summed over
    categories. Only p values measured on *every* category are eligible."""
    shared = set.intersection(*(set(c) for c in curves.values())) if curves else set()
    scored = []
    for p in sorted(x for x in shared if x <= MAX_NOISE):
        err = sum(abs(curves[cat][p] - targets[cat]) for cat in curves)
        scored.append((err, p))
    if not scored:
        return None, None
    err, p = min(scored)
    return p, err


def calibrate_joint(model, categories, sim, gpu, log_dir, threads):
    """One noise level per (model, sim), fitted across all categories at once.

    Coarse sweep over the full range, then a fine sweep around the coarse
    winner -- the same coarse->fine idea the per-category calibration used, but
    the choice is made on the summed deviation from the human targets rather
    than on one category crossing its own target.
    """
    fit_cats = EXP["calib_categories"].get(sim, categories)
    targets = {c: target_for(sim, c) for c in fit_cats}
    # A category fitted on but never run still has to be measured; make sure it
    # is in the residual list too.
    categories = list(dict.fromkeys(list(categories) + list(fit_cats)))

    if CALIB_RANGE:
        lo, hi, step = CALIB_RANGE
        n = int(round((hi - lo) / step))
        coarse = [round(lo + i * step, 2) for i in range(n + 1)]
    else:
        coarse = [round(x * 0.05, 2) for x in range(0, 11)]      # 0.00 .. 0.50
    curves = {c: probe_grid(model, c, sim, coarse, gpu, log_dir, "coarse", threads)
              for c in fit_cats}
    p, err = joint_fit(curves, targets)
    if p is None:
        return None, {}
    print(f"[calib] {model} {sim}: coarse p={p:.2f} (total err {err:.3f})", flush=True)

    fine = [round(x, 2) for x in
            [p - 0.04 + 0.01 * i for i in range(9)] if 0.0 <= x <= MAX_NOISE]
    fine = [x for x in fine if x not in coarse]
    if fine:
        for c in fit_cats:
            curves[c].update(probe_grid(model, c, sim, fine, gpu, log_dir, "fine", threads))
        p, err = joint_fit(curves, targets)
    # Categories excluded from the fit still get their accuracy at the chosen p
    # measured, so every category stays reportable.
    residuals = {}
    for c in categories:
        if c in curves and p in curves[c]:
            residuals[c] = curves[c][p]
        else:
            residuals[c] = probe_grid(model, c, sim, [p], gpu, log_dir,
                                      "atp", threads).get(p)
    return p, residuals


def parse_rows(sim, out):
    """(study, test, accuracy) triples for yin; (presentation, accuracy) for kanwisher."""
    if sim == "yin":
        return [(s, t, float(a)) for s, t, a in RE_YIN.findall(out)]
    return [(p, p, float(a)) for p, a, _ in RE_KANW.findall(out)]


def main():
    global EXP, CALIB_SEEDS, CALIB_BASE_SEED, CALIB_RANGE, LAYER
    args = parse_args()
    CALIB_SEEDS = args.calib_seeds
    LAYER = args.layer
    # A layer probe is a different measurement of the same model, so its rows
    # must not land in the bottleneck-code out-dir under the same filename.
    if LAYER != "h" and not args.out_dir:
        raise SystemExit(f"--layer {LAYER} requires an explicit --out-dir "
                         f"(else it would overwrite the 'h' results).")
    CALIB_RANGE = args.calib_range
    if args.calib_base_seed is not None:
        CALIB_BASE_SEED = args.calib_base_seed
    EXP = EXPERIMENTS[args.experiment]
    if args.calib_label is not None:
        # Copy, so overriding the fit point never mutates the shared preset.
        EXP = {**EXP, "kanw_calib_label": args.calib_label}
    args.out_dir = args.out_dir or EXP["out_dir"]
    args.sims = args.sims or EXP["sims"]
    seeds = expand_seeds(args.seeds)
    if args.calib_on_eval_seeds:
        # Fit on EXACTLY the reported seeds: base = the first one, and as many
        # samples as there are seeds. Using more (or fewer) samples than the
        # sweep reports puts the fit back out of sample, which is the whole
        # problem the flag exists to remove -- measured: 40 samples from seed
        # 101 fitted p=0.28 at 96.4%, and seeds 101-120 then read 98.9%.
        CALIB_BASE_SEED = seeds[0]
        if args.calib_seeds != len(seeds):
            print(f"[calib] --calib-on-eval-seeds: using {len(seeds)} samples "
                  f"(the evaluation seeds), not --calib-seeds {args.calib_seeds}",
                  flush=True)
        CALIB_SEEDS = len(seeds)
    print(f"[exp] {args.experiment}: sims {' '.join(args.sims)} -> {args.out_dir}",
          flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ------- phase 1: ONE noise level per (model, sim), fitted jointly --------
    noise_path = os.path.join(args.out_dir, f"noise_{args.model}.json")
    noise = json.load(open(noise_path)) if os.path.isfile(noise_path) else {}
    fixed = dict(kv.split("=", 1) for kv in args.fixed_noise)
    for sim in args.sims:
        if sim in noise:
            continue
        t0 = time.time()
        if sim in fixed:
            # p is imposed rather than fitted, but still measure where each
            # category actually lands so the residuals stay reportable.
            p = float(fixed[sim])
            residuals = {c: probe_grid(args.model, c, sim, [p], args.gpu,
                                       log_dir, "fixed", args.threads).get(p)
                         for c in cats_for(sim)}
        else:
            p, residuals = calibrate_joint(args.model, cats_for(sim), sim, args.gpu,
                                           log_dir, args.threads)
        noise[sim] = p
        noise[f"{sim}_upright_at_p"] = residuals
        res = "  ".join(f"{c} {a*100:.1f}%" if a is not None else f"{c} n/a"
                        for c, a in residuals.items())
        print(f"[calib] {args.model} {sim} -> p={p} ({time.time()-t0:.0f}s) | "
              f"upright: {res}", flush=True)
        with open(noise_path, "w") as f:
            json.dump(noise, f, indent=2)

    # ---------------- phase 2: seeds at fixed noise ---------------------------
    csv_path = os.path.join(args.out_dir, f"results_{args.model}.csv")
    fresh = not os.path.isfile(csv_path)
    fh = open(csv_path, "a", newline="")
    w = csv.writer(fh)
    if fresh:
        # The label column names the Kanwisher sweep point -- the identity count
        # for the count sweep, the celeb/Set_A composition for the ratio sweep --
        # and is empty for every unit that is not a sweep point.
        w.writerow(["model", "sim", "category", EXP["label_col"], "seed", "noise",
                    "study", "test", "accuracy_pct"])
        fh.flush()

    units = [(sim, cat, extra, label)
             for sim in args.sims for cat, extra, label in work_units(sim)]
    total = len(units) * len(seeds)
    done = 0
    t_start = time.time()
    for seed in seeds:
        for sim, cat, extra, label in units:
            p = noise.get(sim)
            done += 1
            if p is None:
                print(f"[skip] {sim}: no calibrated noise", flush=True)
                continue
            cmd = base_cmd(args.model, cat, sim, extra) + [
                "--noise", str(p), "--seed", str(seed)]
            rc, out = run(cmd, args.gpu, args.threads)
            rows = parse_rows(sim, out)
            tag = f"{cat}_{label}" if label != "" else cat
            if rc != 0 or not rows:
                print(f"[warn] {sim}/{tag}/seed{seed} rc={rc}, {len(rows)} rows",
                      flush=True)
                with open(os.path.join(
                        log_dir, f"fail_{sim}_{args.model}_{tag}_{seed}.log"), "w") as f:
                    f.write(out)
                continue
            for study, test, acc in rows:
                w.writerow([args.model, sim, cat, label, seed, p, study, test, acc])
            fh.flush()
            el = time.time() - t_start
            print(f"[{done}/{total}] {sim} {tag} seed {seed} "
                  f"({el/60:.1f}m elapsed, eta {el/done*(total-done)/60:.0f}m)",
                  flush=True)
    fh.close()
    print(f"DONE {args.model} in {(time.time()-t_start)/60:.1f}m -> {csv_path}")


if __name__ == "__main__":
    main()
