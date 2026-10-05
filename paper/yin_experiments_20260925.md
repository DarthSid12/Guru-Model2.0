# Yin experiment results — 25 September 2026

Checked **2026-09-25 09:15 PDT**. This is a saved snapshot of results available at that time.

Scope: the new two-calibration runs for four models, all three saved r20cyl variants, and the earlier Model 2 runs discussed in this session. Final-checkpoint runs for the three recently training models are still pending.

## Models

| Name used below | Run ID | One-line description | Checkpoint in new/comparison results |
| --- | --- | --- | --- |
| Model 2 | `r21dev_aa_s42` | Log-polar antialiased ResNet18 trained with a ten-stage developmental curriculum of VGGFace2/CelebA/RFW faces, objects, and houses. | Best epoch 117 |
| r20cylblur | `r20_cylblur_s42` | Log-polar ResNet18 with circular angular padding and BlurPool, trained with the six-stage r20 mixed face/object/house curriculum. | Snapshot epoch 69 |
| Developmental VGG | `r21dev_vgg16_bn_aa_s42` | Log-polar antialiased VGG16-BN trained with the same ten-stage mixed-source developmental curriculum as Model 2. | Snapshot epoch 111 |
| VGG2k | `r21vgg2k_vgg16_bn_aa_s42` | Log-polar antialiased VGG16-BN using the developmental schedule with faces exclusively from 2,048 VGGFace2 identities, plus objects and houses. | Snapshot epoch 105 |
| r20cyl | `r20_cyl_s42` | Log-polar ResNet18 with circular angular padding and no BlurPool, trained with the six-stage r20 mixed face/object/house curriculum. | Final epoch 80 (older runs) |

All five use training seed 42, 16 training fixations, and a 256-bit representation. **r20cyl and r20cylblur are different models.** “Snapshot” means the checkpoint captured when this queue was prepared; it is not the model’s eventual final checkpoint.

## Exact developmental diets and epochs

### Model 2 and Developmental VGG: 10 stages, 124 epochs

These two models use the same diet. Each number below is the **cumulative number of active identities/classes at that stage**, not the number newly added. `VGG` means `faces_vgg`; `CelebA` means `faces`; `RFW-W` and `RFW-O` are the white and other-race RFW training stores. `ZuBuD` is `houses_zubud137_41`. The generic `houses` category is one class beginning at stage 5; its image cap grows from 4 to 128 training photos.

| Stage | Global epochs | Epochs in stage | VGG | CelebA | RFW-W | RFW-O | Total faces | Objects | ZuBuD | Generic house photos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1–2 | 2 | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| 2 | 3–4 | 2 | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| 3 | 5–8 | 4 | 16 | 0 | 0 | 0 | 16 | 4 | 0 | 0 |
| 4 | 9–12 | 4 | 32 | 0 | 0 | 0 | 32 | 8 | 4 | 0 |
| 5 | 13–20 | 8 | 48 | 16 | 0 | 0 | 64 | 16 | 8 | 4 |
| 6 | 21–28 | 8 | 96 | 32 | 0 | 0 | 128 | 32 | 16 | 8 |
| 7 | 29–44 | 16 | 160 | 64 | 32 | 0 | 256 | 64 | 32 | 16 |
| 8 | 45–60 | 16 | 288 | 96 | 128 | 0 | 512 | 64 | 64 | 32 |
| 9 | 61–92 | 32 | 400 | 128 | 448 | 48 | 1,024 | 64 | 137 | 64 |
| 10 | 93–124 | 32 | 480 | 128 | 1,152 | 288 | 2,048 | 64 | 137 | 128 |

The first eight VGG identities are pinned. RFW-W enters at stage 7 and RFW-O at stage 9. Identities remain available in later stages. Each epoch draws exactly **2,000 batches of 256** training examples, so the full schedule specifies **248,000 optimizer steps**. Category sampling follows the square-root rule in the training script: the four face stores form one domain, and the domain weights change as the active class counts grow. There are no fixed per-category percentages. A global cosine learning-rate schedule spans all 124 epochs, with 500 warm-up steps after each stage change.

### VGG2k: the same 10 stages and 124 epochs

The face domain instead contains only `faces_vgg2k` identities. Its face counts are exactly the `Total faces` column above: **4, 8, 16, 32, 64, 128, 256, 512, 1,024, 2,048**. The object, ZuBuD, and generic-house columns, image caps, stage lengths, epoch ranges, 2,000 steps per epoch, square-root sampling, and 500-step stage warm-ups are the same. The VGG2k identity order is pinned in [`faces_vgg2k_order.txt`](../data/vggface2_raw/faces_vgg2k_order.txt); the first eight identities match the pins used in the mixed-source developmental diet.

### r20cyl and r20cylblur: 6 stages, 80 epochs

These two models share the older r20 diet. Counts are cumulative active classes. `ZuBuD` here is `houses_zubud137`, and the generic `houses` category is one class throughout.

| Stage | Global epochs | Epochs in stage | CelebA | VGG | RFW-W | RFW-O | Total faces | Objects | ZuBuD | Generic houses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1–6 | 6 | 4 | 16 | 50 | 0 | 70 | 4 | 0 | 1 |
| 2 | 7–12 | 6 | 8 | 32 | 150 | 0 | 190 | 16 | 0 | 1 |
| 3 | 13–20 | 8 | 16 | 64 | 400 | 0 | 480 | 32 | 24 | 1 |
| 4 | 21–28 | 8 | 32 | 160 | 700 | 75 | 967 | 64 | 48 | 1 |
| 5 | 29–40 | 12 | 64 | 320 | 1,000 | 180 | 1,564 | 64 | 88 | 1 |
| 6 | 41–80 | 40 | 128 | 480 | 1,200 | 300 | 2,108 | 64 | 137 | 1 |

The fixed final-stage sampling shares are CelebA 14%, VGG 26%, RFW-W 16%, RFW-O 4%, objects 19%, ZuBuD 16%, and generic houses 5%; absent categories are renormalized in earlier stages. r20 uses a global 80-epoch cosine schedule with a 200-step warm-up at each stage change. Its epochs are based on the active training pool rather than a fixed 2,000-step budget.

**Checkpoint position:** the new Yin runs tested Model 2 best epoch 117 (inside stage 10), Developmental VGG snapshot epoch 111 (stage 10), VGG2k snapshot epoch 105 (stage 10), and r20cylblur snapshot epoch 69 (stage 6). The older r20cyl runs tested final epoch 80. The saved `config.json` files for the 10-stage runs show the parser default `epochs: 50`; [`train.py`](../training/train.py) replaces that value with the sum of `curriculum_epochs`, which is **124**.

Sources: [`train_r21_dev.sh`](../training/train_r21_dev.sh), [`train_r21_vgg2k.sh`](../training/train_r21_vgg2k.sh), [`train_r20.sh`](../scripts/train_r20.sh), and the saved run configs linked by the [queue manifest](../runs/yin_orientation_20260925/manifest.json).

## Reading the tables

Values are **mean accuracy (%) ± sample standard deviation across simulation seeds**. Completed runs use seeds 101–150, with 50 seeds per condition. These are repeated simulations of one trained checkpoint, not 50 independently trained models. Older logs/CSVs saved each seed’s accuracy rounded to two decimal places.

| Condition | Study orientation | Test orientation |
| --- | --- | --- |
| UU | Upright | Upright |
| II | Inverted | Inverted |
| UI | Upright | Inverted |
| IU | Inverted | Upright |

The face benchmark uses RFW WM64 and CFD WM64. Human face values below are the repository’s Yin reference; they are the same reference for both image sets, not separate human measurements on RFW and CFD.

## 1. New two-calibration experiment

For each checkpoint and face set, fit **p1 on UU** toward **96.29%** and **p2 on II** toward **81.88%**, averaging each candidate over 50 seeds. The two fits are independent; p2 is allowed to be smaller than p1. The search uses a 0–0.50 coarse grid in 0.05 steps, then a local refinement in 0.01 steps.

Noise is independent XOR bit-flip noise applied to the sampled 256-bit model representation after image encoding. It is applied when encoding study items into memory and when encoding both test alternatives. It is not pixel noise. The noise probability follows the orientation of the image in that phase:

| Condition | Study noise | Test noise |
| --- | --- | --- |
| UU | p1 | p1 |
| II | p2 | p2 |
| UI | p1 | p2 |
| IU | p2 | p1 |

Each seed reshuffles the 40 study / 24 new-item split. The evaluation uses 10 study fixations, 32 test fixations, and KDE width σ = 2. Calibration and reporting use the same seed set, so UU and II are fitted conditions; UI and IU are the cross-orientation predictions.

### RFW WM64

| Model | Epoch | UU | II | UI | IU |
| --- | --- | --- | --- | --- | --- |
| Human reference | — | 96.29 | 81.88 | 84.13 | 78.58 |
| Model 2 | 117 | 95.92 ± 3.90 | 81.33 ± 7.77 | 60.00 ± 10.75 | 60.67 ± 9.89 |
| r20cylblur | 69 | 96.58 ± 3.64 | 81.58 ± 7.20 | 81.83 ± 7.84 | 79.50 ± 7.18 |
| Developmental VGG | 111 | 96.33 ± 3.54 | 80.00 ± 9.03 | 84.58 ± 7.49 | 80.83 ± 7.34 |
| VGG2k | 105 | 96.00 ± 3.67 | 79.00 ± 8.16 | 86.58 ± 7.16 | 81.08 ± 6.59 |

### CFD WM64

| Model | Epoch | UU | II | UI | IU |
| --- | --- | --- | --- | --- | --- |
| Human reference | — | 96.29 | 81.88 | 84.13 | 78.58 |
| Model 2 | 117 | 96.33 ± 3.54 | 82.08 ± 6.69 | 63.83 ± 9.47 | 65.75 ± 8.64 |
| r20cylblur | 69 | 96.25 ± 3.69 | 81.25 ± 8.13 | 86.67 ± 6.30 | 82.08 ± 8.30 |
| Developmental VGG | 111 | 96.17 ± 3.75 | 81.17 ± 9.00 | 86.58 ± 6.37 | 81.58 ± 9.11 |
| VGG2k | 105 | 96.67 ± 4.04 | 80.67 ± 7.42 | 87.92 ± 5.27 | 86.08 ± 6.92 |

### Fitted noise probabilities and remaining calibration error

| Model | Face set | p1 (upright) | p2 (inverted) | UU − human (pp) | II − human (pp) |
| --- | --- | --- | --- | --- | --- |
| Model 2 | RFW WM64 | 0.36 | 0.36 | -0.37 | -0.55 |
| Model 2 | CFD WM64 | 0.20 | 0.02 | +0.04 | +0.20 |
| r20cylblur | RFW WM64 | 0.36 | 0.39 | +0.29 | -0.30 |
| r20cylblur | CFD WM64 | 0.11 | 0.33 | -0.04 | -0.63 |
| Developmental VGG | RFW WM64 | 0.36 | 0.39 | +0.04 | -1.88 |
| Developmental VGG | CFD WM64 | 0.30 | 0.35 | -0.12 | -0.71 |
| VGG2k | RFW WM64 | 0.37 | 0.40 | -0.29 | -2.88 |
| VGG2k | CFD WM64 | 0.34 | 0.38 | +0.38 | -1.21 |

All targets lie within the sampled calibration ranges. The finite grid does not match every target exactly: the largest remaining error is VGG2k RFW II at **−2.88 percentage points**.

**Main observation:** Model 2 reaches roughly the fitted UU/II levels but its UI/IU accuracies remain low (60.00/60.67% on RFW and 63.83/65.75% on CFD). The other three checkpoints show substantially higher cross-orientation accuracy. The calibration fit alone therefore does not establish a human-like cross-orientation pattern.

## 2. r20cyl: all saved variants

All rows below use **r20cyl final epoch 80**, with 50 seeds per condition. These older calibrated runs fit p1/p2 **separately for each seed**, using targets 96%/82% and a constraint p2 ≥ p1 + 0.01. This differs from the new independent fits over the mean of 50 seeds; the results are useful comparisons but do not isolate model architecture under an identical calibration procedure.

| Noise variant | Face set | UU | II | UI | IU |
| --- | --- | --- | --- | --- | --- |
| Both phases | RFW WM64 | 99.92 ± 0.59 | 85.17 ± 4.47 | 78.25 ± 8.72 | 73.00 ± 9.46 |
| Both phases | CFD WM64 | 95.08 ± 4.81 | 83.50 ± 2.91 | 80.58 ± 6.55 | 77.75 ± 5.69 |
| Study only | RFW WM64 | 99.83 ± 0.83 | 84.33 ± 2.98 | 82.75 ± 8.03 | 68.00 ± 8.18 |
| Study only | CFD WM64 | 95.08 ± 4.81 | 84.00 ± 1.76 | 82.67 ± 7.16 | 74.83 ± 8.42 |
| Zero added noise | RFW WM64 | 99.83 ± 0.83 | 99.75 ± 1.00 | 93.08 ± 5.09 | 94.08 ± 3.77 |
| Zero added noise | CFD WM64 | 94.92 ± 4.71 | 93.25 ± 4.44 | 84.08 ± 6.77 | 85.08 ± 6.30 |

“Both phases” follows the same orientation routing table as the new experiment. “Study only” applies p1 to upright study images and p2 to inverted study images, with zero added noise at test. “Zero added noise” sets both rates to zero; the model’s Bernoulli representation sampling still remains stochastic.

| r20cyl variant | Face set | Mean fitted p1 across seeds | Mean fitted p2 across seeds |
| --- | --- | --- | --- |
| Both phases | RFW WM64 | 0.30950 | 0.38324 |
| Both phases | CFD WM64 | 0.04726 | 0.28726 |
| Study only | RFW WM64 | 0.34760 | 0.42552 |
| Study only | CFD WM64 | 0.05450 | 0.31514 |

These are averages of per-seed probabilities, not one fixed probability used for the entire experiment. The achieved older fits are visibly imperfect—for example, RFW UU in the both-phase run is 99.92% despite its 96% target.

## 3. Earlier Model 2 experiments

### Completed single-noise run

This run used a single **p = 0.36**, calibrated on RFW UU and then reused across all categories/orientations, with noise in both study and test. It completed 200 category/seed jobs (800 condition rows). The historical CSV does not record a checkpoint epoch, so its epoch is not asserted here.

| Stimulus set | Seeds per condition | UU | II | UI | IU |
| --- | --- | --- | --- | --- | --- |
| RFW WM64 | 50 | 95.83 ± 4.21 | 81.58 ± 7.77 | 59.92 ± 10.37 | 60.50 ± 10.25 |
| CFD WM64 | 50 | 81.58 ± 6.53 | 59.42 ± 9.33 | 57.17 ± 9.85 | 59.33 ± 9.01 |
| Houses | 50 | 67.67 ± 7.83 | 68.08 ± 6.97 | 62.00 ± 8.01 | 59.00 ± 7.45 |
| Objects | 50 | 94.00 ± 3.96 | 91.42 ± 4.41 | 87.00 ± 5.37 | 91.25 ± 5.14 |

Source: [per-seed CSV](../runs/sim_seeds_wm64_rfwcal/results_r21dev_aa_s42.csv).

### Interrupted rerun — partial results only

The archived rerun stopped after 181/200 category/seed jobs (724 condition rows). CFD contains 46 seeds; the other sets contain 45. It also used p = 0.36. Its log is labeled “final,” but the CSV does not establish the checkpoint epoch. These partial values are not completed experiment results.

| Stimulus set | Seeds per condition | UU | II | UI | IU |
| --- | --- | --- | --- | --- | --- |
| RFW WM64 | 45 | 96.57 ± 3.47 | 83.43 ± 6.68 | 61.02 ± 10.17 | 62.78 ± 9.63 |
| CFD WM64 | 46 | 81.25 ± 6.85 | 58.97 ± 9.54 | 57.34 ± 10.25 | 56.61 ± 10.08 |
| Houses | 45 | 67.50 ± 8.17 | 67.68 ± 7.49 | 61.67 ± 11.12 | 60.56 ± 7.83 |
| Objects | 45 | 93.80 ± 4.13 | 91.57 ± 4.57 | 86.76 ± 5.85 | 90.46 ± 5.93 |

Source: [per-seed CSV](../runs/trash_killed_final_r21dev_aa/results_r21dev_aa_s42.csv).

### Attempts without scientific results

| Attempt | Saved outcome |
| --- | --- |
| Earlier RFW own-calibration attempt | No data rows; failed because `antialiased_cnns` was missing from the execution environment. |
| Earlier CFD own-calibration attempt | No data rows; same dependency failure. |
| New queue smoke checks | Eight model/category probes passed, including resume checks; these small validation runs are excluded from the experiment averages. |

The environment issue was fixed in Conda `themodel2`; the completed new two-calibration runs above supersede those failed attempts.

## 4. Pending final-checkpoint experiments

| Model | Scheduled face sets | Status at report check |
| --- | --- | --- |
| r20cylblur | RFW WM64 + CFD WM64 | Pending: queue waits for all three training runs to finish |
| Developmental VGG | RFW WM64 + CFD WM64 | Pending: queue waits for all three training runs to finish |
| VGG2k | RFW WM64 + CFD WM64 | Pending: queue waits for all three training runs to finish |

Queue state: `waiting_for_all_training`; last state update `2026-09-25T09:15:04.950070-07:00`. The eight initial model/category jobs all finished successfully on their first attempt. The six final-checkpoint jobs have no results yet. The detached tmux queue is `yin_orientation_20260925`. This report does not automatically update when they finish.

## 5. Sources and reproducibility

| Experiment | Files |
| --- | --- |
| Model 2 — RFW WM64 | [Summary](../runs/yin_orientation_20260925/results/model2/r21dev_aa_s42/faces_rfwWM64/summary.json) · [Per-seed results](../runs/yin_orientation_20260925/results/model2/r21dev_aa_s42/faces_rfwWM64/results.csv) · [Calibration](../runs/yin_orientation_20260925/results/model2/r21dev_aa_s42/faces_rfwWM64/calibration.json) |
| Model 2 — CFD WM64 | [Summary](../runs/yin_orientation_20260925/results/model2/r21dev_aa_s42/faces_cfdWM64/summary.json) · [Per-seed results](../runs/yin_orientation_20260925/results/model2/r21dev_aa_s42/faces_cfdWM64/results.csv) · [Calibration](../runs/yin_orientation_20260925/results/model2/r21dev_aa_s42/faces_cfdWM64/calibration.json) |
| r20cylblur — RFW WM64 | [Summary](../runs/yin_orientation_20260925/results/latest/r20_cylblur_s42/faces_rfwWM64/summary.json) · [Per-seed results](../runs/yin_orientation_20260925/results/latest/r20_cylblur_s42/faces_rfwWM64/results.csv) · [Calibration](../runs/yin_orientation_20260925/results/latest/r20_cylblur_s42/faces_rfwWM64/calibration.json) |
| r20cylblur — CFD WM64 | [Summary](../runs/yin_orientation_20260925/results/latest/r20_cylblur_s42/faces_cfdWM64/summary.json) · [Per-seed results](../runs/yin_orientation_20260925/results/latest/r20_cylblur_s42/faces_cfdWM64/results.csv) · [Calibration](../runs/yin_orientation_20260925/results/latest/r20_cylblur_s42/faces_cfdWM64/calibration.json) |
| Developmental VGG — RFW WM64 | [Summary](../runs/yin_orientation_20260925/results/latest/r21dev_vgg16_bn_aa_s42/faces_rfwWM64/summary.json) · [Per-seed results](../runs/yin_orientation_20260925/results/latest/r21dev_vgg16_bn_aa_s42/faces_rfwWM64/results.csv) · [Calibration](../runs/yin_orientation_20260925/results/latest/r21dev_vgg16_bn_aa_s42/faces_rfwWM64/calibration.json) |
| Developmental VGG — CFD WM64 | [Summary](../runs/yin_orientation_20260925/results/latest/r21dev_vgg16_bn_aa_s42/faces_cfdWM64/summary.json) · [Per-seed results](../runs/yin_orientation_20260925/results/latest/r21dev_vgg16_bn_aa_s42/faces_cfdWM64/results.csv) · [Calibration](../runs/yin_orientation_20260925/results/latest/r21dev_vgg16_bn_aa_s42/faces_cfdWM64/calibration.json) |
| VGG2k — RFW WM64 | [Summary](../runs/yin_orientation_20260925/results/latest/r21vgg2k_vgg16_bn_aa_s42/faces_rfwWM64/summary.json) · [Per-seed results](../runs/yin_orientation_20260925/results/latest/r21vgg2k_vgg16_bn_aa_s42/faces_rfwWM64/results.csv) · [Calibration](../runs/yin_orientation_20260925/results/latest/r21vgg2k_vgg16_bn_aa_s42/faces_rfwWM64/calibration.json) |
| VGG2k — CFD WM64 | [Summary](../runs/yin_orientation_20260925/results/latest/r21vgg2k_vgg16_bn_aa_s42/faces_cfdWM64/summary.json) · [Per-seed results](../runs/yin_orientation_20260925/results/latest/r21vgg2k_vgg16_bn_aa_s42/faces_cfdWM64/results.csv) · [Calibration](../runs/yin_orientation_20260925/results/latest/r21vgg2k_vgg16_bn_aa_s42/faces_cfdWM64/calibration.json) |
| r20cyl — Both phases | [Saved logs/results](../runs/r20cyl_bothnoise_bytest) |
| r20cyl — Study only | [Saved logs/results](../runs/cyl_study) |
| r20cyl — Zero added noise | [Saved logs/results](../runs/zeronoise_cyl) |

- [Human reference values](figures/yin_human_vs_models.csv).
- [Queue manifest](../runs/yin_orientation_20260925/manifest.json) and [queue state](../runs/yin_orientation_20260925/queue_state.json) record immutable checkpoint paths, captured epochs, and SHA-256 hashes.
- [Queue setup and commands](../runs/yin_orientation_20260925/README.md); [frozen source](../runs/yin_orientation_20260925/source); [smoke checks](../runs/yin_orientation_20260925/smoke_results.json); [unit-test log](../runs/yin_orientation_20260925/unit_tests.log).
- Model definitions: [model.py](../training/model.py), [cylconv.py](../training/cylconv.py), [r20 training](../scripts/train_r20.sh), [developmental training](../training/train_r21_dev.sh), [VGG2k training](../training/train_r21_vgg2k.sh).
- Older r20cyl launchers: both phases (retired launcher), [study only](../yin_tests/run_cyl_battery.sh), [zero added noise](../yin_tests/run_zeronoise_cyl.sh).
- Failed older calibration log: [own-calibration log](../runs/logs_yin_r21dev_aa_owncal.log).

Verification: recomputed every new-run mean from its per-seed CSV and checked it against the saved summary; checked all 300 r20cyl logs for four conditions and seeds 101–150; matched the older both-phase CSV against its raw logs; checked the complete and partial historical row counts.
