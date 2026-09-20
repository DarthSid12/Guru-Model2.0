# Model catalogue — every training round, r3 to r18

Generated 2026-09-12 from `runs/*/config.json`, `scripts/train_*.sh`, `run_sim_seeds.MODELS`, and every `runs/sim_seeds_*/results_*.csv`.


## How to read the "best Yin" column

It is the **largest face UU−II this model achieved in any simulation run in this
repo**, with the store, experiment and seed count that produced it. It is a
best-case cherry-pick across stores and noise levels, useful for "what is this
model capable of", **useless for model-to-model comparison** — different rows
sit at different calibrated `p`, on different face stores, some of which are
in-domain for some models. For a like-for-like comparison of r15/r16/r17 use
[results_rfwcal_battery.md](results_rfwcal_battery.md), where every row is
fitted to the same anchor on the same store.

Human reference (Yin 1969, faces): UU 96.29, II 81.88, **UU−II +14.41**.

## Conventions shared by most rounds

- ResNet-18 backbone trained **from scratch**, never ImageNet-initialised.
- `--variant lp`: log-polar foveated crops around 16 Gabor-selected fixation
  points, 180 px. (`plain` = no foveation/log-polar; `cnn` = foveation only.)
- 256-unit sigmoid bottleneck `h`, Bernoulli-sampled; the Yin/NIMBLE memory
  model reads `h`.
- From r7 on, a **curriculum**: classes are introduced in stages rather than all
  at once. From r15 on this is 6 stages / 80 epochs (6/6/8/8/12/40) with a
  global cosine LR from 1e-3.
- **No inverted views in training** — a deliberate safeguard so the Yin
  manipulation stays uncontaminated. Broken on purpose only by r18.


---

## Master table

| round | date | face arm | house arm | objects | key variable | best face UU−II |
|---|---|---|---|---|---|---|
| `r3` | 2026-07-16 | 128 CelebA | `houses` generic | 64 | first LP baseline; 30/60 ep, no curriculum | — |
| `r4` | 2026-07-17 | 128 CelebA | `houses_ident` per-photo ids | 64 | house identity as per-photo, not per-building | — |
| `r5` | 2026-07-18 | 128 CelebA | `houses_zubud` (ZuBuD ids) | 64 | ZuBuD buildings as house identities | — |
| `r5b` | 2026-07-19 | 128 CelebA | `houses_zubud` | 64 | **all-at-once** control for the curriculum | +11.11 |
| `r6` | 2026-07-30 | 128 CelebA | `houses_zubud` | 64 | `plain` variant — no log-polar, no foveation | — |
| `r7` | 2026-08-07 | 128 CelebA | `houses_zubud` | 64 | **first curriculum** (developmental) run | +5.87 |
| `r8` | 2026-08-12 | 128 CelebA | `houses_zubud` | 64 | developmental, refined schedule | +8.96 |
| `r9` | 2026-08-12 | 128 CelebA | `houses_zubud` | 64 | developmental + per-category **weighting** | +6.79 |
| `r10` | 2026-08-23 | 128 CelebA | 80 ZuBuD ids @20% | 64 | house ladder grows **gradually** | +1.42 |
| `r11` | 2026-08-23 | 128 CelebA | 80 ZuBuD + generic `houses` | 64 | + basic-level houseness from epoch 1 | +0.92 |
| `r12` | 2026-08-23 | 128 CelebA | 80 ZuBuD **en bloc** | 64 | all 80 house ids arrive in the final stage | +1.67 |
| `r7ft / r8ft / r11ft` | 2026-08-27/30 | +159 new face ids (CFD + Set_A) | inherited | inherited | 20-epoch fine-tunes at lr 1e-4 | +22.00 |
| `r13` | 2026-08-30 | 128 CelebA + CFD | 40 ZuBuD + `houses` | 64 | CFD identities added, house arm at 40 | +8.42 |
| `r14` | 2026-08-30 | 128 CelebA + CFD | ZuBuD + `houses`, low | 64 | r13 with a lower house weight | +8.25 |
| `r15` | 2026-08-31 | 128 CelebA + **480 VGGFace2** + 119 CFD | 40 ZuBuD + `houses` | 64 | **VGGFace2 added** — natural photo variation | +4.00 |
| `r16_base` | 2026-09-01 | 128 CelebA + 480 VGGFace2 | 40 ZuBuD + **`houses_gen`** | 64 | reference for the r16 round; no CFD | +10.75 |
| `r16_acuity` | 2026-09-01 | as r16_base | as r16_base | 64 | **blur→sharp** acuity schedule (σ 8→0) | +9.42 |
| `r16_rfw` | 2026-09-01 | + **1500 RFW** (1200 W, 300 other) | as r16_base | 64 | race-balanced face diet, white-first | +16.25 |
| `r16rfwftH` | 2026-09-03 | warm-start r16_rfw | **137 ZuBuD** @45%, `houses` | 64 | house fine-tune to repair the control | +20.33 |
| `r17_base` | 2026-09-03 | 128 CelebA + 480 VGGFace2 | 40 ZuBuD + **`houses`** | 64 | single-variable: drop ZuBuD from generic | +9.50 |
| `r17_houses` | 2026-09-03 | VGGFace2 at 0.33 | **137 ZuBuD** @0.20, `houses` | 0.22 | house arm scaled on identities AND weight | +3.83 |
| `r18_inv02/05/20` | 2026-09-10/12 | as r16_rfw | as r16_rfw | 64 | **2/5/20% inverted** training exposure | -2.67 |

`house_control_r1` ("David's model") appears throughout the sims as an external comparison. It lives in the Guru-Model2.0 tree, not this repo, and is not part of the r-series. **There is no r1 or r2** in this repo — the numbering starts at r3.


---

## Best Yin face result per model, with provenance

| model | best face UU−II | UU at that point | store / experiment / seeds |
|---|---|---|---|
| `r11ftA` | **+22.00** | 94.25 | `cfdWM64` / `cfdWM64_ctrl` / 50s |
| `r16rfwftH` | **+20.33** | 97.00 | `cfdWM64` / `wm64` / 50s |
| `house_control_r1` | **+17.22** | 97.22 | `celeb24` / `celeb24` / 20s |
| `r16_rfw` | **+16.25** | 81.42 | `cfdWM64` / `wm64_rfwcal` / 50s |
| `r5b_allatonce` | **+11.11** | 98.33 | `celeb24` / `celeb24` / 20s |
| `r11ftB` | **+10.83** | 89.08 | `cfdWM64` / `cfdWM64_p010` / 50s |
| `r16_base` | **+10.75** | 85.25 | `cfdWM64` / `wm64_rfwcal` / 50s |
| `r7ftA` | **+9.67** | 90.08 | `cfdWM64` / `cfdWM64_p010` / 50s |
| `r17_base` | **+9.50** | 84.58 | `cfdWM64` / `wm64_rfwcal_layer4` / 50s |
| `r8ftB` | **+9.42** | 89.08 | `cfdWM64` / `cfdWM64_p010` / 50s |
| `r16_acuity` | **+9.42** | 77.75 | `cfdWM64` / `wm64_rfwcal` / 50s |
| `r8_developmental` | **+8.96** | 63.92 | `u40k24` / `yin4024` / 100s |
| `r8ftA` | **+8.75** | 92.58 | `cfdWM64` / `cfdWM64_ctrl` / 50s |
| `r13_cfd_h40` | **+8.42** | 83.33 | `cfdWM64` / `cfd_h40_p010` / 50s |
| `r14_cfd_lo` | **+8.25** | 82.17 | `cfdWM64` / `cfd_h40_p010` / 50s |
| `r7ftB` | **+7.92** | 92.33 | `cfdWM64` / `cfdWM64_p010` / 50s |
| `r9_developmental_weighted` | **+6.79** | 95.46 | `k64` / `familiarity4` / 100s |
| `r17_base_interim_ep63` | **+6.75** | 88.67 | `cfdWM64` / `wm64` / 50s |
| `r7_curriculum` | **+5.87** | 80.27 | `setA_ho` / `setA_ho` / 50s |
| `r17_houses_interim_ep63` | **+5.67** | 92.67 | `cfdWM64` / `wm64` / 50s |
| `r15_vgg` | **+4.00** | 89.17 | `cfdWM64` / `wm64_rfwcal` / 50s |
| `r17_houses` | **+3.83** | 97.08 | `rfwWM64` / `wm64` / 50s |
| `r12_dev_h80_enbloc` | **+1.67** | 93.88 | `mixu16` / `joint_zubud_mixu16` / 100s |
| `r10_dev_h80` | **+1.42** | 65.75 | `cfdWM64` / `cfdWM64` / 50s |
| `r11_dev_h80_generic` | **+0.92** | 93.04 | `mixu16` / `joint_zubud_mixu16` / 100s |
| `r18_inv20` | **-1.67** | 93.33 | `rfwWM64` / `wm64_rfwcal` / 50s |
| `r18_inv05` | **-2.67** | 91.92 | `rfwWM64` / `wm64_rfwcal` / 50s |

| *Human (Yin 1969)* | **+14.41** | 96.29 | — |

---

## What each round was actually asking

**r3–r6 — getting the stimulus set right.** No curriculum, 30–60 epochs, both
resnet18 and resnet34. The house category was the moving part: `houses` (one
generic class) → `houses_ident` (per-photo identities) → `houses_zubud` (ZuBuD
buildings as identities). **r6** is the ablation that matters: `plain` variant,
no log-polar and no foveation, i.e. the control for whether the front end is
doing the work. **r5b** is the all-at-once control that the developmental rounds
are read against.

**r7–r9 — the developmental curriculum.** r7 introduces staged class
introduction; r8 refines the schedule; r9 adds per-category sampling weights.
This is the round the "developmental vs all-at-once" claim rests on
([curriculum_results_summary.md](curriculum_results_summary.md)): developmental
reaches +12.10 face UU−II against all-at-once's +9.12, human +14.41.

**r10–r12 — how the house arm should grow.** All three hold houses at 20% of
every batch with 80 ZuBuD identities. The single variable is *timing*: r10 grows
the house-class ladder gradually, r12 drops all 80 in at the final stage, r11
adds a generic `houses` class from epoch 1 (basic-level houseness before any
individuation). Face results are weak across all three (+0.92 to +1.67) — this
round is about the control category, not the faces.

**The fine-tunes (r7ft, r8ft, r11ft).** 20 epochs at lr 1e-4 on 159 new face
identities — 90 CFD white females, 29 CFD white males outside the held-out 64,
and 40 Set_A identities — with the original categories capped at 40 img/class.
`r11ftA` holds the highest face UU−II anywhere in the project (**+22.00** on
cfdWM64), though at a p and store that make it non-comparable to the calibrated
battery. Note Set_A is now *familiar* to these models, so the mix64 / growA /
familiarity4 experiments do not apply to them.

**r13–r14 — CFD in the diet.** Adds Chicago Face Database identities to the
training set, differing in the house weight. These are the models r15 was built
to improve on.

**r15_vgg — natural photo variation.** Adds 480 VGGFace2 identities for 727
total face identities against r13/r14's 247. Aimed at the one gap CFD training
did not close: Kanwisher matching under natural within-identity variation, where
CFD's expression-only variation gave the models nothing. Carried through the
r16/r17 battery as the reference model.

**r16 — three-way round, and the `houses_gen` mistake.** All three share a
generic house class `houses_gen` = 97 leftover ZuBuD buildings **+** 635
Houses-dataset buildings. That turned out to be a serious error: folding ZuBuD
into a class the model never individuates collapses ZuBuD representations, and
all three models read held-out `houses_yin64` at **33.9–43.1%**, at or below the
50% 2AFC floor. The house control is dead in every r16 model. `r16_acuity` adds
a blur→sharp schedule; `r16_rfw` adds 1500 RFW identities and is the round's
best face model.

**r16rfwftH — the repair, and the project's best result.** 20 epochs at lr 1e-4
on r16_rfw with 137 ZuBuD identities at 45% of the diet and the generic class
switched back to `houses`. It restores the house control (59.9% upright, clear
of the floor) while keeping the face effect, and in the calibrated battery it is
the only model with a clean dissociation on **both** face stores at once:
cfd +16.25, rfw +11.92, houses −0.92, objects +2.75.

**r17 — the single-variable fix.** `r17_base` is `r16_base` with exactly one
change, `houses_gen` → `houses`. It restores held-out ZuBuD from 33.92% to
85.17% — and, unexpectedly, also lifts *face*-representation robustness
(calibrated p 0.08 → 0.20), so the `houses_gen` damage was never confined to
houses. `r17_houses` additionally scales the house arm to 137 identities at
weight 0.20, paid for from `faces_vgg` and `objects` — so its **object rows are
not comparable** to the other models.

**r18 — inverted-exposure arms (in progress).** `r16_rfw`'s recipe, byte-identical
apart from `--invert-p`, so `r16_rfw` is the matched 0% control. These
**deliberately break the upright-only training invariant**, so their Yin rows are
interpretable only against each other and `r16_rfw`. Result so far: 5% inverted
exposure abolishes the classification inversion effect outright (faces
valid/test-inverted 90/62 → 86/86) and drives the 2AFC congruence term from
+18.17 to +4.38 — but takes the orientation main effects negative with it, so the
model becomes orientation-indifferent rather than human-like. `r18_inv02` is
still training.

---

## Caveats

1. **The "best Yin" column is a cherry-pick.** Different stores, different
   calibrated `p`, different seed counts. Never compare two rows of it directly.
2. **`houses_gen` models have no usable house control** — r16_base, r16_acuity,
   r16_rfw, and all r18 arms read held-out ZuBuD at or below chance.
3. **Face stores are not domain-neutral.** `faces_cfdWM64` is in-domain for
   `r15_vgg` (119 CFD ids) and for the r7/r8/r11 fine-tunes; `faces_rfwWM64` is
   in-domain for `r16_rfw` and `r16rfwftH`.
4. **r10–r12 have no `run_tag`** in their configs and live at `runs/r1*_*`
   rather than the usual long descriptive path.
5. **Several early rounds have duplicate directories** (resnet18 and resnet34
   variants of the same tag) — r3, r5, r5b, r6 each have two.

