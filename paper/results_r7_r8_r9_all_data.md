# r7, r8, r9, house_control — every Yin and Kanwisher result, against the human anchors

> ## ⚠ Correction — 2026-08-21
>
> **`house_control_r1` ("David's model") trained on 40 ZuBuD buildings.** Its
> houses are a 40-building subset of the same ZuBuD store the other models use
> (train stems `object0101.view02`, 3 views each), *not* a separate Houses
> dataset. Every statement in this file that it "never trained on ZuBuD" is
> wrong. It is also a curriculum model, on a two-stage schedule (2 epochs of
> 256 face+object classes, then all 296 with houses added) driven by
> `houses_delay_epochs: 2` rather than the class-count ladder that
> `config.json`'s `curriculum: false` refers to.
>
> Consequence: its ZuBuD rows were scored on **all 201 buildings including the 40
> it trained on** — the held-out filter matched nothing because its label_map
> files houses under `houses/` while the packed store is `houses_zubud`, and its
> class names are bare ids (`101`) against the store's `house0101`. The filter
> now canonicalises names across stores and errors out on a zero-match instead of
> passing silently. **Treat every `house_control_r1` ZuBuD number below as
> provisional**; corrected runs are in `runs/sim_seeds_joint_zubud_fix` and
> `runs/sim_seeds_kanw3_fix`.


Complete inventory of the simulation data collected for all four models,
aggregated from the raw CSVs under `runs/` (not transcribed from earlier
write-ups). Every row states **which stimulus set it was scored on** and whether
that set was trained on. Rows scored on different sets are not comparable to
each other; compare within a dataset.

Generated 2026-08-21.

## Models

| key | run dir | training data |
| --- | --- | --- |
| `r7_curriculum` | `runs/…_resnet18_r7_curriculum` | faces 128, objects 64, **houses_zubud 201 (all)**; global curriculum `4→8→…→all` |
| `r8_developmental` | `runs/…_resnet18_r8_developmental` | faces 128, objects 64, **houses_zubud 40 (capped)**; per-category curriculum, houses withheld until stage 3 |
| `r9_developmental_weighted` | `runs/…_resnet18_r9_developmental_weighted` | identical to r8 plus `category_weights: faces=0.45 objects=0.45 houses_zubud=0.10` |
| `house_control_r1` | `/home/d1deutsch/Guru-Model2.0/runs/…_resnet18_house_control_r1` | faces 128, **objects 128**, **houses_zubud 40 (same dataset and count as r8)**; two-stage curriculum (2 epochs of 256 face+object classes, then all 296 with houses added, via `houses_delay_epochs: 2`); `category_ratios faces=0.40 objects=0.40 houses=0.20`, no LR schedule |

r7 and r8 differ **only** in house class count and curriculum shape — face and
object training are identical. r8 and r9 differ **only** in the category
weights. `house_control_r1` trains on the *same* house dataset and the same 40
buildings as r8, but still differs in three ways at once — **object class count
(128 vs 64), curriculum shape (2 stages vs 6), and no LR schedule** — so it is
**not** a single-variable control. Its `config.json` records `curriculum: false`,
which refers only to the class-count ladder; the house delay it does run comes
from `houses_delay_epochs: 2` and is visible as `stage_spec` `all-prehouse` →
`all` in its training history.

## Human anchors

| Simulation | Faces | Objects | Houses |
| --- | --- | --- | --- |
| Yin (1969) upright-upright | **96.29** | **84.79** | **90.71** |
| Yin inversion cost (UU − UI) | **+12.16** | **−1.92** | **+2.63** |
| Dobs/Kanwisher (2023) upright | **87.50** | — | — |
| Dobs/Kanwisher inversion effect | **+10.70** between-subj / **+11.60** within-subj | — | — |

Yin: Tables 1–2, accuracy = (24 − mean errors) × 100 / 24. Dobs et al.: Exp. 1
and 5. Inversion cost is **UU − UI**, paired per seed; `UU − II` is not the
informative contrast for this design.

## Face stimulus sets

| Store | Composition | Seen in training? |
| --- | --- | --- |
| `faces` | the 128 CelebA training identities | **yes** (images from held-out `valid`) |
| `faces_celeb24` | the 24 named celebs | **yes** |
| `faces_k64` | 24 named + 40 more trained | **yes** |
| `faces_mixu8` | 56 known + 4 Set_E + 4 Set_A | mixed |
| `faces_mixu16` | 48 known + 8 Set_E + 8 Set_A | mixed |
| `faces_mixed64` (= `mixu24`) | 40 known (24 named + 16 other) + 12 Set_E + 12 Set_A | mixed |
| `faces_k32u32` | 32 known + 32 Set_E | mixed |
| `faces_mixu32` | 32 known + 16 Set_E + 16 Set_A | mixed |
| `faces_mixu40` | 24 known + 20 Set_E + 20 Set_A | mixed |
| `faces_mixu48` | 16 known + 24 Set_E + 24 Set_A | mixed |
| `faces_u64` / `faces_setE` | 64 Set_E identities | **no** |
| `faces_setA` | 40 Set_A identities | **no** — but broken, see §6 |

All ladder stores are 64 identities × 5 photos, so the Yin 40/24 split is
identical across them and only the known/unknown ratio moves.

---

## Set_A alone — every run we have

`faces_setA`: 40 novel identities × 5 photos, 250×250, seen by no model. Pulled
out separately here because it is the one stimulus set that fails on both
simulations, and because **all of its Yin data comes from runs the project has
already excluded**. r7 was never run on Set_A at all.

### Kanwisher on Set_A — valid, and far below the anchor

Same-store triplet matching, so no cross-store artefact is possible. These runs
are sound; the numbers are simply bad.

| Model | p | Upright (**87.50**) | Effect (**+10.70/+11.60**) | n | Source |
| --- | --- | --- | --- | --- | --- |
| r8 | 0.00 | 71.25 ±0.02 | +2.90 ±0.03 | 100 | `sim_seeds/kanwisher_setA` |
| r9 | 0.00 | 69.86 ±0.02 | +3.41 ±0.03 | 100 | `sim_seeds/kanwisher_setA` |
| house_control_r1 | 0.00 | 71.18 ±0.03 | +2.69 ±0.03 | 100 | `sim_seeds/kanwisher_setA` |
| r8 | 0.39 | 61.74 ±0.20 | +2.24 ±0.26 | 30 | `calibrated_p_partial` |
| r9 | 0.38 | 61.81 ±0.23 | +2.46 ±0.29 | 29 | `calibrated_p_partial` |
| house_control_r1 | 0.35 | 63.69 ±0.19 | +3.12 ±0.26 | 29 | `calibrated_p_partial` |
| r8 | 0.40 | 60.67 ±0.23 | +1.75 ±1.02 | 2 | `fixed_p03_partial` |
| r9 | 0.40 | 59.45 ±1.66 | +1.84 ±0.50 | 2 | `fixed_p03_partial` |
| house_control_r1 | 0.40 | 60.14 ±0.65 | +3.85 ±0.10 | 2 | `fixed_p03_partial` |

Upright tops out at **69.9–71.3% against an 87.50 anchor even at zero noise**,
so `p` pins at the floor for every model and nothing here is calibrated. Adding
noise only makes it worse (~60% at p≈0.4). The inversion effects of +1.8 to
+3.9 are roughly a quarter of the human value. Set_A cannot support a Kanwisher
result at any noise level.

### Yin on Set_A — ⚠ every row is from an excluded run

**Do not cite any number in this table.** All three source directories are on
the excluded list in `results_all_stimulus_sets.md` §4: study items were drawn
from Set_A while the "new" distractors came from the `faces` store, so the 2AFC
was decided by **dataset appearance, not memory** — studied Set_A scored 90%
against old-store distractors and *unstudied* Set_A scored 90% too.

| Model | p | UU (96.29) | Cost (+12.16) | n | Source ⚠ |
| --- | --- | --- | --- | --- | --- |
| r8 | 0.00 | 95.42 ±0.50 | +4.86 ±0.90 | 30 | `calibrated_p_partial` |
| r8 | 0.30 | 93.05 ±2.78 | +1.39 ±1.39 | 3 | `fixed_p03_partial` |
| r8 | 0.32 | 91.25 ±1.04 | +2.50 ±1.15 | 20 | `yin_scan/p0.32` |
| r8 | 0.34 | 90.21 ±1.18 | +2.92 ±1.60 | 20 | `yin_scan/p0.34` |
| r8 | 0.36 | 89.38 ±1.26 | +2.92 ±1.60 | 20 | `yin_scan/p0.36` |
| r8 | 0.38 | 87.71 ±1.33 | +1.67 ±1.69 | 20 | `yin_scan/p0.38` |
| r9 | 0.08 | 93.75 ±0.68 | +4.58 ±0.78 | 30 | `calibrated_p_partial` |
| r9 | 0.30 | 93.06 ±1.39 | +0.00 ±4.16 | 3 | `fixed_p03_partial` |
| r9 | 0.32 | 89.79 ±1.26 | +2.08 ±1.47 | 20 | `yin_scan/p0.32` |
| r9 | 0.34 | 89.58 ±1.19 | +1.67 ±1.43 | 20 | `yin_scan/p0.34` |
| r9 | 0.36 | 86.88 ±1.26 | +0.21 ±1.95 | 20 | `yin_scan/p0.36` |
| r9 | 0.38 | 85.83 ±1.22 | +2.09 ±1.90 | 20 | `yin_scan/p0.38` |
| house_control_r1 | 0.10 | 97.08 ±0.50 | +6.67 ±0.95 | 30 | `calibrated_p_partial` |
| house_control_r1 | 0.30 | 100.00 ±0.00 | +5.56 ±3.67 | 3 | `fixed_p03_partial` |
| house_control_r1 | 0.32 | 95.42 ±0.67 | +6.46 ±1.30 | 20 | `yin_scan/p0.32` |
| house_control_r1 | 0.34 | 94.37 ±0.82 | +5.83 ±1.30 | 20 | `yin_scan/p0.34` |
| house_control_r1 | 0.36 | 92.92 ±1.14 | +6.46 ±1.75 | 20 | `yin_scan/p0.36` |
| house_control_r1 | 0.38 | 91.04 ±0.97 | +6.25 ±1.75 | 20 | `yin_scan/p0.38` |

The tell is the accuracy: 87–100% UU on a stimulus set whose **honest same-store
Yin score is 53.33% at p=0 — chance**. A 40-point gap between the excluded runs
and the same-store measurement is the artefact, not model skill. The inversion
costs of +0.2 to +6.7 are also well under the human +12.16 despite the inflated
accuracy.

**There is no valid Yin Set_A-only result in the repo, and there cannot be one:**
Set_A has exactly 40 identities, so it cannot fill Yin's 40 study + 24 test
design from a single store, which is what forced the cross-store distractors in
the first place. Any Yin condition using Set_A must embed it in a larger store
(`faces_mixed64`, `faces_mix64`, the `mixu` ladder), where it contributes items
but is not measured alone.

### Why Set_A fails

Its photos of one identity barely resemble each other (within-identity pixel
correlation **+0.078** vs Set_E's **+0.113**) while every identity is a young
female celebrity, so between-identity variability is low too — both directions
hurt an identity-matching task. It is **not** a resolution effect: Set_E has
*more* sub-224px images. Set_E is the working alternative for any condition that
needs unfamiliar faces.

---

## 1. ★ The results that match humans on both axes

### 1a. Kanwisher, `faces_mixed64` — all three models replicate

20 seeds each. One Kanwisher seed scores ~100k triplets, hence the tight SEMs.

| Source | Dataset | p | Upright | Inversion effect |
| --- | --- | --- | --- | --- |
| **Human (between-subjects)** | — | — | **87.50** | **+10.70** |
| **Human (within-subject)** | — | — | **87.50** | **+11.60** |
| **r7_curriculum** | `faces_mixed64` | 0.40 | **87.33 ±0.30** | **+11.09 ±0.30** |
| **r8_developmental** | `faces_mixed64` | 0.40 | **87.13 ±0.32** | **+10.79 ±0.38** |
| **house_control_r1** | `faces_mixed64` | 0.39 | **87.33 ±0.35** | **+12.17 ±0.26** |

Every model lands on the 87.50 upright anchor and inside (or within half a
point of) the two human effects. Face-specificity in the same runs, same p:

| Model | Dataset | Upright | Effect |
| --- | --- | --- | --- |
| r7 | `objects` (64, trained) | 85.35 ±0.61 | +3.21 ±0.35 |
| r7 | `houses_zubud` (201, **trained**) | 95.14 ±0.37 | +1.27 ±0.53 |
| r8 | `objects` (64, trained) | 84.75 ±0.51 | +2.93 ±0.38 |
| r8 | `houses_zubud` (161, held-out) | 86.40 ±0.61 | +0.57 ±0.72 |
| house_control_r1 | `objects` (128, trained) | 81.94 ±0.35 | +3.11 ±0.30 |
| house_control_r1 | `houses_zubud` (161 held-out, **corrected**) | 87.31 ±0.73 | +2.03 ±0.78 |

Faces ~+11 to +12, objects ~+3, houses ~+1. The inversion effect is roughly ten
times larger for faces than for houses, in all three models. **This is the
strongest result in the project.**

### 1b. Yin, `faces_mixu16`, r8, 100 seeds

| Source | Dataset | p | UU | Cost (UU−UI) |
| --- | --- | --- | --- | --- |
| **Human (Yin faces)** | — | — | **96.29** | **+12.16** |
| **r8_developmental** | `faces_mixu16` (48 known + 16 unknown) | 0.23 | **95.75 ±0.42** | **+12.17 ±0.65** |

---

## 2. Yin — faces, every dataset

| Model | Dataset | p | UU (96.29) | Cost (+12.16) | n | Source |
| --- | --- | --- | --- | --- | --- | --- |
| r8 | `faces_mixu16` | 0.23 | **95.75 ±0.42** | **+12.17 ±0.65** | 100 | `sim_seeds_ladder/u16` |
| r8 | `faces_mixed64` | 0.23 | 89.63 ±0.62 | **+12.21 ±0.70** | 100 | `sim_seeds_mixed64_ctrl` |
| r8 | `faces_mixed64` | 0.30 | 87.17 ±0.62 | +12.00 ±0.75 | 100 | `sim_seeds_familiarity4` |
| r9 | `faces_mixed64` | 0.30 | 86.96 ±0.60 | **+12.87 ±0.79** | 100 | `sim_seeds_familiarity4` |
| house_control_r1 | `faces_mixed64` | 0.30 | 87.08 ±0.63 | +14.21 ±0.83 | 100 | `sim_seeds_familiarity4` |
| house_control_r1 | `faces_mixed64` | 0.00 | 94.73 ±0.52 | +15.36 ±0.85 | 64* | `sim_seeds_joint_common` |
| r8 | `faces_mixu8` | 0.23 | 95.96 ±0.39 | +9.04 ±0.60 | 100 | `sim_seeds_ladder/u8` |
| r8 | `faces_mixu32` | 0.23 | 87.83 ±0.65 | +10.83 ±0.66 | 100 | `sim_seeds_ladder/u32` |
| r8 | `faces_mixu40` | 0.23 | 85.67 ±0.64 | +11.79 ±0.78 | 100 | `sim_seeds_ladder/u40` |
| r8 | `faces_mixu48` | 0.23 | 81.75 ±0.71 | +10.33 ±0.82 | 100 | `sim_seeds_ladder/u48` |
| r8 | `faces_k64` (all known) | 0.30 | 95.00 ±0.42 | +6.92 ±0.56 | 100 | `sim_seeds_familiarity4` |
| r9 | `faces_k64` (all known) | 0.30 | 95.46 ±0.41 | +10.46 ±0.64 | 100 | `sim_seeds_familiarity4` |
| house_control_r1 | `faces_k64` (all known) | 0.30 | 97.83 ±0.25 | +9.12 ±0.61 | 100 | `sim_seeds_familiarity4` |
| r8 | `faces_k32u32` | 0.30 | 86.37 ±0.60 | +12.00 ±0.85 | 100 | `sim_seeds_familiarity4` |
| r9 | `faces_k32u32` | 0.30 | 85.17 ±0.62 | +12.83 ±0.86 | 100 | `sim_seeds_familiarity4` |
| house_control_r1 | `faces_k32u32` | 0.30 | 90.42 ±0.55 | +16.00 ±0.87 | 100 | `sim_seeds_familiarity4` |
| r8 | `faces_u64` (all Set_E) | 0.30 | 88.29 ±0.65 | +15.92 ±0.84 | 100 | `sim_seeds_familiarity4` |
| r9 | `faces_u64` (all Set_E) | 0.30 | 83.58 ±0.72 | +14.50 ±0.99 | 100 | `sim_seeds_familiarity4` |
| house_control_r1 | `faces_u64` (all Set_E) | 0.30 | 87.12 ±0.66 | +17.50 ±0.92 | 100 | `sim_seeds_familiarity4` |
| r7 | `faces` (128, **trained**) | 0.38 | 93.10 ±0.29 | +11.56 ±0.55 | 200 | `sim_seeds` |
| r8 | `faces_celeb24` (24 known) | 0.30 | 96.33 ±0.61 | +7.00 ±0.85 | 100 | `sim_seeds_celeb24_ctrl_p030` |
| house_control_r1 | `faces_celeb24` (24 known) | 0.30 | 98.56 ±0.41 | +10.55 ±1.10 | 100 | `sim_seeds_celeb24_ctrl_p030_david` |
| r8 | `faces_setE` (64 novel) | 0.00 | 96.29 ±0.31 | +23.67 ±0.76 | 100 | `sim_seeds/yin_setE` |
| r9 | `faces_setE` (64 novel) | 0.00 | 92.58 ±0.43 | +24.00 ±0.75 | 100 | `sim_seeds/yin_setE` |
| house_control_r1 | `faces_setE` (64 novel) | 0.04 | 96.83 ±0.30 | +18.46 ±0.65 | 100 | `sim_seeds/yin_setE` |
| r8 | `faces_mixE64` | 0.04 | 91.46 ±1.26 | +16.46 ±2.45 | 20 | `sim_seeds_mixE64` |
| house_control_r1 | `faces_mixE64` | 0.03 | 93.96 ±1.15 | +17.08 ±2.32 | 20 | `sim_seeds_mixE64` |
| r8 | `faces_mix64` (24 known + 40 Set_A) | 0.00 | 83.96 ±1.29 | +11.46 ±1.28 | 20 | `sim_seeds_mix64` |
| house_control_r1 | `faces_mix64` (24 known + 40 Set_A) | 0.03 | 84.17 ±1.44 | +13.96 ±2.01 | 20 | `sim_seeds_mix64` |

\* still running at time of writing.

### 2a. The familiarity ladder at one fixed noise (r8, p=0.23, 100 seeds)

Only the known/unknown ratio moves; the design is identical at every rung.

| Unknown identities | 8 | 16 | 24 | 32 | 40 | 48 |
| --- | --- | --- | --- | --- | --- | --- |
| UU | 95.96 | 95.75 | 89.63 | 87.83 | 85.67 | 81.75 |
| Cost | +9.04 | **+12.17** | **+12.21** | +10.83 | +11.79 | +10.33 |

Upright accuracy falls monotonically with unfamiliarity, as expected. The cost
does **not** rise monotonically — it peaks around 16–24 unknown and is flat to
mildly falling thereafter. Both rungs that hit the human cost do so, but only
`mixu16` also holds the accuracy anchor.

### 2b. Familiarity at matched design (`familiarity4`, p=0.30, 100 seeds)

| Store | r8 UU / cost | r9 UU / cost | house_control_r1 UU / cost |
| --- | --- | --- | --- |
| 64 known (`faces_k64`) | 95.00 / +6.92 | 95.46 / +10.46 | 97.83 / +9.12 |
| 32 known + 32 unknown | 86.37 / +12.00 | 85.17 / +12.83 | 90.42 / +16.00 |
| mixed 40 known + 24 unknown | 87.17 / +12.00 | 86.96 / +12.87 | 87.08 / +14.21 |
| 64 unknown (`faces_u64`) | 88.29 / +15.92 | 83.58 / +14.50 | 87.12 / +17.50 |
| objects (64 known) | 95.75 / +1.54 | 96.37 / +3.96 | 98.29 / +3.04 |

Under Yin the inversion cost **rises as faces get less familiar**, in all three
models. Note this is the opposite direction to Kanwisher (§5a), where the
largest effects come from the more familiar stores. None of these four stores is
calibrated, so part of the gradient is the accuracy-versus-cost trade-off
described in §7. house_control_r1 runs the largest cost at every rung.

---

## 3. Yin — objects (always trained; 64 classes, or 128 for house_control_r1)

| Model | p | UU (84.79) | Cost (−1.92) | n |
| --- | --- | --- | --- | --- |
| r8 | 0.00 | 98.29 ±0.25 | +1.17 ±0.26 | 100 |
| r8 | 0.23 | 97.42 ±0.29 | +1.75 ±0.35 | 100 |
| r8 | 0.30 | 95.75 ±0.36 | +1.54 ±0.41 | 100 |
| r9 | 0.00 | 98.12 ±0.23 | +4.58 ±0.33 | 100 |
| r9 | 0.30 | 96.37 ±0.33 | +3.96 ±0.47 | 100 |
| r7 | 0.43 | 72.62 ±0.59 | +1.42 ±0.81 | 200 |
| house_control_r1 | 0.04 | 99.33 ±0.15 | +1.67 ±0.25 | 100 |
| house_control_r1 | 0.30 | 98.29 ±0.27 | +3.04 ±0.41 | 100 |

Costs are near-zero, which is the human pattern. Accuracy sits 11–14 points
**above** the human anchor because every object class was trained on, and it
cannot be calibrated down without collapsing faces at the same p. r9's object
cost (+4.0 to +4.6) is consistently larger than r8's (+1.2 to +1.8).

---

## 4. Yin — houses (the one unsolved arm)

| Model | Dataset | p | UU (90.71) | Cost (+2.63) | n |
| --- | --- | --- | --- | --- | --- |
| house_control_r1 ⚠ | ZuBuD, ⚠ scored on all 201, 40 trained-on | 0.04 | 97.08 ±0.27 | +10.58 ±0.61 | 100 |
| house_control_r1 ⚠ | ZuBuD, ⚠ scored on all 201, 40 trained-on | 0.10 | 96.39 ±0.62 | +10.83 ±1.03 | 30 |
| house_control_r1 ⚠ | ZuBuD, ⚠ scored on all 201, 40 trained-on | 0.30 | 86.96 ±0.61 | +8.04 ±0.85 | 100 |
| r9 | ZuBuD 161 **held-out** | 0.00 | **92.00 ±0.43** | +10.75 ±0.61 | 100 |
| r9 | ZuBuD held-out | 0.08 | 89.17 ±0.86 | +9.31 ±0.99 | 30 |
| r8 | ZuBuD 161 **held-out** | 0.00 | 84.83 ±0.60 | +10.96 ±0.64 | 100 |
| r8 | ZuBuD held-out | 0.23 | 79.29 ±0.68 | +9.33 ±0.59 | 100 |
| r8 | ZuBuD held-out | 0.30 | 75.54 ±0.88 | +8.46 ±0.88 | 100 |
| r7 | ZuBuD 201 **trained-on** | 0.32 | 90.29 ±0.40 | +11.21 ±0.47 | 200 |
| r8 | `houses_common` (94, unseen by all) | 0.00 | 86.78 ±0.64 | **+22.70 ±0.92** | 58* |
| r7 | `houses_common` (94, unseen by all) | 0.01 | 84.84 ±0.67 | **+26.65 ±1.10** | 58* |
| house_control_r1 | `houses_common` (94, unseen by all) | 0.00 | 75.39 ±0.76 | +12.63 ±1.07 | 64* |

\* still running at time of writing.

**Every model, every house dataset, is 3–10× the human +2.63.** Three further
points:

- **house_control_r1 has the highest house accuracy anywhere** (97.08 on ZuBuD),
  but ⚠ that row includes the 40 buildings it trained on, so it is not a
  held-out number and not comparable to r8/r9's. **r9 is the model to quote
  here**: it reaches the accuracy anchor (92.00 vs 90.71) on 161 genuinely
  held-out buildings, and it is exactly the r8 variant with houses
  *down-weighted* to 0.10.
- **`houses_common` is worse, not better.** It is the only store held out for
  all three models (94 Houses-dataset exteriors, built by
  `make_houses_common.py` — a different dataset from ZuBuD, which no model
  trained on), which finally makes them comparable — but its inversion costs are
  +12.6 to +26.7, far worse than ZuBuD's +8 to +11.
- **Nothing calibrates the house cost down.** Raising p lowers accuracy and the
  cost together; no p makes any model resemble the human +2.63.

### 4a. Held-out house accuracy at p=0 (screen, 12 draws)

| Model | `houses_common` (unseen by all) | ZuBuD |
| --- | --- | --- |
| r7 | 85.07 | 96.88 *(trained on all 201)* |
| r8 | **89.24** | 83.68 *(161 held-out)* |
| house_control_r1 ⚠ | 77.08 | 97.57 *(scored on all 201, 40 trained-on)* |

Only ZuBuD rows reach the 90.71 anchor, and both of those are trained-on to some
degree.

**Retracted 2026-08-21 — the "training hurts held-out performance" reading.**
This section previously argued that *training on a house dataset makes held-out
performance within that dataset worse*, on the strength of house_control_r1
scoring 97.57 on "the dataset it never saw" and 77.08 on the one it trained on.
Both halves of that sentence were backwards: it trained on ZuBuD, and never saw
the Houses-dataset. Read correctly its numbers run the *ordinary* way — high on
what it trained on (97.57, and inflated further by the 40 trained buildings in
the item pool), low on the dataset it never saw (77.08). The ordering
r8 84.83 → r9 92.00 → house_control_r1 97.08 is likewise not "monotone in less
house training", since house_control_r1 is not the zero-house-training end of it.

What survives is r8 on its own: **89.24 on a dataset it never saw vs 83.68 on
held-out classes of the one it trained on.** One model, one comparison, ~5.6
points — suggestive, not a finding. Testing it properly needs a model trained on
one house dataset and scored on both, with the trained classes actually excluded.

---

## 5. Kanwisher — every dataset

| Model | Dataset | p | Upright (87.50) | Effect (+10.70/+11.60) | n |
| --- | --- | --- | --- | --- | --- |
| **r7** | `faces_mixed64` | 0.40 | **87.33 ±0.30** | **+11.09 ±0.30** | 20 |
| **r8** | `faces_mixed64` | 0.40 | **87.13 ±0.32** | **+10.79 ±0.38** | 20 |
| **house_control_r1** | `faces_mixed64` | 0.39 | **87.33 ±0.35** | **+12.17 ±0.26** | 20 |
| r7 | `faces` (128, **trained**) | 0.43 | 85.44 ±0.10 | +14.63 ±0.13 | 200 |
| r8 | `faces_setE` (64 novel) | 0.28 | 87.76 ±0.11 | +6.45 ±0.10 | 100 |
| r9 | `faces_setE` (64 novel) | 0.09 | 88.98 ±0.11 | +5.92 ±0.11 | 100 |
| house_control_r1 | `faces_setE` (64 novel) | 0.24 | 88.01 ±0.11 | +7.46 ±0.12 | 100 |
| r8 | `faces_mix64`, 24 known only | 0.04 | 95.81 ±0.03 | +6.08 ±0.04 | 20 |
| house_control_r1 | `faces_mix64`, 24 known only | 0.09 | 96.01 ±0.04 | +6.26 ±0.08 | 20 |
| r8 | `faces_setA` | 0.00 | 71.25 ±0.02 | +2.90 ±0.03 | 100 |
| r9 | `faces_setA` | 0.00 | 69.86 ±0.02 | +3.41 ±0.03 | 100 |
| house_control_r1 | `faces_setA` | 0.00 | 71.18 ±0.03 | +2.69 ±0.03 | 100 |
| r7 | `objects` (64, trained) | 0.40 | 85.35 ±0.61 | +3.21 ±0.35 | 20 |
| r8 | `objects` (64, trained) | 0.40 | 84.75 ±0.51 | +2.93 ±0.38 | 20 |
| r8 | `objects` | 0.28 | 92.52 ±0.15 | +2.07 ±0.09 | 100 |
| r9 | `objects` | 0.09 | 93.77 ±0.14 | +1.77 ±0.08 | 100 |
| house_control_r1 | `objects` (128, trained) | 0.39 | 81.94 ±0.35 | +3.11 ±0.30 | 20 |
| house_control_r1 | `objects` | 0.24 | 91.36 ±0.13 | +2.00 ±0.09 | 100 |
| r7 | `houses_zubud` (201, **trained**) | 0.40 | 95.14 ±0.37 | +1.27 ±0.53 | 20 |
| r8 | `houses_zubud` (161, held-out) | 0.40 | 86.40 ±0.61 | +0.57 ±0.72 | 20 |
| r8 | `houses_zubud` (held-out) | 0.28 | 96.63 ±0.13 | +0.33 ±0.10 | 100 |
| r9 | `houses_zubud` (held-out) | 0.09 | 97.58 ±0.11 | +0.15 ±0.09 | 100 |
| house_control_r1 | `houses_zubud` (161 held-out, **corrected**) | 0.39 | 87.31 ±0.73 | +2.03 ±0.78 | 20 |
| house_control_r1 ⚠ | `houses_zubud` (⚠ scored on all 201, 40 trained-on) | 0.24 | 98.11 ±0.12 | +0.40 ±0.07 | 100 |

### 5a. Identity-count and ratio sweeps, `faces_mix64` (24 known + Set_A)

Identity-count sweep, pool grows from 24 to 49:

| Identity pool | 24 | 29 | 34 | 39 | 44 | 49 |
| --- | --- | --- | --- | --- | --- | --- |
| r8 upright (p=0.04) | 95.81 | 93.59 | 91.37 | 89.49 | 87.99 | 86.77 |
| r8 effect | +6.08 | +5.34 | +5.06 | +4.76 | +4.59 | +4.44 |
| house_control_r1 upright (p=0.09) | 96.01 | 94.25 | 92.03 | 90.29 | 88.63 | 87.34 |
| house_control_r1 effect | +6.26 | +5.76 | +5.45 | +5.14 | +5.03 | +4.90 |

Ratio sweep, pool fixed at 40, composition swept:

| Composition | 24+16 | 20+20 | 16+24 | 12+28 | 8+32 | 4+36 | 0+40 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r8 upright (p=0) | 89.33 | 86.97 | 84.36 | 81.92 | 78.41 | 75.02 | 71.19 |
| r8 effect | +4.78 | +4.53 | +4.26 | +4.12 | +3.71 | +3.35 | +2.83 |
| house_control_r1 upright (p=0) | 90.25 | 87.75 | 84.94 | 82.12 | 78.88 | 75.23 | 71.20 |
| house_control_r1 effect | +4.87 | +4.61 | +4.50 | +4.03 | +3.63 | +3.17 | +2.67 |

Both sweeps fall monotonically as Set_A identities are added, but they are
confounded with Set_A's stimulus quality (§6) and were run near the noise floor.
Refitted where the anchor is reachable (`ratio_hi`, p≈0.30–0.33) the 24-celeb
point reads +6.11 (r8), +6.26 (r5b) and +7.84 (house_control_r1) — the same
model ordering, with the effects roughly a third larger than the floor-pinned
sweep implied.

---

## 6. Bad and invalid results, recorded so they are not repeated

**`faces_setA` is a broken stimulus set.** Yin: r8 UU 95.42 with cost only
+4.86 ±0.90 (30 seeds, p=0); r9 93.75 / +4.58 ±0.78; house_control_r1 97.08 /
+6.67 ±0.95. Kanwisher: upright tops out at 69.86–71.25 across all models
against an 87.50 anchor even at zero noise, so p pins at the floor and nothing
is calibrated. Its photos of one identity barely resemble each other
(within-identity pixel corr +0.078 vs Set_E's +0.113) while every identity is a
young female celebrity, so between-identity variability is low too.

**`faces_k40u24` (40 known studied / 24 unknown distractors) is invalid.**
r8 reads UU 96.50 ±0.34 with cost +4.17 ±0.61 at p=0.36 — but the accuracy-vs-p
curve is flat at ~98.6% from p=0.00 to 0.30 before falling off a cliff, and the
reversed-role control `faces_u40k24` (40 unknown studied, 24 known distractors)
reads UU 63.92 ±0.88 with cost +25.04 ±1.08 at the same p. A 33-point swing from
flipping which side familiarity favours: the 2AFC is being won by recognising
trained identities, not by remembering the study episode. **Do not report the
`k40u24` number as a Yin result.**

**Objects can never be held out.** All packed object classes were trained on by
every model, so object accuracy is inflated by exposure and cannot reach the
84.79 anchor at any usable noise.

**`houses_ident` cannot be used for Kanwisher.** Its packed `valid` and `test`
splits are bit-identical (verified), so a triplet match would compare an image
with itself. Kanwisher houses must use ZuBuD, whose splits are genuinely
different views (view04 vs view05).

**r7 and r5b have no held-out ZuBuD.** Both trained on all 201 packed classes,
so the `--exclude-seen-from` rule leaves them zero items and the condition
scores a structural 0.0% rather than erroring. Both are now listed in
`EXCLUDE_SEEN_WAIVED`; their ZuBuD rows are **trained-on** and labelled as such.

**Cross-store Yin distractors are forbidden.** An earlier design drew study
items from Set_A and distractors from the `faces` store; studied *and unstudied*
Set_A both scored 90% against old-store distractors while the honest same-store
test sat at chance. Study and distractor pools must come from one store.

---

## 7. Reading the tables

**Calibration was fitted out of sample until 2026-08-20.** `p` was averaged over
8 item draws from base seed 42 while results were reported over seeds 101–120 —
different draws, so the fitted accuracy was not the reported accuracy. On a
24-identity store the gap ran 1–2.6 points; r8's `faces_celeb24` moved from
p=0.33 / UU 97.22 / cost +8.33 ±2.40 to p=0.30 / UU 96.33 ±0.61 / cost
+7.00 ±0.85 when fixed, and house_control_r1's fitted p moved 0.34 → ≈0.30. Rows
from `sim_seeds_celeb24`, `sim_seeds_mix64`, `sim_seeds_mixE64`,
`sim_seeds_ratio*` and the older `sim_seeds` sweeps carry this error and should
be treated as provisional. `--calib-base-seed`, `--calib-seeds` and
`--calib-on-eval-seeds` in `run_sim_seeds.py` now control it.

**Inversion cost depends on where you read it on the accuracy curve.** Holding
p fixed while a store gets harder manufactures a cost gradient out of an
accuracy gradient: on the Set_A growth sweep the cost appeared to climb from
+6.87 toward +12.63 at fixed p=0.30, but refitting p at every point so upright
stayed on 96.29 flattened it to +7.5 to +9.1. Compare **costs at matched upright
accuracy**, and treat any uncalibrated gradient with suspicion.

**`p` differs across rows** and is fitted per (model, experiment). Compare
inversion costs, not raw accuracies, across rows with different `p`.

**Where a fit pins at p=0** the row is a ceiling, not a calibration — the model
cannot reach the human anchor at any noise level. Check `yin_upright_at_p` in
the run's `noise_<model>.json` before reporting such a row.

**house_control_r1 is not a single-variable control** (see §Models). It trains on
the same house dataset and the same 40 buildings as r8, but differs in object
class count (128 vs 64), curriculum shape (2 stages vs 6) and LR schedule (none
vs cosine), so a difference between it and r8 cannot be attributed to any one of
them. For a clean curriculum contrast use r7 vs r5b; for a clean house-weight
contrast use r8 vs r9.

---

## 8. Summary

| Arm | Status | Best result |
| --- | --- | --- |
| **Kanwisher faces** | **matched, all 3 models** | r7 87.33 / +11.09 · r8 87.13 / +10.79 · house_control 87.33 / +12.17, `faces_mixed64`, p≈0.40 |
| **Kanwisher face-specificity** | **matched, all 3 models** | same runs: houses +0.6 to +1.3, objects +2.9 to +3.2 |
| **Yin faces** | **matched** | r8 `faces_mixu16` p=0.23: 95.75 / +12.17 |
| **Yin objects** | cost matched, accuracy inflated | +1.2 to +4.6 vs human −1.92; UU 11–14 pts high (all trained) |
| **Yin houses** | **not matched** | every model/dataset +8 to +27 vs human +2.63; best *held-out* accuracy is r9's 92.00 on 161 unseen ZuBuD buildings (house_control_r1's 97.08 includes 40 trained-on) |

The open question worth putting to a reader: **the same models get houses right
under Kanwisher (+0.15 to +1.27) and badly wrong under Yin (+8 to +27).** The
failure is specific to Yin's study/test memory design, not to the models' house
representations as such — which also rules out "the log-polar front end is
orientation-sensitive for buildings" as the explanation, since that would break
both tasks equally.
