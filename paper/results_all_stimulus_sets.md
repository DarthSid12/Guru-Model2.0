# Yin (1969) & Dobs/Kanwisher (2023): all stimulus sets

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


Every model row states **which stimulus set it was scored on**. Rows scored on different sets are not comparable to each other; compare within a stimulus set.

Model rows are means ± SEM over simulation seeds (`run_sim_seeds.py`); each seed redraws the study/test items. Retrieval noise `p` is calibrated so upright accuracy matches the human anchor, then held fixed across seeds.

## Stimulus sets

| Set | Contents | Seen in training? | Used for |
|---|---|---|---|
| `faces` | 128 celebrity identities | **YES — training classes** | earlier sweep (r7, r5b), 200 seeds |
| **Set_A** | 40 female-celebrity identities × 5 photos, 250×250 | no | Kanwisher 100 seeds; Yin spot-checks |
| **Set_E** | 64 identities × 10 photos | no | Yin + Kanwisher 100 seeds (primary) |
| objects | 64 ImageNet classes | **YES — training classes** | all runs |
| ZuBuD | 201 buildings × 5 exterior views | varies by model (see rows) | all runs |

**Set_A is unusable** — see §4. It is reported for completeness, not for conclusions.

---

## 1. Yin (1969) — study/test orientation matching

UU/II = same orientation at study and test; UI/IU = mismatched. Inversion cost = `UU − UI`, paired per seed.

### 1a. Faces

| Source | Stimulus set | p | UU | II | UI | IU | Inversion cost (UU−UI) | seeds |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** |  |
| LP-Net **curriculum** (r7, r18) | `faces` (128, **trained**) | 0.38 | 93.10 ±0.29 | 81.00 ±0.51 | 81.54 ±0.49 | 81.90 ±0.49 | **+11.56** ±0.55 | 200 |
| LP-Net **all-at-once** (r5b, r18) | `faces` (128, **trained**) | 0.20 | 96.10 ±0.25 | 86.98 ±0.43 | 85.48 ±0.46 | 81.31 ±0.48 | **+10.62** ±0.45 | 200 |
| LP-Net **developmental** (r8, r18) | **Set_E** (64 novel) | 0.00 | 96.29 ±0.31 | 95.46 ±0.32 | 72.63 ±0.67 | 80.38 ±0.58 | **+23.67** ±0.76 | 100 |
| LP-Net **developmental+weighted** (r9, r18) | **Set_E** (64 novel) | 0.00 | 92.58 ±0.43 | 96.17 ±0.35 | 68.58 ±0.70 | 68.96 ±0.77 | **+24.00** ±0.75 | 100 |
| **David's model** (curriculum, r18)† | **Set_E** (64 novel) | 0.04 | 96.83 ±0.30 | 98.54 ±0.23 | 78.37 ±0.65 | 83.58 ±0.66 | **+18.46** ±0.65 | 100 |
| LP-Net **curriculum** (r7, r18) | **Set_A** (40 novel) | 0.00 | 68.89 | 66.67 | 44.44 | 42.22 | **+24.45** | 3 |
| LP-Net **curriculum** (r7, r18) | **Set_A** (40 novel) | 0.20 | 53.33 | 53.33 | 40.00 | 42.22 | **+13.33** | 3 |
| LP-Net **curriculum** (r7, r18) | **Set_E** (64 novel) | 0.00 | 97.22 | 91.67 | 80.56 | 73.61 | **+16.67** | 3 |
| LP-Net **curriculum** (r7, r18) | **Set_E** (64 novel) | 0.20 | 94.44 | 87.50 | 72.22 | 77.78 | **+22.22** | 3 |
| LP-Net **all-at-once** (r5b, r18) | **Set_A** (40 novel) | 0.00 | 48.89 | 40.00 | 28.89 | 31.11 | **+20.00** | 3 |
| LP-Net **all-at-once** (r5b, r18) | **Set_A** (40 novel) | 0.20 | 53.33 | 35.56 | 24.44 | 31.11 | **+28.89** | 3 |
| LP-Net **all-at-once** (r5b, r18) | **Set_E** (64 novel) | 0.00 | 84.72 | 86.11 | 66.67 | 56.94 | **+18.05** | 3 |
| LP-Net **all-at-once** (r5b, r18) | **Set_E** (64 novel) | 0.20 | 76.39 | 77.78 | 65.28 | 52.78 | **+11.11** | 3 |
| LP-Net **developmental** (r8, r18) | **Set_E** (64 novel) | 0.20 | 91.67 | 90.28 | 73.61 | 77.78 | **+18.06** | 3 |
| LP-Net **developmental** (r8, r18) | **Set_E** (64 novel) | 0.30 | 86.11 | 84.72 | 65.28 | 77.78 | **+20.83** | 3 |
| LP-Net **developmental+weighted** (r9, r18) | **Set_E** (64 novel) | 0.20 | 87.50 | 90.28 | 70.83 | 68.06 | **+16.67** | 3 |
| LP-Net **developmental+weighted** (r9, r18) | **Set_E** (64 novel) | 0.30 | 77.78 | 83.33 | 62.50 | 62.50 | **+15.28** | 3 |
| LP-Net **developmental** (r8, r18) | **Set_A** (40 novel) | 0.00 | 61.67 | — | — | — | — | 8 |
| LP-Net **developmental+weighted** (r9, r18) | **Set_A** (40 novel) | 0.00 | 66.67 | — | — | — | — | 8 |
| **David's model** (curriculum, r18)† | **Set_A** (40 novel) | 0.00 | 68.33 | — | — | — | — | 8 |

*Set_A rows use a 25 study / 15 test design (only 40 identities available), not the standard 40/24 — so they are not directly comparable to Set_E rows even within a model.*

### 1b. Objects

| Source | Stimulus set | p | UU | II | UI | IU | Inversion cost (UU−UI) | seeds |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 84.79 | 83.96 | 86.71 | 82.75 | **-1.92** |  |
| LP-Net **curriculum** (r7, r18) | objects (64, **trained**) | 0.43 | 72.62 ±0.59 | 69.29 ±0.68 | 71.21 ±0.65 | 70.48 ±0.71 | **+1.42** ±0.81 | 200 |
| LP-Net **all-at-once** (r5b, r18) | objects (64, **trained**) | 0.00 | 82.06 ±0.40 | 74.52 ±0.37 | 73.04 ±0.41 | 75.96 ±0.41 | **+9.02** ±0.32 | 200 |
| LP-Net **developmental** (r8, r18) | objects (64, **trained**) | 0.00 | 98.29 ±0.25 | 99.17 ±0.18 | 97.12 ±0.28 | 97.00 ±0.32 | **+1.17** ±0.26 | 100 |
| LP-Net **developmental+weighted** (r9, r18) | objects (64, **trained**) | 0.00 | 98.12 ±0.23 | 99.50 ±0.14 | 93.54 ±0.37 | 96.87 ±0.29 | **+4.58** ±0.33 | 100 |
| **David's model** (curriculum, r18)† | objects (64, **trained**) | 0.04 | 99.33 ±0.15 | 99.50 ±0.14 | 97.67 ±0.27 | 98.21 ±0.23 | **+1.67** ±0.25 | 100 |

### 1c. Houses

| Source | Stimulus set | p | UU | II | UI | IU | Inversion cost (UU−UI) | seeds |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** |  |
| LP-Net **curriculum** (r7, r18) | ZuBuD (201, **trained**) | 0.32 | 90.29 ±0.40 | 82.00 ±0.46 | 79.08 ±0.50 | 81.33 ±0.51 | **+11.21** ±0.47 | 200 |
| LP-Net **all-at-once** (r5b, r18) | ZuBuD (201, **trained**) | 0.29 | 90.79 ±0.35 | 81.92 ±0.42 | 80.71 ±0.46 | 83.06 ±0.43 | **+10.08** ±0.46 | 200 |
| LP-Net **developmental** (r8, r18) | ZuBuD (161 **held-out**) | 0.00 | 84.83 ±0.60 | 81.46 ±0.64 | 73.87 ±0.71 | 74.33 ±0.75 | **+10.96** ±0.64 | 100 |
| LP-Net **developmental+weighted** (r9, r18) | ZuBuD (161 **held-out**) | 0.00 | 92.00 ±0.43 | 95.50 ±0.33 | 81.25 ±0.55 | 78.87 ±0.60 | **+10.75** ±0.61 | 100 |
| **David's model** (curriculum, r18)† ⚠ | ZuBuD (scored on all 201; 40 trained-on) | 0.04 | 97.08 ±0.27 | 95.00 ±0.37 | 86.50 ±0.65 | 83.83 ±0.65 | **+10.58** ±0.61 | 100 |

---

## 2. Dobs / Kanwisher (2023) — upright vs. inverted, faces only

Three-image identity matching, no memory phase, ~156k triplets per run — hence SEMs ~5× tighter than Yin's.

| Source | Stimulus set | p | Upright | Inverted | Inversion effect | seeds |
|---|---|---|---|---|---|---|
| **Human (between-subjects, n=1,532/1,219)** | — | — | 87.50 | 76.80 | **+10.70** |  |
| **Human (within-subject, n=364)** | — | — | 87.50 | 75.90 | **+11.60** |  |
| **Dobs et al. Face-ID CNN (Fig. 3B)** | — | — | 86.90 | 66.40 | **+20.50** |  |
| LP-Net **curriculum** (r7, r18) | `faces` (128, **trained**) | 0.43 | 85.44 ±0.10 | 70.81 ±0.13 | **+14.63** ±0.13 | 200 |
| LP-Net **all-at-once** (r5b, r18) | `faces` (128, **trained**) | 0.42 | 83.45 ±0.09 | 70.97 ±0.12 | **+12.48** ±0.12 | 200 |
| LP-Net **developmental** (r8, r18) | **Set_E** (64 novel) | 0.28 | 87.76 ±0.11 | 81.31 ±0.13 | **+6.45** ±0.10 | 100 |
| LP-Net **developmental+weighted** (r9, r18) | **Set_E** (64 novel) | 0.09 | 88.98 ±0.11 | 83.06 ±0.13 | **+5.92** ±0.11 | 100 |
| **David's model** (curriculum, r18)† | **Set_E** (64 novel) | 0.24 | 88.01 ±0.11 | 80.55 ±0.16 | **+7.46** ±0.12 | 100 |
| LP-Net **developmental** (r8, r18) | **Set_A** (40 novel) | 0.00 | 71.25 ±0.02 | 68.35 ±0.03 | **+2.90** ±0.03 | 100 |
| LP-Net **developmental+weighted** (r9, r18) | **Set_A** (40 novel) | 0.00 | 69.86 ±0.02 | 66.46 ±0.03 | **+3.41** ±0.03 | 100 |
| **David's model** (curriculum, r18)† | **Set_A** (40 novel) | 0.00 | 71.18 ±0.03 | 68.49 ±0.03 | **+2.69** ±0.03 | 100 |

Kanwisher objects and houses (Set_E sweep, 100 seeds), for the face-specific contrast:

| Model | Category | Stimulus set | p | Upright | Inverted | Inversion effect |
|---|---|---|---|---|---|---|
| LP-Net **developmental** (r8, r18) | objects | objects (64, **trained**) | 0.28 | 92.52 ±0.15 | 90.45 ±0.18 | **+2.07** ±0.09 |
| LP-Net **developmental** (r8, r18) | houses_zubud | ZuBuD (161 **held-out**) | 0.28 | 96.63 ±0.13 | 96.29 ±0.15 | **+0.33** ±0.10 |
| LP-Net **developmental+weighted** (r9, r18) | objects | objects (64, **trained**) | 0.09 | 93.77 ±0.14 | 92.00 ±0.16 | **+1.77** ±0.08 |
| LP-Net **developmental+weighted** (r9, r18) | houses_zubud | ZuBuD (161 **held-out**) | 0.09 | 97.58 ±0.11 | 97.43 ±0.11 | **+0.15** ±0.09 |
| **David's model** (curriculum, r18)† | objects | objects (64, **trained**) | 0.24 | 91.36 ±0.13 | 89.36 ±0.16 | **+2.00** ±0.09 |
| **David's model** (curriculum, r18)† ⚠ | houses_zubud | ZuBuD (scored on all 201; 40 trained-on) | 0.24 | 98.11 ±0.12 | 97.72 ±0.12 | **+0.40** ±0.07 |

---

## 3. Caveats

**1. Face stimuli differ between blocks — the single biggest confound.** The earlier r7/r5b sweep scored faces on the 128 identities those models *trained on*; the r8/r9/David rows use novel Set_E. Trained identities give the model identity-specific, orientation-tuned features that inversion disrupts, so the earlier Yin costs (+12.10, +9.12) are probably inflated relative to a held-out test. **Do not read the drop from r7's +12.10 to r8's +0.83 as a model effect** — the stimuli changed too. The 3-seed spot-checks of r7/r5b *on Set_E* are the only same-stimulus old-vs-new comparison here, and they suggest r7 retains a face inversion cost (+5.6) where r8/r9 do not.

**2. Inversion cost here is `UU − UI`** — study upright, then test in the same vs the opposite orientation. Note this is a *different* contrast from `UU − II`: on Set_E the models show essentially **no** UU−II cost (II tracks UU within a few points, sometimes above it) because a consistently inverted study+test pair is just as matchable as an upright one. The orientation cost lands almost entirely on the **mismatched** conditions, which is what UU−UI measures.

**3. Objects are never held out.** All 64 packed object classes were trained on by every model, so object accuracy is inflated by exposure and cannot reach the 84.79% human anchor at any usable noise.

**4. Set_A is a broken stimulus set.** Same-store Yin at chance (48.9–68.9% across five models) and Kanwisher ~70% against an 87.5% anchor. Its photos of one person barely resemble each other (within-identity pixel corr +0.078 vs Set_E's +0.113) while all identities are demographically similar, so between-identity variability is low too. Not a resolution effect — Set_E has *more* sub-224px images.

**5. "David's model" is not a clean control.** It is a curriculum model like r8 and trains on the **same house dataset and the same 40 ZuBuD buildings**, but differs in three further ways: **128 object classes** (vs 64), **two curriculum stages** (vs six), and **no LR schedule** (vs cosine). Its inversion effects therefore cannot be attributed to any single design difference. For a clean curriculum vs no-curriculum contrast use **r7 vs r5b**, which differ only in that; for a clean house-diet contrast use **r8 vs r9**.

† **David's model is a curriculum model**, on a two-stage schedule: 2 epochs of 256 classes (all 128 faces + all 128 objects, no houses) then 29 epochs of all 296 with the 40 houses added — `stage_spec` `all-prehouse` → `all` in its training history. The `curriculum: False` in its `config.json` (and the absence of `stage*.pth` checkpoints) refers only to the class-count ladder r8/r9 use; its house delay runs off `houses_delay_epochs: 2` instead. It differs from r8 in object-class count (128 vs 64), curriculum shape (2 stages vs 6) and LR schedule (none vs cosine) — but **not** in house dataset: both train on 40 ZuBuD buildings.

**6. `p` differs across rows** and is calibrated per model so upright accuracy matches the human anchor. On Set_E, Yin pins at p≈0 (models only just reach 96.29% unaided) while Kanwisher retains headroom (p = 0.09–0.28). Compare inversion **costs**, not raw accuracies, across rows with different `p`.

---

## 4. Excluded runs

Recorded so they are not silently repeated:

- **Yin faces_setA with cross-store distractors** (`runs/sim_seeds/yin_scan/p0.3{2,4,6,8}`, `calibrated_p_partial/`, `fixed_p03_partial/`). Study items came from Set_A and the 'new' distractors from the `faces` store, so the 2AFC was won on **dataset appearance**, not memory: studied Set_A vs old-store scored 90%, and *unstudied* Set_A vs old-store also scored 90%, while the honest same-store test sat at 45% (chance). Symptom was upright accuracy flat at 95.83% across every noise level. **All faces numbers from these runs are invalid.** Objects and houses in those same files are same-store and remain valid.

- **Kanwisher faces_setA (100 seeds)** is reported above but should not be used for conclusions — it is 16–18 points below the human anchor for the reasons in §4 of the caveats.

---

Generated by `paper/make_setE_summary.py`. Earlier 200-seed rows transcribed from [curriculum_results_summary.md](curriculum_results_summary.md).
