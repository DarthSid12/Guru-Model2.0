# r10, r11, r12 — every Yin and Kanwisher result, against the human anchors

Round launched 2026-08-23, all simulations complete 2026-08-24 02:56. Three
developmental models trained concurrently on GPUs 0/1/2 via
`scripts/run_model.sh {r10|r11|r12} GPU`. Training finished 12:19 (~9.5 h);
all three reached 80/80 epochs with no early stopping.

Every number in this document was regenerated from the raw per-seed CSVs, not
copied from run logs. Sources:

| Simulation | Preset | Directory | Seeds |
| --- | --- | --- | --- |
| Yin, arm 1 | `joint_zubud` | `runs/sim_seeds_joint_zubud/` | 100 |
| Yin, arm 2 | `joint_zubud_mixu16` | `runs/sim_seeds_joint_zubud_mixu16/` | 100 |
| Kanwisher | `kanw3` | `runs/sim_seeds_kanw3/` | 20 |

All cells are complete: 3 categories × 4 orientation conditions × 100 seeds =
1200 rows per model per Yin arm; 3 × 2 × 20 = 120 rows per model for Kanwisher.

---

## Models

All three share: `resnet18`, `lp` (log-polar) variant, 16 fixations, batch 256,
lr 1e-3 cosine, weight decay 0.05, dropout 0.3, seed 42, `--patience 40`,
curriculum epochs `6/6/8/8/12/40` (**80 total**), 128 faces + 64 objects +
**80** individuated ZuBuD buildings (the remaining 121 held out for the sims).

They differ only in the house arm:

| key | run dir | house arm | classes |
| --- | --- | --- | --- |
| `r10_dev_h80` | `runs/r10_dev_h80` | **ladder**: house classes grow `0/0/8/16/40/80` across stages | 272 |
| `r11_dev_h80_generic` | `runs/r11_dev_h80_generic` | same ladder **plus** a generic one-class `houses` category present from epoch 1 (basic-level houseness before any individuation) | 273 |
| `r12_dev_h80_enbloc` | `runs/r12_dev_h80_enbloc` | houses withheld entirely, **all 80 arrive at once in stage 6** | 272 |

Category weights: r10 and r12 `faces=0.40 objects=0.40 houses_zubud=0.20`;
r11 `faces=0.35 objects=0.35 houses_zubud=0.20 houses=0.10`.

Houses are **20%** of every batch in all three (house_control_r1's proportion,
not r9's 10%). Because they train under the category name `houses_zubud`,
`EXCLUDE_SEEN` in `run_sim_seeds.py` holds out the 121 unseen buildings
automatically — no `LABELMAP_CATEGORY` entry needed.

**r10 vs r12 is the intended single-variable contrast**: gradual individuation
versus late en-bloc introduction. See §6 for why it is not readable as measured.

---

## Human anchors

| Simulation | Faces | Objects | Houses |
| --- | --- | --- | --- |
| Yin (1969) UU | **96.29** | **84.79** | **90.71** |
| Yin cost (UU − UI) | **+12.16** | **−1.92** | **+2.63** |
| Yin (UU − II), for reference | **+14.41** | **+0.83** | **+4.96** |
| Dobs/Kanwisher (2023) upright | **87.50** | — | — |
| Dobs/Kanwisher inversion effect | **+10.70** between-subj / **+11.60** within-subj | — | — |

UU/II = same orientation at study and test; UI/IU = mismatched. The inversion
cost reported throughout is **UU − UI, paired per seed**, matching every earlier
table in this project. Yin's Exp. II puts the orientation change between study
and test, so the cost lands on the mismatched conditions; UU − II is printed for
reference only. ± values are standard errors of the mean across seeds.

---

## 1. Yin, arm 1 — `joint_zubud` (faces_mixed64), houses held out, n=100

Bold marks the headline house costs. `p` is the jointly fitted noise level.

| model | p | faces UU (96.29) | faces cost (+12.16) | houses UU (90.71) | houses cost (+2.63) | objects UU (84.79) | objects cost (−1.92) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **r10** ladder | 0.01 | 86.50 ±0.64 | +11.00 ±0.76 | **91.38 ±0.50** | **+15.00 ±0.66** | 96.50 ±0.31 | +2.00 ±0.32 |
| **r11** generic | 0.01 | 88.25 ±0.64 | +10.54 ±0.72 | **89.71 ±0.49** | **+8.50 ±0.67** | 97.79 ±0.29 | **+0.46 ±0.24** |
| **r12** enbloc | 0.13 | 88.62 ±0.53 | +11.71 ±0.75 | **90.75 ±0.46** | **+10.75 ±0.69** | 97.46 ±0.29 | +3.62 ±0.39 |
| r8 (prior round) | 0.00 | 92.12 ±0.53 | +12.62 ±0.68 | 84.83 ±0.60 | +10.96 ±0.64 | 98.29 ±0.25 | +1.17 ±0.26 |
| r7 (prior round) | 0.30 | 86.25 ±0.61 | +12.04 ±0.69 | 91.38 ±0.50 | +10.38 ±0.59 | 98.04 ±0.25 | +2.46 ±0.37 |
| house_control_r1 | 0.26 | 89.25 ±0.62 | +14.00 ±0.78 | 90.04 ±0.52 | +8.38 ±0.76 | 98.83 ±0.21 | +3.00 ±0.38 |

All four orientation cells, same run:

| model | category | UU | II | UI | IU |
| --- | --- | --- | --- | --- | --- |
| r10 | faces_mixed64 | 86.50 | 89.62 | 75.50 | 76.46 |
| r10 | houses_zubud | 91.38 | 94.58 | 76.38 | 80.71 |
| r10 | objects | 96.50 | 97.71 | 94.50 | 95.79 |
| r11 | faces_mixed64 | 88.25 | 89.37 | 77.71 | 75.58 |
| r11 | houses_zubud | 89.71 | 91.21 | 81.21 | 79.13 |
| r11 | objects | 97.79 | 99.42 | 97.33 | 96.96 |
| r12 | faces_mixed64 | 88.62 | 88.04 | 76.92 | 75.96 |
| r12 | houses_zubud | 90.75 | 89.50 | 80.00 | 81.79 |
| r12 | objects | 97.46 | 98.00 | 93.83 | 95.25 |

---

## 2. Yin, arm 2 — `joint_zubud_mixu16`, the calibrated face arm, n=100

Arm 1 pinned the joint fit near p=0 for all three because `faces_mixed64` tops
out at 88–90% upright, ~8 pts under the 96.29 anchor. `faces_mixu16` (48 known
+ 8 Set_E + 8 Set_A) is the Step 10 rung where faces actually reach the anchor,
so this arm is the one to quote for face numbers.

| model | p | faces UU (96.29) | faces cost (+12.16) | houses UU (90.71) | houses cost (+2.63) | objects UU (84.79) | objects cost (−1.92) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **r10** ladder | 0.06 | 93.12 ±0.50 | +8.50 ±0.76 | 91.04 ±0.50 | **+15.21 ±0.71** | 96.29 ±0.33 | +1.67 ±0.30 |
| **r11** generic | 0.00 | 93.04 ±0.52 | +9.92 ±0.66 | 89.79 ±0.50 | **+8.67 ±0.67** | 97.83 ±0.29 | **+0.38 ±0.24** |
| **r12** enbloc | 0.13 | 93.88 ±0.45 | +9.79 ±0.67 | 90.75 ±0.46 | +10.75 ±0.69 | 97.46 ±0.29 | +3.62 ±0.39 |

All four cells:

| model | category | UU | II | UI | IU |
| --- | --- | --- | --- | --- | --- |
| r10 | faces_mixu16 | 93.12 | 93.21 | 84.63 | 82.21 |
| r10 | houses_zubud | 91.04 | 93.46 | 75.83 | 79.83 |
| r10 | objects | 96.29 | 97.58 | 94.62 | 95.71 |
| r11 | faces_mixu16 | 93.04 | 92.12 | 83.13 | 82.38 |
| r11 | houses_zubud | 89.79 | 91.46 | 81.12 | 78.96 |
| r11 | objects | 97.83 | 99.37 | 97.46 | 97.08 |
| r12 | faces_mixu16 | 93.88 | 92.21 | 84.08 | 83.46 |
| r12 | houses_zubud | 90.75 | 89.50 | 80.00 | 81.79 |
| r12 | objects | 97.46 | 98.00 | 93.83 | 95.25 |

Two notes on reading this arm:

- **Face costs read 2–4 pts BELOW human here** (+8.50/+9.92/+9.79 vs +12.16),
  where arm 1 read close to human. This is expected per Step 10: `mixu16` is a
  *more familiar* rung than `mixed64`, and the cost peaks around 16–24 unknown
  identities. It is a different point on the familiarity ladder, not a regression.
- **r12's house and object rows are numerically identical across both arms**
  because its joint fit landed on p=0.13 in both, so the same seeds at the same
  noise reproduce exactly. r10 (0.01 → 0.06) and r11 (0.01 → 0.00) do move.

**The house ranking is unchanged between arms — r11 best (+8.50 / +8.67), r12
middle (+10.75), r10 worst (+15.00 / +15.21).** That it survives a change of
which face store calibrates `p` is the main reason to trust it: it is a property
of these models' houses, not an artifact of the `mixed64` ceiling.

---

## 3. Kanwisher `kanw3`, n=20

| model | p | faces upright (87.50) | faces effect (+10.70/+11.60) | houses upright | houses effect | objects upright | objects effect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **r10** ladder | 0.39 | **87.31 ±0.33** | +9.94 ±0.32 | 84.65 ±0.51 | +0.07 ±0.77 | 87.01 ±0.42 | +2.95 ±0.28 |
| **r11** generic | 0.40 | 85.50 ±0.35 | +13.49 ±0.37 | 80.90 ±0.55 | +2.43 ±0.90 | 85.85 ±0.50 | +3.57 ±0.38 |
| **r12** enbloc | 0.39 | **87.51 ±0.31** | **+10.16 ±0.23** | 85.11 ±0.44 | +0.14 ±0.88 | 87.17 ±0.46 | +3.34 ±0.29 |
| r7 | 0.40 | 87.33 ±0.30 | +11.09 ±0.30 | 95.14 ±0.37 | +1.27 ±0.53 | 85.35 ±0.61 | +3.21 ±0.35 |
| r8 | 0.40 | 87.13 ±0.32 | +10.79 ±0.38 | 86.40 ±0.61 | +0.57 ±0.72 | 84.75 ±0.51 | +2.93 ±0.38 |
| r9 | 0.40 | 87.08 ±0.36 | +11.15 ±0.38 | 82.36 ±0.73 | +0.90 ±0.74 | 84.89 ±0.34 | +3.22 ±0.31 |
| house_control_r1 | 0.39 | 87.33 ±0.35 | +12.17 ±0.26 | 88.85 ±0.72 | +0.71 ±0.86 | 81.94 ±0.35 | +3.11 ±0.30 |

**r12 lands on the anchor on both axes** (87.51 upright, +10.16 effect).
r11 is the weakest face model in the round on this task: it buys its house
result with the worst Kanwisher faces (85.50, +13.49 — over-inverted).

Houses are essentially flat under Kanwisher in all three (+0.07 / +2.43 / +0.14).

---

## 4. Training-time accuracy and inversion effect

Final epoch, **trained** buildings, n=80 so ±1.25 pt. This is classification,
not the Yin memory task — it is not comparable to the tables above.

| model | faces | objects | houses_zubud | generic `houses` |
| --- | --- | --- | --- | --- |
| r10 | 91.80 | 75.32 | 97.50 | — |
| r11 | 90.95 | 75.50 | 93.75 | 95.00 |
| r12 | 91.09 | 75.18 | 97.50 | — |

Upright − inverted at training time:

| model | faces | objects | houses_zubud | generic `houses` |
| --- | --- | --- | --- | --- |
| r8 (prior) | +20.5 | — | +30.0 | — |
| r10 | +17.04 | +5.42 | +25.00 | — |
| **r11** | **+18.67** | +5.49 | **+15.00** | +32.00 |
| r12 | +17.33 | +5.35 | +21.25 | — |

**r11 is the only model where faces > houses at training time**, which is the
human ordering. Figure: `paper/fig_r10_r11_r12_inversion.png`.

---

## 5. What the round establishes

1. **House accuracy is solved.** All three land within ~1 pt of the 90.71 anchor
   on held-out buildings (91.38 / 89.71 / 90.75). r8 was 84.83 and pinned at
   p=0. 80 identities plus the 20% batch share does it, without r9's
   object-costing diet change. All three now accept a real joint `p`, which
   before this round only house_control_r1 could.
2. **House cost is not solved.** The best is r11 at +8.50 against human +2.63 —
   still >3× the anchor. No configuration in this round produces human-like
   house orientation tolerance.
3. **r11 (generic house category) is the only model with the human ordering**
   faces > houses > objects (+10.54 / +8.50 / +0.46), and it posts the best
   object cost in the project. It pays with the worst Kanwisher faces
   (85.50 / +13.49).
4. **r12 is the best face model**: Kanwisher 87.51 / +10.16, on the anchor on
   both axes, with human-level house accuracy 90.75 — but house cost +10.75.
5. **Sharpest Yin/Kanwisher dissociation yet.** r10's houses cost +15.00 under
   Yin and +0.07 under Kanwisher — same model, same held-out buildings, same
   checkpoint. Whatever the Yin task is measuring in houses, Kanwisher's task
   does not see it at all.

---

## 6. Caveats that block interpretation

- **r10 vs r12 is NOT readable as measured.** The joint fit put them at p=0.01
  vs p=0.13 in arm 1 and 0.06 vs 0.13 in arm 2, so their house costs sit at
  different task difficulties. The gradual-vs-en-bloc question — the reason the
  round exists — needs a matched-`p` rerun in the style of Step 10's matched
  p=0.30 table. **Do not claim en-bloc beats the ladder from these numbers.**
  This rerun has not been done.
- **No model reproduces the UU − II contrast.** Human UU − II is +14.41 for
  faces; r10 is **−3.12**, r11 −1.12, r12 +0.58 (r7 +5.71 and house_control_r1
  +6.92 do better). At the near-zero `p` that arm 1 fits, the II condition
  actually beats UU for r10 on both faces and houses. The round's face results
  rest entirely on UU − UI; the second contrast in Yin's own data does not
  replicate here. This is not tracked in earlier summaries and warrants a look.
- **Each model differs from r8 in several ways at once** (house count 40→80,
  batch share 10%→20%, 50→80 epochs). Only the internal r10/r11/r12 contrasts
  are clean; r8 comparisons are directional at best.
- **Cosine LR was held identical across all three**, so r12's houses arrive at
  epoch 41 with LR ~5e-4 already decaying to 0. r12 did not come out flat, so
  the pre-registered `r12b --lr-schedule none` fallback was not triggered — but
  the LR confound remains in any r10-vs-r12 comparison.
- **Face accuracy in arm 1 is ~8 pts under the anchor** for all three, because
  `faces_mixed64` tops out at 88–90%. Quote arm 2 for face accuracy.

---

## 7. Outstanding work

1. **The matched-`p` r10-vs-r12 rerun.** The round's headline question is
   formally unanswered without it.
2. **Investigate the UU − II failure** — whether it is a property of these
   models or of how the Yin simulation constructs the II condition.
3. Nothing else is pending: 3 models trained, 2 Yin arms, Kanwisher, training
   figures, all completed 2026-08-23/24 in one overnight-plus-day window with
   no manual intervention after launch.
