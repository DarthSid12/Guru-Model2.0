# Celeb-identity simulations — r5b, r8 developmental, house control

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


20 seeds (101–120) per model per experiment. Models:

| key | run | in earlier write-ups |
| --- | --- | --- |
| `r5b_allatonce` | `runs/…_resnet18_r5b_houses_zubud_30ep` | LP-Net all-at-once |
| `r8_developmental` | `runs/…_resnet18_r8_developmental` | LP-Net developmental |
| `house_control_r1` | `Guru-Model2.0/runs/…_resnet18_house_control_r1` | "David's model" |

The 24 identities are the requested list (Adele … Tilda Swinton), packed as
`celeb_*` from the held-out `valid` split of the 128-identity training store.
**All three models trained on these 24 identities** — the house control differs
only in its house training, not its face training. Stores are built by
`make_faces_mix64.py --preset {celeb24,mixE64}`; sweeps by
`run_sim_seeds.py --experiment {celeb24,mixE64,ratio}`; tables by
`paper/aggregate_celeb_experiments.py`.

**Inversion cost = `UU − UI`, paired per seed** — the contrast the other results
tables use. A consistently inverted study+test pair is about as matchable as an
upright one, so the orientation cost lands on the *mismatched* conditions;
`UU − II` is reported alongside only for reference and is not the informative
contrast (see `results_faces_mix64.md` §1d).

Human anchors: Yin (1969) Tables 1–2, accuracy = (24 − mean errors) × 100 / 24;
Dobs et al. (2023) Exp. 5. Retrieval noise `p` is fitted once per (model, sim)
and held fixed across seeds and sweep points.

---

## 1. Yin — 24 celeb identities only

24 items cannot fill Yin's 40/24 design, so the sim scales to **study 15 / test
9**, the same 5:3 ratio. Nine pairs = 11.1 points per pair; read the SEM.

Calibration is clean here, and this is the first store where it is: every model
lands at **95.8%** UU against Yin's 96.29% anchor, at a real noise level
(p = 0.17 / 0.33 / 0.34) rather than pinned at the floor.

| Source | p | UU | II | UI | IU | Inversion cost (UU−UI) |
| --- | --- | --- | --- | --- | --- | --- |
| **Human (Yin 1969, faces)** | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** |
| r5b_allatonce | 0.17 | 98.3 ±0.9 | 87.2 ±1.7 | 82.8 ±2.2 | 86.7 ±1.7 | **+15.55** ±2.19 |
| r8_developmental | 0.33 | 97.2 ±1.1 | 91.1 ±2.1 | 88.9 ±2.0 | 86.1 ±2.1 | **+8.33** ±2.40 |
| house_control_r1 | 0.34 | 97.2 ±1.1 | 80.0 ±3.0 | 82.8 ±3.4 | 81.7 ±2.3 | **+14.44** ±3.61 |

`UU − II` for reference: human +14.41; r5b +11.11 ±1.61, developmental
+6.11 ±2.61, house control +17.22 ±3.07.

Upright accuracy matches the human anchor and the inversion costs bracket it:
r5b and the house control run a few points above +12.16, the developmental model
a few below, all within ~1.5 SEM of the human value. This is the closest any of
our stores has come to Yin's faces result on both axes at once.

## 2. Yin — 24 celeb + 40 Set_E

Same design, 64 items, study 40 / test 24. Noise fits at p = 0.03–0.04: the 40
unfamiliar identities cost enough accuracy that there is no headroom left to
calibrate with, so this is a near-floor calibration.

| Source | p | UU | II | UI | IU | Inversion cost (UU−UI) |
| --- | --- | --- | --- | --- | --- | --- |
| **Human (Yin 1969, faces)** | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** |
| r5b_allatonce | 0.04 | 87.9 ±1.0 | 82.1 ±1.7 | 65.4 ±2.3 | 66.5 ±2.3 | **+22.50** ±2.29 |
| r8_developmental | 0.04 | 91.5 ±1.3 | 90.0 ±1.1 | 75.0 ±2.5 | 72.9 ±2.4 | **+16.46** ±2.45 |
| house_control_r1 | 0.03 | 94.0 ±1.1 | 93.5 ±1.4 | 76.9 ±2.7 | 78.1 ±2.3 | **+17.08** ±2.32 |

`UU − II` for reference: human +14.41; r5b +5.83 ±1.55, developmental
+1.46 ±1.36, house control +0.42 ±1.31.

All three inversion costs **exceed** the human +12.16, in the same model order
as experiment 1 (r5b largest, developmental smallest). The near-zero `UU − II`
values are the expected signature described above — matched-orientation pairs
stay matchable — and must not be read as an absent inversion effect.

## 3. Dobs/Kanwisher — familiar/unfamiliar ratio at a fixed pool of 40

Pool size held at 40 throughout, composition swept 24 celeb + 16 Set_A → 0 + 40.
`p` fitted at the all-unfamiliar end (the condition matching Dobs et al.'s human
anchor, who matched faces they had never seen); no model reaches 87.5% there, so
`p` pins at ~0 and every point runs at essentially zero noise.

| Source | Upright | Inverted | Inversion effect |
| --- | --- | --- | --- |
| **Human (between-subjects, n=1,532/1,219)** | 87.50 | 76.80 | **+10.70** |
| **Human (within-subject, n=364)** | 87.50 | 75.90 | **+11.60** |
| **Dobs et al. Face-ID CNN (Fig. 3B)** | 86.90 | 66.40 | **+20.50** |

| n_celeb | r5b U / I | r8_dev U / I | house_ctrl U / I |
| --- | --- | --- | --- |
| 24 | 90.7 / 85.9 | 89.3 / 84.5 | 90.2 / 85.4 |
| 20 | 88.4 / 83.8 | 87.0 / 82.4 | 87.7 / 83.1 |
| 16 | 85.8 / 81.5 | 84.4 / 80.1 | 84.9 / 80.4 |
| 12 | 83.1 / 78.9 | 81.9 / 77.8 | 82.1 / 78.1 |
| 8 | 79.6 / 75.8 | 78.4 / 74.7 | 78.9 / 75.3 |
| 4 | 76.1 / 72.7 | 75.0 / 71.7 | 75.2 / 72.1 |
| 0 | 72.3 / 69.4 | 71.2 / 68.4 | 71.2 / 68.5 |

Inversion effect (upright − inverted), points — human +10.70 / +11.60:

| model | 24 | 20 | 16 | 12 | 8 | 4 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r5b_allatonce | 4.84 | 4.63 | 4.31 | 4.22 | 3.76 | 3.39 | 2.90 |
| r8_developmental | 4.78 | 4.53 | 4.26 | 4.12 | 3.71 | 3.35 | 2.83 |
| house_control_r1 | 4.87 | 4.61 | 4.50 | 4.03 | 3.63 | 3.17 | 2.67 |

Both accuracy and the inversion effect fall monotonically as celeb identities
are replaced (~18 points of accuracy, ~2 points of effect, SEMs 0.1–0.2). Every
point sits **below both human effects** at every composition — the same shortfall
the identity-count sweep showed.

**The three models are indistinguishable** (within ~1.5 points everywhere). That
is the important negative result: this sweep cannot separate them, because all
three trained on the same 128 CelebA identities, so familiarity with the celeb
half is identical across models. What the sweep varies is *which stimulus set*
the pool is drawn from, and Set_A is simply harder (its photos of one identity
barely resemble each other). Attributing the gradient to familiarity would need
a model that never saw the celeb identities.

### 3b. The same task, properly calibrated (24 and 20 celeb only)

The sweep above pins `p` at 0 because it is fitted at the all-unfamiliar end,
where no model reaches 87.5%. Refitting at a point the models *can* reach makes
the anchor bind: `run_sim_seeds.py --experiment ratio_hi`, optionally with
`--calib-label 20celeb_20setA`.

**Fit at 24 celeb + 16 Set_A** (`runs/sim_seeds_ratio_hi`) — p = 0.30 / 0.31 /
0.33, upright lands on 87.5% at the fit point:

| model | p | n_celeb | Upright | Inverted | Inversion effect |
| --- | --- | --- | --- | --- | --- |
| **Human (between-subjects)** | — | — | 87.50 | 76.80 | **+10.70** |
| **Human (within-subject)** | — | — | 87.50 | 75.90 | **+11.60** |
| r5b_allatonce | 0.30 | 24 | 87.50 ±0.13 | 81.24 ±0.20 | **+6.26** ±0.23 |
| | | 20 | 84.80 ±0.21 | 79.13 ±0.23 | +5.67 ±0.25 |
| r8_developmental | 0.31 | 24 | 86.83 ±0.22 | 80.71 ±0.23 | **+6.11** ±0.23 |
| | | 20 | 84.03 ±0.18 | 78.41 ±0.22 | +5.62 ±0.27 |
| house_control_r1 | 0.33 | 24 | 86.13 ±0.22 | 78.29 ±0.15 | **+7.84** ±0.28 |
| | | 20 | 82.93 ±0.17 | 75.63 ±0.29 | +7.30 ±0.28 |

**Fit at 20 celeb + 20 Set_A** (`runs/sim_seeds_ratio_hi20`) — p = 0.23 / 0.04 /
0.16, upright ≈ 87.0% at the fit point:

| model | p | n_celeb 24 (U / I / effect) | n_celeb 20 (U / I / effect) |
| --- | --- | --- | --- |
| r5b_allatonce | 0.23 | 89.39 / 83.90 / **+5.49** ±0.18 | 86.93 / 81.79 / +5.14 ±0.19 |
| r8_developmental | 0.04 | 89.24 / 84.43 / **+4.81** ±0.15 | 86.89 / 82.25 / +4.64 ±0.21 |
| house_control_r1 | 0.16 | 89.63 / 84.24 / **+5.39** ±0.12 | 87.03 / 81.86 / +5.17 ±0.14 |

Two things calibration changes:

- **The effects grow.** At p ≈ 0 every model sat at +4.8; anchored at 87.5% they
  rise to +6.1 to +7.8. Still well below the human +10.70 / +11.60, but the gap
  is roughly half what the uncalibrated run implied.
- **The models separate.** house_control_r1 shows a clearly larger inversion
  effect (+7.84 ±0.28) than r5b (+6.26 ±0.23) or r8 (+6.11 ±0.23) — the same
  ordering as experiment 1's Yin result, and outside the SEMs. The p ≈ 0 sweep
  could not show this because all three were compressed near ceiling.

Prefer the fit-at-24 table for cross-model comparison: all three models sit at
similar p (0.30–0.33), so they are compared at matched task difficulty. In the
fit-at-20 table r8 needs only p = 0.04 (it is already at the anchor unaided)
while r5b needs 0.23, so those rows compare models at different noise levels.

## 4. Yin — houses (from the mix64 sweep, for comparison)

Not a new run: these are the `houses_zubud` rows of `runs/sim_seeds_mix64`, same
20 seeds, `p` fitted on faces.

| Source | Stimulus set | p | UU | II | UI | IU | Inversion cost (UU−UI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Human (Yin 1969, houses)** | — | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** |
| r5b_allatonce | ZuBuD 201, **trained-on** | 0.02 | 96.0 ±0.7 | 86.5 ±1.4 | 86.0 ±1.2 | 88.1 ±1.1 | **+10.00** ±1.02 |
| r8_developmental | ZuBuD 161 held-out | 0.00 | 83.7 ±1.3 | 80.2 ±1.2 | 73.5 ±1.7 | 74.0 ±1.8 | **+10.21** ±1.15 |
| house_control_r1 ⚠ | ZuBuD, scored on all 201 (40 trained-on) | 0.03 | 98.3 ±0.6 | 95.2 ±0.7 | 87.5 ±1.3 | 85.8 ±1.3 | **+10.83** ±1.37 |

r8_developmental trained on 40 of the 201 packed ZuBuD classes and is scored on
the other 161. house_control_r1 also trained on 40 ZuBuD buildings, but ⚠ was
scored on all 201 including those (the held-out filter matched nothing — see the
correction at the top), so its row is inflated. **r5b_allatonce has no held-out
houses at all** — it trained on all 201 and is scored on its own training
classes, so its numbers are the most inflated of the three.

All three show ~+10 points of house inversion cost against a human +2.63 — the
models are far more orientation-sensitive for houses than people are, and the
effect does not distinguish the house-trained models from the control.

---

## Caveats

- Experiment 1 scores 9 pairs per condition per seed. Single seeds are
  uninformative; ±SEM is over 20 item draws.
- Experiments 2 and 3 are near-floor calibrations (p ≈ 0). Accuracies there are
  the models' ceiling on the task, not a match to a human anchor.
- Experiment 3's familiarity gradient is confounded with stimulus set.
- Experiment 4's r5b row is trained-on, not held-out.
- `celeb_` photos come from the training identities' held-out `valid` split: the
  identities were trained on, the exact images were not.
