# Yin (1969) & Dobs/Kanwisher (2023) on `faces_mix64` — model vs. human

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


All model rows are means ± SEM over **20 seeds** (`run_sim_seeds.py --seeds 101-120`,
2026-08-15). Raw rows: `runs/sim_seeds_mix64/results_*.csv`; regenerate the tables with
`python runs/sim_seeds_mix64/aggregate_mix64.py`.

Retrieval noise `p` is calibrated **once** per (model, simulation) on `faces_mix64`, then
held fixed across seeds and across every sweep point.

## Stimulus set

| Set | Contents | Seen in training? | Used for |
|---|---|---|---|
| **`faces_mix64`** | 24 CelebA identities + 40 Set_A identities, 5 photos each = 320 photos | **mixed** — the 24 celeb identities are training classes for all three models; the 40 Set_A identities are novel to all three | Yin (all 64 items), Kanwisher (identity-count sweep) |
| objects | 64 ImageNet classes | **YES — training classes** | Yin control |
| ZuBuD | 201 buildings × 5 exterior views | varies by model (see rows) | Yin control |

Built by `make_faces_mix64.py` as a pure re-pack of the existing packed stores (same 224px
crops, same 32 fixation coords). Celeb photos come from the held-out `valid` split, so the
identity was seen in training but the exact image was not.

---

## 1. Yin (1969) — study/test orientation matching

UU/II = same orientation at study and test; UI/IU = mismatched. **Inversion cost = `UU − UI`,
paired per seed** — the same contrast the other results tables use. `UU − II` is reported in
§1d because it is the contrast that is *not* informative here (see caveat 2).

### 1a. Faces

| Source | Stimulus set | p | UU | II | UI | IU | Inversion cost (UU−UI) | seeds |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** |  |
| LP-Net **all-at-once** (r5b, r18) | `faces_mix64` (24 trained + 40 novel) | 0.02 | 74.58 ±1.68 | 72.50 ±2.27 | 65.42 ±2.46 | 63.54 ±2.34 | **+9.17** ±1.59 | 20 |
| LP-Net **developmental** (r8, r18) | `faces_mix64` (24 trained + 40 novel) | 0.00 | 83.96 ±1.29 | 83.54 ±1.15 | 72.50 ±1.58 | 70.42 ±1.79 | **+11.46** ±1.28 | 20 |
| **David's model** (curriculum, r18)† | `faces_mix64` (24 trained + 40 novel) | 0.03 | 84.17 ±1.44 | 77.71 ±1.46 | 70.21 ±1.87 | 70.00 ±2.11 | **+13.96** ±2.01 | 20 |

All three face costs are significant (paired t = 5.8 / 8.9 / 7.0). **Upright accuracy misses the
96.29% human anchor by 12–22 points at every noise level** — see caveat 1.

### 1b. Objects

| Source | Stimulus set | p | UU | II | UI | IU | Inversion cost (UU−UI) | seeds |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 84.79 | 83.96 | 86.71 | 82.75 | **−1.92** |  |
| LP-Net **all-at-once** (r5b, r18) | objects (64, **trained**) | 0.02 | 81.46 ±1.37 | 73.96 ±1.24 | 72.50 ±1.15 | 74.58 ±1.51 | **+8.96** ±0.82 | 20 |
| LP-Net **developmental** (r8, r18) | objects (64, **trained**) | 0.00 | 99.37 ±0.34 | 99.17 ±0.38 | 97.92 ±0.48 | 97.92 ±0.64 | **+1.46** ±0.46 | 20 |
| **David's model** (curriculum, r18)† | objects (64, **trained**) | 0.03 | 99.17 ±0.38 | 99.17 ±0.38 | 97.50 ±0.56 | 97.29 ±0.55 | **+1.67** ±0.47 | 20 |

### 1c. Houses

| Source | Stimulus set | p | UU | II | UI | IU | Inversion cost (UU−UI) | seeds |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** |  |
| LP-Net **all-at-once** (r5b, r18) | ZuBuD (201, **trained** — see caveat 3) | 0.02 | 96.04 ±0.71 | 86.46 ±1.38 | 86.04 ±1.18 | 88.12 ±1.06 | **+10.00** ±1.02 | 20 |
| LP-Net **developmental** (r8, r18) | ZuBuD (161 **held-out**) | 0.00 | 83.75 ±1.35 | 80.21 ±1.17 | 73.54 ±1.66 | 73.96 ±1.84 | **+10.21** ±1.15 | 20 |
| **David's model** (curriculum, r18)† ⚠ | ZuBuD (scored on all 201; 40 trained-on) | 0.03 | 98.33 ±0.56 | 95.21 ±0.69 | 87.50 ±1.32 | 85.83 ±1.26 | **+10.83** ±1.37 | 20 |

### 1d. `UU − II` for reference (not the inversion cost)

| Source | Faces | Objects | Houses |
|---|---|---|---|
| **Human (Yin 1969)** | **+14.41** | **+0.83** | **+4.96** |
| LP-Net **all-at-once** (r5b, r18) | +2.08 ±1.72 (t=1.2, n.s.) | +7.50 ±0.89 | +9.58 ±1.17 |
| LP-Net **developmental** (r8, r18) | +0.42 ±1.13 (t=0.4, n.s.) | +0.21 ±0.37 (t=0.6, n.s.) | +3.54 ±1.10 |
| **David's model** (curriculum, r18)† | +6.46 ±1.92 | 0.00 ±0.30 (t=0.0, n.s.) | +3.13 ±0.85 |

Consistent with caveat 2 of `results_all_stimulus_sets.md`: a consistently inverted study+test
pair is about as matchable as an upright one, so the orientation cost lands on the mismatched
conditions. **Do not read the n.s. face entries here as "no inversion effect"** — the same
models show +9.2 to +14.0 on UU−UI.

---

## 2. Dobs / Kanwisher (2023) — identity-count sweep, faces only

Three-image identity matching, no memory phase. The identity pool is held at the **24 CelebA
identities** and Set_A identities are added 5 at a time. Selections nest as the pool grows, so
at a fixed seed each point adds to the set the previous point used — the sweep is a
within-sample comparison, not six independent draws. `p` is fitted once at n=49 and held fixed
across all six points.

| Source | Identity pool | p | Upright | Inverted | Inversion effect | seeds |
|---|---|---|---|---|---|---|
| **Human (between-subjects, n=1,532/1,219)** | — | — | 87.50 | 76.80 | **+10.70** |  |
| **Human (within-subject, n=364)** | — | — | 87.50 | 75.90 | **+11.60** |  |
| **Dobs et al. Face-ID CNN (Fig. 3B)** | — | — | 86.90 | 66.40 | **+20.50** |  |
| LP-Net **all-at-once** (r5b, r18) | 24 celeb + 0 Set_A | 0.24 | 96.29 ±0.08 | 89.74 ±0.11 | **+6.56** ±0.13 | 20 |
| | 24 celeb + 5 Set_A | 0.24 | 93.87 ±0.11 | 87.69 ±0.16 | **+6.17** ±0.15 | 20 |
| | 24 celeb + 10 Set_A | 0.24 | 91.47 ±0.17 | 85.74 ±0.17 | **+5.73** ±0.16 | 20 |
| | 24 celeb + 15 Set_A | 0.24 | 89.59 ±0.15 | 83.98 ±0.17 | **+5.62** ±0.18 | 20 |
| | 24 celeb + 20 Set_A | 0.24 | 87.94 ±0.17 | 82.49 ±0.17 | **+5.44** ±0.16 | 20 |
| | 24 celeb + 25 Set_A | 0.24 | 86.63 ±0.16 | 81.30 ±0.16 | **+5.33** ±0.19 | 20 |
| LP-Net **developmental** (r8, r18) | 24 celeb + 0 Set_A | 0.04 | 95.81 ±0.03 | 89.74 ±0.04 | **+6.08** ±0.04 | 20 |
| | 24 celeb + 5 Set_A | 0.04 | 93.59 ±0.12 | 88.25 ±0.17 | **+5.34** ±0.14 | 20 |
| | 24 celeb + 10 Set_A | 0.04 | 91.37 ±0.18 | 86.30 ±0.13 | **+5.06** ±0.17 | 20 |
| | 24 celeb + 15 Set_A | 0.04 | 89.49 ±0.16 | 84.73 ±0.12 | **+4.76** ±0.15 | 20 |
| | 24 celeb + 20 Set_A | 0.04 | 87.99 ±0.13 | 83.40 ±0.13 | **+4.59** ±0.17 | 20 |
| | 24 celeb + 25 Set_A | 0.04 | 86.77 ±0.13 | 82.33 ±0.09 | **+4.44** ±0.13 | 20 |
| **David's model** (curriculum, r18)† | 24 celeb + 0 Set_A | 0.09 | 96.01 ±0.04 | 89.75 ±0.06 | **+6.26** ±0.08 | 20 |
| | 24 celeb + 5 Set_A | 0.09 | 94.25 ±0.15 | 88.49 ±0.17 | **+5.76** ±0.09 | 20 |
| | 24 celeb + 10 Set_A | 0.09 | 92.03 ±0.16 | 86.58 ±0.16 | **+5.45** ±0.12 | 20 |
| | 24 celeb + 15 Set_A | 0.09 | 90.29 ±0.13 | 85.15 ±0.13 | **+5.14** ±0.13 | 20 |
| | 24 celeb + 20 Set_A | 0.09 | 88.63 ±0.12 | 83.60 ±0.11 | **+5.03** ±0.09 | 20 |
| | 24 celeb + 25 Set_A | 0.09 | 87.34 ±0.10 | 82.44 ±0.10 | **+4.90** ±0.09 | 20 |

Every point is significant (paired t = 27–152). All three models land **below both human
inversion effects** (+10.70 / +11.60) at every pool size, and far below the Dobs Face-ID CNN's
+20.50.

### 2a. Effect vs. pool size

| Model | n=24 | n=29 | n=34 | n=39 | n=44 | n=49 | Slope per +5 identities |
|---|---|---|---|---|---|---|---|
| LP-Net **all-at-once** (r5b, r18) | +6.56 | +6.17 | +5.73 | +5.62 | +5.44 | +5.33 | **−0.241** ±0.033 (t=−7.3) |
| LP-Net **developmental** (r8, r18) | +6.08 | +5.34 | +5.06 | +4.76 | +4.59 | +4.44 | **−0.307** ±0.031 (t=−10.0) |
| **David's model** (curriculum, r18)† | +6.26 | +5.76 | +5.45 | +5.14 | +5.03 | +4.90 | **−0.265** ±0.015 (t=−17.6) |

The ordering r5b > David's > r8 holds at every one of the six points. Upright accuracy falls
~96% → ~87% over the same range.

---

## 3. Caveats

**1. Yin faces is NOT calibrated to the human anchor, and cannot be.** Set_A is at chance under
Yin's same-store 2AFC. Measured at p=0 on r5b: `faces` (128 trained celeb) 95.83%, `faces_setE`
95.83%, **`faces_setA` 53.33%** (46.7–66.7% across seeds, with or without item shuffling). So 40
of the 64 `faces_mix64` items carry no signal, no noise level reaches 96.29%, and `p` pinned at
the floor for all three models (0.02 / 0.00 / 0.03). Read the Yin face rows as a mix of ~24
usable items and 40 near-chance ones. This reproduces caveat 4 of `results_all_stimulus_sets.md`;
**Set_E remains the working alternative** if a calibrated Yin face condition is needed.
Kanwisher is unaffected — all three fit the 87.5% anchor exactly.

**2. `UU − II` is the wrong contrast for this design** (see §1d). Use `UU − UI`.

**3. r5b house numbers are trained-on.** r5b trained on all 201 packed ZuBuD classes, so it has
no held-out houses and `--exclude-seen-from` leaves it zero items (a structural 0.0%). It is
therefore run on its *own training classes* via `EXCLUDE_SEEN_WAIVED` in `run_sim_seeds.py`. Its
house row is inflated and **not comparable** to r8's (161 held out). David's model ⚠ is not a
clean comparison either: it trained on 40 ZuBuD buildings and was scored on all 201, so its house
row is inflated too, just less so.

**4. Objects are never held out** for any model, so object accuracy is inflated by exposure and
cannot reach the 84.79% human anchor. r5b is the outlier at 81.5% UU — that is r5b being a weaker
model overall, not object-specific inversion.

**5. The Kanwisher sweep confounds pool size with stimulus set.** Adding identities makes every
trial harder regardless of *which* identities are added, so the −0.24 to −0.31 slope is not by
itself evidence that Set_A specifically shrinks the inversion effect. The clean control is a
matched 24→49 sweep drawing from Set_E instead; it has not been run.

**6. Seeds are not independent across a sweep row.** Every point shares the same 24 celeb
identities by construction, and the pools nest. SEMs are small (±0.04–0.19) partly because a
Kanwisher point scores 80k–235k triplets.

**7. `p` differs across rows.** Compare inversion **costs**, not raw accuracies, across rows with
different `p`.

† **David's model is a curriculum model**, on a two-stage schedule: 2 epochs of 256 classes (all 128 faces + all 128 objects, no houses) then 29 epochs of all 296 with the 40 houses added — `stage_spec` `all-prehouse` → `all` in its training history. The `curriculum: False` in its `config.json` (and the absence of `stage*.pth` checkpoints) refers only to the class-count ladder r8/r9 use; its house delay runs off `houses_delay_epochs: 2` instead. It differs from r8 in object-class count (128 vs 64), curriculum shape (2 stages vs 6) and LR schedule (none vs cosine) — but **not** in house dataset: both train on 40 ZuBuD buildings. See `results_all_stimulus_sets.md` for the full note.

---

## 4. Run record

- 3 models × 20 seeds × 9 units = **540 simulation runs, 1,440 condition rows, 0 failures**.
- Wall clock: 72–85 min per model, GPUs 0/1/2 in parallel.
- Calibration: `noise_<model>.json` in `runs/sim_seeds_mix64/`, with the upright accuracy each
  category reaches at the chosen `p` recorded alongside it.
