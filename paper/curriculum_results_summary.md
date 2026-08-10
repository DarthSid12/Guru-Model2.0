# developmental vs. all-at-once: Yin (1969) and Kanwisher (2023)

Model rows are **means ± SEM over independent simulation seeds** (`run_sim_seeds.py`); each seed redraws the study/test items, so a single seed is not meaningful on its own — one Yin test pair is 4.2 points. Retrieval noise `p` is calibrated **once** per model and category so that upright accuracy matches the human target, then held fixed across seeds.

Seeds: **n=200 (seeds 101–300)** for every model × simulation × category cell — the sweep is complete, with no cell short. Both models are LP-Net ResNet-18, 393 classes.

Reference data transcribed from [results_summary.md](results_summary.md).

---

## 1. Yin (1969) — study/test orientation matching

UU/II = same orientation at study and test; UI/IU = mismatched. The face signature is a large UU→II drop that objects and houses do not show.

### Faces

| Source | Noise p | UU | II | UI | IU | Inversion cost (UU−II) |
|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | 96.29% | 81.88% | 84.13% | 78.58% | **+14.41** |
| LP-Net **developmental** (r7, r18) | 0.38 | 93.10% ±0.29 | 81.00% ±0.51 | 81.54% ±0.49 | 81.90% ±0.49 | **+12.10** ±0.52 |
| LP-Net **all-at-once** (r5b, r18) | 0.20 | 96.10% ±0.25 | 86.98% ±0.43 | 85.48% ±0.46 | 81.31% ±0.48 | **+9.12** ±0.43 |

### Objects

| Source | Noise p | UU | II | UI | IU | Inversion cost (UU−II) |
|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | 84.79% | 83.96% | 86.71% | 82.75% | **+0.83** |
| LP-Net **developmental** (r7, r18) | 0.43 | 72.62% ±0.59 | 69.29% ±0.68 | 71.21% ±0.65 | 70.48% ±0.71 | **+3.33** ±0.87 |
| LP-Net **all-at-once** (r5b, r18) | 0.00 | 82.06% ±0.40 | 74.52% ±0.37 | 73.04% ±0.41 | 75.96% ±0.41 | **+7.54** ±0.37 |

### Houses

| Source | Noise p | UU | II | UI | IU | Inversion cost (UU−II) |
|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | 90.71% | 85.75% | 88.08% | 85.71% | **+4.96** |
| LP-Net **developmental** (r7, r18) | 0.32 | 90.29% ±0.40 | 82.00% ±0.46 | 79.08% ±0.50 | 81.33% ±0.51 | **+8.29** ±0.50 |
| LP-Net **all-at-once** (r5b, r18) | 0.29 | 90.79% ±0.35 | 81.92% ±0.42 | 80.71% ±0.46 | 83.06% ±0.43 | **+8.88** ±0.44 |

---

## 2. Dobs / Kanwisher (2023) — upright vs. inverted matching

A three-image identity-matching task with no memory phase, scored over ~156k triplets per run — which is why these SEMs are ~5× tighter than Yin's.

### Faces

| Source | Noise p | Upright | Inverted | Inversion effect |
|---|---|---|---|---|
| **Human (between-subjects, n=1,532/1,219)** | — | 87.50% | 76.80% | **+10.70** |
| **Human (within-subject, n=364)** | — | 87.50% | 75.90% | **+11.60** |
| **Dobs et al. Face-ID CNN (Fig. 3B)** | — | 86.90% | 66.40% | **+20.50** |
| LP-Net **developmental** (r7, r18) | 0.43 | 85.44% ±0.10 | 70.81% ±0.13 | **+14.63** ±0.13 |
| LP-Net **all-at-once** (r5b, r18) | 0.42 | 83.45% ±0.09 | 70.97% ±0.12 | **+12.48** ±0.12 |

### Objects

| Source | Noise p | Upright | Inverted | Inversion effect |
|---|---|---|---|---|
| LP-Net **developmental** (r7, r18) | 0.40 | 85.02% ±0.13 | 82.19% ±0.13 | **+2.83** ±0.10 |
| LP-Net **all-at-once** (r5b, r18) | 0.37 | 84.58% ±0.14 | 81.72% ±0.13 | **+2.86** ±0.11 |

*Dobs et al. (2023) Exp. 5 tested faces only, so there is no human or Face-ID CNN reference for objects; the model rows stand alone.*

### Houses

| Source | Noise p | Upright | Inverted | Inversion effect |
|---|---|---|---|---|
| LP-Net **developmental** (r7, r18) | 0.43 | 85.57% ±0.20 | 84.44% ±0.21 | **+1.13** ±0.26 |
| LP-Net **all-at-once** (r5b, r18) | 0.42 | 89.06% ±0.18 | 87.48% ±0.19 | **+1.58** ±0.24 |

*Dobs et al. (2023) Exp. 5 tested faces only, so there is no human or Face-ID CNN reference for houses; the model rows stand alone.*

---

## 3. Face-specific inversion cost

The claim in both papers is not that inversion hurts — it is that it hurts **faces more than other within-category identity tasks**. Scored per seed as `cost(faces) − mean[cost(objects), cost(houses)]`, so a model with a large but uniform inversion cost scores zero.

Paired by seed, since every category shares a seed's sampling draw. CIs are 20k-resample bootstraps over the 200 per-seed contrasts.

| Simulation | Model | Face-specific cost | 95% CI | Verdict |
|---|---|---|---|---|
| Yin (1969) | LP-Net **developmental** (r7, r18) | **+6.29** ±0.67 | [+4.96, +7.58] | excludes 0 |
| Yin (1969) | LP-Net **all-at-once** (r5b, r18) | **+0.92** ±0.51 | [−0.06, +1.91] | **includes 0** |
| Kanwisher (2023) | LP-Net **developmental** (r7, r18) | **+12.65** ±0.19 | [+12.28, +13.02] | excludes 0 |
| Kanwisher (2023) | LP-Net **all-at-once** (r5b, r18) | **+10.26** ±0.17 | [+9.92, +10.60] | excludes 0 |
| Yin (1969) | **Human** | **+11.52** | — | — |

At the full 200 seeds the all-at-once model's Yin face-specific cost is not distinguishable from zero — it fails Yin's actual claim — while the developmental model clears it by a wide margin. This is the one place where the extra seeds changed a qualitative conclusion rather than a decimal.

Per-seed values are in `figures/sim_seeds_face_specific.csv` (`figures/aggregate_sim_seeds.py`).

