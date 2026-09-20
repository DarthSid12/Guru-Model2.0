# Log-polar developmental models r16–r17: face inversion, and what the house control is worth

Rounds r16 (2026-09-01) and r17 (2026-09-03), plus one house fine-tune, with
r15_vgg carried through as the reference model. All models are ResNet-18
backbones trained from scratch on log-polar foveated fixation crops (16
fixations, 180 px), never ImageNet-initialised.

**Status.** r16 and its simulations are complete. r17_base and r17_houses were
still training at ~epoch 64/80 when this was written and their simulation rows
are interim — marked ⚠ throughout. Re-score both on final weights.

---

## 1. Models

| model | face arm | object arm | house identities | generic house class | notes |
|---|---|---|---|---|---|
| `r15_vgg` | 128 CelebA + 480 VGGFace2 + 119 CFD | 64 | 40 ZuBuD | `houses` (635 Houses-dataset) | reference |
| `r16_base` | 128 CelebA + 480 VGGFace2 | 64 | 40 ZuBuD | `houses_gen` (635 HD **+ 97 ZuBuD**) | |
| `r16_acuity` | as r16_base | 64 | 40 ZuBuD | `houses_gen` | **+ blur→sharp schedule**, σ 8/6/4/2/1/0 per stage, train-time only |
| `r16_rfw` | as r16_base **+ 1500 RFW** (1200 Caucasian from stage 1, 300 other-race from stage 4) | 64 | 40 ZuBuD | `houses_gen` | |
| `r17_base` | as r16_base | 64 | 40 ZuBuD | `houses` (HD only) | single variable vs r16_base |
| `r17_houses` | as r16_base | 64 | **137 ZuBuD** @ weight 0.20 | `houses` | objects weight 0.25→0.22, faces_vgg 0.40→0.33 |
| `r16rfwftH` | warm-start from `r16_rfw` | 64 | **137 ZuBuD** @ 0.45 | `houses` | 20 epochs, lr 1e-4, face/object capped 40 img/class |

All from-scratch runs use one 6-stage / 80-epoch curriculum (6/6/8/8/12/40
epochs), global cosine LR from 1e-3, and per-category sampling weights. No model
except r15_vgg trains on CFD.

**Held-out throughout:** `faces_cfdWM64` (64 white-male CFD, studio),
`faces_rfwWM64` (64 white-male RFW, in-the-wild, built by
`scripts/make_rfw_wm64.py`), `houses_yin64` (64 ZuBuD), `faces_vggHO`,
`faces_setA`.

**Simulation.** Yin (1969) 2AFC: study 40 items, test 24 old-vs-new pairs, four
orientation conditions — UU/II/UI/IU (study/test). One binomial-noise level `p`
per model, calibrated so upright accuracy meets the human anchor. **Compare
inversion costs, not raw accuracies, across rows with different `p`.**

---

## 2. Training accuracy (upright valid / inverted test, %)

| model | faces | faces_vgg | objects | house ids | generic houses | RFW |
|---|---|---|---|---|---|---|
| r16_base | 93.1 / 68.1 | 89.1 / 63.4 | 74.9 / 68.3 | 87.5 / 37.5 | 97.5 / 88.3 | — |
| r16_acuity | 92.9 / 57.2 | 89.2 / 54.4 | 73.9 / 66.4 | 90.0 / 45.0 | 98.0 / 87.8 | — |
| r16_rfw | 90.5 / 63.3 | 85.2 / 52.5 | 74.0 / 67.1 | 87.5 / 40.0 | 99.5 / 88.8 | W 9.5, O 18.7 |
| r16rfwftH | 89.1 / 57.6 | 79.4 / 39.4 | 66.3 / 60.2 | **97.1 / 83.2** (137-way) | 100.0 / 100.0 | W 6.6, O 12.7 |
| r17_base ⚠ ep64 | 92.0 | 86.5 | 73.0 | 95.0 | 96.0 | — |
| r17_houses ⚠ ep64 | 93.4 | 86.2 | 70.8 | **94.9** (137-way) | 96.0 | — |

`r16_rfw`'s RFW pair (W 9.5 vs O 18.7) is **not** an other-race measure: rfwW is
1200-way and rfwO 300-way, so chance differs 4×, and the top-100-per-race "other"
identities are deeper (5.03 photos/id vs 3.23). The reversal is a class-count
artefact.

On this readout `r16_acuity` looked like a clean win — faces lost ~10 extra
points to inversion with upright accuracy unchanged, objects and generic houses
moving ~1. **Section 3 does not reproduce it.**

---

## 3. Results
## A. r16 round — wm64 battery, 50 seeds, per-model calibrated p

**faces_cfdWM64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | +14.41 |
| r16_base | 0.01 | 50 | 86.67 ±1.57 | 78.75 ±2.00 | 64.92 ±2.26 | 63.83 ±2.76 | **+21.75** ±2.30 | +7.92 ±2.28 |
| r16_acuity | 0.0 | 50 | 87.50 ±1.30 | 89.33 ±1.67 | 69.50 ±2.41 | 67.25 ±2.42 | **+18.00** ±2.59 | -1.83 ±2.29 |
| r16_rfw | 0.1 | 50 | 95.42 ±1.10 | 81.17 ±2.15 | 65.58 ±2.82 | 67.33 ±2.50 | **+29.83** ±2.62 | +14.25 ±1.98 |
| r15_vgg (ref) | 0.0 | 50 | 91.33 ±1.42 | 89.17 ±1.44 | 70.33 ±2.32 | 63.25 ±2.43 | **+21.00** ±2.19 | +2.17 ±1.59 |

**faces_rfwWM64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | +14.41 |
| r16_base | 0.01 | 50 | 96.67 ±1.09 | 97.08 ±0.97 | 80.33 ±2.14 | 78.08 ±1.63 | **+16.33** ±2.40 | -0.42 ±1.35 |
| r16_acuity | 0.0 | 50 | 97.08 ±0.97 | 96.42 ±0.81 | 75.67 ±1.92 | 74.67 ±2.35 | **+21.42** ±1.97 | +0.67 ±0.97 |
| r16_rfw | 0.1 | 50 | 99.83 ±0.23 | 99.58 ±0.35 | 88.42 ±1.60 | 87.25 ±1.87 | **+11.42** ±1.63 | +0.25 ±0.36 |
| r15_vgg (ref) | 0.0 | 50 | 98.17 ±0.71 | 99.08 ±0.48 | 78.08 ±2.32 | 78.75 ±2.42 | **+20.08** ±2.53 | -0.92 ±0.85 |

**houses_yin64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** | +4.96 |
| r16_base | 0.01 | 50 | 33.92 ±1.82 | 36.33 ±2.06 | 32.58 ±2.11 | 33.83 ±1.96 | **+1.33** ±2.27 | -2.42 ±2.41 |
| r16_acuity | 0.0 | 50 | 38.08 ±2.07 | 37.00 ±1.79 | 34.58 ±1.84 | 37.75 ±2.00 | **+3.50** ±1.72 | +1.08 ±1.80 |
| r16_rfw | 0.1 | 50 | 43.08 ±2.02 | 51.17 ±1.62 | 49.75 ±1.81 | 40.92 ±2.07 | **-6.67** ±2.44 | -8.08 ±2.25 |
| r15_vgg (ref) | 0.0 | 50 | 90.58 ±1.21 | 85.50 ±1.42 | 71.67 ±2.01 | 69.50 ±1.89 | **+18.92** ±2.32 | +5.08 ±1.65 |

**objects**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 84.79 | 83.96 | 86.71 | 82.75 | **-1.92** | +0.83 |
| r16_base | 0.01 | 50 | 98.75 ±0.63 | 99.17 ±0.47 | 96.17 ±0.99 | 97.67 ±0.78 | **+2.58** ±0.90 | -0.42 ±0.58 |
| r16_acuity | 0.0 | 50 | 98.25 ±0.74 | 99.42 ±0.41 | 96.17 ±1.01 | 98.83 ±0.57 | **+2.08** ±1.10 | -1.17 ±0.70 |
| r16_rfw | 0.1 | 50 | 97.83 ±0.63 | 97.92 ±0.67 | 93.50 ±1.02 | 95.50 ±0.80 | **+4.33** ±0.90 | -0.08 ±0.89 |
| r15_vgg (ref) | 0.0 | 50 | 99.42 ±0.41 | 100.00 ±0.00 | 96.92 ±0.87 | 98.08 ±0.67 | **+2.50** ±0.81 | -0.58 ±0.41 |

## B. r17 round + fine-tune — wm64 battery, all at p=0.10, 50 seeds

▲ = r17 models scored on `best_model.pth` at ~epoch 64/80; re-score on final weights.

**faces_cfdWM64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | +14.41 |
| r16_rfw (parent) | 0.1 | 50 | 95.42 ±1.10 | 81.17 ±2.15 | 65.58 ±2.82 | 67.33 ±2.50 | **+29.83** ±2.62 | +14.25 ±1.98 |
| r16rfwftH | 0.1 | 50 | 97.00 ±0.87 | 76.67 ±2.06 | 66.17 ±2.62 | 67.67 ±2.41 | **+30.83** ±2.63 | +20.33 ±2.27 |
| r17_base ▲ | 0.1 | 50 | 88.67 ±2.14 | 81.92 ±1.80 | 70.17 ±2.52 | 66.83 ±2.42 | **+18.50** ±2.42 | +6.75 ±2.58 |
| r17_houses ▲ | 0.1 | 50 | 92.67 ±1.35 | 87.00 ±1.69 | 72.42 ±2.25 | 73.25 ±1.93 | **+20.25** ±2.26 | +5.67 ±2.11 |

**faces_rfwWM64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | +14.41 |
| r16_rfw (parent) | 0.1 | 50 | 99.83 ±0.23 | 99.58 ±0.35 | 88.42 ±1.60 | 87.25 ±1.87 | **+11.42** ±1.63 | +0.25 ±0.36 |
| r16rfwftH | 0.1 | 50 | 99.92 ±0.16 | 98.75 ±0.54 | 89.25 ±1.63 | 88.92 ±1.68 | **+10.67** ±1.64 | +1.17 ±0.52 |
| r17_base ▲ | 0.1 | 50 | 99.00 ±0.50 | 94.08 ±1.32 | 81.00 ±2.05 | 80.08 ±1.87 | **+18.00** ±2.07 | +4.92 ±1.37 |
| r17_houses ▲ | 0.1 | 50 | 97.92 ±0.94 | 93.92 ±1.46 | 76.42 ±2.40 | 77.17 ±2.27 | **+21.50** ±2.38 | +4.00 ±1.63 |

**houses_yin64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** | +4.96 |
| r16_rfw (parent) | 0.1 | 50 | 43.08 ±2.02 | 51.17 ±1.62 | 49.75 ±1.81 | 40.92 ±2.07 | **-6.67** ±2.44 | -8.08 ±2.25 |
| r16rfwftH | 0.1 | 50 | 76.75 ±2.15 | 76.67 ±1.90 | 68.92 ±2.08 | 66.33 ±2.02 | **+7.83** ±2.04 | +0.08 ±2.21 |
| r17_base ▲ | 0.1 | 50 | 85.50 ±1.88 | 91.42 ±1.46 | 77.67 ±2.06 | 74.67 ±1.92 | **+7.83** ±2.17 | -5.92 ±2.14 |
| r17_houses ▲ | 0.1 | 50 | 95.83 ±0.90 | 95.17 ±1.03 | 85.25 ±1.72 | 85.33 ±1.70 | **+10.58** ±1.88 | +0.67 ±1.31 |

**objects**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 84.79 | 83.96 | 86.71 | 82.75 | **-1.92** | +0.83 |
| r16_rfw (parent) | 0.1 | 50 | 97.83 ±0.63 | 97.92 ±0.67 | 93.50 ±1.02 | 95.50 ±0.80 | **+4.33** ±0.90 | -0.08 ±0.89 |
| r16rfwftH | 0.1 | 50 | 96.58 ±0.83 | 93.25 ±1.14 | 92.17 ±1.31 | 92.25 ±1.28 | **+4.42** ±1.43 | +3.33 ±1.26 |
| r17_base ▲ | 0.1 | 50 | 97.50 ±0.74 | 98.42 ±0.61 | 91.83 ±1.09 | 94.33 ±0.95 | **+5.66** ±1.16 | -0.92 ±0.82 |
| r17_houses ▲ | 0.1 | 50 | 98.75 ±0.54 | 98.58 ±0.55 | 93.67 ±1.10 | 96.92 ±1.11 | **+5.08** ±1.17 | +0.17 ±0.81 |

## C. houses_ho64 (Houses-dataset held-out) — negative control

**houses_ho64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** | +4.96 |
| r16_base | 0.01 | 50 | 49.17 ±2.69 | 56.25 ±2.56 | 50.08 ±2.13 | 49.33 ±2.42 | **-0.92** ±2.95 | -7.08 ±3.04 |
| r16_acuity | 0.0 | 50 | 49.25 ±2.49 | 57.08 ±2.67 | 51.58 ±2.72 | 49.25 ±2.50 | **-2.33** ±3.28 | -7.83 ±3.24 |
| r16_rfw | 0.1 | 50 | 48.92 ±2.31 | 55.25 ±2.68 | 52.33 ±2.66 | 49.92 ±2.07 | **-3.42** ±2.80 | -6.33 ±2.81 |
| r15_vgg | 0.0 | 50 | 50.50 ±2.48 | 62.75 ±2.28 | 52.17 ±2.29 | 50.50 ±2.51 | **-1.67** ±3.05 | -12.25 ±3.12 |

## D. r15_vgg earlier sweeps — per-experiment calibration

**vggHO_ctrl / faces_vggHO**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | +14.41 |
| r15_vgg [vggHO_ctrl] | 0.27 | 50 | 94.78 ±0.84 | 91.13 ±1.33 | 78.35 ±2.21 | 77.39 ±2.09 | **+16.43** ±2.17 | +3.65 ±1.41 |

**vggHO_ctrl / houses_yin64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** | +4.96 |
| r15_vgg [vggHO_ctrl] | 0.27 | 50 | 75.75 ±2.34 | 75.08 ±1.77 | 64.33 ±2.04 | 61.92 ±2.56 | **+11.42** ±2.90 | +0.67 ±2.27 |

**vggHO_ctrl / objects**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 84.79 | 83.96 | 86.71 | 82.75 | **-1.92** | +0.83 |
| r15_vgg [vggHO_ctrl] | 0.27 | 50 | 98.33 ±0.66 | 99.58 ±0.35 | 97.17 ±0.79 | 96.25 ±0.97 | **+1.17** ±0.81 | -1.25 ±0.71 |

**cfd_h40_ctrl / faces_cfdWM64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | +14.41 |
| r15_vgg [cfd_h40_ctrl] | 0.0 | 50 | 91.33 ±1.42 | 89.17 ±1.44 | 70.33 ±2.32 | 63.25 ±2.43 | **+21.00** ±2.19 | +2.17 ±1.59 |

**cfd_h40_ctrl / houses_yin64**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** | +4.96 |
| r15_vgg [cfd_h40_ctrl] | 0.0 | 50 | 90.58 ±1.21 | 85.50 ±1.42 | 71.67 ±2.01 | 69.50 ±1.89 | **+18.92** ±2.32 | +5.08 ±1.65 |

**cfd_h40_ctrl / objects**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 84.79 | 83.96 | 86.71 | 82.75 | **-1.92** | +0.83 |
| r15_vgg [cfd_h40_ctrl] | 0.0 | 50 | 99.42 ±0.41 | 100.00 ±0.00 | 96.92 ±0.87 | 98.08 ±0.67 | **+2.50** ±0.81 | -0.58 ±0.41 |

**rfwHO / faces_rfwHO**

| model | p | n | UU | II | UI | IU | UU−UI | UU−II |
|---|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | +14.41 |
| r15_vgg [rfwHO] | 0.0 | 50 | 95.25 ±1.23 | 92.33 ±1.92 | 76.92 ±2.59 | 76.50 ±2.53 | **+18.33** ±2.21 | +2.92 ±1.52 |

---

## 4. Findings

**4.1 The acuity schedule does not survive the Yin measure.** Against `r16_base`
it goes the wrong way on CFD (+18.00 vs +21.75) and the right way on RFW (+21.42
vs +16.33) — no consistent direction, both within ~2 CIs. The +10-point
face-specific signal in §2 is most likely the acuity model being worse at
inverted *classification*, not configural processing. Vogelsang et al. (2018) is
not supported by this round.

**4.2 The models show an orientation-MISMATCH effect, not an inversion effect.**
Visible only with all four conditions. Almost everywhere UU ≈ II — studying *and*
testing inverted costs nothing (`r16_base`/rfwWM64 96.67 vs 97.08). Humans lose
14.4 points there. The models' entire UU−UI effect comes from UI and IU dropping
together, i.e. from study/test orientation disagreeing. The human ordering also
differs in kind: for humans II (81.88) sits *below* UI (84.13); every model
reverses it.

One exception: **`r16_rfw` on `faces_cfdWM64`, UU−II = +14.25 ±1.98 against a
human +14.41** — the only complete cell reproducing the Yin signature. It is also
the only r16 model at p=0.10, and higher noise degrades the harder condition
faster, so part of the gap may be the noise level. `r16_base` gives +7.92 at
p=0.01, suggesting a real effect that p amplifies rather than manufactures.

**4.3 `faces_rfwWM64` does not measure face processing.** UU 96.67–100.00 with
II 95.08–99.58: inverting barely matters. Consistent with the 2AFC being won on
non-facial cues — background, lighting, pose, clothing all differ between
in-the-wild photos and survive inversion. `faces_cfdWM64` (studio-controlled, so
identities differ mainly in facial structure) is where every UU−II effect
appears, and is the store to prefer.

**4.4 RFW training makes a markedly better face model.** `r16_rfw` needs p=0.10
to be pulled *down* to the anchor where `r16_base` needs 0.01, and reaches 95.42%
upright on `faces_cfdWM64` — a domain it never trained on — against `r16_base`'s
86.67%. Largest CFD inversion effect in the round.

**4.5 `houses_gen` destroyed the house arm, and removing it restores it.** All
three r16 models score `houses_yin64` at 33.9–43.1% upright — at or below the 50%
2AFC floor — against `r15_vgg`'s 90.58% on the identical store. The only relevant
difference is that `houses_gen` folds 97 leftover **ZuBuD** buildings into a
single generic `house` label, collapsing ZuBuD representations. Dose-response:

| model | generic class | house identities | `houses_yin64` UU | UU−II |
|---|---|---|---|---|
| **Human (Yin 1969)** | — | — | 90.71 | +4.96 |
| r16_rfw | HD + 97 ZuBuD | 40 | 43.08 ±2.02 | −8.08 ±2.25 |
| r16rfwftH | HD only | 137 (fine-tune) | 76.75 | +0.08 ±2.21 |
| r17_base ▲ | HD only | 40 | 85.50 | −5.92 ±2.14 |
| r17_houses ▲ | HD only | 137 | 95.83 | +0.67 ±1.31 |
| r15_vgg | HD only | 40 | 90.58 ±1.21 | +5.08 ±1.70 |

Removing ZuBuD from the generic class restores individuation; adding identities
(40→137) pushes it to ceiling. A 20-epoch fine-tune recovers most of it without
an 18 h retrain. Held-out ZuBuD has always worked in models without ZuBuD in a
generic class — r8 84.83, r9 92.00 on 161 held-out buildings.

**4.5b `r16rfwftH` is the best face-specificity result in the project.** With a
working house arm restored, its classic Yin effect (UU−II) dissociates cleanly by
category, and its upright CFD accuracy lands on the human anchor:

| | UU (faces) | faces UU−II | houses UU−II | objects UU−II |
|---|---|---|---|---|
| **Human (Yin 1969)** | 96.29 | **+14.41** | +4.96 | +0.83 |
| r16rfwftH | **97.00** | **+20.33** ±2.27 | +0.08 ±2.21 | +3.33 ±1.26 |
| r16_rfw | 95.42 | +14.25 ±1.98 | −8.08 ±2.25 | −0.08 ±0.89 |
| r17_base ▲ | 88.67 | +6.75 ±2.58 | −5.92 ±2.14 | −0.92 ±0.82 |
| r17_houses ▲ | 92.67 | +5.67 ±2.11 | +0.67 ±1.31 | +0.17 ±0.81 |

Faces +20.33 against houses +0.08 and objects +3.33, with houses individuable
(UU 76.75, well clear of the 50% floor) rather than at ceiling or dead. The
face effect overshoots the human +14.41 and the house effect undershoots +4.96,
but the ordering and the size of the gap are right. On UU−UI the same model gives
faces +30.83 / houses +7.83, a 3.9× ratio against the human 4.6× — the closest
any model here comes to the human separation.

**4.5c RFW training, not the house fix, drives the face inversion effect.** The
two models with RFW in the diet (`r16_rfw` +14.25, `r16rfwftH` +20.33 UU−II on
CFD) are far above the two without (`r17_base` +6.75, `r17_houses` +5.67), even
though the r17 pair has the better house arm. 1500 in-the-wild identities appear
to be what builds an orientation-sensitive face representation.

**Correction.** An interim read at ~10 seeds put `r17_base` at faces +20.8 /
houses +3.7 (a 5.6× ratio, near-human). At the full 50 seeds it is +18.50 /
+7.83, a 2.4× ratio. The apparent near-human dissociation was small-sample noise.

**4.6 More house training makes houses more face-like, not less.** `r17_houses`
(137 ids, UU−UI +10.58) shows a *larger* house inversion effect than
`r17_base` (40 ids, +7.83), and both exceed the human +2.63.
Consistent with inversion cost tracking expertise rather than stimulus class —
which is a problem for any face-specificity claim resting on a house control.

**4.7 Objects behave correctly everywhere** — UU−UI +2.08 to +7.8 against a human
−1.92, i.e. near zero in every model. But they sit at 95.8–100.0% upright, so
they are a ceiling-limited control.

**4.8 `houses_ho64` is an unusable store.** All four r16-era models read chance
on it (48.92–50.50% UU) *including* `r15_vgg`. Every model was trained to collapse
the Houses-dataset into one generic `house` label, so held-out buildings from that
dataset inherit "→ one label". A held-out house store must come from a dataset the
model was never trained to collapse — i.e. ZuBuD, as `houses_yin64` does. §C is
retained as a negative control only.

---

## 5. Caveats

1. **r17 rows are interim** — `best_model.pth` at ~epoch 63/80, LR still ~1e-4,
   partial seeds. Re-score on final weights.
2. **`p` differs across models** (0.00–0.10). Compare inversion costs, not raw
   accuracies. §B holds p=0.10 fixed so those four rows are mutually comparable.
3. **`faces_cfdWM64` is a novel domain for r16/r17 but not for `r15_vgg`**, which
   trained on 119 CFD identities. r16-vs-r16 is clean; r16-vs-r15 on CFD is not.
4. **`r17_houses` moved the objects weight** (0.25→0.22), so its object row is not
   directly comparable to r16/`r17_base`.
5. **`faces_rfwWM64`'s 64 males were identified by eye** from contact sheets —
   RFW ships no sex labels. Visual judgements, not ground-truth metadata.
6. **No other-race effect was measured.** `faces_rfwWM64` is white males only, so
   there is no matched non-white store to contrast at the same `p`.
7. **Houses are trained in a memorisation regime.** `houses_zubud` supplies 120
   images (0.067% of unique training content) but takes 10% of every batch — each
   unique house crop is drawn ~11,909× over the run against 60 for `faces_vgg`.
   `r17_houses` (137 ids, 411 images) partly addresses this.

## 6. Open

- Re-score r17_base / r17_houses on final weights.
- Matched-`p` run of `r16_base` and `r16_rfw` to test whether the +14.25 UU−II is
  representational or a noise-level artefact.
- A non-face control that is neither at ceiling (objects) nor expertise-confounded
  (houses) — currently the main obstacle to a face-specificity claim.
- Scale control: face arm cut to house scale (40 ids × 3 photos) to test whether
  the house arm's behaviour is data scale rather than stimulus class.
