# Yin (1969) battery — rfwWM64-calibrated, all models

Generated 2026-09-09 from `runs/sim_seeds_wm64_rfwcal`.

**Calibration.** One retrieval-noise level `p` per model, fitted on `faces_rfwWM64` **alone** against the 96.29 human upright anchor (`--experiment wm64_rfwcal`). This replaces the older joint fit over `faces_cfdWM64` + `faces_rfwWM64`, which railed to the p=0 grid floor on models where CFD cannot reach its anchor at any noise level.

**Design.** Study 40 items, test 24 old-vs-new 2AFC pairs, 50 seeds (101–150), 4 orientation conditions = 800 rows per model. Study/test orientation: U = upright, I = inverted.

All values are mean ± SEM over 50 seeds, in percent.


---

## 1. Calibration summary

| model | fitted p | `faces_rfwWM64` UU achieved | off anchor (96.29) | rows | warns |
|---|---|---|---|---|---|
| `r15_vgg` | 0.09 | 97.83 | +1.54 | 800/800 | 0 |
| `r16_base` | 0.08 | 96.42 | +0.13 | 800/800 | 0 |
| `r16_acuity` | 0.28 | 93.08 | -3.21 ⚠ | 800/800 | 0 |
| `r16_rfw` | 0.33 | 96.58 | +0.29 | 800/800 | 0 |
| `r16rfwftH` | 0.37 | 91.75 | -4.54 ⚠ | 800/800 | 0 |
| `r17_base` | 0.2 | 97.25 | +0.96 | 800/800 | 0 |
| `r17_houses` | 0.06 | 97.33 | +1.04 | 800/800 | 21 |

⚠ = achieved upright accuracy more than 2 points off the anchor. `p` is fitted on 8 probe seeds at base seed 42 but reported on seeds 101–150, so the fit is out-of-sample and drifts. `--calib-on-eval-seeds` closes this gap.


> Note on `r17_houses`: its first run lost 21 sims (seeds 118–124) to a CUDA OOM caused by a co-tenant process on the GPU. Those seeds were deleted and regenerated at the same p=0.06; the file is complete at 800 rows.


---

## 2. Model configurations

| model | face arm | generic house class | other |
|---|---|---|---|
| `r15_vgg` | 128 CelebA + 480 VGGFace2 + 119 CFD | houses (HD only) | — |
| `r16_base` | 128 CelebA + 480 VGGFace2 | houses_gen (HD + 97 ZuBuD) | — |
| `r16_acuity` | 128 CelebA + 480 VGGFace2 | houses_gen (HD + 97 ZuBuD) | blur→sharp σ 8/6/4/2/1/0 |
| `r16_rfw` | 128 CelebA + 480 VGGFace2 + 1500 RFW | houses_gen (HD + 97 ZuBuD) | — |
| `r16rfwftH` | warm-start r16_rfw, faces capped 40img/cls | houses (137 ZuBuD @0.45) | 20-epoch fine-tune, lr 1e-4 |
| `r17_base` | 128 CelebA + 480 VGGFace2 | houses (HD only) | — |
| `r17_houses` | 128 CelebA + 480 VGGFace2 @0.33 | houses (137 ZuBuD @0.20) | objects 0.25→0.22 |

---

## 3. Full condition tables, per model


### `r15_vgg`  — p = 0.09

| category | UU | UI | IU | II | UU−II | UU−UI | UU−IU |
|---|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 89.17 ±0.90 | 69.50 ±1.07 | 63.25 ±1.24 | 85.17 ±0.90 | **+4.00** | +19.67 | +25.92 |
| `faces_rfwWM64` | 97.83 ±0.48 | 76.33 ±1.13 | 77.50 ±1.22 | 97.50 ±0.46 | **+0.33** | +21.50 | +20.33 |
| `houses_yin64` | 87.75 ±0.70 | 70.58 ±0.99 | 68.75 ±0.94 | 83.58 ±0.77 | **+4.17** | +17.17 | +19.00 |
| `objects` | 99.17 ±0.24 | 97.25 ±0.41 | 97.67 ±0.34 | 99.83 ±0.12 | **-0.67** | +1.92 | +1.50 |
| | | | | | | | |
| *human — `faces_cfdWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `faces_rfwWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `houses_yin64`* | 90.71 | — | — | — | *+4.96* | *+2.63* | — |
| *human — `objects`* | 84.79 | — | — | — | *+0.83* | *-1.92* | — |

### `r16_base`  — p = 0.08

| category | UU | UI | IU | II | UU−II | UU−UI | UU−IU |
|---|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 85.25 ±0.97 | 63.83 ±1.32 | 63.67 ±1.22 | 74.50 ±1.04 | **+10.75** | +21.42 | +21.58 |
| `faces_rfwWM64` | 96.42 ±0.55 | 78.92 ±1.25 | 77.17 ±0.87 | 96.25 ±0.55 | **+0.17** | +17.50 | +19.25 |
| `houses_yin64` | 33.92 ±0.82 | 32.17 ±1.04 | 34.25 ±0.90 | 35.42 ±0.99 | **-1.50** | +1.75 | -0.33 |
| `objects` | 98.67 ±0.32 | 96.50 ±0.48 | 97.50 ±0.38 | 99.17 ±0.24 | **-0.50** | +2.17 | +1.17 |
| | | | | | | | |
| *human — `faces_cfdWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `faces_rfwWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `houses_yin64`* | 90.71 | — | — | — | *+4.96* | *+2.63* | — |
| *human — `objects`* | 84.79 | — | — | — | *+0.83* | *-1.92* | — |

### `r16_acuity`  — p = 0.28

| category | UU | UI | IU | II | UU−II | UU−UI | UU−IU |
|---|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 77.75 ±1.04 | 59.42 ±1.17 | 58.42 ±1.29 | 68.33 ±1.50 | **+9.42** | +18.33 | +19.33 |
| `faces_rfwWM64` | 93.08 ±0.74 | 67.50 ±1.42 | 66.25 ±1.44 | 88.92 ±0.96 | **+4.17** | +25.58 | +26.83 |
| `houses_yin64` | 38.17 ±1.02 | 33.83 ±1.01 | 37.42 ±0.99 | 35.33 ±1.00 | **+2.83** | +4.33 | +0.75 |
| `objects` | 97.25 ±0.42 | 93.92 ±0.66 | 96.33 ±0.46 | 98.25 ±0.36 | **-1.00** | +3.33 | +0.92 |
| | | | | | | | |
| *human — `faces_cfdWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `faces_rfwWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `houses_yin64`* | 90.71 | — | — | — | *+4.96* | *+2.63* | — |
| *human — `objects`* | 84.79 | — | — | — | *+0.83* | *-1.92* | — |

### `r16_rfw`  — p = 0.33

| category | UU | UI | IU | II | UU−II | UU−UI | UU−IU |
|---|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 81.42 ±0.86 | 59.58 ±1.13 | 59.42 ±1.38 | 65.17 ±1.54 | **+16.25** | +21.83 | +22.00 |
| `faces_rfwWM64` | 96.58 ±0.46 | 76.75 ±1.01 | 74.58 ±1.19 | 91.08 ±0.74 | **+5.50** | +19.83 | +22.00 |
| `houses_yin64` | 42.17 ±0.94 | 48.33 ±1.09 | 40.83 ±0.97 | 50.75 ±0.99 | **-8.58** | -6.17 | +1.33 |
| `objects` | 95.92 ±0.48 | 89.00 ±0.78 | 93.00 ±0.68 | 93.92 ±0.65 | **+2.00** | +6.92 | +2.92 |
| | | | | | | | |
| *human — `faces_cfdWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `faces_rfwWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `houses_yin64`* | 90.71 | — | — | — | *+4.96* | *+2.63* | — |
| *human — `objects`* | 84.79 | — | — | — | *+0.83* | *-1.92* | — |

### `r16rfwftH`  — p = 0.37

| category | UU | UI | IU | II | UU−II | UU−UI | UU−IU |
|---|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 73.25 ±1.21 | 56.33 ±1.36 | 55.92 ±1.36 | 57.00 ±1.46 | **+16.25** | +16.92 | +17.33 |
| `faces_rfwWM64` | 91.75 ±0.75 | 70.50 ±1.32 | 68.08 ±1.25 | 79.83 ±1.19 | **+11.92** | +21.25 | +23.67 |
| `houses_yin64` | 59.92 ±1.09 | 57.25 ±1.35 | 53.00 ±1.07 | 60.83 ±1.50 | **-0.92** | +2.67 | +6.92 |
| `objects` | 86.67 ±0.70 | 81.42 ±1.03 | 83.42 ±1.20 | 83.92 ±0.86 | **+2.75** | +5.25 | +3.25 |
| | | | | | | | |
| *human — `faces_cfdWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `faces_rfwWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `houses_yin64`* | 90.71 | — | — | — | *+4.96* | *+2.63* | — |
| *human — `objects`* | 84.79 | — | — | — | *+0.83* | *-1.92* | — |

### `r17_base`  — p = 0.2

| category | UU | UI | IU | II | UU−II | UU−UI | UU−IU |
|---|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 70.17 ±0.89 | 61.67 ±1.28 | 56.50 ±1.34 | 77.25 ±0.92 | **-7.08** | +8.50 | +13.67 |
| `faces_rfwWM64` | 97.25 ±0.53 | 75.17 ±1.27 | 73.08 ±1.12 | 88.67 ±0.95 | **+8.58** | +22.08 | +24.17 |
| `houses_yin64` | 85.17 ±1.02 | 73.75 ±0.93 | 77.08 ±1.10 | 86.17 ±0.77 | **-1.00** | +11.42 | +8.08 |
| `objects` | 97.50 ±0.39 | 92.33 ±0.56 | 95.08 ±0.46 | 97.67 ±0.42 | **-0.17** | +5.17 | +2.42 |
| | | | | | | | |
| *human — `faces_cfdWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `faces_rfwWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `houses_yin64`* | 90.71 | — | — | — | *+4.96* | *+2.63* | — |
| *human — `objects`* | 84.79 | — | — | — | *+0.83* | *-1.92* | — |

### `r17_houses`  — p = 0.06

| category | UU | UI | IU | II | UU−II | UU−UI | UU−IU |
|---|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 84.50 ±0.95 | 69.58 ±1.18 | 65.00 ±1.30 | 88.58 ±0.81 | **-4.08** | +14.92 | +19.50 |
| `faces_rfwWM64` | 97.33 ±0.50 | 75.00 ±1.27 | 74.83 ±1.18 | 93.83 ±0.64 | **+3.50** | +22.33 | +22.50 |
| `houses_yin64` | 96.50 ±0.55 | 83.58 ±0.92 | 83.75 ±0.95 | 94.83 ±0.48 | **+1.67** | +12.92 | +12.75 |
| `objects` | 98.08 ±0.32 | 96.58 ±0.44 | 96.58 ±0.46 | 98.92 ±0.29 | **-0.83** | +1.50 | +1.50 |
| | | | | | | | |
| *human — `faces_cfdWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `faces_rfwWM64`* | 96.29 | — | — | — | *+14.41* | *+12.10* | — |
| *human — `houses_yin64`* | 90.71 | — | — | — | *+4.96* | *+2.63* | — |
| *human — `objects`* | 84.79 | — | — | — | *+0.83* | *-1.92* | — |

---

## 4. Cross-model matrices


### 4.1 UU − II  (upright superiority)

| model | p | `faces_cfdWM64` | `faces_rfwWM64` | `houses_yin64` | `objects` |
|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | +14.41 | +14.41 | +4.96 | +0.83 |
| `r15_vgg` | 0.09 | +4.00 | +0.33 | +4.17 | -0.67 |
| `r16_base` | 0.08 | +10.75 | +0.17 | -1.50 | -0.50 |
| `r16_acuity` | 0.28 | +9.42 | +4.17 | +2.83 | -1.00 |
| `r16_rfw` | 0.33 | +16.25 | +5.50 | -8.58 | +2.00 |
| `r16rfwftH` | 0.37 | +16.25 | +11.92 | -0.92 | +2.75 |
| `r17_base` | 0.2 | -7.08 | +8.58 | -1.00 | -0.17 |
| `r17_houses` | 0.06 | -4.08 | +3.50 | +1.67 | -0.83 |

### 4.2 UU − UI  (orientation-mismatch cost)

| model | p | `faces_cfdWM64` | `faces_rfwWM64` | `houses_yin64` | `objects` |
|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | +12.10 | +12.10 | +2.63 | -1.92 |
| `r15_vgg` | 0.09 | +19.67 | +21.50 | +17.17 | +1.92 |
| `r16_base` | 0.08 | +21.42 | +17.50 | +1.75 | +2.17 |
| `r16_acuity` | 0.28 | +18.33 | +25.58 | +4.33 | +3.33 |
| `r16_rfw` | 0.33 | +21.83 | +19.83 | -6.17 | +6.92 |
| `r16rfwftH` | 0.37 | +16.92 | +21.25 | +2.67 | +5.25 |
| `r17_base` | 0.2 | +8.50 | +22.08 | +11.42 | +5.17 |
| `r17_houses` | 0.06 | +14.92 | +22.33 | +12.92 | +1.50 |

### 4.3 Raw upright accuracy (UU)

| model | p | `faces_cfdWM64` | `faces_rfwWM64` | `houses_yin64` | `objects` |
|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | 96.29 | 96.29 | 90.71 | 84.79 |
| `r15_vgg` | 0.09 | 89.17 | 97.83 | 87.75 | 99.17 |
| `r16_base` | 0.08 | 85.25 | 96.42 | 33.92 | 98.67 |
| `r16_acuity` | 0.28 | 77.75 | 93.08 | 38.17 | 97.25 |
| `r16_rfw` | 0.33 | 81.42 | 96.58 | 42.17 | 95.92 |
| `r16rfwftH` | 0.37 | 73.25 | 91.75 | 59.92 | 86.67 |
| `r17_base` | 0.2 | 70.17 | 97.25 | 85.17 | 97.50 |
| `r17_houses` | 0.06 | 84.50 | 97.33 | 96.50 | 98.08 |

---

## 5. Supplementary — `r17_base` checkpoint contrast at matched p=0.10

Both arms at p=0.10, 50 seeds; only the checkpoint differs. `interim` is the 2026-09-04 mid-run scoring (epoch ~60 for the first ~78% of seeds, epoch ~67 after — `best_model.pth` was being overwritten by the still-running training job); `final` is `final_model_20260904_151045.pth` (epoch 80).

| category | UU interim | UU final | ΔUU | UU−II interim | UU−II final | Δ |
|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 88.67 | 73.50 | -15.17 | +6.75 | -7.92 | -14.67 |
| `faces_rfwWM64` | 99.00 | 97.83 | -1.17 | +4.92 | +5.75 | +0.83 |
| `houses_yin64` | 85.50 | 89.25 | +3.75 | -5.92 | -0.50 | +5.42 |
| `objects` | 97.50 | 98.08 | +0.58 | -0.92 | -0.08 | +0.83 |

The interim rows are **not reproducible** — the epoch-60 weights were overwritten and no longer exist. Retained for reference only.


### 5.1 `r17_base` full condition table at forced p=0.10 (final weights)

| category | UU | UI | IU | II | UU−II | UU−UI |
|---|---|---|---|---|---|---|
| `faces_cfdWM64` | 73.50 ±0.99 | 64.17 ±1.28 | 57.25 ±1.17 | 81.42 ±0.93 | **-7.92** | +9.33 |
| `faces_rfwWM64` | 97.83 ±0.42 | 77.25 ±1.18 | 76.75 ±1.16 | 92.08 ±0.75 | **+5.75** | +20.58 |
| `houses_yin64` | 89.25 ±0.91 | 76.08 ±0.85 | 78.75 ±0.79 | 89.75 ±0.60 | **-0.50** | +13.17 |
| `objects` | 98.08 ±0.38 | 92.25 ±0.66 | 95.25 ±0.53 | 98.17 ±0.34 | **-0.08** | +5.83 |

---

## 6. Caveats

1. **Models are not matched at the anchor.** Achieved `faces_rfwWM64` UU spans 91.75–97.83 against a 96.29 target (see §1). Models further below ceiling have more room for a large UU−II, so cross-model UU−II is partly confounded with ceiling distance.

2. **Neither face store is domain-neutral.** `faces_rfwWM64` is drawn from `faces_rfwHO` — identities held out from every model, but the RFW *dataset* is in-distribution for `r16_rfw` and `r16rfwftH`, which train on 1500 RFW identities. `faces_cfdWM64` is in-distribution for `r15_vgg` (119 CFD identities). Calibrating on either store advantages some models.

3. **`houses_yin64` is uninterpretable for the `houses_gen` models.** `r16_base` (33.92), `r16_acuity` (38.17) and `r16_rfw` (42.17) read at or below the 50% 2AFC floor upright.

4. **UU−UI is not a specificity measure.** It conflates upright superiority with orientation-mismatch cost, which every category shows. UU−II isolates upright superiority.

5. **`objects` are ceiling-limited** — 86.67–99.17 upright against an 84.79 human anchor.

6. **`r17` CFD anomaly.** Both r17 models show negative CFD UU−II while all r15/r16 models are positive. Not a face-diet effect: `r16_base` has `r17_base`'s exact face diet and reads +10.75. §5 shows `r17_base` CFD was +6.75 at epoch ~60 and −7.92 at epoch 80, so it is a late-training effect specific to the r17 runs; the cause is open.

