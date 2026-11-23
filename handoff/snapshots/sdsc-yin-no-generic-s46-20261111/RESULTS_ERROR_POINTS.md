# Yin results: LP/CNN trained without generic houses

Error points on a **12-decision scale**, mean ± sample SD. Lower means fewer errors. Columns are UU,UI,IU,II. Both models are finalepoch124,trainingseed46. Model uncertainty is across50simulation seeds; human uncertainty is across participants.

Faces use40faces studied and24test pairs, normalized to12decisions. Houses use20faces+20houses studied; objects use20faces+20objects studied, with12test pairs/category. The human reference studied20/category in mixed blocks; it is not matched human data for the40-face design.

Human values come from the frozen Yin1969 extraction:UU/II n26,UI/IU n21. Object human rows are historical **airplane** references. The fixed model object bank differs and includes16categories absent from the revised training diet.

Face UU is fitted in every protocol; II is also fitted in the two-noise protocols. These anchor outcomes are calibration results. Human contrast SD is unavailable without paired covariance; model contrast SD is calculated from paired simulation seeds.

## Single noise

Study/test noise=p1/p1 in every condition.

### Faces

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human | — | 0.89 ± 1.09 | 3.81 ± 1.71 | 5.14 ± 1.39 | 4.35 ± 1.41 | -3.54 (SD unavailable) |
| LP | 0.36 | 0.90 ± 0.59 | 4.59 ± 1.25 | 4.87 ± 1.03 | 4.22 ± 1.23 | -3.66 ± 1.02 |
| CNN | 0.36 | 0.93 ± 0.45 | 5.81 ± 1.28 | 5.97 ± 1.28 | 4.87 ± 1.09 | -4.62 ± 0.84 |

### Houses

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 | -1.01 (SD unavailable) |
| LP | 0.38 | 5.38 ± 1.58 | 5.76 ± 1.72 | 5.52 ± 1.66 | 5.26 ± 1.86 | -0.13 ± 1.36 |
| CNN | 0.36 | 5.44 ± 1.86 | 5.88 ± 1.67 | 6.04 ± 1.82 | 4.36 ± 1.82 | 0.01 ± 2.08 |

### Objects

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human airplanes | — | 3.65 ± 1.69 | 3.19 ± 1.94 | 4.14 ± 1.58 | 3.85 ± 2.03 | -0.08 (SD unavailable) |
| LP | 0.37 | 1.60 ± 1.29 | 3.04 ± 1.41 | 3.18 ± 1.34 | 1.86 ± 1.05 | -1.09 ± 1.35 |
| CNN | 0.37 | 1.80 ± 1.29 | 3.76 ± 1.51 | 3.58 ± 1.49 | 2.02 ± 1.19 | -1.32 ± 1.45 |

## Fitted study-only noise

UU/UI:p1/p1;II/IU:p2/p1 (study/test).

### Faces

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human | — | 0.89 ± 1.09 | 3.81 ± 1.71 | 5.14 ± 1.39 | 4.35 ± 1.41 | -3.54 (SD unavailable) |
| LP | 0.36/0.37 | 0.90 ± 0.59 | 4.59 ± 1.25 | 4.81 ± 1.06 | 4.48 ± 1.37 | -3.73 ± 1.06 |
| CNN | 0.36/0.32 | 0.93 ± 0.45 | 5.81 ± 1.28 | 6.03 ± 1.22 | 4.39 ± 1.08 | -4.48 ± 0.83 |

### Houses

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 | -1.01 (SD unavailable) |
| LP | 0.38/0.37 | 5.38 ± 1.58 | 5.76 ± 1.72 | 5.40 ± 1.46 | 5.26 ± 1.82 | -0.09 ± 1.44 |
| CNN | 0.36/0.35 | 5.44 ± 1.86 | 5.88 ± 1.67 | 5.88 ± 1.69 | 4.38 ± 1.79 | 0.06 ± 2.19 |

### Objects

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human airplanes | — | 3.65 ± 1.69 | 3.19 ± 1.94 | 4.14 ± 1.58 | 3.85 ± 2.03 | -0.08 (SD unavailable) |
| LP | 0.37/0.38 | 1.60 ± 1.29 | 3.04 ± 1.41 | 3.06 ± 1.33 | 2.02 ± 1.19 | -1.11 ± 1.41 |
| CNN | 0.37/0.37 | 1.80 ± 1.29 | 3.76 ± 1.51 | 3.58 ± 1.49 | 2.02 ± 1.19 | -1.32 ± 1.45 |

## Two-cross noise

UU:p1/p1;UI:p1/p2;IU:p2/p1;II:p2/p2 (study/test).

### Faces

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human | — | 0.89 ± 1.09 | 3.81 ± 1.71 | 5.14 ± 1.39 | 4.35 ± 1.41 | -3.54 (SD unavailable) |
| LP | 0.36/0.36 | 0.90 ± 0.59 | 4.59 ± 1.25 | 4.87 ± 1.03 | 4.22 ± 1.23 | -3.66 ± 1.02 |
| CNN | 0.36/0.34 | 0.93 ± 0.45 | 5.79 ± 1.17 | 5.99 ± 1.14 | 4.47 ± 0.97 | -4.49 ± 0.67 |

### Houses

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 | -1.01 (SD unavailable) |
| LP | 0.38/0.38 | 5.38 ± 1.58 | 5.76 ± 1.72 | 5.52 ± 1.66 | 5.26 ± 1.86 | -0.13 ± 1.36 |
| CNN | 0.36/0.35 | 5.44 ± 1.86 | 5.76 ± 1.67 | 5.88 ± 1.69 | 4.42 ± 1.89 | 0.09 ± 2.20 |

### Objects

| Source | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---|---:|---:|---:|---:|---:|
| Human airplanes | — | 3.65 ± 1.69 | 3.19 ± 1.94 | 4.14 ± 1.58 | 3.85 ± 2.03 | -0.08 (SD unavailable) |
| LP | 0.37/0.38 | 1.60 ± 1.29 | 2.96 ± 1.38 | 3.06 ± 1.33 | 2.22 ± 1.33 | -1.15 ± 1.40 |
| CNN | 0.37/0.37 | 1.80 ± 1.29 | 3.76 ± 1.51 | 3.58 ± 1.49 | 2.02 ± 1.19 | -1.32 ± 1.45 |

[All seven study designs, original accuracy tables](/Users/darthsid/Documents/ChatGPT/GURU/operations/sdsc-yin-no-generic-s46-20261111/completed-sdsc/results/SUMMARY_LOCAL.md) · [Independent verification](/Users/darthsid/Documents/ChatGPT/GURU/operations/sdsc-yin-no-generic-s46-20261111/INDEPENDENT_VERIFICATION.json)
