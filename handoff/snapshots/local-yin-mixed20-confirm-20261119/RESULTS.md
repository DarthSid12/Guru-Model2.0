# Confirmed mixed-category Yin rerun

Errors out of 12, mean ± sample SD across 50 simulation seeds (101–150). Same seeds used for face calibration; this is not independent validation. Both experiments use the same epoch-124, training-seed-46 LP/CNN checkpoints.

Study 20 faces + 20 houses first; transfer the fitted noise unchanged to a separate study of 20 faces + 20 objects. Each test has 12 old/new pairs per category, within-category comparisons only. Ten study fixations and 32 test fixations per image.

Every category count reproduced the previous results exactly.

## fixed sigma

| Model | Study group | Outcome | σ | p1 / p2 | UU | UI | IU | II |
|---|---|---|---:|---|---:|---:|---:|---:|
| LP | 20 faces + 20 houses | faces | 2.0000 | 0.373 / 0.369 | 0.90 ± 0.79 | 4.36 ± 1.52 | 4.40 ± 1.36 | 4.32 ± 1.80 |
| CNN | 20 faces + 20 houses | faces | 2.0000 | 0.364 / 0.354 | 0.88 ± 0.77 | 6.36 ± 1.55 | 6.04 ± 1.78 | 4.38 ± 1.32 |
| LP | 20 faces + 20 houses | houses | 2.0000 | 0.373 / 0.369 | 5.12 ± 1.52 | 5.62 ± 1.52 | 5.28 ± 1.55 | 4.98 ± 1.63 |
| CNN | 20 faces + 20 houses | houses | 2.0000 | 0.364 / 0.354 | 5.70 ± 1.90 | 5.98 ± 1.57 | 5.90 ± 1.62 | 4.60 ± 1.81 |
| LP | 20 faces + 20 objects | faces | 2.0000 | 0.373 / 0.369 | 1.06 ± 0.98 | 4.62 ± 1.96 | 4.58 ± 2.08 | 4.14 ± 1.62 |
| CNN | 20 faces + 20 objects | faces | 2.0000 | 0.364 / 0.354 | 0.58 ± 0.78 | 5.92 ± 1.83 | 6.20 ± 1.95 | 4.24 ± 1.73 |
| LP | 20 faces + 20 objects | objects | 2.0000 | 0.373 / 0.369 | 1.70 ± 1.30 | 2.98 ± 1.46 | 3.10 ± 1.50 | 1.76 ± 1.04 |
| CNN | 20 faces + 20 objects | objects | 2.0000 | 0.364 / 0.354 | 1.64 ± 1.26 | 3.70 ± 1.52 | 3.48 ± 1.43 | 2.04 ± 1.14 |

## adjusted sigma

| Model | Study group | Outcome | σ | p1 / p2 | UU | UI | IU | II |
|---|---|---|---:|---|---:|---:|---:|---:|
| LP | 20 faces + 20 houses | faces | 4.5096 | 0.377 / 0.378 | 0.90 ± 0.84 | 4.54 ± 1.73 | 4.66 ± 1.49 | 4.34 ± 1.87 |
| CNN | 20 faces + 20 houses | faces | 6.7540 | 0.366 / 0.365 | 0.90 ± 0.86 | 6.16 ± 1.72 | 5.90 ± 1.89 | 4.34 ± 1.47 |
| LP | 20 faces + 20 houses | houses | 4.5096 | 0.377 / 0.378 | 4.96 ± 1.69 | 5.60 ± 1.62 | 5.42 ± 1.31 | 4.72 ± 1.64 |
| CNN | 20 faces + 20 houses | houses | 6.7540 | 0.366 / 0.365 | 5.72 ± 1.62 | 5.76 ± 1.80 | 5.88 ± 1.75 | 4.44 ± 1.64 |
| LP | 20 faces + 20 objects | faces | 4.5096 | 0.377 / 0.378 | 1.08 ± 1.08 | 4.58 ± 1.75 | 4.76 ± 1.96 | 4.10 ± 1.63 |
| CNN | 20 faces + 20 objects | faces | 6.7540 | 0.366 / 0.365 | 0.64 ± 0.83 | 5.84 ± 1.66 | 6.00 ± 1.81 | 4.22 ± 1.85 |
| LP | 20 faces + 20 objects | objects | 4.5096 | 0.377 / 0.378 | 1.76 ± 1.12 | 2.88 ± 1.38 | 3.36 ± 1.32 | 2.18 ± 1.17 |
| CNN | 20 faces + 20 objects | objects | 6.7540 | 0.366 / 0.365 | 1.56 ± 1.21 | 3.74 ± 1.47 | 3.60 ± 1.40 | 1.90 ± 1.13 |

UU/UI use p1 at study and test. IU/II use p2 at study and p1 at test. Noise and sigma remain identical across the two study groups within each model/experiment.

The object bank includes 16 categories outside the revised training diet. Adjusted sigma was previously selected using face and house outcomes.
