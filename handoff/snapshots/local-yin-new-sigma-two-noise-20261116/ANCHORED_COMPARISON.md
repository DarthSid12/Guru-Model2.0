# Face-anchored comparison on the fitting seeds

This is the corrected presentation for the requested two-anchor comparison. It follows the older screenshot convention: report the same 50 simulation seeds used to fit face UU and II (101–150). The earlier fresh-seed table used 201–250, so its displayed II values were not the fitted anchors. No parameters or data have changed.

Original KDE; newer Delta no-generic models, training seed 46, epoch 124. LP σ = 4.5096, p1/p2 = .377/.378; CNN σ = 6.7540, p1/p2 = .366/.365. UU/UI use (p1,p1); IU/II use (p2,p1), ordered study/test. Face targets are 92.5833% UU and 63.75% II. Both models achieve 92.50% and 63.83% on these fitting seeds.

Errors out of 12, mean ± sample SD. Model SD describes 50 simulation seeds; human SD describes participants. Face UU and II were fitted; UI/IU, houses and objects were not targets. These fitting-set outcomes are not independent validation. The separate fresh-seed results remain available and show that the anchors do not stay exact on new simulations.

| Row | p1/p2 | UU | UI | IU | II |
|---|---|---:|---:|---:|---:|
| Human faces | — | 0.89 ± 1.09 | 3.81 ± 1.71 | 5.14 ± 1.39 | 4.35 ± 1.41 |
| LP faces | 0.377/0.378 | 0.90 ± 0.84 | 4.54 ± 1.73 | 4.66 ± 1.49 | 4.34 ± 1.87 |
| CNN faces | 0.366/0.365 | 0.90 ± 0.86 | 6.16 ± 1.72 | 5.90 ± 1.89 | 4.34 ± 1.47 |
| Human houses | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP houses | 0.377/0.378 | 4.96 ± 1.69 | 5.60 ± 1.62 | 5.42 ± 1.31 | 4.72 ± 1.64 |
| CNN houses | 0.366/0.365 | 5.72 ± 1.62 | 5.76 ± 1.80 | 5.88 ± 1.75 | 4.44 ± 1.64 |
| Human airplanes reference | — | 3.65 ± 1.69 | 3.19 ± 1.94 | 4.14 ± 1.58 | 3.85 ± 2.03 |
| LP objects | 0.377/0.378 | 1.76 ± 1.12 | 2.88 ± 1.38 | 3.36 ± 1.32 | 2.18 ± 1.17 |
| CNN objects | 0.366/0.365 | 1.56 ± 1.21 | 3.74 ± 1.47 | 3.60 ± 1.40 | 1.90 ± 1.13 |

Faces/houses use 20 faces + 20 houses studied, 12 pairs per category. Objects use 20 faces + 20 objects with the face/house noise pair transferred unchanged. Historical airplanes are reference only; the frozen object bank retains the previously documented 16 out-of-diet categories.

[Separate fresh-seed evaluation and controls](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/RESULTS.md) · [Fitting-set raw records](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/results/CALIBRATION_SET_DIAGNOSTIC.csv) · [Calibration LP](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/calibration/lp.json) · [Calibration CNN](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/calibration/cnn.json)
