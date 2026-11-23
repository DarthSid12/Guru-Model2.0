# New-model KDE with new sigma and two face anchors

The experiment is complete and independently verified. It uses the original KDE mechanism and the completed Delta no-generic ResNet-18 LP/CNN checkpoints, epoch 124, training seed 46. No retraining or nearest-prototype readout is involved.

The new noise pairs are nearly equal. Compared with the original σ = 2 two-noise control, upright-house errors on fresh seeds are essentially unchanged: LP 5.34 → 5.36 and CNN 5.38 → 5.42. Inverted houses remain easier than upright houses. CNN objects show a modest reduction in orientation dependence; LP objects do not improve against the original baseline.

## Calibration and evaluation

The rule is the same as the screenshot: UU/UI use (p1, p1), while IU/II use (p2, p1), ordered study/test. Fit p1 to human face UU = 92.5833%; then fit p2 to face II = 63.75%, holding test noise at p1. All other categories and conditions are unfitted. Absolute σ stays fixed throughout noise fitting.

Calibration uses 50 seeds, 101–150. Evaluation uses 50 new seeds, 201–250, outside the 101–200 seeds used to select σ. This is a holdout of simulation randomness and study/probe selections within the same image bank, not an independent stimulus dataset or training replication.

| Model | σ | p1 | p2 | Fitting UU | Fitting II | Fresh UU | Fresh II |
|---|---:|---:|---:|---:|---:|---:|---:|
| LP | 4.5096 | 0.377 | 0.378 | 92.50% | 63.83% | 94.50% | 70.17% |
| CNN | 6.7540 | 0.366 | 0.365 | 92.50% | 63.83% | 94.83% | 65.83% |

The fitting anchors are within 0.084 percentage points of their targets. The fresh-seed values are predictions and were not refitted. In particular, held-out LP face II is 70.17%, above its 63.75% target. These results should not be described as an exact human match on independent simulations.

## Fresh evaluation: errors out of 12

Faces and houses use 20 faces + 20 houses studied. Objects use 20 faces + 20 objects studied, with the face/house noise pair transferred unchanged. Both experiments have 40 study images, 12 old/new pairs per category, 10 study fixations and 32 test fixations.

Values are mean ± sample SD across 50 simulation seeds. Human SD is across participants. The historical airplane row is a reference on different stimuli, not matched human data for the object bank.

| Row | p1/p2 | UU | UI | IU | II |
|---|---|---:|---:|---:|---:|
| Human faces | — | 0.89 ± 1.09 | 3.81 ± 1.71 | 5.14 ± 1.39 | 4.35 ± 1.41 |
| LP faces | 0.377/0.378 | 0.66 ± 0.82 | 4.12 ± 1.90 | 3.96 ± 1.56 | 3.58 ± 1.55 |
| CNN faces | 0.366/0.365 | 0.62 ± 0.88 | 5.80 ± 1.90 | 5.96 ± 1.60 | 4.10 ± 1.68 |
| Human houses | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP houses | 0.377/0.378 | 5.36 ± 1.80 | 5.60 ± 1.96 | 5.64 ± 1.50 | 4.66 ± 1.77 |
| CNN houses | 0.366/0.365 | 5.42 ± 1.82 | 6.00 ± 1.82 | 6.16 ± 1.62 | 4.48 ± 1.39 |
| Human airplanes reference | — | 3.65 ± 1.69 | 3.19 ± 1.94 | 4.14 ± 1.58 | 3.85 ± 2.03 |
| LP objects | 0.377/0.378 | 1.76 ± 1.04 | 2.64 ± 1.34 | 3.08 ± 1.56 | 1.98 ± 1.25 |
| CNN objects | 0.366/0.365 | 1.56 ± 1.30 | 3.56 ± 1.61 | 3.40 ± 1.55 | 2.20 ± 1.58 |

## Controls

All controls use the same checkpoints, pictures, study/probe plans and common random draws. The original σ = 2 fit uses LP p1/p2 = .38/.37 and CNN .36/.35. The matched-fit σ = 2 control uses the newly fitted rates, isolating the bandwidth change. The new-σ single-p1 control uses p1 for both study and test, isolating the effect of p2.

### Faces

| Model | Control/readout | σ | p1/p2 | UU | UI | IU | II |
|---|---|---:|---|---:|---:|---:|---:|
| LP | New σ, two noise | 4.5096 | 0.377/0.378 | 0.66 ± 0.82 | 4.12 ± 1.90 | 3.96 ± 1.56 | 3.58 ± 1.55 |
| LP | σ = 2, original two-noise fit | 2.0000 | 0.380/0.370 | 1.02 ± 0.98 | 4.46 ± 1.86 | 4.14 ± 1.70 | 3.98 ± 1.65 |
| LP | σ = 2, new two-noise fit | 2.0000 | 0.377/0.378 | 0.88 ± 0.98 | 4.32 ± 1.90 | 4.00 ± 1.80 | 4.02 ± 1.61 |
| LP | New σ, p1 only | 4.5096 | 0.377/0.377 | 0.66 ± 0.82 | 4.12 ± 1.90 | 3.80 ± 1.53 | 3.56 ± 1.46 |
| CNN | New σ, two noise | 6.7540 | 0.366/0.365 | 0.62 ± 0.88 | 5.80 ± 1.90 | 5.96 ± 1.60 | 4.10 ± 1.68 |
| CNN | σ = 2, original two-noise fit | 2.0000 | 0.360/0.350 | 0.72 ± 0.83 | 6.12 ± 1.78 | 5.92 ± 1.69 | 3.98 ± 1.70 |
| CNN | σ = 2, new two-noise fit | 2.0000 | 0.366/0.365 | 0.84 ± 0.82 | 5.82 ± 1.77 | 5.90 ± 1.73 | 4.20 ± 1.86 |
| CNN | New σ, p1 only | 6.7540 | 0.366/0.366 | 0.62 ± 0.88 | 5.80 ± 1.90 | 5.94 ± 1.65 | 4.06 ± 1.72 |

### Houses

| Model | Control/readout | σ | p1/p2 | UU | UI | IU | II |
|---|---|---:|---|---:|---:|---:|---:|
| LP | New σ, two noise | 4.5096 | 0.377/0.378 | 5.36 ± 1.80 | 5.60 ± 1.96 | 5.64 ± 1.50 | 4.66 ± 1.77 |
| LP | σ = 2, original two-noise fit | 2.0000 | 0.380/0.370 | 5.34 ± 1.98 | 5.48 ± 2.12 | 5.48 ± 1.76 | 4.84 ± 1.81 |
| LP | σ = 2, new two-noise fit | 2.0000 | 0.377/0.378 | 5.30 ± 1.87 | 5.72 ± 1.88 | 5.52 ± 1.72 | 4.74 ± 1.72 |
| LP | New σ, p1 only | 4.5096 | 0.377/0.377 | 5.36 ± 1.80 | 5.60 ± 1.96 | 5.56 ± 1.51 | 4.66 ± 1.84 |
| CNN | New σ, two noise | 6.7540 | 0.366/0.365 | 5.42 ± 1.82 | 6.00 ± 1.82 | 6.16 ± 1.62 | 4.48 ± 1.39 |
| CNN | σ = 2, original two-noise fit | 2.0000 | 0.360/0.350 | 5.38 ± 1.70 | 6.02 ± 1.46 | 6.04 ± 1.77 | 4.34 ± 1.53 |
| CNN | σ = 2, new two-noise fit | 2.0000 | 0.366/0.365 | 5.48 ± 1.64 | 5.90 ± 1.57 | 6.20 ± 1.74 | 4.48 ± 1.47 |
| CNN | New σ, p1 only | 6.7540 | 0.366/0.366 | 5.42 ± 1.82 | 6.00 ± 1.82 | 6.16 ± 1.68 | 4.42 ± 1.47 |

### Objects

| Model | Control/readout | σ | p1/p2 | UU | UI | IU | II |
|---|---|---:|---|---:|---:|---:|---:|
| LP | New σ, two noise | 4.5096 | 0.377/0.378 | 1.76 ± 1.04 | 2.64 ± 1.34 | 3.08 ± 1.56 | 1.98 ± 1.25 |
| LP | σ = 2, original two-noise fit | 2.0000 | 0.380/0.370 | 2.02 ± 1.25 | 2.86 ± 1.28 | 3.24 ± 1.51 | 2.06 ± 1.22 |
| LP | σ = 2, new two-noise fit | 2.0000 | 0.377/0.378 | 1.86 ± 1.21 | 2.84 ± 1.27 | 3.40 ± 1.50 | 2.26 ± 1.24 |
| LP | New σ, p1 only | 4.5096 | 0.377/0.377 | 1.76 ± 1.04 | 2.64 ± 1.34 | 3.06 ± 1.45 | 2.04 ± 1.26 |
| CNN | New σ, two noise | 6.7540 | 0.366/0.365 | 1.56 ± 1.30 | 3.56 ± 1.61 | 3.40 ± 1.55 | 2.20 ± 1.58 |
| CNN | σ = 2, original two-noise fit | 2.0000 | 0.360/0.350 | 1.46 ± 1.22 | 3.88 ± 1.57 | 3.32 ± 1.57 | 2.14 ± 1.54 |
| CNN | σ = 2, new two-noise fit | 2.0000 | 0.366/0.365 | 1.44 ± 1.11 | 3.86 ± 1.63 | 3.54 ± 1.50 | 2.32 ± 1.42 |
| CNN | New σ, p1 only | 6.7540 | 0.366/0.366 | 1.56 ± 1.30 | 3.56 ± 1.61 | 3.42 ± 1.53 | 2.24 ± 1.57 |

## Object orientation gaps

Range = maximum minus minimum of the four condition mean errors. Lower means flatter orientation performance. These are descriptive ranges, not tests of statistical significance.

| Model | Readout | Range, error points | Range, accuracy percentage points |
|---|---|---:|---:|
| LP | New σ, two noise | 1.32 | 11.00 |
| LP | σ = 2, original two-noise fit | 1.22 | 10.17 |
| LP | σ = 2, new two-noise fit | 1.54 | 12.83 |
| LP | New σ, p1 only | 1.30 | 10.83 |
| CNN | New σ, two noise | 2.00 | 16.67 |
| CNN | σ = 2, original two-noise fit | 2.42 | 20.17 |
| CNN | σ = 2, new two-noise fit | 2.42 | 20.17 |
| CNN | New σ, p1 only | 2.00 | 16.67 |

Against the original σ = 2 two-noise fit, CNN narrows from 2.42 to 2.00 error points; LP changes from 1.22 to 1.32. At matched new noise, changing σ alone narrows LP’s range from 1.54 to 1.32. The added p2 parameter itself makes almost no difference because p1 and p2 converge to nearly equal values.

## Fitting-seed results: older screenshot convention

The earlier screenshot reported the same 50 seeds used to fit face UU and II. For a transparent comparison, this table evaluates the new checkpoint and σ on those same seeds, 101–150, after freezing the parameters. These are in-sample results, not independent validation. Only face UU and II were fitted.

| Row | UU | UI | IU | II |
|---|---:|---:|---:|---:|
| LP faces | 0.90 ± 0.84 | 4.54 ± 1.73 | 4.66 ± 1.49 | 4.34 ± 1.87 |
| CNN faces | 0.90 ± 0.86 | 6.16 ± 1.72 | 5.90 ± 1.89 | 4.34 ± 1.47 |
| LP houses | 4.96 ± 1.69 | 5.60 ± 1.62 | 5.42 ± 1.31 | 4.72 ± 1.64 |
| CNN houses | 5.72 ± 1.62 | 5.76 ± 1.80 | 5.88 ± 1.75 | 4.44 ± 1.64 |
| LP objects | 1.76 ± 1.12 | 2.88 ± 1.38 | 3.36 ± 1.32 | 2.18 ± 1.17 |
| CNN objects | 1.56 ± 1.21 | 3.74 ± 1.47 | 3.60 ± 1.40 | 1.90 ± 1.13 |

## Scope and verification

The older screenshot used Anvil seed-45 checkpoints with an older training recipe. This experiment uses newer Delta seed-46 no-generic checkpoints. The comparison controls above hold the newer checkpoint fixed; changes relative to the older screenshot cannot be attributed to σ alone.

The frozen object bank includes 16 categories absent from the revised training diet. This inherited limitation is retained so the mechanism comparisons use identical inputs. No images, coordinates, checkpoints or training jobs changed. The active baseline remains original KDE with σ = 2; this is an experimental variant.

Independent verification passed 6,400 category/seed records, 76,800 trial records, 128 condition cells and 400 original calibration controls. All 19,200 primary trial scores were replayed exactly through the immutable original engine. All 200 selected calibration-seed outcomes were reproduced with the full probe batch. Noise-grid selection, noise assignment, object transfer, counts, means, sample SD/SEM, paired effects and source/cache/checkpoint hashes were checked. Another 1,600 fitting-seed diagnostic rows were checked for internal consistency and agreement with the fitted anchors.

[Verification receipt](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/INDEPENDENT_VERIFICATION.json) · [Full summary](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/results/SUMMARY.json) · [LP calibration](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/calibration/lp.json) · [CNN calibration](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/calibration/cnn.json) · [Raw records](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-new-sigma-two-noise-20261116/results/all_records.csv)
