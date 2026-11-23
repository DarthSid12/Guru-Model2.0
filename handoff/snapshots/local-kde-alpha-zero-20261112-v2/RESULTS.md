# Alpha search at zero added XOR noise

The zero-XOR sweep found lower upright-house errors with small alpha. The development-selected alpha.005 gives sigma.0534807 forLP and.0528662 forCNN. On50fresh simulation seeds, LP houseUU errors fell3.40to3.00 and CNN5.16to4.74, compared with sigma2 at the same zero noise. Paired improvements were LP0.40 ±0.103SEM and CNN0.42 ±0.099SEM. Very small alpha values form a plateau; .005is a tie-broken grid endpoint, not an established unique optimum. LP alpha.005,.01,.025have identical development houseUU means; CNN does too. Neither model achieves human-like UU/II ordering at any tested alpha on validation.

This completed on the user's Mac using saved probabilities and torch2.5.1 CPU. Independent verification passed38,400rows/768cells/50seeds,800sigma2controls and4,800exact earlier correctness masks. The original strict float comparison failed only on about1e-15 Mac/Linux distance-summary differences; that attempt is preserved separately. Version2 verifies identical Hamming histograms, checks moments within1e-10, and reuses the stored original distance scale exactly. No SDSC password/code was needed and no remote job was submitted.

Completed local Mac CPU-only exploration on the same latest no-generic LP/CNN models, trainingseed46 epoch124. Same20faces+20houses study,12old/new pairs/category,10study/32testfixations, all four orientations; zero added XOR noise for both models; stochastic Bernoulli sampling remains on. No noise fitting or retraining. This is a zero-XOR diagnostic, not a face-calibrated human simulation. Scoring uses the stable unfloored KDE variant from the previous sweep.

We tested23 alpha values from.005to4, plus sigma2. Sigma=alpha times the exact previous upright-study-only mean binary-code distance (reference seed42). Each model has one shared sigma across faces/houses and orientations. Development seeds101–150 select the alpha with minimum mean houseUU errors, breaking ties by smaller alpha. Validation uses50new seeds151–200. All grid values are retained on validation for transparency, but the chosen alpha is never reselected from validation. This is a holdout of simulation randomness and study/probe splits within the same64-photo banks, not an independent image dataset or training replication.

Values below are errors out of12, mean ± sampleSD. HumanSD is across participants; modelSD across simulation seeds.

## Development — baseline and development-selected alpha

| Row | Noise | Alpha | Sigma | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|---:|---:|
| Human houses | — | — | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP baseline | 0.00 | — | 2.0000 | 3.50 ± 1.64 | 5.02 ± 1.53 | 5.12 ± 1.71 | 2.50 ± 1.52 |
| LP selected on development | 0.00 | 0.005 | 0.0535 | 3.12 ± 1.66 | 4.94 ± 1.57 | 5.08 ± 1.74 | 2.08 ± 1.44 |
| CNN baseline | 0.00 | — | 2.0000 | 5.00 ± 1.55 | 5.90 ± 1.79 | 5.78 ± 1.58 | 1.74 ± 1.27 |
| CNN selected on development | 0.00 | 0.005 | 0.0529 | 4.52 ± 1.49 | 5.78 ± 1.80 | 5.72 ± 1.62 | 1.44 ± 1.25 |

## Validation — baseline and development-selected alpha

| Row | Noise | Alpha | Sigma | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|---:|---:|
| Human houses | — | — | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP baseline | 0.00 | — | 2.0000 | 3.40 ± 1.70 | 5.04 ± 1.48 | 5.18 ± 1.47 | 2.38 ± 1.32 |
| LP selected on development | 0.00 | 0.005 | 0.0535 | 3.00 ± 1.54 | 4.86 ± 1.39 | 5.16 ± 1.56 | 1.86 ± 1.23 |
| CNN baseline | 0.00 | — | 2.0000 | 5.16 ± 1.73 | 5.78 ± 1.46 | 5.90 ± 1.69 | 1.64 ± 1.16 |
| CNN selected on development | 0.00 | 0.005 | 0.0529 | 4.74 ± 1.58 | 5.70 ± 1.49 | 5.90 ± 1.54 | 1.44 ± 1.05 |

## Paired validation effects

Positive improvement means fewer errors with the selected alpha than sigma2 on the same seed. Selection occurred on development only. The SD/SEM here describe paired differences, not the individual condition distribution.

| Model | Category | Condition | Mean improvement | Paired SD | Paired SEM |
|---|---|---|---:|---:|---:|
| LP | faces | UU | 0.0000 | 0.0000 | 0.0000 |
| LP | faces | UI | 0.0000 | 0.4949 | 0.0700 |
| LP | faces | IU | -0.0600 | 0.6197 | 0.0876 |
| LP | faces | II | 0.2600 | 0.4870 | 0.0689 |
| LP | houses | UU | 0.4000 | 0.7284 | 0.1030 |
| LP | houses | UI | 0.1800 | 0.4819 | 0.0682 |
| LP | houses | IU | 0.0200 | 0.7420 | 0.1049 |
| LP | houses | II | 0.5200 | 0.7624 | 0.1078 |
| CNN | faces | UU | 0.0400 | 0.1979 | 0.0280 |
| CNN | faces | UI | -0.1000 | 0.6468 | 0.0915 |
| CNN | faces | IU | 0.1400 | 1.0500 | 0.1485 |
| CNN | faces | II | 0.3400 | 0.5194 | 0.0735 |
| CNN | houses | UU | 0.4200 | 0.7025 | 0.0993 |
| CNN | houses | UI | 0.0800 | 0.7783 | 0.1101 |
| CNN | houses | IU | 0.0000 | 0.7284 | 0.1030 |
| CNN | houses | II | 0.2000 | 0.6389 | 0.0904 |

## Face outcomes at the selected alpha

| Model | Split | UU | UI | IU | II |
|---|---|---:|---:|---:|---:|
| LP | development | 0.00 ± 0.00 | 2.56 ± 1.34 | 2.54 ± 1.45 | 0.82 ± 0.87 |
| LP | validation | 0.00 ± 0.00 | 2.34 ± 1.22 | 2.64 ± 1.17 | 0.50 ± 0.79 |
| CNN | development | 0.04 ± 0.20 | 5.88 ± 1.86 | 6.00 ± 1.82 | 1.12 ± 0.98 |
| CNN | validation | 0.04 ± 0.20 | 6.12 ± 1.77 | 5.86 ± 2.14 | 1.76 ± 1.13 |

## Full grid

House human RMSE is sqrt(mean squared error in the four condition means against historical human error means). It is descriptive only and was not the alpha-selection criterion. UU<II checks that single piece of the human ordering; all condition means are supplied so it cannot conceal other ordering failures.

### LP development

| Alpha | Sigma | UU | UI | IU | II | Human RMSE | UU < II? | Faces UU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed sigma2 | 2.0000 | 3.50 ± 1.64 | 5.02 ± 1.53 | 5.12 ± 1.71 | 2.50 ± 1.52 | 1.5796 | no | 0.00 ± 0.00 |
| 0.005 | 0.0535 | 3.12 ± 1.66 | 4.94 ± 1.57 | 5.08 ± 1.74 | 2.08 ± 1.44 | 1.5521 | no | 0.00 ± 0.00 |
| 0.01 | 0.1070 | 3.12 ± 1.66 | 4.94 ± 1.57 | 5.08 ± 1.74 | 2.08 ± 1.44 | 1.5521 | no | 0.00 ± 0.00 |
| 0.025 | 0.2674 | 3.12 ± 1.66 | 4.94 ± 1.57 | 5.08 ± 1.74 | 2.08 ± 1.44 | 1.5521 | no | 0.00 ± 0.00 |
| 0.05 | 0.5348 | 3.14 ± 1.63 | 4.94 ± 1.57 | 5.06 ± 1.75 | 2.08 ± 1.44 | 1.5498 | no | 0.00 ± 0.00 |
| 0.075 | 0.8022 | 3.16 ± 1.62 | 4.98 ± 1.53 | 5.06 ± 1.75 | 2.08 ± 1.44 | 1.5662 | no | 0.00 ± 0.00 |
| 0.1 | 1.0696 | 3.20 ± 1.62 | 5.00 ± 1.54 | 5.08 ± 1.76 | 2.10 ± 1.46 | 1.5800 | no | 0.00 ± 0.00 |
| 0.125 | 1.3370 | 3.22 ± 1.62 | 4.96 ± 1.52 | 5.12 ± 1.77 | 2.14 ± 1.46 | 1.5720 | no | 0.00 ± 0.00 |
| 0.15 | 1.6044 | 3.38 ± 1.61 | 4.98 ± 1.52 | 5.12 ± 1.77 | 2.28 ± 1.53 | 1.5790 | no | 0.00 ± 0.00 |
| 0.175 | 1.8718 | 3.46 ± 1.63 | 5.00 ± 1.54 | 5.08 ± 1.69 | 2.42 ± 1.51 | 1.5664 | no | 0.00 ± 0.00 |
| 0.2 | 2.1392 | 3.56 ± 1.63 | 5.04 ± 1.55 | 5.12 ± 1.71 | 2.54 ± 1.53 | 1.5931 | no | 0.00 ± 0.00 |
| 0.25 | 2.6740 | 3.62 ± 1.68 | 5.04 ± 1.48 | 5.06 ± 1.66 | 2.66 ± 1.51 | 1.5747 | no | 0.00 ± 0.00 |
| 0.3 | 3.2088 | 3.74 ± 1.71 | 5.08 ± 1.47 | 5.02 ± 1.65 | 2.82 ± 1.48 | 1.5888 | no | 0.00 ± 0.00 |
| 0.35 | 3.7437 | 3.76 ± 1.72 | 5.10 ± 1.49 | 5.06 ± 1.63 | 2.90 ± 1.46 | 1.6036 | no | 0.00 ± 0.00 |
| 0.4 | 4.2785 | 3.78 ± 1.72 | 5.10 ± 1.49 | 5.08 ± 1.64 | 3.00 ± 1.48 | 1.6062 | no | 0.00 ± 0.00 |
| 0.45 | 4.8133 | 3.82 ± 1.75 | 5.12 ± 1.55 | 5.06 ± 1.65 | 3.02 ± 1.45 | 1.6165 | no | 0.00 ± 0.00 |
| 0.5 | 5.3481 | 3.84 ± 1.77 | 5.10 ± 1.57 | 5.08 ± 1.63 | 3.02 ± 1.39 | 1.6196 | no | 0.02 ± 0.14 |
| 0.6 | 6.4177 | 3.84 ± 1.77 | 5.06 ± 1.56 | 5.12 ± 1.66 | 3.02 ± 1.39 | 1.6162 | no | 0.02 ± 0.14 |
| 0.75 | 8.0221 | 3.84 ± 1.80 | 5.06 ± 1.56 | 5.12 ± 1.66 | 3.08 ± 1.37 | 1.6127 | no | 0.02 ± 0.14 |
| 1.0 | 10.6961 | 3.84 ± 1.80 | 5.06 ± 1.58 | 5.08 ± 1.65 | 3.12 ± 1.36 | 1.6004 | no | 0.02 ± 0.14 |
| 1.5 | 16.0442 | 3.84 ± 1.80 | 5.06 ± 1.58 | 5.12 ± 1.64 | 3.12 ± 1.36 | 1.6108 | no | 0.02 ± 0.14 |
| 2.0 | 21.3923 | 3.84 ± 1.80 | 5.06 ± 1.58 | 5.12 ± 1.64 | 3.12 ± 1.36 | 1.6108 | no | 0.02 ± 0.14 |
| 3.0 | 32.0884 | 3.84 ± 1.80 | 5.06 ± 1.58 | 5.12 ± 1.64 | 3.12 ± 1.36 | 1.6108 | no | 0.02 ± 0.14 |
| 4.0 | 42.7846 | 3.84 ± 1.80 | 5.06 ± 1.58 | 5.12 ± 1.64 | 3.12 ± 1.36 | 1.6108 | no | 0.02 ± 0.14 |

### LP validation

| Alpha | Sigma | UU | UI | IU | II | Human RMSE | UU < II? | Faces UU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed sigma2 | 2.0000 | 3.40 ± 1.70 | 5.04 ± 1.48 | 5.18 ± 1.47 | 2.38 ± 1.32 | 1.6020 | no | 0.00 ± 0.00 |
| 0.005 | 0.0535 | 3.00 ± 1.54 | 4.86 ± 1.39 | 5.16 ± 1.56 | 1.86 ± 1.23 | 1.5827 | no | 0.00 ± 0.00 |
| 0.01 | 0.1070 | 3.00 ± 1.54 | 4.86 ± 1.39 | 5.16 ± 1.56 | 1.86 ± 1.23 | 1.5827 | no | 0.00 ± 0.00 |
| 0.025 | 0.2674 | 3.00 ± 1.54 | 4.86 ± 1.39 | 5.16 ± 1.56 | 1.84 ± 1.20 | 1.5876 | no | 0.00 ± 0.00 |
| 0.05 | 0.5348 | 3.00 ± 1.54 | 4.86 ± 1.39 | 5.18 ± 1.55 | 1.82 ± 1.17 | 1.5981 | no | 0.00 ± 0.00 |
| 0.075 | 0.8022 | 3.02 ± 1.55 | 4.86 ± 1.39 | 5.16 ± 1.61 | 1.82 ± 1.19 | 1.5951 | no | 0.00 ± 0.00 |
| 0.1 | 1.0696 | 3.04 ± 1.52 | 4.90 ± 1.36 | 5.14 ± 1.55 | 1.88 ± 1.15 | 1.5901 | no | 0.00 ± 0.00 |
| 0.125 | 1.3370 | 3.08 ± 1.56 | 4.88 ± 1.38 | 5.14 ± 1.54 | 2.00 ± 1.25 | 1.5607 | no | 0.00 ± 0.00 |
| 0.15 | 1.6044 | 3.08 ± 1.56 | 4.96 ± 1.43 | 5.16 ± 1.53 | 2.22 ± 1.31 | 1.5464 | no | 0.00 ± 0.00 |
| 0.175 | 1.8718 | 3.28 ± 1.67 | 5.02 ± 1.49 | 5.18 ± 1.45 | 2.32 ± 1.28 | 1.5843 | no | 0.00 ± 0.00 |
| 0.2 | 2.1392 | 3.44 ± 1.75 | 4.98 ± 1.50 | 5.20 ± 1.48 | 2.42 ± 1.43 | 1.5883 | no | 0.00 ± 0.00 |
| 0.25 | 2.6740 | 3.60 ± 1.78 | 5.00 ± 1.54 | 5.14 ± 1.55 | 2.48 ± 1.34 | 1.6019 | no | 0.02 ± 0.14 |
| 0.3 | 3.2088 | 3.80 ± 1.76 | 5.02 ± 1.48 | 5.12 ± 1.53 | 2.56 ± 1.34 | 1.6375 | no | 0.02 ± 0.14 |
| 0.35 | 3.7437 | 3.84 ± 1.80 | 5.04 ± 1.52 | 5.08 ± 1.51 | 2.66 ± 1.33 | 1.6313 | no | 0.02 ± 0.14 |
| 0.4 | 4.2785 | 3.90 ± 1.80 | 5.04 ± 1.52 | 5.06 ± 1.49 | 2.66 ± 1.33 | 1.6413 | no | 0.04 ± 0.20 |
| 0.45 | 4.8133 | 3.96 ± 1.86 | 5.10 ± 1.49 | 5.04 ± 1.50 | 2.72 ± 1.33 | 1.6653 | no | 0.04 ± 0.20 |
| 0.5 | 5.3481 | 4.06 ± 1.80 | 5.10 ± 1.49 | 5.00 ± 1.44 | 2.74 ± 1.34 | 1.6803 | no | 0.04 ± 0.20 |
| 0.6 | 6.4177 | 4.04 ± 1.82 | 5.12 ± 1.48 | 5.00 ± 1.46 | 2.76 ± 1.33 | 1.6796 | no | 0.04 ± 0.20 |
| 0.75 | 8.0221 | 4.06 ± 1.81 | 5.12 ± 1.48 | 5.02 ± 1.46 | 2.76 ± 1.33 | 1.6897 | no | 0.06 ± 0.24 |
| 1.0 | 10.6961 | 4.08 ± 1.85 | 5.12 ± 1.48 | 5.04 ± 1.48 | 2.76 ± 1.33 | 1.6998 | no | 0.08 ± 0.27 |
| 1.5 | 16.0442 | 4.08 ± 1.85 | 5.12 ± 1.48 | 5.06 ± 1.46 | 2.78 ± 1.34 | 1.7027 | no | 0.08 ± 0.27 |
| 2.0 | 21.3923 | 4.08 ± 1.85 | 5.12 ± 1.47 | 5.06 ± 1.46 | 2.78 ± 1.34 | 1.7027 | no | 0.08 ± 0.27 |
| 3.0 | 32.0884 | 4.08 ± 1.85 | 5.12 ± 1.47 | 5.06 ± 1.46 | 2.78 ± 1.34 | 1.7027 | no | 0.08 ± 0.27 |
| 4.0 | 42.7846 | 4.08 ± 1.85 | 5.12 ± 1.47 | 5.06 ± 1.46 | 2.78 ± 1.34 | 1.7027 | no | 0.08 ± 0.27 |

### CNN development

| Alpha | Sigma | UU | UI | IU | II | Human RMSE | UU < II? | Faces UU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed sigma2 | 2.0000 | 5.00 ± 1.55 | 5.90 ± 1.79 | 5.78 ± 1.58 | 1.74 ± 1.27 | 2.5129 | no | 0.14 ± 0.40 |
| 0.005 | 0.0529 | 4.52 ± 1.49 | 5.78 ± 1.80 | 5.72 ± 1.62 | 1.44 ± 1.25 | 2.3945 | no | 0.04 ± 0.20 |
| 0.01 | 0.1057 | 4.52 ± 1.49 | 5.80 ± 1.76 | 5.72 ± 1.62 | 1.44 ± 1.25 | 2.4006 | no | 0.04 ± 0.20 |
| 0.025 | 0.2643 | 4.52 ± 1.49 | 5.78 ± 1.80 | 5.72 ± 1.62 | 1.44 ± 1.25 | 2.3945 | no | 0.04 ± 0.20 |
| 0.05 | 0.5287 | 4.54 ± 1.49 | 5.78 ± 1.80 | 5.72 ± 1.62 | 1.44 ± 1.25 | 2.3993 | no | 0.04 ± 0.20 |
| 0.075 | 0.7930 | 4.54 ± 1.49 | 5.78 ± 1.80 | 5.74 ± 1.63 | 1.42 ± 1.25 | 2.4082 | no | 0.04 ± 0.20 |
| 0.1 | 1.0573 | 4.60 ± 1.53 | 5.76 ± 1.82 | 5.78 ± 1.59 | 1.44 ± 1.23 | 2.4223 | no | 0.04 ± 0.20 |
| 0.125 | 1.3217 | 4.76 ± 1.61 | 5.84 ± 1.80 | 5.80 ± 1.58 | 1.48 ± 1.30 | 2.4830 | no | 0.06 ± 0.24 |
| 0.15 | 1.5860 | 4.90 ± 1.61 | 5.84 ± 1.80 | 5.78 ± 1.57 | 1.60 ± 1.28 | 2.4922 | no | 0.06 ± 0.24 |
| 0.175 | 1.8503 | 4.96 ± 1.58 | 5.90 ± 1.82 | 5.76 ± 1.66 | 1.72 ± 1.28 | 2.5007 | no | 0.12 ± 0.39 |
| 0.2 | 2.1146 | 5.00 ± 1.55 | 5.96 ± 1.76 | 5.76 ± 1.59 | 1.76 ± 1.27 | 2.5233 | no | 0.16 ± 0.47 |
| 0.25 | 2.6433 | 5.06 ± 1.52 | 5.90 ± 1.78 | 5.72 ± 1.68 | 1.92 ± 1.37 | 2.4872 | no | 0.16 ± 0.47 |
| 0.3 | 3.1720 | 5.06 ± 1.52 | 5.88 ± 1.77 | 5.80 ± 1.70 | 2.04 ± 1.35 | 2.4825 | no | 0.18 ± 0.52 |
| 0.35 | 3.7006 | 5.08 ± 1.52 | 5.88 ± 1.77 | 5.78 ± 1.69 | 2.10 ± 1.34 | 2.4753 | no | 0.18 ± 0.52 |
| 0.4 | 4.2293 | 5.08 ± 1.54 | 5.88 ± 1.77 | 5.84 ± 1.71 | 2.18 ± 1.29 | 2.4793 | no | 0.18 ± 0.52 |
| 0.45 | 4.7580 | 5.10 ± 1.56 | 5.88 ± 1.77 | 5.88 ± 1.66 | 2.28 ± 1.33 | 2.4829 | no | 0.18 ± 0.52 |
| 0.5 | 5.2866 | 5.10 ± 1.56 | 5.84 ± 1.78 | 5.88 ± 1.66 | 2.28 ± 1.33 | 2.4708 | no | 0.18 ± 0.52 |
| 0.6 | 6.3439 | 5.10 ± 1.56 | 5.82 ± 1.78 | 5.94 ± 1.63 | 2.34 ± 1.38 | 2.4731 | no | 0.18 ± 0.52 |
| 0.75 | 7.9299 | 5.12 ± 1.56 | 5.80 ± 1.77 | 5.94 ± 1.63 | 2.38 ± 1.37 | 2.4687 | no | 0.18 ± 0.52 |
| 1.0 | 10.5732 | 5.12 ± 1.56 | 5.80 ± 1.77 | 5.94 ± 1.65 | 2.44 ± 1.42 | 2.4625 | no | 0.18 ± 0.52 |
| 1.5 | 15.8598 | 5.12 ± 1.56 | 5.80 ± 1.77 | 5.94 ± 1.65 | 2.44 ± 1.42 | 2.4625 | no | 0.18 ± 0.52 |
| 2.0 | 21.1465 | 5.12 ± 1.56 | 5.80 ± 1.77 | 5.94 ± 1.65 | 2.44 ± 1.42 | 2.4625 | no | 0.18 ± 0.52 |
| 3.0 | 31.7197 | 5.12 ± 1.56 | 5.80 ± 1.77 | 5.94 ± 1.65 | 2.44 ± 1.42 | 2.4625 | no | 0.18 ± 0.52 |
| 4.0 | 42.2929 | 5.12 ± 1.56 | 5.80 ± 1.77 | 5.94 ± 1.65 | 2.44 ± 1.42 | 2.4625 | no | 0.18 ± 0.52 |

### CNN validation

| Alpha | Sigma | UU | UI | IU | II | Human RMSE | UU < II? | Faces UU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed sigma2 | 2.0000 | 5.16 ± 1.73 | 5.78 ± 1.46 | 5.90 ± 1.69 | 1.64 ± 1.16 | 2.5681 | no | 0.08 ± 0.27 |
| 0.005 | 0.0529 | 4.74 ± 1.58 | 5.70 ± 1.49 | 5.90 ± 1.54 | 1.44 ± 1.05 | 2.4692 | no | 0.04 ± 0.20 |
| 0.01 | 0.1057 | 4.74 ± 1.58 | 5.70 ± 1.49 | 5.90 ± 1.54 | 1.44 ± 1.05 | 2.4692 | no | 0.04 ± 0.20 |
| 0.025 | 0.2643 | 4.74 ± 1.58 | 5.70 ± 1.49 | 5.90 ± 1.54 | 1.44 ± 1.05 | 2.4692 | no | 0.04 ± 0.20 |
| 0.05 | 0.5287 | 4.76 ± 1.57 | 5.70 ± 1.49 | 5.90 ± 1.54 | 1.44 ± 1.05 | 2.4743 | no | 0.04 ± 0.20 |
| 0.075 | 0.7930 | 4.76 ± 1.57 | 5.74 ± 1.45 | 5.88 ± 1.57 | 1.46 ± 1.05 | 2.4769 | no | 0.04 ± 0.20 |
| 0.1 | 1.0573 | 4.78 ± 1.64 | 5.78 ± 1.42 | 5.90 ± 1.56 | 1.48 ± 1.05 | 2.4947 | no | 0.04 ± 0.20 |
| 0.125 | 1.3217 | 4.92 ± 1.66 | 5.82 ± 1.38 | 5.96 ± 1.64 | 1.50 ± 1.07 | 2.5537 | no | 0.04 ± 0.20 |
| 0.15 | 1.5860 | 5.04 ± 1.67 | 5.76 ± 1.39 | 5.96 ± 1.71 | 1.46 ± 1.07 | 2.5763 | no | 0.04 ± 0.20 |
| 0.175 | 1.8503 | 5.14 ± 1.70 | 5.78 ± 1.45 | 5.90 ± 1.66 | 1.54 ± 1.13 | 2.5802 | no | 0.06 ± 0.24 |
| 0.2 | 2.1146 | 5.18 ± 1.75 | 5.80 ± 1.50 | 5.92 ± 1.71 | 1.66 ± 1.15 | 2.5809 | no | 0.08 ± 0.27 |
| 0.25 | 2.6433 | 5.16 ± 1.68 | 5.80 ± 1.48 | 5.94 ± 1.63 | 1.84 ± 1.20 | 2.5507 | no | 0.10 ± 0.30 |
| 0.3 | 3.1720 | 5.30 ± 1.72 | 5.76 ± 1.52 | 5.88 ± 1.64 | 1.94 ± 1.22 | 2.5509 | no | 0.10 ± 0.30 |
| 0.35 | 3.7006 | 5.36 ± 1.71 | 5.78 ± 1.53 | 5.82 ± 1.52 | 2.02 ± 1.22 | 2.5493 | no | 0.10 ± 0.30 |
| 0.4 | 4.2293 | 5.36 ± 1.69 | 5.78 ± 1.50 | 5.80 ± 1.50 | 2.06 ± 1.25 | 2.5392 | no | 0.10 ± 0.30 |
| 0.45 | 4.7580 | 5.36 ± 1.69 | 5.74 ± 1.54 | 5.80 ± 1.50 | 2.10 ± 1.30 | 2.5224 | no | 0.10 ± 0.30 |
| 0.5 | 5.2866 | 5.34 ± 1.66 | 5.72 ± 1.54 | 5.82 ± 1.49 | 2.14 ± 1.31 | 2.5101 | no | 0.10 ± 0.30 |
| 0.6 | 6.3439 | 5.36 ± 1.65 | 5.72 ± 1.54 | 5.82 ± 1.49 | 2.18 ± 1.27 | 2.5113 | no | 0.10 ± 0.30 |
| 0.75 | 7.9299 | 5.34 ± 1.66 | 5.72 ± 1.54 | 5.82 ± 1.49 | 2.22 ± 1.31 | 2.5002 | no | 0.10 ± 0.30 |
| 1.0 | 10.5732 | 5.36 ± 1.61 | 5.72 ± 1.54 | 5.82 ± 1.49 | 2.26 ± 1.34 | 2.5017 | no | 0.10 ± 0.30 |
| 1.5 | 15.8598 | 5.40 ± 1.64 | 5.72 ± 1.54 | 5.78 ± 1.47 | 2.28 ± 1.33 | 2.5025 | no | 0.10 ± 0.30 |
| 2.0 | 21.1465 | 5.40 ± 1.64 | 5.72 ± 1.54 | 5.78 ± 1.47 | 2.28 ± 1.33 | 2.5025 | no | 0.10 ± 0.30 |
| 3.0 | 31.7197 | 5.40 ± 1.64 | 5.74 ± 1.55 | 5.80 ± 1.46 | 2.28 ± 1.33 | 2.5130 | no | 0.10 ± 0.30 |
| 4.0 | 42.2929 | 5.40 ± 1.64 | 5.74 ± 1.55 | 5.80 ± 1.46 | 2.28 ± 1.33 | 2.5130 | no | 0.10 ± 0.30 |

## Verification

Require the independent verification receipt in this operation folder:38,400unique category records,768condition cells,50seeds/cell,192paired orientation contrasts,800exact legacy controls and4,800overlapping parent correctness masks; source/cache/spec/checkpoint provenance and cross-platform replay of original SDSC decisions, unchanged distance-reference records, count consistency, all means/sampleSD/SEM, development-only selection and paired validation effects independently recomputed. The selected alpha is best only for the declared development houseUU criterion among this finite grid. No global optimum or improvement on independent house images is established.
