# Finer alpha search for house recognition

The finer search found development winners LP alpha0.40 (sigma4.5096) and CNN alpha0.15 (sigma1.6885). On50fresh simulation seeds, LP houseUU errors changed from5.36to5.38, while CNN changed from5.60to5.42. Paired mean improvement ± Monte CarloSEM was LP-0.02 ±0.163 and CNN+0.18 ±0.098 errors. The development LP gain did not persist; the CNN gain is small and uncertain. No alpha in the entire grid produced mean houseUU errors below houseII errors on validation. There is no demonstrated reason from this sweep to replace the registered sigma2 baseline with an established optimum.

Completed CPU-only exploration on the same latest no-generic LP/CNN models, trainingseed46 epoch124. Same20faces+20houses study,12old/new pairs/category,10study/32testfixations, all four orientations; fixed single noise LP.38/CNN.36. No noise refit or retraining. Scoring uses the stable unfloored KDE variant from the previous sweep.

We tested20 alpha values from.05to4, plus sigma2. Sigma=alpha times the exact previous upright-study-only mean binary-code distance (reference seed42). Each model has one shared sigma across faces/houses and orientations. Development seeds101–150 select the alpha with minimum mean houseUU errors, breaking ties by smaller alpha. Validation uses50new seeds151–200. All grid values are retained on validation for transparency, but the chosen alpha is never reselected from validation. This is a holdout of simulation randomness and study/probe splits within the same64-photo banks, not an independent image dataset or training replication.

Values below are errors out of12, mean ± sampleSD. HumanSD is across participants; modelSD across simulation seeds.

## Development — baseline and development-selected alpha

| Row | Noise | Alpha | Sigma | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|---:|---:|
| Human houses | — | — | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP baseline | 0.38 | — | 2.0000 | 5.38 ± 1.58 | 5.76 ± 1.72 | 5.52 ± 1.66 | 5.26 ± 1.86 |
| LP selected on development | 0.38 | 0.4 | 4.5096 | 5.02 ± 1.73 | 5.64 ± 1.66 | 5.40 ± 1.47 | 4.76 ± 1.66 |
| CNN baseline | 0.36 | — | 2.0000 | 5.44 ± 1.86 | 5.88 ± 1.67 | 6.04 ± 1.82 | 4.36 ± 1.82 |
| CNN selected on development | 0.36 | 0.15 | 1.6885 | 5.38 ± 1.83 | 5.92 ± 1.78 | 6.14 ± 1.76 | 4.54 ± 1.82 |

## Validation — baseline and development-selected alpha

| Row | Noise | Alpha | Sigma | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|---:|---:|
| Human houses | — | — | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP baseline | 0.38 | — | 2.0000 | 5.36 ± 1.51 | 5.58 ± 1.49 | 5.64 ± 1.65 | 4.78 ± 1.52 |
| LP selected on development | 0.38 | 0.4 | 4.5096 | 5.38 ± 1.32 | 5.56 ± 1.37 | 5.66 ± 1.87 | 4.82 ± 1.35 |
| CNN baseline | 0.36 | — | 2.0000 | 5.60 ± 1.59 | 6.04 ± 1.46 | 6.08 ± 1.78 | 4.46 ± 1.63 |
| CNN selected on development | 0.36 | 0.15 | 1.6885 | 5.42 ± 1.49 | 6.04 ± 1.59 | 6.08 ± 1.59 | 4.56 ± 1.63 |

## Paired validation effects

Positive improvement means fewer errors with the selected alpha than sigma2 on the same seed. Selection occurred on development only. The SD/SEM here describe paired differences, not the individual condition distribution.

| Model | Category | Condition | Mean improvement | Paired SD | Paired SEM |
|---|---|---|---:|---:|---:|
| LP | faces | UU | 0.0800 | 0.7516 | 0.1063 |
| LP | faces | UI | -0.0200 | 1.1337 | 0.1603 |
| LP | faces | IU | -0.0400 | 1.1599 | 0.1640 |
| LP | faces | II | 0.0800 | 1.4263 | 0.2017 |
| LP | houses | UU | -0.0200 | 1.1516 | 0.1629 |
| LP | houses | UI | 0.0200 | 1.1156 | 0.1578 |
| LP | houses | IU | -0.0200 | 1.1865 | 0.1678 |
| LP | houses | II | -0.0400 | 1.2610 | 0.1783 |
| CNN | faces | UU | 0.0000 | 0.2857 | 0.0404 |
| CNN | faces | UI | -0.0400 | 1.0490 | 0.1484 |
| CNN | faces | IU | -0.0800 | 0.7516 | 0.1063 |
| CNN | faces | II | -0.0800 | 0.6652 | 0.0941 |
| CNN | houses | UU | 0.1800 | 0.6908 | 0.0977 |
| CNN | houses | UI | 0.0000 | 0.6999 | 0.0990 |
| CNN | houses | IU | 0.0000 | 0.6701 | 0.0948 |
| CNN | houses | II | -0.1000 | 0.7626 | 0.1079 |

## Face outcomes at the selected alpha

| Model | Split | UU | UI | IU | II |
|---|---|---:|---:|---:|---:|
| LP | development | 0.96 ± 0.95 | 4.58 ± 1.60 | 4.68 ± 1.61 | 4.42 ± 1.87 |
| LP | validation | 0.94 ± 0.93 | 4.46 ± 1.99 | 4.84 ± 1.39 | 3.84 ± 1.39 |
| CNN | development | 0.90 ± 0.86 | 6.34 ± 1.48 | 6.22 ± 1.45 | 4.48 ± 1.42 |
| CNN | validation | 0.70 ± 0.84 | 5.98 ± 1.67 | 6.22 ± 1.61 | 4.52 ± 1.49 |

## Full grid

House human RMSE is sqrt(mean squared error in the four condition means against historical human error means). It is descriptive only and was not the alpha-selection criterion. UU<II checks that single piece of the human ordering; all condition means are supplied so it cannot conceal other ordering failures.

### LP development

| Alpha | Sigma | UU | UI | IU | II | Human RMSE | UU < II? | Faces UU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed sigma2 | 2.0000 | 5.38 ± 1.58 | 5.76 ± 1.72 | 5.52 ± 1.66 | 5.26 ± 1.86 | 2.5537 | no | 1.00 ± 0.83 |
| 0.05 | 0.5637 | 5.54 ± 1.58 | 5.90 ± 1.49 | 5.56 ± 1.77 | 5.62 ± 1.64 | 2.7191 | yes | 2.04 ± 1.19 |
| 0.075 | 0.8455 | 5.56 ± 1.64 | 5.82 ± 1.49 | 5.62 ± 1.69 | 5.52 ± 1.78 | 2.6952 | no | 1.94 ± 1.11 |
| 0.1 | 1.1274 | 5.58 ± 1.64 | 5.94 ± 1.58 | 5.54 ± 1.67 | 5.50 ± 1.85 | 2.7151 | no | 1.70 ± 1.04 |
| 0.125 | 1.4092 | 5.60 ± 1.69 | 5.86 ± 1.53 | 5.42 ± 1.65 | 5.40 ± 1.81 | 2.6569 | no | 1.38 ± 0.95 |
| 0.15 | 1.6911 | 5.26 ± 1.59 | 5.72 ± 1.69 | 5.54 ± 1.55 | 5.34 ± 1.78 | 2.5248 | yes | 1.24 ± 0.89 |
| 0.175 | 1.9729 | 5.40 ± 1.56 | 5.70 ± 1.69 | 5.54 ± 1.63 | 5.32 ± 1.87 | 2.5582 | no | 1.04 ± 0.83 |
| 0.2 | 2.2548 | 5.38 ± 1.63 | 5.68 ± 1.67 | 5.46 ± 1.62 | 5.16 ± 1.74 | 2.5012 | no | 1.00 ± 0.86 |
| 0.25 | 2.8185 | 5.20 ± 1.70 | 5.68 ± 1.71 | 5.44 ± 1.53 | 5.06 ± 1.75 | 2.4240 | no | 1.00 ± 0.88 |
| 0.3 | 3.3822 | 5.06 ± 1.81 | 5.70 ± 1.63 | 5.50 ± 1.43 | 4.90 ± 1.73 | 2.3743 | no | 0.96 ± 0.92 |
| 0.35 | 3.9459 | 5.08 ± 1.79 | 5.64 ± 1.68 | 5.40 ± 1.43 | 4.80 ± 1.69 | 2.3257 | no | 0.98 ± 0.94 |
| 0.4 | 4.5096 | 5.02 ± 1.73 | 5.64 ± 1.66 | 5.40 ± 1.47 | 4.76 ± 1.66 | 2.3016 | no | 0.96 ± 0.95 |
| 0.45 | 5.0733 | 5.02 ± 1.73 | 5.66 ± 1.70 | 5.36 ± 1.51 | 4.76 ± 1.67 | 2.2992 | no | 0.96 ± 0.92 |
| 0.5 | 5.6370 | 5.08 ± 1.71 | 5.64 ± 1.70 | 5.40 ± 1.55 | 4.80 ± 1.73 | 2.3257 | no | 1.00 ± 0.95 |
| 0.6 | 6.7643 | 5.12 ± 1.72 | 5.58 ± 1.68 | 5.44 ± 1.55 | 4.78 ± 1.68 | 2.3260 | no | 1.02 ± 0.98 |
| 0.75 | 8.4554 | 5.16 ± 1.75 | 5.58 ± 1.72 | 5.46 ± 1.55 | 4.76 ± 1.66 | 2.3399 | no | 1.00 ± 0.99 |
| 1.0 | 11.2739 | 5.16 ± 1.75 | 5.60 ± 1.71 | 5.50 ± 1.63 | 4.72 ± 1.60 | 2.3488 | no | 1.02 ± 0.98 |
| 1.5 | 16.9109 | 5.14 ± 1.73 | 5.60 ± 1.71 | 5.50 ± 1.63 | 4.70 ± 1.67 | 2.3398 | no | 1.02 ± 0.98 |
| 2.0 | 22.5478 | 5.14 ± 1.73 | 5.60 ± 1.71 | 5.50 ± 1.63 | 4.70 ± 1.67 | 2.3398 | no | 1.00 ± 0.97 |
| 3.0 | 33.8217 | 5.14 ± 1.73 | 5.60 ± 1.71 | 5.52 ± 1.63 | 4.70 ± 1.67 | 2.3443 | no | 1.00 ± 0.97 |
| 4.0 | 45.0956 | 5.14 ± 1.73 | 5.60 ± 1.71 | 5.52 ± 1.63 | 4.70 ± 1.67 | 2.3443 | no | 1.00 ± 0.97 |

### LP validation

| Alpha | Sigma | UU | UI | IU | II | Human RMSE | UU < II? | Faces UU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed sigma2 | 2.0000 | 5.36 ± 1.51 | 5.58 ± 1.49 | 5.64 ± 1.65 | 4.78 ± 1.52 | 2.4459 | no | 1.02 ± 0.82 |
| 0.05 | 0.5637 | 5.70 ± 1.83 | 5.64 ± 1.63 | 5.90 ± 1.83 | 5.14 ± 1.48 | 2.6846 | no | 1.92 ± 1.14 |
| 0.075 | 0.8455 | 5.64 ± 1.85 | 5.70 ± 1.58 | 6.00 ± 1.84 | 5.04 ± 1.44 | 2.6890 | no | 1.88 ± 1.10 |
| 0.1 | 1.1274 | 5.60 ± 1.70 | 5.72 ± 1.54 | 5.84 ± 1.71 | 4.96 ± 1.46 | 2.6323 | no | 1.74 ± 1.14 |
| 0.125 | 1.4092 | 5.72 ± 1.64 | 5.68 ± 1.50 | 5.70 ± 1.52 | 4.88 ± 1.49 | 2.6181 | no | 1.44 ± 1.03 |
| 0.15 | 1.6911 | 5.58 ± 1.60 | 5.62 ± 1.60 | 5.72 ± 1.58 | 4.76 ± 1.49 | 2.5436 | no | 1.12 ± 0.94 |
| 0.175 | 1.9729 | 5.42 ± 1.51 | 5.58 ± 1.53 | 5.66 ± 1.66 | 4.76 ± 1.52 | 2.4669 | no | 1.04 ± 0.86 |
| 0.2 | 2.2548 | 5.32 ± 1.53 | 5.54 ± 1.40 | 5.72 ± 1.73 | 4.66 ± 1.49 | 2.4245 | no | 1.06 ± 0.82 |
| 0.25 | 2.8185 | 5.34 ± 1.42 | 5.42 ± 1.34 | 5.72 ± 1.86 | 4.78 ± 1.50 | 2.4145 | no | 1.02 ± 0.89 |
| 0.3 | 3.3822 | 5.28 ± 1.37 | 5.54 ± 1.28 | 5.72 ± 1.82 | 4.86 ± 1.43 | 2.4394 | no | 1.00 ± 0.93 |
| 0.35 | 3.9459 | 5.40 ± 1.31 | 5.52 ± 1.33 | 5.68 ± 1.83 | 4.86 ± 1.36 | 2.4628 | no | 1.00 ± 0.93 |
| 0.4 | 4.5096 | 5.38 ± 1.32 | 5.56 ± 1.37 | 5.66 ± 1.87 | 4.82 ± 1.35 | 2.4569 | no | 0.94 ± 0.93 |
| 0.45 | 5.0733 | 5.38 ± 1.35 | 5.58 ± 1.44 | 5.66 ± 1.88 | 4.80 ± 1.36 | 2.4596 | no | 0.96 ± 0.92 |
| 0.5 | 5.6370 | 5.38 ± 1.41 | 5.58 ± 1.49 | 5.70 ± 1.84 | 4.78 ± 1.39 | 2.4659 | no | 0.94 ± 0.91 |
| 0.6 | 6.7643 | 5.34 ± 1.53 | 5.52 ± 1.47 | 5.74 ± 1.83 | 4.78 ± 1.40 | 2.4461 | no | 0.92 ± 0.88 |
| 0.75 | 8.4554 | 5.36 ± 1.56 | 5.50 ± 1.40 | 5.76 ± 1.80 | 4.80 ± 1.41 | 2.4546 | no | 0.98 ± 0.94 |
| 1.0 | 11.2739 | 5.38 ± 1.55 | 5.50 ± 1.40 | 5.82 ± 1.77 | 4.76 ± 1.42 | 2.4698 | no | 0.98 ± 0.89 |
| 1.5 | 16.9109 | 5.40 ± 1.56 | 5.54 ± 1.39 | 5.86 ± 1.76 | 4.78 ± 1.46 | 2.4993 | no | 0.98 ± 0.89 |
| 2.0 | 22.5478 | 5.42 ± 1.49 | 5.52 ± 1.37 | 5.86 ± 1.76 | 4.74 ± 1.48 | 2.4949 | no | 1.00 ± 0.88 |
| 3.0 | 33.8217 | 5.42 ± 1.49 | 5.52 ± 1.37 | 5.86 ± 1.76 | 4.74 ± 1.48 | 2.4949 | no | 1.00 ± 0.88 |
| 4.0 | 45.0956 | 5.42 ± 1.49 | 5.52 ± 1.37 | 5.86 ± 1.76 | 4.74 ± 1.48 | 2.4949 | no | 1.00 ± 0.88 |

### CNN development

| Alpha | Sigma | UU | UI | IU | II | Human RMSE | UU < II? | Faces UU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed sigma2 | 2.0000 | 5.44 ± 1.86 | 5.88 ± 1.67 | 6.04 ± 1.82 | 4.36 ± 1.82 | 2.6039 | no | 0.78 ± 0.82 |
| 0.05 | 0.5628 | 5.46 ± 1.75 | 5.74 ± 1.60 | 6.04 ± 1.83 | 5.24 ± 1.96 | 2.6857 | no | 1.28 ± 0.93 |
| 0.075 | 0.8442 | 5.54 ± 1.80 | 5.76 ± 1.67 | 6.08 ± 1.75 | 5.06 ± 2.00 | 2.6962 | no | 1.18 ± 0.94 |
| 0.1 | 1.1257 | 5.60 ± 1.77 | 5.90 ± 1.75 | 6.16 ± 1.71 | 4.96 ± 2.05 | 2.7579 | no | 1.16 ± 0.91 |
| 0.125 | 1.4071 | 5.48 ± 1.79 | 5.74 ± 1.72 | 6.14 ± 1.77 | 4.74 ± 1.85 | 2.6431 | no | 1.06 ± 0.84 |
| 0.15 | 1.6885 | 5.38 ± 1.83 | 5.92 ± 1.78 | 6.14 ± 1.76 | 4.54 ± 1.82 | 2.6403 | no | 0.90 ± 0.86 |
| 0.175 | 1.9699 | 5.44 ± 1.86 | 5.90 ± 1.67 | 6.08 ± 1.84 | 4.34 ± 1.78 | 2.6179 | no | 0.80 ± 0.83 |
| 0.2 | 2.2513 | 5.48 ± 1.79 | 5.82 ± 1.75 | 6.04 ± 1.68 | 4.30 ± 1.71 | 2.5938 | no | 0.68 ± 0.77 |
| 0.25 | 2.8142 | 5.48 ± 1.78 | 5.82 ± 1.66 | 5.94 ± 1.43 | 4.44 ± 1.69 | 2.5819 | no | 0.72 ± 0.73 |
| 0.3 | 3.3770 | 5.40 ± 1.68 | 5.80 ± 1.64 | 5.98 ± 1.56 | 4.40 ± 1.55 | 2.5571 | no | 0.72 ± 0.70 |
| 0.35 | 3.9398 | 5.48 ± 1.68 | 5.76 ± 1.66 | 5.88 ± 1.57 | 4.32 ± 1.57 | 2.5389 | no | 0.74 ± 0.75 |
| 0.4 | 4.5027 | 5.44 ± 1.69 | 5.82 ± 1.60 | 5.84 ± 1.66 | 4.28 ± 1.60 | 2.5305 | no | 0.70 ± 0.76 |
| 0.45 | 5.0655 | 5.42 ± 1.67 | 5.84 ± 1.63 | 5.84 ± 1.68 | 4.22 ± 1.68 | 2.5251 | no | 0.68 ± 0.77 |
| 0.5 | 5.6283 | 5.46 ± 1.62 | 5.82 ± 1.66 | 5.84 ± 1.69 | 4.24 ± 1.72 | 2.5335 | no | 0.68 ± 0.79 |
| 0.6 | 6.7540 | 5.44 ± 1.59 | 5.82 ± 1.72 | 5.90 ± 1.73 | 4.24 ± 1.70 | 2.5416 | no | 0.70 ± 0.79 |
| 0.75 | 8.4425 | 5.46 ± 1.61 | 5.88 ± 1.75 | 5.92 ± 1.72 | 4.30 ± 1.73 | 2.5753 | no | 0.72 ± 0.81 |
| 1.0 | 11.2566 | 5.48 ± 1.63 | 5.88 ± 1.77 | 5.88 ± 1.72 | 4.28 ± 1.68 | 2.5703 | no | 0.70 ± 0.76 |
| 1.5 | 16.8850 | 5.48 ± 1.63 | 5.88 ± 1.77 | 5.88 ± 1.72 | 4.28 ± 1.68 | 2.5703 | no | 0.72 ± 0.78 |
| 2.0 | 22.5133 | 5.48 ± 1.63 | 5.90 ± 1.79 | 5.88 ± 1.72 | 4.28 ± 1.68 | 2.5762 | no | 0.72 ± 0.78 |
| 3.0 | 33.7699 | 5.48 ± 1.63 | 5.90 ± 1.79 | 5.88 ± 1.72 | 4.28 ± 1.68 | 2.5762 | no | 0.72 ± 0.78 |
| 4.0 | 45.0266 | 5.48 ± 1.63 | 5.90 ± 1.79 | 5.88 ± 1.72 | 4.28 ± 1.68 | 2.5762 | no | 0.72 ± 0.78 |

### CNN validation

| Alpha | Sigma | UU | UI | IU | II | Human RMSE | UU < II? | Faces UU |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed sigma2 | 2.0000 | 5.60 ± 1.59 | 6.04 ± 1.46 | 6.08 ± 1.78 | 4.46 ± 1.63 | 2.7191 | no | 0.70 ± 0.89 |
| 0.05 | 0.5628 | 5.66 ± 1.85 | 6.12 ± 1.65 | 6.10 ± 1.54 | 5.06 ± 1.81 | 2.8377 | no | 1.20 ± 0.99 |
| 0.075 | 0.8442 | 5.72 ± 1.76 | 6.02 ± 1.68 | 6.12 ± 1.53 | 5.08 ± 1.91 | 2.8354 | no | 1.04 ± 0.90 |
| 0.1 | 1.1257 | 5.64 ± 1.63 | 6.06 ± 1.67 | 6.14 ± 1.67 | 5.00 ± 1.99 | 2.8155 | no | 1.02 ± 0.98 |
| 0.125 | 1.4071 | 5.58 ± 1.60 | 6.06 ± 1.60 | 6.10 ± 1.68 | 4.70 ± 1.82 | 2.7491 | no | 0.94 ± 0.84 |
| 0.15 | 1.6885 | 5.42 ± 1.49 | 6.04 ± 1.59 | 6.08 ± 1.59 | 4.56 ± 1.63 | 2.6744 | no | 0.70 ± 0.84 |
| 0.175 | 1.9699 | 5.58 ± 1.57 | 6.02 ± 1.49 | 6.10 ± 1.80 | 4.48 ± 1.63 | 2.7139 | no | 0.72 ± 0.90 |
| 0.2 | 2.2513 | 5.64 ± 1.68 | 6.12 ± 1.48 | 6.02 ± 1.55 | 4.36 ± 1.66 | 2.7316 | no | 0.62 ± 0.85 |
| 0.25 | 2.8142 | 5.58 ± 1.69 | 6.08 ± 1.45 | 6.00 ± 1.62 | 4.50 ± 1.72 | 2.7093 | no | 0.52 ± 0.79 |
| 0.3 | 3.3770 | 5.58 ± 1.69 | 6.08 ± 1.40 | 5.94 ± 1.70 | 4.44 ± 1.69 | 2.6894 | no | 0.46 ± 0.73 |
| 0.35 | 3.9398 | 5.62 ± 1.70 | 6.16 ± 1.45 | 5.94 ± 1.63 | 4.34 ± 1.78 | 2.7170 | no | 0.44 ± 0.70 |
| 0.4 | 4.5027 | 5.62 ± 1.66 | 6.14 ± 1.46 | 5.96 ± 1.64 | 4.36 ± 1.75 | 2.7173 | no | 0.44 ± 0.70 |
| 0.45 | 5.0655 | 5.60 ± 1.64 | 6.20 ± 1.48 | 5.92 ± 1.64 | 4.32 ± 1.71 | 2.7167 | no | 0.44 ± 0.70 |
| 0.5 | 5.6283 | 5.56 ± 1.68 | 6.22 ± 1.47 | 5.88 ± 1.62 | 4.32 ± 1.71 | 2.7014 | no | 0.48 ± 0.74 |
| 0.6 | 6.7540 | 5.56 ± 1.69 | 6.18 ± 1.49 | 5.82 ± 1.59 | 4.32 ± 1.72 | 2.6755 | no | 0.48 ± 0.74 |
| 0.75 | 8.4425 | 5.60 ± 1.70 | 6.20 ± 1.55 | 5.86 ± 1.54 | 4.38 ± 1.74 | 2.7083 | no | 0.48 ± 0.76 |
| 1.0 | 11.2566 | 5.58 ± 1.73 | 6.22 ± 1.56 | 5.84 ± 1.56 | 4.42 ± 1.74 | 2.7074 | no | 0.48 ± 0.76 |
| 1.5 | 16.8850 | 5.64 ± 1.74 | 6.18 ± 1.59 | 5.82 ± 1.57 | 4.44 ± 1.77 | 2.7112 | no | 0.50 ± 0.76 |
| 2.0 | 22.5133 | 5.64 ± 1.74 | 6.16 ± 1.58 | 5.80 ± 1.56 | 4.44 ± 1.77 | 2.7007 | no | 0.50 ± 0.76 |
| 3.0 | 33.7699 | 5.64 ± 1.74 | 6.16 ± 1.58 | 5.80 ± 1.56 | 4.44 ± 1.77 | 2.7007 | no | 0.50 ± 0.76 |
| 4.0 | 45.0266 | 5.64 ± 1.74 | 6.16 ± 1.58 | 5.80 ± 1.56 | 4.44 ± 1.77 | 2.7007 | no | 0.50 ± 0.76 |

## Verification

Require the independent verification receipt in this operation folder:33,600unique category records,672condition cells,50seeds/cell,168paired orientation contrasts,800exact legacy controls and4,800overlapping parent correctness masks; source/cache/spec/checkpoint provenance, unchanged distance-reference records, count consistency, all means/sampleSD/SEM, development-only selection and paired validation effects independently recomputed. The selected alpha is best only for the declared development houseUU criterion among this finite grid. No global optimum or improvement on independent house images is established.
