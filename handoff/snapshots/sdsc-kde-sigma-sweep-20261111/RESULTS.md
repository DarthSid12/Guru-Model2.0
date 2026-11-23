# KDE sigma: distance-scaled exploration

The completed experiment supports measuring sigma relative to code distances, but does not establish one optimal bandwidth. At fixed face-calibrated noise, broadening sigma gives a small LP upright-house improvement and essentially none for CNN. Every tested setting still has better II than UU house recognition. The LP improvement is small compared with Monte Carlo uncertainty; these data do not show a resolved improvement in learned house representations.

One bounded SDSC CPU job55694699 completed in46seconds. Independently verified19,200 category records,384cells,50simulation seeds101–150 per cell,16 actual-cache pilot score checks and1,600 exact legacy-sigma2 control replays. Both models use epoch124/trainingseed46; this uncertainty is over simulation seeds, not independently trained networks.

## Definition and scale

The Gaussian kernel is exp(-||x-y||²/(2 sigma²)). Its inputs are256-bit fixation codes after Bernoulli sampling and XOR noise. For these codes, squared Euclidean distance equals Hamming distance. We used sigma = alpha * d_bar, where d_bar is the mean Euclidean distance across79,800 unordered distinct pairs of400 upright study fixation codes:20face plus20house images,1xx0fixations/image. Reference seed42 is separate from evaluation seeds; no probe vectors, human outcomes or test performance were used to set this scale. One shared sigma per model/noise arm is held across categories and orientations.

Mean-distance scaling supplies units; alpha is still a free parameter and is not automatically validated by this normalization. Distance-based kernel bandwidth heuristics are established starting points ([Garreau et al.](https://arxiv.org/abs/1707.07269)); this experiment deliberately uses a mean rather than claiming their median convention.

| Model | Added XOR noise | Mean binary-code distance | Tested sigmas |
|---|---:|---:|---|
| LP | 0.38 | 11.2739 | 2.0000, 1.4092, 2.8185, 5.6370, 11.2739, 22.5478 |
| CNN | 0.36 | 11.2566 | 2.0000, 1.4071, 2.8142, 5.6283, 11.2566, 22.5133 |
| LP | 0.00 | 10.6961 | 2.0000, 1.3370, 2.6740, 5.3481, 10.6961, 21.3923 |
| CNN | 0.00 | 10.5732 | 2.0000, 1.3217, 2.6433, 5.2866, 10.5732, 21.1465 |

## Houses at existing face-calibrated noise

Errors out of12, mean ± sampleSD. Lower is better. HumanSD is across participants; modelSD is across50Monte Carlo seeds. This section uses the stable unfloored readout. At sigma2 it gives exactly the original decisions; the small-sigma difference is reported separately below.

| Row | Noise | Sigma | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|---:|
| Human houses | — | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP | 0.38 | 2.0000 (fixed_2) | 5.38 ± 1.58 | 5.76 ± 1.72 | 5.52 ± 1.66 | 5.26 ± 1.86 |
| LP | 0.38 | 1.4092 (distance_x0.125) | 5.60 ± 1.69 | 5.86 ± 1.53 | 5.42 ± 1.65 | 5.40 ± 1.81 |
| LP | 0.38 | 2.8185 (distance_x0.25) | 5.20 ± 1.70 | 5.68 ± 1.71 | 5.44 ± 1.53 | 5.06 ± 1.75 |
| LP | 0.38 | 5.6370 (distance_x0.5) | 5.08 ± 1.71 | 5.64 ± 1.70 | 5.40 ± 1.55 | 4.80 ± 1.73 |
| LP | 0.38 | 11.2739 (distance_x1.0) | 5.16 ± 1.75 | 5.60 ± 1.71 | 5.50 ± 1.63 | 4.72 ± 1.60 |
| LP | 0.38 | 22.5478 (distance_x2.0) | 5.14 ± 1.73 | 5.60 ± 1.71 | 5.50 ± 1.63 | 4.70 ± 1.67 |
| CNN | 0.36 | 2.0000 (fixed_2) | 5.44 ± 1.86 | 5.88 ± 1.67 | 6.04 ± 1.82 | 4.36 ± 1.82 |
| CNN | 0.36 | 1.4071 (distance_x0.125) | 5.48 ± 1.79 | 5.74 ± 1.72 | 6.14 ± 1.77 | 4.74 ± 1.85 |
| CNN | 0.36 | 2.8142 (distance_x0.25) | 5.48 ± 1.78 | 5.82 ± 1.66 | 5.94 ± 1.43 | 4.44 ± 1.69 |
| CNN | 0.36 | 5.6283 (distance_x0.5) | 5.46 ± 1.62 | 5.82 ± 1.66 | 5.84 ± 1.69 | 4.24 ± 1.72 |
| CNN | 0.36 | 11.2566 (distance_x1.0) | 5.48 ± 1.63 | 5.88 ± 1.77 | 5.88 ± 1.72 | 4.28 ± 1.68 |
| CNN | 0.36 | 22.5133 (distance_x2.0) | 5.48 ± 1.63 | 5.90 ± 1.79 | 5.88 ± 1.72 | 4.28 ± 1.68 |

For the illustrative half-distance candidate, LP reduces UU errors by0.30 with paired Monte CarloSEM0.16 (pairedSD1.13); CNN changes by-0.02 with SEM0.12 (pairedSD0.84). These are exploratory comparisons on the same seeds, not held-out evidence that a chosen sigma generalizes. LP faceUU stays1.00errors; CNN faceUU changes0.78to0.68errors. Noise was held fixed and was not freshly fitted to humans after changing sigma.

## Why noise matters

| Model | Noise | Within same house photo: mean distance | Between different house photos: mean distance |
|---|---:|---:|---:|
| LP | 0.00 | 8.7739 | 9.6616 |
| LP | 0.38 | 11.1929 | 11.2171 |
| CNN | 0.00 | 8.7543 | 9.5664 |
| CNN | 0.36 | 11.1151 | 11.1841 |

These reference distributions use upright study codes only; they do not constitute an independent identity test or a claim that all vector information is erased. Nonetheless, the much smaller within/between mean-distance gap after added noise helps explain why bandwidth alone has limited benefit. For independent XOR flips with probability p, expected Hamming distance transforms as E[H_after]=2p(1-p)D+(1-2p)^2 E[H_before]. Thus differences of expected Hamming distances shrink by(1-2p)^2:0.0576forLPp=.38 and0.0784forCNNp=.36. The mean Euclidean distances above are measured directly; they are not square roots of mean Hamming distance.

## Separate zero-XOR diagnostic

Zero XOR retains stochastic Bernoulli encoding. It is not a human-calibrated replacement.

| Model | Sigma | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|
| LP | 2.0000 (fixed_2) | 3.50 ± 1.64 | 5.02 ± 1.53 | 5.12 ± 1.71 | 2.50 ± 1.52 |
| LP | 1.3370 (distance_x0.125) | 3.22 ± 1.62 | 4.96 ± 1.52 | 5.12 ± 1.77 | 2.14 ± 1.46 |
| LP | 2.6740 (distance_x0.25) | 3.62 ± 1.68 | 5.04 ± 1.48 | 5.06 ± 1.66 | 2.66 ± 1.51 |
| LP | 5.3481 (distance_x0.5) | 3.84 ± 1.77 | 5.10 ± 1.57 | 5.08 ± 1.63 | 3.02 ± 1.39 |
| LP | 10.6961 (distance_x1.0) | 3.84 ± 1.80 | 5.06 ± 1.58 | 5.08 ± 1.65 | 3.12 ± 1.36 |
| LP | 21.3923 (distance_x2.0) | 3.84 ± 1.80 | 5.06 ± 1.58 | 5.12 ± 1.64 | 3.12 ± 1.36 |
| CNN | 2.0000 (fixed_2) | 5.00 ± 1.55 | 5.90 ± 1.79 | 5.78 ± 1.58 | 1.74 ± 1.27 |
| CNN | 1.3217 (distance_x0.125) | 4.76 ± 1.61 | 5.84 ± 1.80 | 5.80 ± 1.58 | 1.48 ± 1.30 |
| CNN | 2.6433 (distance_x0.25) | 5.06 ± 1.52 | 5.90 ± 1.78 | 5.72 ± 1.68 | 1.92 ± 1.37 |
| CNN | 5.2866 (distance_x0.5) | 5.10 ± 1.56 | 5.84 ± 1.78 | 5.88 ± 1.66 | 2.28 ± 1.33 |
| CNN | 10.5732 (distance_x1.0) | 5.12 ± 1.56 | 5.80 ± 1.77 | 5.94 ± 1.65 | 2.44 ± 1.42 |
| CNN | 21.1465 (distance_x2.0) | 5.12 ± 1.56 | 5.80 ± 1.77 | 5.94 ± 1.65 | 2.44 ± 1.42 |

## Numerical floor control

The historical implementation computes log(mean(exp(-distance²/(2sigma²)))+1e-12) in float32. The comparison readout uses float64 logsumexp minus log(10), with no additive floor. Logsumexp provides stable log-of-sum computation ([SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html)); removing the explicit floor changes the scoring model and is therefore a separate recorded arm.

Both versions have identical correctness decisions at sigma2 for all1,600 control records. At small sigma the floor changes many decisions without necessarily causing exact score ties. For example, LP with noise.38 and sigma1.4092 has faceUU2.78errors under the old floor versus1.38under the unfloored calculation; housesUU5.66versus5.60. This is a readout artifact and should not be interpreted as a property of the representation alone.

## Interpretation

Use the dimensionless alpha parameterization for future experiments, with the code-space/distance-estimation procedure specified and frozen. There is no support here for simply replacing sigma2 with the full average distance or selecting a universal optimum from these evaluation pictures. Choosing alpha using independent recognition-validation stimuli and then fitting/freezing face noise would avoid choosing a house bandwidth on the same houses used for reporting. Jointly fitting noise and sigma to a single faceUU target would be underconstrained. Existing model training and registered evaluation baselines remain preserved.

Full face/house tables for every noise, bandwidth and readout are in completed-sdsc/results/TABLES.md. Raw masks, ties, means, sampleSD/SEM, paired contrasts and distance histograms are retained in the same results folder. The independent verification receipt is INDEPENDENT_VERIFICATION.json.
