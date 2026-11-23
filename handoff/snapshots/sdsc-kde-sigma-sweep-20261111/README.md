# Distance-scaled KDE bandwidth diagnostic

Authorized exploration of sigma using the completed no-generic-house LP and CNN
ResNet18 epoch124 training-seed46 checkpoints. No training, new images, cache
extraction, noise calibration or change to previous registrations.

Design: 20 faces plus 20 houses studied, 12 old/new pairs per category, shared
40-item memory, 10 study and 32 probe fixations, UU/UI/IU/II, 50 seeds101-150.
Two separate fixed-noise arms: existing face-calibrated single noise (LP.38,
CNN.36), and zero added XOR (Bernoulli sampling remains active).

For each model/noise arm, estimate d_bar from all 79,800 unordered distinct
pairs among its 400 upright study fixation codes from independent seed42.
The study bank has equal numbers of face and house items. No probe vectors,
human outcomes, or test accuracy are used to set this reference. Codes are
256-dimensional binary vectors after Bernoulli and XOR. Euclidean distance
is sqrt(Hamming). Report within-photo, between-photo/same-category and cross-
category distributions, and raw sigmoid-vector distances separately.

Frozen sigma candidates: 2, and d_bar times .125,.25,.5,1,2. Same scalar per
model/noise arm for all categories, seeds and orientations. This is an
exploratory mean-distance scaling rule, not a claim of optimal bandwidth.

At each sigma compare original float32 exp/mean/log(+1e-12) scoring against
float64 logsumexp without that additive floor. Removing the floor changes
readout semantics; it is explicitly a separate arm, not silently substituted.
Both retain sum across probe fixations and maximum across 40 memory items.
The Gaussian normalization cancels because sigma/width/fixation counts are
shared across candidates within a comparison. Strict old>new defines correct.

Require exact replay of 1,600 original category/condition/seed counts at
legacy sigma2 (including both noise arms). Retain per-pair correctness/ties,
legacy2 decision changes, floor-dominated fractions, distance histograms,
source/cache hashes, all 19,200 category records, sample SD/SEM, paired
inversion contrasts. Separate synthetic numerical tests and a small actual-
cache pilot run on allocated CPUs before the full sweep. Two CPU cores,
8GiB, bounded30minutes. Never edit or resubmit the parent jobs.

Background: Gaussian bandwidth heuristics based on pairwise distances are
studied by Garreau et al., https://arxiv.org/abs/1707.07269 . Our reference
uses the mean, not their median convention. Stable log-sum-exp documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
