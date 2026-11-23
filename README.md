# Guru Model 2.0: ResNet18 and Yin agent handoff

**Handoff date: November 23, 2026.** This branch captures the code and verified results behind the latest conversation. Start here instead of treating an older launcher, report, or snapshot status as current.

The current question is why the models distinguish novel and studied houses poorly in the Yin memory task, despite strong classification of their training house identities. The latest requested baseline uses **fixed KDE sigma = 2 and one noise parameter per model, calibrated only on upright-study/upright-test faces (UU)**. The latest diagnostic pools all studied-house fixations into one KDE. Pooling did not resolve the house problem. It has **not replaced** the original readout.

## Start here

1. Read this README and [AGENTS.md](AGENTS.md).
2. Use [the portable evaluator](yin_tests/current_kde/run.py) for current cached-feature experiments.
3. Read the [single-noise results](handoff/snapshots/local-yin-sigma2-single-uu-20261119/RESULTS_ACCURACY.md) and [pooled-house results](handoff/snapshots/local-yin-pooled-house-kde-20261119/RESULTS.md).
4. Use [data and cluster locations](handoff/DATA_AND_OPERATIONS.md) to find artifacts. Check live jobs before acting. Historical status files are not current scheduler evidence.
5. Preserve registered experiments. Create a new output directory/version for a scientific change. Do not launch, cancel, or duplicate training merely to resume this handoff.

This branch starts from `biggerDatasets` commit `2c362b4f56399edf737af22d9f39413e6334862c`. It includes the verified training resume fix, the later experiment sources, calibration evidence, result records, and a portable entry point. It does not contain training photographs, feature caches, or model weights. Those need separate authorized access.

## Models behind the latest results

Both are scratch **ResNet18**, final epoch **124**, **training seed 46**, trained on Delta with no generic-house category. Do not confuse them with the earlier seed-45 models or the older VGG16-BN experiments.

| Model | Checkpoint SHA-256 |
|---|---|
| LP | `cf516526e48eed6b6769fef3faf85b94cf3192527b29c38a988d36a3825a05ea` |
| CNN | `e39d54ad778c50df1c0a4820836762916c73f2bac702f13c3ef50321f50b90ad` |

The LP model uses the log-polar transform. Both models use foveation and a shared 256-dimensional sigmoid representation, temperature 2, dropout 0.3, and a classifier trained over 2,240 labels. Behavioral evaluation uses the representation, not the classifier's class confidence.

Training: AdamW, learning rate 0.001, cosine scheduling, weight decay 0.05 on weights rather than bias/norm parameters, batch size 256 fixation crops, 2,000 optimizer steps per epoch, 16 random fixations per image from 32 stored coordinates, 180-pixel crops, bfloat16 AMP, channels-last, eight workers, square-root category sampling, patience 20. Later curriculum stages use a 500-optimizer-step learning-rate warmup. Training uses upright images with scale/rotation augmentation, not 180-degree inversion. See the [exact registered arguments](handoff/snapshots/resnet18-no-generic-20261109/cluster/recipe.py).

`training/train.py` includes the recorded resume repair: if an early-stopped stage's last checkpoint is resumed before stage advancement was saved, skip its remaining epochs rather than restarting that stage. The file matches the completed training source hash. Older VGG shell launchers remain historical; they are not the current ResNet18 recipe.

### Training diet

| Category | Classes | Train / validation / test images per class | Source |
|---|---:|---|---|
| Faces | 2,048 identities | 200 / 12 / 12 | VGGFace2 |
| Objects | 64 categories | 500 / 100 / 100 | Revised Wordbank/Open Images bank |
| Houses | 128 identities | 3 / 1 / 1 | Houses535 originals and synthetic viewpoints |

House training views are original, left 15°, and right 15°. Left 30° is validation; right 30°, inverted, is the classifier test. The five later random synthetic views are excluded. There is no generic house class. Synthetic views can drift architecturally; their generation is not evidence of true multiview identity consistency.

| Epochs | Face identities | Object categories | House identities |
|---|---:|---:|---:|
| 1–2 | 4 | 0 | 0 |
| 3–4 | 8 | 0 | 0 |
| 5–8 | 16 | 4 | 0 |
| 9–12 | 32 | 8 | 4 |
| 13–20 | 64 | 16 | 8 |
| 21–28 | 128 | 32 | 16 |
| 29–44 | 256 | 64 | 32 |
| 45–60 | 512 | 64 | 64 |
| 61–92 | 1,024 | 64 | 128 |
| 93–124 | 2,048 | 64 | 128 |

The object curriculum uses MacArthur–Bates/Wordbank **understands**, not produces. The 4/8/16/32/64 selection checkpoints are 8/10/12/15/18 months, not asserted ages of visual learning. [Exact words introduced at each stage](handoff/snapshots/resnet18-no-generic-20261109/DIET.md#exact-object-introductions). That frozen document's old job-state prose is historical.

## Current Yin design

The fixed evaluation banks contain 64 CFD white male neutral faces, 64 DalHouses photographs, and 64 Wordbank object pictures. DalHouses evaluation identities are separate from the synthetic training houses. The corrected house bank includes DalHouse018 and excludes duplicate DalHouse071. Image bytes and coordinate hashes are pinned in [PACKED_HOUSES.json](handoff/snapshots/sdsc-yin-no-generic-s46-20261111/completed-sdsc/PACKED_HOUSES.json).

For each simulation seed:

* Study **20 faces + 20 houses**, interleaved into one 40-image memory.
* Test **12 old/new pairs per category**, comparing faces with faces and houses with houses.
* Separately study **20 faces + 20 objects**, using exactly the same fitted noise. Test 12 pairs per category.
* Each studied image contributes 10 fixation vectors. Each test image contributes 32.
* The 64-image banks stay fixed. Selection of studied images, novel images, pairings, and ordering changes deterministically with seed. The two study designs have distinct deterministic RNG streams; they do not necessarily use identical face selections.

**One noise value: LP 0.373, CNN 0.364.** Each model's rate is used for every study and test presentation in UU, UI, IU, and II. Only face UU in the faces/houses design is fitted to 92.5833%; II is a prediction. Noise transfers unchanged to faces/objects.

Simulation seeds are 101–150. These are the same seeds used for calibration and reporting, so the tables are **not independent validation**. Model SD measures variation across simulation seeds, not across training seeds or human subjects. Use accuracy percentages, mean ± sample SD, in **UU, UI, IU, II** order. The composite is UU minus the average of UI, IU, II. Human composite SD is unavailable without covariance information.

Noise means independent XOR bit flips after Bernoulli sampling of sigmoid outputs. Zero added noise still retains Bernoulli variability. Human airplane values are a historical reference on different stimuli. The fixed object evaluation bank contains **16 categories outside the revised training diet**; preserve that limitation when discussing transfer.

### Original memory readout

For each test image, fit/evaluate a KDE separately against the 10 fixation vectors of each of the 40 stored images. For each stored image, sum log densities over all 32 test fixations. Then take the maximum across the 40 stored images:

```text
identity_score[i] = sum_t log(mean_j exp(-||q[t]-memory[i,j]||² / (2*sigma²)) + 1e-12)
familiarity(q) = max_i identity_score[i]
```

Distances between binary vectors are squared Euclidean/Hamming distances. Preserve the original float32 kernel/mean/log, `1e-12` density floor, and float64 fixation-score sum. The old picture wins only when its familiarity is strictly larger than the novel picture's; ties count as incorrect.

This already computes familiarity **conditional on each stored image**, then picks the highest. With one photograph per identity in the study bank, these are also identity-specific memories. Correct retrieval of an old house's own memory does not guarantee Yin success: a novel house can obtain a higher score against another memory. Familiarity scores are uncalibrated log densities, not confidence percentages.

### Latest pooled-house diagnostic

Pool the 200 stored fixation vectors from the 20 studied houses into one KDE. Average kernels across all 200 vectors **before** taking each test fixation's log density, then sum across its 32 fixations. There is no maximum over house identities. No novel/test vectors enter the memory.

The paired controls are the original best-of-40 readout and best-of-20 house memories. Category-restricted scoring assumes the category is known. Sigma, noise, images, seeds, and fixation draws stay unchanged.

| House accuracy, mean % | UU | UI | IU | II |
|---|---:|---:|---:|---:|
| LP original | 57.33 | 53.17 | 54.00 | 58.33 |
| LP pooled | 53.83 | 51.83 | 53.17 | 55.50 |
| CNN original | 52.50 | 50.17 | 50.17 | 62.00 |
| CNN pooled | 53.50 | 51.17 | 50.33 | 56.17 |

Pooling does not fix the house problem at these settings. Excluding face memories alone changes at most one of 600 decisions in a condition. These are descriptive paired simulation results, not a new validated model selection.

## Reproduce the latest results

Use Python 3.12 and a CPU environment with the pinned versions below. GPU training dependencies remain in the root `requirements.txt`; this replay needs only cached probabilities.

```bash
python3.12 -m venv .venv-yin
source .venv-yin/bin/activate
python -m pip install -r yin_tests/current_kde/requirements.txt
python yin_tests/current_kde/run.py \
  --cache-root /path/to/completed-sdsc/cache \
  --output /path/to/new-yin-output \
  --pooled-houses
```

Add `--calibrate-uu` to recompute the one-parameter face fit using the original coarse 0.05, fine 0.005, and ultrafine 0.001 search. Calibration should recover LP 0.373 and CNN 0.364. Without that flag, use the verified saved rates. The program verifies cache hashes, rejects an existing output directory, and saves the run configuration, exact stimulus plans, category records, means/SD/SEM, and validation receipt. It does not submit cluster jobs or modify a checkpoint.

Then compare every category decision against the archived latest runs:

```bash
python handoff/verify_replay.py /path/to/new-yin-output
```

This branch was checked with the full 50-seed replay, including fresh UU calibration and the pooled-house controls. [Validation receipt](handoff/VALIDATION.json).

## Experiment history and unresolved work

* **Fixed sigma, one noise:** current requested presentation results. [Source and full tables](handoff/snapshots/local-yin-sigma2-single-uu-20261119/RESULTS_ACCURACY.md).
* **Two-noise study-only protocol:** prior comparison used UU/UI `(p1,p1)`, IU/II `(p2,p1)` for study/test. Do not mix it with the current single-noise experiment or `two_cross` (which also changes test noise). Source and fits are in `handoff/snapshots/local-yin-two-sigma-tables-20261116/`.
* **Adjusted sigma:** previous exploratory values were LP 4.5095643231 and CNN 6.7539878491. They arose from pooled face/house human-similarity ranking. They were not a house-UU-only fit and not a mean-within-identity-distance estimate. The distance reference used all pairs among 400 upright study fixation codes, including pairs across identities/categories. These values are not the current default.
* **Nearest-memory prototype:** the user rejected this alternative. It is archived in `local-yin-nearest-memory-20261113-v2`, not active.
* **House PCA/retrieval diagnostics:** source snapshots in `handoff/analysis/`. Two-dimensional overlap alone does not establish confusion in 256 dimensions. These diagnostics used their documented older noise settings, not automatically the latest 0.373/0.364 rates.
* **Kanwisher Set A:** completed SDSC evaluation used earlier seed-45 epoch-124 checkpoints, original OSF 40 identities × 5 photos, 16 fixations, 50 simulation seeds, 156,000 ordered triplets/orientation. LP accuracy upright/inverted was 87.3637/76.5864%; CNN 89.0962/60.8522%. Preserve its separate provenance. It is not a fresh Kanwisher evaluation of the latest no-generic seed-46 pair.
* **Cluster status:** Delta completion is verified in saved receipts. Anvil's last local check was November 13; do not claim it is still running or complete without a new scheduler/log check. No live cluster check was performed to create this branch.

The next scientific step has not been chosen. Do not claim the house issue is solved, substitute a different house bank silently, or change noise/sigma while calling a readout comparison controlled. If evaluating new seeds, keep that independent validation separate from fitted-anchor tables.

## What is preserved

`handoff/snapshots/` holds byte-preserved historical source/metadata plus compact result records. [Snapshot hashes](handoff/SNAPSHOT_MANIFEST.json) identify the copied files. Some snapshots contain absolute laptop/cluster paths and stale status descriptions; they are audit evidence, not portable launchers. The Nov19 single-noise simulation finished successfully, then a report-formatting error was repaired by `finish_report.py` without rerunning or changing its raw results.

The [previous branch README](handoff/README_PREVIOUS_BRANCH.md), historical VGG sources, and old reports remain available. They describe earlier experiments. Do not mistake old `paper/` outputs or `training/train_r21_vgg2k.sh` for the current handoff.
