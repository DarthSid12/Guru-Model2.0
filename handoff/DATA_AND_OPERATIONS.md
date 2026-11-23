# Artifact locations and safe continuation

These paths are provenance and retrieval pointers, not guaranteed current availability. No credentials are included. Obtain fresh authentication through the user when required. Do not reuse old one-time codes or store passwords in scripts/logs.

## Laptop

Workspace: `/Users/darthsid/Documents/ChatGPT/GURU`.

Current cached representations:

```text
operations/sdsc-yin-no-generic-s46-20261111/completed-sdsc/cache/
  resnet18_lp_nogeneric_e124_s46/
  resnet18_cnn_nogeneric_e124_s46/
```

Each contains `DONE.json` and six float32 `(64,32,256)` arrays:
`faces_cfdWM64_{U,I}.npy`, `houses_dal64_{U,I}.npy`, `objects_wordbank64_{U,I}.npy`.
The portable evaluator verifies every file against `yin_tests/current_kde/registry.json`.
These caches are model sigmoid outputs before Bernoulli sampling and XOR noise. Do not regenerate them with different coordinates or preprocessing and claim an exact replay.

Source image bank: `/Users/darthsid/Downloads/Yin_Image_Bank_192_2026-11-06/`.
Its 64 house image hashes and packed-source hashes were checked against the current run. The evaluation house bank is DalHouses, not Houses535 synthetic training images.

Full training source/data registration: `operations/resnet18-no-generic-20261109/`.
Raw object-photo manifests: `operations/wordbank-object-curation-20261106/PHOTO_SPLITS.json` and the older `wordbank-inanimate-20261105/`. These large photo manifests and images are not in this branch. Download and curation scripts are preserved, but they have historical absolute paths and should not be run unreviewed.

Generated house originals and viewpoints live under `output/manual-houses128-20261027/` and the later `output/manual-houses128-random5-20261030/`. Read their original manifests before reusing them. The five random additions were not added to the pinned training diet.

## Delta

SSH alias: `delta-training`. Persistent run root:
`/projects/bjaa/sagrawal5/guru-resnet18-no-generic-20261109`.

Runs:
`runs/resnet18_lp_objects64clean_h128_nogeneric_s46` and
`runs/resnet18_cnn_objects64clean_h128_nogeneric_s46`.

Recorded jobs: preparation 23684223, LP 23684224 with reserved continuation 23684225, CNN 23684226 with reserved continuation 23684227. Both primary training runs completed epoch124. Do not infer continuation state from an old receipt. Inspect `squeue`, `sacct`, logs, and COMPLETE records before resubmitting anything.

## SDSC Expanse

Latest cached-code source root:
`/expanse/lustre/projects/sdp159/sagrawal5/yin-no-generic-s46-20261111`.

Its original GPU/CPU Yin jobs were 55690839/55690840. Recent single-noise and pooled-house runs used the downloaded caches locally, not new SDSC jobs.

Earlier Kanwisher root:
`/expanse/lustre/projects/sdp159/sagrawal5/kanwisher-setA-resnet18-20261106`.
Job 55556897 completed and was independently verified. Its checkpoints are the earlier seed45 models. The user requested cancellation of the duplicate Anvil Kanwisher job; do not revive that old monitoring task.

## Anvil

SSH alias: `anvil`. Run root:
`/anvil/projects/x-cis260738/x-sagrawal5/guru-resnet18-no-generic-20261109`.

Training seed45. Recorded preparation 22111363, LP 22111364/22111365, CNN 22111366/22111367. Last local check (November13) saw primary LP/CNN running. This is stale, not a November23 status report. Use fresh accounting and checkpoint hashes before evaluating Anvil weights. Allocation hours and maintenance dates are also stale until checked live.

## Launching or changing experiments

The root code plus `handoff/snapshots/resnet18-no-generic-20261109/cluster/recipe.py` describe the completed training recipe. Snapshot Slurm scripts contain registered site-specific paths and old job IDs. Do not blindly execute their submit scripts. Prepare a new versioned directory and verify CLI arguments, source hashes, data receipt, allocation, partition, environment, smoke tests, and existing queued/running work before a newly authorized submission.

Follow `instructions/hpc-onboarding.md` for general cluster guidance, checking live official facts as needed. Resume training from full-state `checkpoint_last.pth`, not model-only weights. Do not edit frozen source or data under an existing registered job.
