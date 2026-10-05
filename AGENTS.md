# Instructions for agents working in combined_lpnet

## Start here

Read `README.md` for the pipeline and `instructions/hpc-onboarding.md` before preparing
cluster jobs or transfers. User instructions take precedence over this file.
Inspect `git status --short` before editing; preserve existing user changes.
Use the checked-out code and saved run metadata as evidence, rather than
assuming the latest report contains the newest results.

## Project and code map

The current workflow trains a shared VGG16-BN log-polar model on VGG2k faces,
objects, individual buildings, and generic houses, then evaluates Yin inversion
and Kanwisher/Dobs matching behavior through the shared binary code `h`.

- `training/train.py`: training, curriculum, sampling, checkpointing, and resume.
- `training/model.py`, `training/cylconv.py`: backbones, antialiasing, and model representation.
- `training/datasets.py`, `data_preparation/preprocess_fixations.py`, `training/trans.py`, `training/salience_trans.py`:
  packed data, fixation coordinates, and GPU image transforms.
- `training/train_r21_vgg2k.sh`: canonical current training recipe.
- `yin_tests/run_yin_orientation.py`, `yin_tests/simulate_yin1969.py`,
  `yin_tests/simulate_yin1969_bothnoise.py`: behavioral simulation and noise protocols.
- `yin_tests/launch_latest_vgg_yin.sh`,
  `kanwisher_tests/launch_latest_vgg_kanwisher.sh`, and their summary scripts:
  evaluation orchestration and summaries.
- `yin_tests/run_yin_aa5_new_seeds.py`: final-checkpoint seed 43/44 replications.
- `reporting/build_vgg16bn_report.py`, `reporting/plot_vgg16bn_training_curves.py`,
  `yin_tests/update_yin_replication_html.py`: report generation. Read each
  script before running it; some paths and experiment dates are fixed.
- `paper/`: current reports, methods, figures, and report data.
- `runs/`: run configurations, histories, checkpoints, logs, evaluation records.

## Preserve experimental meaning

The VGG2k recipe uses 16 fixations, batch size 256, 2,000 steps per epoch,
10 curriculum stages / 124 epochs, cosine learning rate, sqrt category
sampling, fixed identity ordering, and staged house image caps. Refer to the
launcher for exact settings. Do not silently change these when refactoring.

Keep VGG16-BN no-AA, AA4 (`vgg16_bn_aa`), and AA5 (`vgg16_bn_aa5`) distinct.
Retain mixed-face VGG support needed by the report comparisons. Older numbered
launchers have been retired; do not restore them as part of routine cleanup.

Distinguish training seeds (e.g. 42–44) from simulation seeds (e.g. 101–150).
Keep `single_uu`, `two_cross`, and `two_shared_ui` protocols distinct. Preserve
calibration category, noise assignment, checkpoint epoch/hash, trial counts,
and uncertainty definition (SD versus SEM) in every summary. Do not label an
intermediate snapshot as a final-checkpoint result.

## Training and HPC operations

Do not restart, pause, terminate, or replace running jobs as a side effect of
code edits. Inspect processes, logs, and GPU allocation before launching work.
Use scheduler-assigned devices on clusters; do not copy a workstation GPU
index into a Slurm job. GPU training belongs on compute nodes.

Follow `instructions/hpc-onboarding.md`, especially its repository-specific section. Verify
current cluster facts using official documentation and cluster commands. Its
generic Slurm template is not directly executable with this project's CLI.

Pin remote jobs to an exact commit and record the SHA, command, data paths,
seed, environment, and checkpoint provenance. Use full-state
`checkpoint_last.pth` for resume, persistent output storage, and a bounded
resubmission chain when needed. Do not invent account names or allocation
credentials. Preserve authentication and keep secrets out of commits/logs.

## Editing and validation

Make focused changes and update callers/docs when removing an option. Keep
checkpoint loading compatible with retained experiments. Do not delete
datasets, saved runs, or report inputs during code cleanup. Check references
before removing paper files or scripts; report builders depend on historical
source documents.

Use the project environment. On the current workstation it is commonly:

```bash
/home/siagrawal/miniconda3/envs/themodel2/bin/python training/train.py --help
/home/siagrawal/miniconda3/envs/themodel2/bin/python -m unittest discover -s tests -q
```

That absolute environment path is local, not portable. On another machine use
its configured interpreter and `requirements.txt`. Run Python syntax checks
and `bash -n` for changed shell scripts. Verify training launcher arguments
without starting a full GPU run. Run relevant tests for behavior changes;
document any check that could not be completed. Documentation-only edits need
link/reference checks, not a training run.

## Reports and Git

Build summaries from completed trial records and run metadata. Check expected
seed coverage, condition counts, checkpoint hashes, and completion records
before publishing results. Report generators may overwrite existing HTML or
replication sections; inspect their outputs and preserve unrelated edits.

Data stores, `runs/`, `logs/`, and model weights are ignored by Git. Do not
force-add them. Local `.vscode/` settings and the separate nested repository
`paper/vgg16bn_share_site/` are outside ordinary code commits.

Commit/push when requested. Inspect staged changes and use a normal push; do
not force-push or discard other work. The established GitHub remote is
`DarthSid12/Guru-Model2.0`, and the working branch has been `biggerDatasets`;
check the actual branch and remote rather than assuming they never change.

## Folder layout

Run commands from the repository root. Edit actual sources in `training/`,
`yin_tests/`, `kanwisher_tests/`, `data_preparation/`, `analysis/`, and
`reporting/`. Python imports use the categorized packages. New snapshots must include the
package sources they import. Keep this root
`AGENTS.md` discoverable; onboarding and longer guidance live in `instructions/`.
The experiment folders are separate from automated regression tests in `tests/`.
