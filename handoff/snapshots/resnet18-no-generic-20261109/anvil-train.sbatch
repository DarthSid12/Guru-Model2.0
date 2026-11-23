#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=128G
#SBATCH --no-requeue
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export TMPDIR="/tmp/guru-nogeneric-${SLURM_JOB_ID:?}"
export MPLCONFIGDIR="$TMPDIR/mpl-cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec /anvil/projects/x-cis260738/x-sagrawal5/guru-cnn/lpnet-env/env/bin/python -u /anvil/projects/x-cis260738/x-sagrawal5/guru-resnet18-no-generic-20261109/operations/runtime.py "$@"
