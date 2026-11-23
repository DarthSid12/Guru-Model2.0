#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=projects&work
#SBATCH --no-requeue
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export TMPDIR="/tmp/guru-nogeneric-${SLURM_JOB_ID:?}"
export MPLCONFIGDIR="$TMPDIR/mpl-cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec /work/nvme/bjaa/sagrawal5/lpnet-env/env/bin/python -u /projects/bjaa/sagrawal5/guru-resnet18-no-generic-20261109/operations/runtime.py "$@"
