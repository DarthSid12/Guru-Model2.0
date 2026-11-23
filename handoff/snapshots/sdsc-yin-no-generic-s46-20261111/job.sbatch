#!/usr/bin/env bash
#SBATCH --account=sdp159
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=lustre
#SBATCH --no-requeue
set -euo pipefail
module purge
if [[ "${1:?mode}" == smoke ]]; then module load gpu slurm; else module load cpu slurm; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /expanse/lustre/projects/sdp159/sagrawal5/yin-no-generic-s46-20261111
export MPLCONFIGDIR="/tmp/guru-yin-nogen46-${SLURM_JOB_ID:?}/mpl-cache"
mkdir -p "$MPLCONFIGDIR"
timeout 90 /expanse/lustre/scratch/sagrawal5/temp_project/lpnet/env/bin/python -c 'import sys; print("PYTHON BOOTSTRAP OK",sys.version,flush=True)'
srun --unbuffered /expanse/lustre/scratch/sagrawal5/temp_project/lpnet/env/bin/python -u run.py "$@"
