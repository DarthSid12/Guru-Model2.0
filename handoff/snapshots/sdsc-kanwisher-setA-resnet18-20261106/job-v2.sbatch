#!/usr/bin/env bash
#SBATCH --account=sdp159
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --constraint=lustre
#SBATCH --no-requeue
set -euo pipefail
module purge
module load gpu slurm
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="/tmp/guru-setA-${SLURM_JOB_ID:?}/mpl-cache"
export PYTHONPYCACHEPREFIX="/tmp/guru-setA-${SLURM_JOB_ID:?}/pycache"
mkdir -p "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"
cd /expanse/lustre/projects/sdp159/sagrawal5/kanwisher-setA-resnet18-20261106
timeout 90 /expanse/lustre/scratch/sagrawal5/temp_project/lpnet/env/bin/python -c 'import sys; print("PYTHON BOOTSTRAP OK", sys.version, flush=True)'
srun --unbuffered /expanse/lustre/scratch/sagrawal5/temp_project/lpnet/env/bin/python -u run.py
