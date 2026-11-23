#!/usr/bin/env bash
#SBATCH --account=sdp159
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --constraint=lustre
#SBATCH --no-requeue
set -euo pipefail
module purge
module load cpu slurm
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
cd /expanse/lustre/projects/sdp159/sagrawal5/yin-kde-sigma-sweep-20261111
timeout 90 /expanse/lustre/scratch/sagrawal5/temp_project/lpnet/env/bin/python -c 'import sys; print("PYTHON BOOTSTRAP OK",sys.version,flush=True)'
srun --unbuffered /expanse/lustre/scratch/sagrawal5/temp_project/lpnet/env/bin/python -u run.py
