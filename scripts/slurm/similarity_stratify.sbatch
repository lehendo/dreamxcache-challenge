#!/bin/bash
#SBATCH --job-name=sim_stratify
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Template: set --account / --partition for your cluster (and drop them if you do not need them),
# then submit from the repository root:  sbatch scripts/slurm/similarity_stratify.sbatch
# Activates the environment given by $DREAMXCACHE_ENV (default ./.venv).
# Run `mkdir -p slurm_logs` BEFORE submitting: SLURM does not create the output directory,
# and a job whose log path cannot be opened fails immediately with no log at all.
#
# GPU nearest-positive similarity stratification (chunked matmul, minutes not hours).

set -euo pipefail
mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"
source "${DREAMXCACHE_ENV:-.venv}/bin/activate"
python scripts/similarity_stratify.py
