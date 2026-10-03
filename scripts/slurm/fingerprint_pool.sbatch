#!/bin/bash
#SBATCH --job-name=fp_pool
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Template: set --account / --partition for your cluster (and drop them if you do not need them),
# then submit from the repository root:  sbatch scripts/slurm/fingerprint_pool.sbatch
# Activates the environment given by $DREAMXCACHE_ENV (default ./.venv).
# Run `mkdir -p slurm_logs` BEFORE submitting: SLURM does not create the output directory,
# and a job whose log path cannot be opened fails immediately with no log at all.
#
# Use sbatch, not srun: this runs 15-60+ minutes and must survive an SSH disconnect.

set -euo pipefail
mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"
source "${DREAMXCACHE_ENV:-.venv}/bin/activate"
python scripts/compute_pool_fingerprints.py --n-workers 32
