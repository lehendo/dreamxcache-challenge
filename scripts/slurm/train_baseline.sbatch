#!/bin/bash
#SBATCH --job-name=train_l0
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Template: set --account / --partition for your cluster (and drop them if you do not need them),
# then submit from the repository root:  sbatch scripts/slurm/train_baseline.sbatch
# Activates the environment given by $DREAMXCACHE_ENV (default ./.venv).
# Run `mkdir -p slurm_logs` BEFORE submitting: SLURM does not create the output directory,
# and a job whose log path cannot be opened fails immediately with no log at all.
#
# The feature matrix is positives + 2x negatives, so this is fast.

set -euo pipefail
mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"
source "${DREAMXCACHE_ENV:-.venv}/bin/activate"
python scripts/train_baseline.py --cluster-id-path data/cache/similarity_cluster_id_full.npy
