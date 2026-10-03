#!/bin/bash
#SBATCH --job-name=score_splits
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Template: set --account / --partition for your cluster (and drop them if you do not need them),
# then submit from the repository root:  sbatch scripts/slurm/score_splits.sbatch
# Activates the environment given by $DREAMXCACHE_ENV (default ./.venv).
# Run `mkdir -p slurm_logs` BEFORE submitting: SLURM does not create the output directory,
# and a job whose log path cannot be opened fails immediately with no log at all.
#
# CPU only: fingerprints both splits and scores them with the trained model.

set -euo pipefail
mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"
source "${DREAMXCACHE_ENV:-.venv}/bin/activate"
python scripts/score_splits.py --save-score-cache
