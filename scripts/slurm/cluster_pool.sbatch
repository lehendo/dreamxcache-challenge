#!/bin/bash
#SBATCH --job-name=bitbirch
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Template: set --account / --partition for your cluster (and drop them if you do not need them),
# then submit from the repository root:  sbatch scripts/slurm/cluster_pool.sbatch
# Activates the environment given by $DREAMXCACHE_ENV (default ./.venv).
# Run `mkdir -p slurm_logs` BEFORE submitting: SLURM does not create the output directory,
# and a job whose log path cannot be opened fails immediately with no log at all.
#
# BitBIRCH.fit() is a single Python loop (no threading): more CPUs do not help. int32 upcast of the full pool needs memory (see negatives/bitbirch_clustering.py).
# Calibrate --threshold on a large sample first. Override via THRESHOLD / BRANCHING env vars.

set -euo pipefail
mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"
source "${DREAMXCACHE_ENV:-.venv}/bin/activate"
python scripts/cluster_pool.py --threshold "${THRESHOLD:-0.30}" --branching-factor "${BRANCHING:-50}"
