#!/bin/bash
#SBATCH --job-name=boltz2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-15
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# One Boltz-2 chunk per array task. Template: set the account and GPU partition, then
#
#   mkdir -p slurm_logs   # SLURM does not create it; a missing log dir fails the job with no log
#   CHUNKS_DIR=data/boltz2/chunks OUT_DIR=data/boltz2/out sbatch scripts/slurm/boltz2_chunks.sbatch
#
# Chunks come from scripts/boltz2_generate_yamls.py (--n-chunks should equal the array size).
# Keep chunks small (~100 compounds) and memory generous: Boltz starts one preprocessing
# thread per input, and a 94-compound chunk was killed at 16 GB on one cluster.
# Redundant submission is fine and cheap insurance against a contended queue: submit the same
# chunks to another partition with a *different* OUT_DIR (two jobs writing one directory race),
# then pool the output directories with scripts/boltz2_aggregate.py. Cancel the losing copies
# once a chunk has a verified result elsewhere.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CHUNK=$(printf "chunk_%02d" "$SLURM_ARRAY_TASK_ID")
bash scripts/slurm/boltz2_guard.sh "${CHUNKS_DIR:?set CHUNKS_DIR}/$CHUNK" "${OUT_DIR:?set OUT_DIR}/$CHUNK"
