#!/bin/bash
# Run `boltz predict` on one chunk directory of YAMLs, and fail loudly instead of
# silently succeeding.
#
#   boltz2_guard.sh CHUNK_DIR OUT_DIR
#
# Two guards, both learned from real failures:
#
# 1. GPU class. A generic GPU request on a mixed partition can land on a slow
#    card (a T4 ran ~8x slower than an H100). Refuse anything but A100/H100/H200.
#    Override with BOLTZ2_ALLOWED_GPUS="H100|A100|H200|L40S" if you know better.
#
# 2. Output count. `boltz predict` exits 0 even when it skips every input (for
#    example a malformed MSA: "Failed to process <file>. Skipping. Error: '\x00'").
#    A job that "completed" in seconds with no predictions is a failure, so after
#    the run, compare the number of affinity results with the number of inputs and
#    exit non-zero if any are missing.
set -euo pipefail

CHUNK_DIR="$1"
OUT_DIR="$2"
ALLOWED="${BOLTZ2_ALLOWED_GPUS:-H100|A100|H200}"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "=== GPU: $GPU_NAME ==="
if ! echo "$GPU_NAME" | grep -Eq "$ALLOWED"; then
  echo "FATAL: expected one of ($ALLOWED), got '$GPU_NAME'. Refusing to run at degraded throughput." >&2
  exit 97
fi

# Optionally activate a dedicated Boltz-2 environment (kept separate from the main one:
# installing boltz into a shared environment silently changed pinned numpy/scipy versions).
if [ -n "${DREAMXCACHE_BOLTZ2_ENV:-}" ]; then source "$DREAMXCACHE_BOLTZ2_ENV/bin/activate"; fi

N_IN=$(find "$CHUNK_DIR" -maxdepth 1 -name '*.yaml' | wc -l)
START=$(date +%s)
boltz predict "$CHUNK_DIR" --out_dir "$OUT_DIR" --accelerator gpu --devices 1 \
  --sampling_steps 50 --diffusion_samples 1 \
  --sampling_steps_affinity 50 --diffusion_samples_affinity 1 \
  --recycling_steps 1
ELAPSED=$(( $(date +%s) - START ))

N_OUT=$(find "$OUT_DIR" -name 'affinity_*.json' | wc -l)
echo "=== inputs: $N_IN, affinity results: $N_OUT, wall time: ${ELAPSED}s ==="
if [ "$N_OUT" -lt "$N_IN" ]; then
  echo "FATAL: only $N_OUT of $N_IN inputs produced a result. Check the log for 'Failed to process' lines (and the MSA for NUL bytes)." >&2
  exit 98
fi
