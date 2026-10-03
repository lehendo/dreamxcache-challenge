#!/usr/bin/env python3
"""GPU nearest-positive-similarity stratification for negative sampling.

Negative sampling needs to know which pool compounds are "similar to the
positives", not a full partition of chemical space. This computes the maximum
Tanimoto similarity from every pool compound to its nearest L0 positive by
chunked GPU matrix multiplication, then thresholds it into a binary stratum
(1 = similar, 0 = elsewhere). The output is a 2-valued ``cluster_id`` array for
``sample_negatives(..., binary_stratum=True)``.

    python scripts/similarity_stratify.py [--threshold T] [--chunk-size 20000]

The threshold is derived, not fixed. The maximum is taken over *all* positives
(tens of thousands), a different quantity from one pairwise comparison: random
pairs exceed a pairwise Tanimoto of 0.30 only ~3% of the time, but the maximum
over tens of thousands of near-independent comparisons exceeds 0.30 for almost
every compound (a 0.30 cutoff flagged ~99.7% of the pool). So the threshold is chosen
as the percentile that makes the stratum ``STRATUM_SIZE_MULTIPLE`` times the
number of "similar" negatives training will draw: selective, but big enough to
sample diversely from. Raw maxima are saved so the threshold can be changed
without recomputing.

Chunked, with a progress line and an incremental checkpoint after every chunk.
Needs a CUDA GPU and ``pip install -e .[gpu]``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from dreamxcache.config import cache_dir
from dreamxcache.ingest.load import load_selection
from dreamxcache.label.recipes import wiki_recipe
from dreamxcache.provenance import write_report

STRATUM_SIZE_MULTIPLE = 5.0  # target stratum size = this x the negative-sampling floor
NEGATIVE_RATIO = 2.0  # must match train_baseline.py
FRACTION_FROM_SIMILAR = 0.8  # must match train_baseline.py
COLS = ["compound", "SMILES", "count_PGK2", "count_PGK2_with_inhibitor", "count_NTC", "historic_hits"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None, help="override the derived percentile threshold")
    parser.add_argument("--chunk-size", type=int, default=20000)
    args = parser.parse_args()

    import torch  # deferred: only this script needs it (pip install -e .[gpu])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this on a GPU node.")
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    fps = np.load(cache_dir() / "ecfp4_assayed_pool_full.npy", mmap_mode="r")
    pool_df = pd.read_parquet(cache_dir() / "ecfp4_assayed_pool_full_smiles.parquet")
    wiki_result = wiki_recipe(load_selection(columns=COLS))
    if not (pool_df["SMILES"].to_numpy() == wiki_result.deduped["SMILES"].to_numpy()).all():
        raise RuntimeError(
            "cached pool SMILES order does not match a fresh L0 deduplication; recompute the fingerprints"
        )
    positive_mask = wiki_result.positive_mask.to_numpy()
    n_positives, n_total = int(positive_mask.sum()), len(pool_df)
    print(f"pool: {n_total:,} compounds, {n_positives:,} L0 positives")

    positive_indices = np.where(positive_mask)[0]
    positives_fp = torch.tensor(np.array(fps[positive_indices]), dtype=torch.float32, device=device)
    positives_popcount = positives_fp.sum(dim=1)

    max_sim = np.zeros(n_total, dtype=np.float32)
    max_sim[positive_indices] = 1.0  # a positive is maximally similar to a positive (itself)
    non_positive = np.where(~positive_mask)[0]
    n_chunks = (len(non_positive) + args.chunk_size - 1) // args.chunk_size
    incomplete = cache_dir() / "similarity_max_to_positive_INCOMPLETE.npy"

    for chunk_num, start in enumerate(range(0, len(non_positive), args.chunk_size)):
        chunk_idx = non_positive[start : start + args.chunk_size]
        chunk_fp = torch.tensor(np.array(fps[chunk_idx]), dtype=torch.float32, device=device)
        chunk_popcount = chunk_fp.sum(dim=1)
        intersection = chunk_fp @ positives_fp.T
        union = chunk_popcount.unsqueeze(1) + positives_popcount.unsqueeze(0) - intersection
        similarity = intersection / union.clamp(min=1.0)
        max_sim[chunk_idx] = similarity.max(dim=1).values.cpu().numpy()
        if chunk_num % 10 == 0 or chunk_num == n_chunks - 1:
            done = start + len(chunk_idx)
            elapsed = time.time() - t0
            print(
                f"  [progress] chunk {chunk_num + 1}/{n_chunks}, {done:,}/{len(non_positive):,} compounds, {elapsed:.0f}s"
            )
            np.save(incomplete, max_sim)  # safe to inspect if the job dies before finishing

    max_sim_path = cache_dir() / "similarity_max_to_positive_full.npy"
    np.save(max_sim_path, max_sim)

    floor = FRACTION_FROM_SIMILAR * NEGATIVE_RATIO * n_positives
    target_size = STRATUM_SIZE_MULTIPLE * floor
    if args.threshold is not None:
        threshold, derivation = args.threshold, "user override"
    else:
        percentile = 100 * (1 - target_size / n_total)
        threshold = float(np.percentile(max_sim, percentile))
        derivation = f"derived: target stratum size {target_size:,.0f} ({percentile:.2f}th percentile)"
    cluster_id = (max_sim >= threshold).astype(np.int64)
    cluster_id_path = cache_dir() / "similarity_cluster_id_full.npy"
    np.save(cluster_id_path, cluster_id)
    incomplete.unlink(missing_ok=True)

    n_similar = int(cluster_id.sum())
    print(f"threshold {threshold:.4f} ({derivation}); similar stratum {n_similar:,}/{n_total:,}; need >= {floor:,.0f}")
    write_report(
        "similarity-stratify",
        {
            "method": "gpu_nearest_positive_similarity",
            "threshold": threshold,
            "threshold_derivation": derivation,
            "negative_sampling_floor": floor,
            "target_stratum_size": target_size,
            "n_positives": n_positives,
            "n_pool": n_total,
            "n_similar_stratum": n_similar,
            "elapsed_seconds": time.time() - t0,
        },
    )


if __name__ == "__main__":
    main()
