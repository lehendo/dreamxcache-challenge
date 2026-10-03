#!/usr/bin/env python3
"""ECFP4 fingerprints for the labeled-and-deduplicated assayed pool, cached to
``<cache>/``. The most expensive step of the baseline; reused by clustering,
stratification and training.

    python scripts/compute_pool_fingerprints.py [--sample-n 100000] [--n-workers 8]

``--sample-n`` runs on a random subsample first to validate the pipeline and
measure real throughput; it uses the same code path as the full run.

Memory: the extraction code returns NaN rows for unparseable SMILES, which
forces float64 on the whole array. Concatenating tens of millions of rows by
2048 float64 chunks would need hundreds of GB. Fingerprints are therefore
downcast to int8 *inside each worker* and written straight into a
pre-allocated int8 memmap on disk. int8 is a storage format only: never take a
dot product of it (see ``negatives/bitbirch_clustering.py``).
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dreamxcache.config import cache_dir
from dreamxcache.ingest.load import load_selection
from dreamxcache.label.recipes import wiki_recipe
from dreamxcache.provenance import write_report

ECFP4_N_BITS = 2048
COLS = ["compound", "SMILES", "count_PGK2", "count_PGK2_with_inhibitor", "count_NTC", "historic_hits"]


def _fingerprint_chunk(args: tuple[int, list[str]]) -> tuple[int, np.ndarray, int]:
    start_offset, smiles_chunk = args
    # Imported in the worker so each process loads the extraction module itself.
    from dreamxcache.features.ecfp4 import compute_ecfp4

    fps = compute_ecfp4(smiles_chunk, use_tqdm=False)
    n_invalid = int(np.isnan(fps).any(axis=1).sum())
    return start_offset, np.nan_to_num(fps, nan=0.0).astype(np.int8), n_invalid


def compute_fingerprints_to_memmap(smiles: list[str], out_path: Path, n_workers: int, chunk_size: int = 5000) -> int:
    """Write int8 fingerprints for `smiles` into a new ``.npy`` memmap at
    `out_path`; returns the number of unparseable SMILES."""
    memmap = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int8, shape=(len(smiles), ECFP4_N_BITS))  # type: ignore[no-untyped-call]
    chunks = [(i, smiles[i : i + chunk_size]) for i in range(0, len(smiles), chunk_size)]
    total_invalid = 0
    with Pool(n_workers) as pool:
        for start_offset, fps_int8, n_invalid in pool.imap_unordered(_fingerprint_chunk, chunks):
            memmap[start_offset : start_offset + len(fps_int8)] = fps_int8
            total_invalid += n_invalid
    memmap.flush()
    return total_invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=None, help="subsample size for a dry run")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading the selection table and deduplicating (recipe L0)...")
    pool_df = wiki_recipe(load_selection(columns=COLS)).deduped[["SMILES"]].reset_index(drop=True)
    print(f"  assayed pool: {len(pool_df):,} unique compounds")

    if args.sample_n is not None:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(pool_df), size=min(args.sample_n, len(pool_df)), replace=False))
        pool_df = pool_df.iloc[idx].reset_index(drop=True)
        print(f"  subsampled to {len(pool_df):,} (seed={args.seed})")

    cache_dir().mkdir(parents=True, exist_ok=True)
    suffix = f"_sample{args.sample_n}" if args.sample_n is not None else "_full"
    fps_path = cache_dir() / f"ecfp4_assayed_pool{suffix}.npy"
    smiles_path = cache_dir() / f"ecfp4_assayed_pool{suffix}_smiles.parquet"

    smiles_list = pool_df["SMILES"].tolist()
    t0 = time.time()
    n_invalid = compute_fingerprints_to_memmap(smiles_list, fps_path, n_workers=args.n_workers)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({len(smiles_list) / elapsed:.0f} mol/s); {n_invalid:,} unparseable SMILES zeroed")
    pool_df.to_parquet(smiles_path)

    write_report(
        "fingerprint-pool",
        {
            "n_compounds": len(smiles_list),
            "n_invalid_smiles": n_invalid,
            "elapsed_seconds": elapsed,
            "n_workers": args.n_workers,
            "sample_n": args.sample_n,
            "seed": args.seed,
            "fps_path": fps_path.name,
            "fps_shape": [len(smiles_list), ECFP4_N_BITS],
        },
    )


if __name__ == "__main__":
    main()
