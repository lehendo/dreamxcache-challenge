#!/usr/bin/env python3
"""BitBIRCH clustering of the cached pool fingerprints.

    python scripts/cluster_pool.py --threshold 0.30 --n-sample 1000000   # calibrate
    python scripts/cluster_pool.py --threshold 0.30                      # full pool

``--n-sample`` clusters a random row-subset of the *already computed* array (no
recomputation) for fast threshold calibration; omit it for the real run.

`--threshold` has no cross-dataset default: it sets how many clusters result, so
calibrate on a sample first. Calibrate on a sample large enough to resemble the
full pool: small-scale granularity does not predict full-scale granularity (the
same ``branching_factor`` gave ~9-member clusters on a 2,060-compound sample and
~600-member clusters on the full 7.5M pool).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from dreamxcache.config import cache_dir
from dreamxcache.negatives.bitbirch_clustering import cluster_fingerprints
from dreamxcache.provenance import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--branching-factor", type=int, default=50)
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fps_path = cache_dir() / "ecfp4_assayed_pool_full.npy"
    if not fps_path.exists():
        raise FileNotFoundError(f"{fps_path} not found: run compute_pool_fingerprints.py first.")
    fps = np.load(fps_path, mmap_mode="r")
    n_total = fps.shape[0]

    if args.n_sample is not None:
        idx = np.sort(np.random.default_rng(args.seed).choice(n_total, size=min(args.n_sample, n_total), replace=False))
        fps_used = np.array(fps[idx])
        print(f"calibration run on {len(fps_used):,} sampled rows")
    else:
        idx = np.arange(n_total)
        fps_used = np.array(fps)

    t0 = time.time()
    clusters, cluster_id_local = cluster_fingerprints(
        fps_used, threshold=args.threshold, branching_factor=args.branching_factor
    )
    elapsed = time.time() - t0
    sizes = np.array([len(c) for c in clusters])
    print(
        f"{len(clusters):,} clusters in {elapsed:.1f}s; size min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()} mean={sizes.mean():.2f}"
    )

    write_report(
        "bitbirch-cluster",
        {
            "threshold": args.threshold,
            "branching_factor": args.branching_factor,
            "n_sample": args.n_sample,
            "seed": args.seed,
            "n_input_rows": len(fps_used),
            "n_clusters": len(clusters),
            "elapsed_seconds": elapsed,
            "cluster_size_min": int(sizes.min()),
            "cluster_size_median": int(np.median(sizes)),
            "cluster_size_max": int(sizes.max()),
            "cluster_size_mean": float(sizes.mean()),
        },
    )
    if args.n_sample is None:
        full = np.full(n_total, -1, dtype=np.int64)
        full[idx] = cluster_id_local
        path = cache_dir() / "bitbirch_cluster_id_full.npy"
        np.save(path, full)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
