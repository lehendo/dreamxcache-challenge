#!/usr/bin/env python3
"""Aggregate Boltz-2 affinity results into a full-length, NaN-safe score array.

    python scripts/boltz2_aggregate.py --out-dirs run_a run_b --split test --cache-name boltz2_score_test.npy
    python scripts/boltz2_aggregate.py --out-dirs run_a --ids-file shortlist.csv --id-col CatalogID --cache-name shortlist_scores.npy

Aligns to either a challenge split (``--split``) or any table of ids
(``--ids-file``). Compounds without a result are NaN, not 0. Output directories
may overlap (redundant copies of a chunk run on different hardware); a compound
found twice keeps its last value. Warns when coverage is below 100%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dreamxcache.config import cache_dir, raw_dir
from dreamxcache.rankers.boltz2 import aggregate_affinities

SPLIT_FILES = {"validation": "PGK2_Validation_split.csv", "test": "PGK2_Test_split.csv"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dirs", nargs="+", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=list(SPLIT_FILES))
    group.add_argument("--ids-file", type=Path)
    parser.add_argument("--id-col", default="CatalogID")
    parser.add_argument("--score-field", default="affinity_probability_binary")
    parser.add_argument("--cache-name", required=True)
    args = parser.parse_args()

    if args.split:
        table = pd.read_csv(raw_dir() / "Val-Test-set" / SPLIT_FILES[args.split])
    else:
        table = pd.read_parquet(args.ids_file) if args.ids_file.suffix == ".parquet" else pd.read_csv(args.ids_file)
    ids = table[args.id_col].astype(str).tolist()

    scores, stats = aggregate_affinities(args.out_dirs, ids, args.score_field)
    n_scored = int(np.isfinite(scores).sum())
    print(
        f"found {stats['n_found']} results across {len(args.out_dirs)} dir(s); "
        f"{stats['n_missing_field']} missing {args.score_field}; {stats['n_unmatched']} ids not in the table"
    )
    print(f"{n_scored}/{len(scores)} compounds scored (the rest are NaN: unscored, not zero)")
    if n_scored:
        finite = scores[np.isfinite(scores)]
        print(f"score range (scored only): [{finite.min():.4f}, {finite.max():.4f}]")
    if n_scored < len(scores):
        print("WARNING: partial coverage; downstream channels must treat NaN as 'not ranked by this channel'.")

    cache_dir().mkdir(parents=True, exist_ok=True)
    out = cache_dir() / args.cache_name
    np.save(out, scores)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
