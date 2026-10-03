#!/usr/bin/env python3
"""Build a known-answer shortlist: every known hit plus a random sample of
non-hits, shuffled.

    python scripts/build_testbed_shortlist.py --library target_easms.parquet --out shortlist.parquet \
        [--n 1500] [--seed 42]

Including *all* hits makes enrichment measurable at a size a slow channel can
afford (e.g. ~1,500 Boltz-2 complexes instead of the whole library). The
permutation null is computed on this exact composition, so the shortlist's
inflated hit rate does not bias the percentile reading. Needs columns
``COMPOUND_ID``, ``SMILES`` and ``LABEL``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dreamxcache.eval.testbed import load_labeled_library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_labeled_library(args.library)
    hits, nonhits = df[df["LABEL"] == 1], df[df["LABEL"] == 0]
    n_sample = args.n - len(hits)
    if n_sample <= 0 or n_sample > len(nonhits):
        raise ValueError(f"cannot build a shortlist of {args.n} from {len(hits)} hits and {len(nonhits)} non-hits")
    rng = np.random.RandomState(args.seed)
    sample_idx = rng.choice(nonhits.index, size=n_sample, replace=False)
    shortlist = (
        df.loc[list(hits.index) + list(sample_idx)].sample(frac=1, random_state=args.seed).reset_index(drop=True)
    )
    cols = [c for c in ("COMPOUND_ID", "SMILES", "LABEL") if c in shortlist.columns]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shortlist[cols].to_parquet(args.out)
    print(
        f"{len(shortlist)} compounds ({int(shortlist['LABEL'].sum())} hits, all {len(hits)} known hits included) -> {args.out}"
    )


if __name__ == "__main__":
    main()
