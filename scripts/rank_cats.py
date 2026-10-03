#!/usr/bin/env python3
"""Rank a library by CATS-like pharmacophore-pair similarity to anchor ligands.

    python scripts/rank_cats.py --query library.csv --anchors anchors.csv --name cats_test [--write-top50]

Folded 2-point pharmacophore count vectors (``rankers/cats.py``), compared with
the generalized Tanimoto. Different pharmacophore typing and topological
encoding from ErG, and neither uses exact atom/bond environments like ECFP4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamxcache.rankers.cats import compute_cats_fingerprints
from dreamxcache.rankers.cli import add_common_args, load_query, save_ranking
from dreamxcache.rankers.similarity import continuous_tanimoto_max_similarity, load_anchors


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--anchor-sources", default=None)
    parser.add_argument("--anchor-details", default=None)
    args = parser.parse_args()

    query = load_query(args)
    anchors = load_anchors(
        args.anchors,
        args.anchor_sources.split(",") if args.anchor_sources else None,
        args.anchor_details.split(",") if args.anchor_details else None,
    )
    anchor_smiles = anchors["smiles"].unique().tolist()
    anchor_fps, anchor_invalid = compute_cats_fingerprints(anchor_smiles, n_jobs=args.n_workers)
    query_fps, query_invalid = compute_cats_fingerprints(query[args.smiles_col].tolist(), n_jobs=args.n_workers)
    scores = continuous_tanimoto_max_similarity(query_fps, anchor_fps)
    save_ranking(
        args,
        query,
        scores,
        "rank-cats",
        {"n_anchors": len(anchor_smiles), "anchor_invalid": anchor_invalid, "query_invalid": query_invalid},
    )


if __name__ == "__main__":
    main()
