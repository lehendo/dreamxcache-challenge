#!/usr/bin/env python3
"""Rank a library by ErG similarity to a set of anchor ligands.

    python scripts/rank_erg.py --query library.csv --anchors anchors.csv --name erg_test \
        [--anchor-sources a,b] [--anchor-details x,y] [--write-top50]

Score = the maximum generalized-Tanimoto similarity between a compound's ErG
vector and any anchor's (``rankers/erg.py``, ``rankers/similarity.py``). ErG is
orthogonal to ECFP4-based models: it matches on pharmacophoric atom types over
reduced-graph distances, so it can match dissimilar scaffolds that share a
pharmacophore. Anchors are a CSV with a ``smiles`` column (DATA.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamxcache.rankers.cli import add_common_args, load_query, save_ranking
from dreamxcache.rankers.erg import compute_erg_fingerprints
from dreamxcache.rankers.similarity import continuous_tanimoto_max_similarity, load_anchors


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--anchor-sources", default=None, help="comma-separated 'source' filter")
    parser.add_argument(
        "--anchor-details", default=None, help="comma-separated 'detail' filter, applied after the source filter"
    )
    args = parser.parse_args()

    query = load_query(args)
    anchors = load_anchors(
        args.anchors,
        args.anchor_sources.split(",") if args.anchor_sources else None,
        args.anchor_details.split(",") if args.anchor_details else None,
    )
    anchor_smiles = anchors["smiles"].unique().tolist()
    print(f"{len(anchor_smiles)} unique anchors; {len(query):,} query compounds")

    anchor_fps, anchor_invalid = compute_erg_fingerprints(anchor_smiles, n_workers=args.n_workers)
    query_fps, query_invalid = compute_erg_fingerprints(query[args.smiles_col].tolist(), n_workers=args.n_workers)
    scores = continuous_tanimoto_max_similarity(query_fps, anchor_fps)
    save_ranking(
        args,
        query,
        scores,
        "rank-erg",
        {"n_anchors": len(anchor_smiles), "anchor_invalid": anchor_invalid, "query_invalid": query_invalid},
    )


if __name__ == "__main__":
    main()
