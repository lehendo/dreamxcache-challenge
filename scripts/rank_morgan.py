#!/usr/bin/env python3
"""Rank a library by ECFP4 or FCFP4 max-Tanimoto similarity to anchor ligands.

    python scripts/rank_morgan.py --query library.csv --anchors anchors.csv --name ecfp4_test --variant ecfp4

The plain nearest-neighbour control for the pharmacophore-style rankers
(``rankers/morgan.py``): no ML and no DEL data. If it scores comparably to ErG
on the same anchors, the anchors matter and the representation does not; if it
scores much worse, the pharmacophore representation is doing real work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamxcache.rankers.cli import add_common_args, load_query, save_ranking
from dreamxcache.rankers.morgan import compute_morgan_fps, max_tanimoto_similarity
from dreamxcache.rankers.similarity import load_anchors


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--variant", choices=["ecfp4", "fcfp4"], default="ecfp4")
    parser.add_argument("--anchor-sources", default=None)
    parser.add_argument("--anchor-details", default=None)
    args = parser.parse_args()

    query = load_query(args)
    anchors = load_anchors(
        args.anchors,
        args.anchor_sources.split(",") if args.anchor_sources else None,
        args.anchor_details.split(",") if args.anchor_details else None,
    )
    use_features = args.variant == "fcfp4"
    anchor_fps = [f for f in compute_morgan_fps(anchors["smiles"].unique().tolist(), use_features) if f is not None]
    query_fps = compute_morgan_fps(query[args.smiles_col].tolist(), use_features, n_workers=args.n_workers)
    scores = max_tanimoto_similarity(query_fps, anchor_fps)
    save_ranking(args, query, scores, f"rank-{args.variant}", {"variant": args.variant, "n_anchors": len(anchor_fps)})


if __name__ == "__main__":
    main()
