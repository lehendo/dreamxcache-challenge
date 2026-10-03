#!/usr/bin/env python3
"""Rank a library with the structure-based pharmacophore score.

    python scripts/rank_pharmacophore.py --query library.csv --name pharm_test \
        --reference-pdb structure.pdb --resname LIG --chain A [--altloc A] [--has-explicit-h] \
        --reference-points points.json

Score = O3A alignment to the reference ligand's bound conformer plus distance
bonuses at the reference points (``rankers/pharmacophore.py`` documents the
mechanism and the JSON schema). Derive reference points from your own structure;
``examples/wdr91_8SHJ_reference_points.json`` is a worked example. Bridging
waters can be found with ``dreamxcache.structure.waters.find_bridging_waters``.
Embedding and alignment per compound is slow (about 150 compounds/s on 16
cores), so budget accordingly for a large library.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamxcache.rankers.cli import add_common_args, load_query, save_ranking
from dreamxcache.rankers.pharmacophore import compute_pharmacophore_scores, load_reference_mol, load_reference_points


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--resname", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--altloc", default="A")
    parser.add_argument("--has-explicit-h", action="store_true")
    parser.add_argument("--reference-points", type=Path, required=True)
    args = parser.parse_args()

    ref_mol = load_reference_mol(args.reference_pdb, args.resname, args.chain, args.altloc, args.has_explicit_h)
    points = load_reference_points(args.reference_points)
    print(f"reference ligand: {ref_mol.GetNumAtoms()} atoms (with Hs); {len(points)} reference points")

    query = load_query(args)
    scores, n_failed = compute_pharmacophore_scores(
        query[args.smiles_col].tolist(), ref_mol, points, n_workers=args.n_workers
    )
    print(f"{n_failed} compounds failed (embedding/alignment/parsing) and scored 0")
    save_ranking(
        args,
        query,
        scores,
        "rank-pharmacophore",
        {"n_failed": n_failed, "n_reference_points": len(points), "reference_points_file": args.reference_points.name},
    )


if __name__ == "__main__":
    main()
