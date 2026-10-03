#!/usr/bin/env python3
"""Anchor-similarity (ErG) channel on a known-answer library, as a percentile of
its own permutation null. Used as the negative control for the other channels.

    python scripts/testbed_erg.py --library shortlist.parquet --target-name LRRK2 \
        (--anchor-smiles SMILES | --anchor-pdb structure.pdb:LIGRES:CHAIN)

Run ``anchor_similarity_test.py`` on the same library first: an anchor that is
anti-correlated with the hits poisons the control (``eval/testbed.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamxcache.eval.testbed import anchor_smiles_from, evaluate_channel, load_labeled_library
from dreamxcache.provenance import write_report
from dreamxcache.rankers.erg import compute_erg_fingerprints
from dreamxcache.rankers.similarity import continuous_tanimoto_max_similarity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--anchor-smiles", default=None)
    parser.add_argument("--anchor-pdb", default=None, help="file:resname:chain[:altloc]")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    library = load_labeled_library(args.library)
    anchor = anchor_smiles_from(args.anchor_smiles, args.anchor_pdb)
    print(f"{args.target_name}: {len(library):,} compounds, {int(library['LABEL'].sum())} hits; anchor {anchor}")

    anchor_fps, _ = compute_erg_fingerprints([anchor], n_workers=1)
    library_fps, n_invalid = compute_erg_fingerprints(library["SMILES"].tolist(), n_workers=args.n_workers)
    scores = continuous_tanimoto_max_similarity(library_fps, anchor_fps)

    result = evaluate_channel(scores, library["LABEL"].to_numpy(), args.n_permutations, args.seed)
    for metric, r in result["metrics"].items():
        print(f"  {metric}: {r['value']:.4f}  (null percentile {r['null_percentile']:.1f})")
    write_report(
        f"testbed-erg-{args.target_name.lower()}",
        {
            "target": args.target_name,
            "channel": "ErG",
            "anchor_smiles": anchor,
            "n_invalid_smiles": n_invalid,
            "n_library": len(library),
            "n_permutations": args.n_permutations,
            "null_seed": args.seed,
            **result,
        },
    )


if __name__ == "__main__":
    main()
