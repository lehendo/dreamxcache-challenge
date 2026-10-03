#!/usr/bin/env python3
"""Structure-based pharmacophore channel on a known-answer library, as a
percentile of its own permutation null.

    python scripts/testbed_pharmacophore.py --library library.parquet --target-name WDR91 \
        --reference-pdb 8SHJ.pdb --resname ZI8 --chain A --reference-points examples/wdr91_8SHJ_reference_points.json

Confounds to state alongside any number from this script: a target whose pocket
has no bridging waters or selectivity residue gives the model fewer point types
than a water-mediated pocket would, and a shallow surface groove is a poorer fit
for a pharmacophore than an enclosed pocket. A weak result is therefore
confounded; a strong one is informative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamxcache.eval.testbed import evaluate_channel, load_labeled_library
from dreamxcache.provenance import write_report
from dreamxcache.rankers.pharmacophore import compute_pharmacophore_scores, load_reference_mol, load_reference_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--resname", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--altloc", default="A")
    parser.add_argument("--has-explicit-h", action="store_true")
    parser.add_argument("--reference-points", type=Path, required=True)
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    library = load_labeled_library(args.library)
    ref_mol = load_reference_mol(args.reference_pdb, args.resname, args.chain, args.altloc, args.has_explicit_h)
    points = load_reference_points(args.reference_points)
    print(
        f"{args.target_name}: {len(library):,} compounds, {int(library['LABEL'].sum())} hits; {len(points)} reference points"
    )

    scores, n_failed = compute_pharmacophore_scores(
        library["SMILES"].tolist(), ref_mol, points, n_workers=args.n_workers
    )
    print(f"{n_failed} compounds failed and scored 0")
    result = evaluate_channel(scores, library["LABEL"].to_numpy(), args.n_permutations, args.seed)
    for metric, r in result["metrics"].items():
        print(f"  {metric}: {r['value']:.4f}  (null percentile {r['null_percentile']:.1f})")
    write_report(
        f"testbed-pharmacophore-{args.target_name.lower()}",
        {
            "target": args.target_name,
            "channel": "pharmacophore",
            "reference_structure": args.reference_pdb.name,
            "reference_ligand": f"{args.resname}:{args.chain}",
            "reference_points_file": args.reference_points.name,
            "n_failed": n_failed,
            "n_library": len(library),
            "n_permutations": args.n_permutations,
            "null_seed": args.seed,
            **result,
        },
    )


if __name__ == "__main__":
    main()
