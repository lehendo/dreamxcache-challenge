#!/usr/bin/env python3
"""Boltz-2 channel on a known-answer shortlist, as a percentile of its own
permutation null, optionally beside an anchor-similarity (ErG) control computed
in the same run so both are read against the *identical* null.

    python scripts/testbed_boltz2.py --shortlist shortlist.parquet --target-name LRRK2 \
        --out-dirs run_a/out run_b/out [--anchor-smiles SMILES]

Needs ``COMPOUND_ID``, ``SMILES`` and ``LABEL`` in the shortlist, and Boltz-2
output directories holding ``affinity_<COMPOUND_ID>.json`` (``scripts/boltz2_*``).
Compounds without a result are excluded from the metric and from the null, and
the coverage is reported. Redundant output directories (the same chunks run on
different hardware) are pooled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamxcache.eval.testbed import evaluate_channel, load_labeled_library
from dreamxcache.provenance import write_report
from dreamxcache.rankers.boltz2 import aggregate_affinities
from dreamxcache.rankers.erg import compute_erg_fingerprints
from dreamxcache.rankers.similarity import continuous_tanimoto_max_similarity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--out-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--score-field", default="affinity_probability_binary")
    parser.add_argument("--anchor-smiles", default=None, help="also score an ErG control against this anchor")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shortlist = load_labeled_library(args.shortlist)
    labels = shortlist["LABEL"].to_numpy()
    scores, stats = aggregate_affinities(args.out_dirs, shortlist["COMPOUND_ID"].astype(str).tolist(), args.score_field)
    print(f"{args.target_name}: {stats['n_found']} results pooled across {len(args.out_dirs)} dir(s)")

    report: dict[str, object] = {
        "target": args.target_name,
        "n_shortlist": len(shortlist),
        "n_hits": int(labels.sum()),
        "score_field": args.score_field,
        "n_permutations": args.n_permutations,
        "null_seed": args.seed,
        "boltz2": evaluate_channel(scores, labels, args.n_permutations, args.seed),
    }
    b = report["boltz2"]
    assert isinstance(b, dict)
    print(f"Boltz-2 scored {b['n_scored']}/{len(shortlist)} compounds ({b['n_hits_scored']}/{int(labels.sum())} hits)")
    for metric, r in b["metrics"].items():
        print(f"  Boltz-2 {metric}: {r['value']:.4f}  (null percentile {r['null_percentile']:.1f})")

    if args.anchor_smiles:
        anchor_fps, _ = compute_erg_fingerprints([args.anchor_smiles], n_workers=1)
        library_fps, _ = compute_erg_fingerprints(shortlist["SMILES"].tolist(), n_workers=8)
        erg = evaluate_channel(
            continuous_tanimoto_max_similarity(library_fps, anchor_fps), labels, args.n_permutations, args.seed
        )
        report["erg_control"] = {"anchor_smiles": args.anchor_smiles, **erg}
        for metric, r in erg["metrics"].items():
            print(f"  ErG     {metric}: {r['value']:.4f}  (null percentile {r['null_percentile']:.1f})")

    write_report(f"testbed-boltz2-{args.target_name.lower()}", report)


if __name__ == "__main__":
    main()
