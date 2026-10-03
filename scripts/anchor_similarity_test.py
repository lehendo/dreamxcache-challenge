#!/usr/bin/env python3
"""Is an anchor ligand informative about a library's true hits? Mean ECFP4
Tanimoto of the real hits to the anchor, against 1,000 random same-size draws
from the *full library*.

    python scripts/anchor_similarity_test.py --library lrrk2.parquet --anchor-smiles 'Fc1ccc2[nH]cc(...)c2c1'

Run this *before* reading an anchor-based negative control. A percentile near
the bottom means the hits are *less* similar to the anchor than almost any random
draw: the anchor is anti-correlated with the hits, and any ranking by similarity
to it will fail by construction. That is a poisoned testbed, not evidence about
the ranking method.

Caveat: this uses plain ECFP4 and the *mean* over hits, not the representation
and aggregation of the ranker under test (ErG uses pharmacophoric features and a
*maximum* over anchors). A ranking can show early recognition when a few hits sit
very close to the anchor even if the mean looks unremarkable, so a null result
here narrows a question without closing it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from dreamxcache.eval.testbed import anchor_smiles_from, load_labeled_library
from dreamxcache.provenance import write_report

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


def ecfp4(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--anchor-smiles", default=None)
    parser.add_argument("--anchor-pdb", default=None, help="file:resname:chain[:altloc]")
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_labeled_library(args.library)
    anchor = anchor_smiles_from(args.anchor_smiles, args.anchor_pdb)
    anchor_fp = ecfp4(anchor)
    sims = np.array(
        [
            DataStructs.TanimotoSimilarity(anchor_fp, fp) if (fp := ecfp4(s)) is not None else np.nan
            for s in df["SMILES"]
        ]
    )
    hit_mask = (df["LABEL"] == 1).to_numpy()
    n_hits = int(hit_mask.sum())
    hit_mean = float(np.nanmean(sims[hit_mask]))
    valid = np.flatnonzero(np.isfinite(sims))
    rng = np.random.default_rng(args.seed)
    null = np.array([np.mean(sims[rng.choice(valid, size=n_hits, replace=False)]) for _ in range(args.n_draws)])
    pct = float(100.0 * np.mean(null <= hit_mean))
    print(
        f"{args.library.name}: {n_hits} hits, mean similarity to the anchor {hit_mean:.4f} -> percentile {pct:.1f} of the random-draw null "
        f"(null mean {null.mean():.4f}, p5 {np.percentile(null, 5):.4f}, p95 {np.percentile(null, 95):.4f})"
    )
    write_report(
        f"anchor-similarity-{args.library.stem.lower()}",
        {
            "anchor_smiles": anchor,
            "n_hits": n_hits,
            "hit_mean_anchor_similarity": hit_mean,
            "random_draw_null": {
                "mean": float(null.mean()),
                "p5": float(np.percentile(null, 5)),
                "p50": float(np.percentile(null, 50)),
                "p95": float(np.percentile(null, 95)),
                "p99": float(np.percentile(null, 99)),
            },
            "hit_percentile_vs_random_draw_null": pct,
            "n_draws": args.n_draws,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
