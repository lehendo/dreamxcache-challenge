#!/usr/bin/env python3
"""Are a library's true hits chemically clustered? Mean pairwise ECFP4 Tanimoto
among the real hits, against 1,000 random draws of the same size from the
*full library* (so shortlist construction cannot create or hide clustering).

    python scripts/hit_clustering_test.py --library target_a.parquet --library target_b.parquet

Each library needs ``SMILES`` and a 0/1 ``LABEL``. Reports the percentile of the
real hits' mean similarity within the random-draw distribution. A high
percentile means the hits cluster; a *low* one means they are more diverse than
random.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from dreamxcache.eval.testbed import load_labeled_library
from dreamxcache.provenance import write_report

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


def ecfp4_fps(smiles_list: list[str]) -> list:
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol is not None else None)
    return fps


def mean_pairwise_tanimoto(fps: list) -> float:
    fps = [f for f in fps if f is not None]
    if len(fps) < 2:
        return float("nan")
    return float(
        np.mean(
            [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(len(fps)) for j in range(i + 1, len(fps))]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", action="append", type=Path, required=True)
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = {}
    for path in args.library:
        df = load_labeled_library(path)
        hits = df[df["LABEL"] == 1]
        n_hits = len(hits)
        hit_mean = mean_pairwise_tanimoto(ecfp4_fps(hits["SMILES"].tolist()))
        all_fps = [f for f in ecfp4_fps(df["SMILES"].tolist()) if f is not None]
        rng = np.random.default_rng(args.seed)
        null = np.array(
            [
                mean_pairwise_tanimoto([all_fps[j] for j in rng.choice(len(all_fps), size=n_hits, replace=False)])
                for _ in range(args.n_draws)
            ]
        )
        pct = float(100.0 * np.mean(null <= hit_mean))
        print(
            f"{path.name}: {n_hits} hits, mean pairwise Tanimoto {hit_mean:.4f} -> percentile {pct:.1f} of the random-draw null "
            f"(null mean {null.mean():.4f}, p95 {np.percentile(null, 95):.4f})"
        )
        results[path.stem] = {
            "n_hits": n_hits,
            "n_library": len(df),
            "hit_mean_pairwise_tanimoto": hit_mean,
            "random_draw_null": {
                "mean": float(null.mean()),
                "p50": float(np.percentile(null, 50)),
                "p95": float(np.percentile(null, 95)),
                "p99": float(np.percentile(null, 99)),
            },
            "hit_percentile_vs_random_draw_null": pct,
        }
    write_report("hit-clustering", {"n_draws": args.n_draws, "seed": args.seed, "libraries": results})


if __name__ == "__main__":
    main()
