#!/usr/bin/env python3
"""Build the Boltz-2 shortlist: the union of the top-N compounds of each cheaper
channel already scored.

Boltz-2 is too slow to score a whole library (hundreds of GPU-hours), so the
two-stage screen runs it only on this union. Writes one row per compound
(``CatalogID``, ``SMILES``, which channel(s) it came from) and a manifest.

    python scripts/boltz2_build_shortlist.py --split test \
        --channels pharmacophore:pharmacophore_score_test.npy:750 model:model_score_test.npy:750

``--channels name:cache_filename:top_n``, each file a full-length score array
aligned to the split in ``<cache>/``. Keep the channels' contributions *balanced*
here. Weight channels against each other at fusion time, not here: if the
shortlist is mostly one channel's picks, the second-stage channel just rescores
that channel's own picks and stops being an independent signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dreamxcache.config import cache_dir, external_dir, raw_dir
from dreamxcache.provenance import write_report

SPLIT_FILES = {"validation": "PGK2_Validation_split.csv", "test": "PGK2_Test_split.csv"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="+", required=True)
    parser.add_argument("--split", choices=list(SPLIT_FILES), required=True)
    parser.add_argument("--out", type=Path, default=external_dir() / "boltz2" / "shortlist.csv")
    args = parser.parse_args()

    channels = []
    for spec in args.channels:
        name, cache_filename, top_n = spec.split(":")
        channels.append((name, cache_filename, int(top_n)))

    template = pd.read_csv(raw_dir() / "Val-Test-set" / SPLIT_FILES[args.split])
    membership: dict[int, list[str]] = {}
    for name, cache_filename, top_n in channels:
        scores = np.load(cache_dir() / cache_filename)
        if len(scores) != len(template):
            raise ValueError(f"{name}: {len(scores)} scores != {len(template)} rows")
        finite = np.flatnonzero(np.isfinite(scores))
        for idx in finite[np.argsort(-scores[finite])[:top_n]]:
            membership.setdefault(int(idx), []).append(name)

    rows = [
        {
            "row_idx": idx,
            "CatalogID": str(template.loc[idx, "CatalogID"]),
            "SMILES": str(template.loc[idx, "SMILES"]),
            "sources": ";".join(sorted(sources)),
            "n_sources": len(sources),
        }
        for idx, sources in membership.items()
    ]
    shortlist = pd.DataFrame(rows).sort_values(["n_sources", "CatalogID"], ascending=[False, True])
    per_channel = {n: int(shortlist["sources"].str.contains(n).sum()) for n, _, _ in channels}
    print(f"shortlist: {len(shortlist)} unique compounds; per-channel membership {per_channel}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(args.out, index=False)
    write_report(
        "boltz2-shortlist",
        {
            "split": args.split,
            "channels": [{"name": n, "cache": c, "top_n": t} for n, c, t in channels],
            "n_shortlist": len(shortlist),
            "per_channel_membership": per_channel,
            "out": args.out.name,
        },
    )


if __name__ == "__main__":
    main()
