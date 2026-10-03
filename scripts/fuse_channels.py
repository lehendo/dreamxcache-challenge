#!/usr/bin/env python3
"""Fuse scoring channels with reciprocal rank fusion, then select the final 50 by
greedy per-series diversity.

    python scripts/fuse_channels.py --split test --name fusion \
        --channels boltz2:boltz2_score_test.npy:1.4 pharmacophore:pharm_test.npy:1.3 model:model_score_test.npy:1.0

``--channels name:cache_filename[:weight]``: each a full-length score array
(higher is better, NaN = not scored) aligned to the split, in ``<cache>/``.
Writes a validation ``.txt`` or a test ``CSV`` (``CatalogID, Sel_50, Score``),
validated locally. Weights are set coarsely *in advance* (``fusion/rrf.py``,
``fusion/diversity.py``); do not tune them against validation feedback, which is
the adaptive-overfitting trap this repository documents.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from dreamxcache.config import cache_dir, raw_dir, submissions_dir
from dreamxcache.fusion.diversity import fuse_and_select
from dreamxcache.provenance import write_report
from dreamxcache.submission import (
    validate_test_format,
    validate_validation_format,
    write_test_csv,
    write_validation_txt,
)

SPLIT_FILES = {"validation": "PGK2_Validation_split.csv", "test": "PGK2_Test_split.csv"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="+", required=True)
    parser.add_argument("--split", choices=list(SPLIT_FILES), required=True)
    parser.add_argument("--name", required=True, help="output filename stem")
    parser.add_argument("--n-candidates-per-channel", type=int, default=500)
    parser.add_argument("--cluster-threshold", type=float, default=0.32)
    args = parser.parse_args()

    template = pd.read_csv(raw_dir() / "Val-Test-set" / SPLIT_FILES[args.split])
    channel_scores, weights, specs = {}, {}, []
    for spec in args.channels:
        parts = spec.split(":")
        name, cache_filename = parts[0], parts[1]
        weight = float(parts[2]) if len(parts) > 2 else 1.0
        scores = np.load(cache_dir() / cache_filename)
        if len(scores) != len(template):
            raise ValueError(f"{name}: {len(scores)} scores != {len(template)} {args.split} rows")
        channel_scores[name], weights[name] = scores, weight
        specs.append(
            {"name": name, "cache": cache_filename, "weight": weight, "n_scored": int(np.isfinite(scores).sum())}
        )
        print(f"channel {name!r}: {specs[-1]['n_scored']:,} scored, weight {weight}")

    result = fuse_and_select(
        template["SMILES"],
        channel_scores,
        weights,
        n_candidates_per_channel=args.n_candidates_per_channel,
        cluster_threshold=args.cluster_threshold,
    )
    print(
        f"{result.n_candidates} unique candidates, {result.n_series_in_pool} proxy series in the pool; "
        f"the 50 picks span {result.n_series_selected} distinct series"
    )

    if args.split == "validation":
        out = submissions_dir() / f"{args.name}.txt"
        selected = template.loc[result.selected_idx, "CatalogID"].tolist()
        problems = write_validation_txt(selected, out) + validate_validation_format(selected, template)
    else:
        out = submissions_dir() / f"{args.name}_test.csv"
        write_test_csv(template["CatalogID"], result.selected_idx, result.combined_score, out)
        problems = validate_test_format(pd.read_csv(out), template)
    print(f"Wrote {out.name}: {'format VALID' if not problems else problems}")

    write_report(
        "fuse-channels",
        {
            "split": args.split,
            "channels": specs,
            "n_candidates_per_channel": args.n_candidates_per_channel,
            "cluster_threshold": args.cluster_threshold,
            "n_candidates": result.n_candidates,
            "n_series_in_pool": result.n_series_in_pool,
            "n_series_selected": result.n_series_selected,
            "output": out.name,
            "format_problems": problems,
        },
    )


if __name__ == "__main__":
    main()
