#!/usr/bin/env python3
"""Positive counts for labeling recipes L0-L8 on your copy of the data, plus the
dedup-order diagnostic.

    python scripts/recipe_yields.py \
        [--z-cutoff 0.07] [--per-library-thresholds thresholds.json] \
        [--joint-z-cutoff 0.5] [--disynthon-threshold 50]

L3, L6, L7 and L8 only run when their data-dependent parameter is supplied;
derive them with ``scripts/derive_thresholds.py``.

The diagnostic applies L0's and L1's thresholds to *raw* (pre-deduplication)
rows to show how many positives are lost by thresholding before summing
duplicate rows. That is the wrong order for real labeling, which is why no
recipe function exposes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamxcache.ingest.audit import check_positives_only_from_summing
from dreamxcache.ingest.load import filter_valid_smiles, load_selection
from dreamxcache.label import recipes as R
from dreamxcache.provenance import write_report

COLS = ["compound", "SMILES", "count_PGK2", "count_PGK2_with_inhibitor", "count_NTC", "historic_hits"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--z-cutoff", type=float, default=None, help="L3")
    parser.add_argument("--per-library-thresholds", type=Path, default=None, help="L6: JSON {library: count floor}")
    parser.add_argument("--joint-z-cutoff", type=float, default=None, help="L7")
    parser.add_argument("--disynthon-threshold", type=float, default=None, help="L8")
    args = parser.parse_args()

    need_z = args.z_cutoff is not None or args.joint_z_cutoff is not None
    selection = load_selection(columns=[*COLS, "zscore_PGK2"] if need_z else COLS, path=args.selection)

    runs = [
        ("L0", R.wiki_recipe, ()),
        ("L1", R.strict_recipe, ()),
        ("L2", R.wiki_recipe_individually_evidenced, ()),
        ("L4", R.wiki_recipe_count_ge_10, ()),
        ("L5", R.wiki_recipe_no_competition, ()),
    ]
    if args.z_cutoff is not None:
        runs.append(("L3", R.wiki_recipe_zscore, (args.z_cutoff,)))
    if args.per_library_thresholds is not None:
        thresholds = {k: int(v) for k, v in json.loads(args.per_library_thresholds.read_text()).items()}
        runs.append(("L6", R.wiki_recipe_per_library, (thresholds,)))
    if args.joint_z_cutoff is not None:
        runs.append(("L7", R.wiki_recipe_joint_zscore, (args.joint_z_cutoff,)))
    if args.disynthon_threshold is not None:
        runs.append(("L8", R.wiki_recipe_disynthon_competitive_hit, (args.disynthon_threshold,)))

    results = {}
    for label, fn, extra in sorted(runs, key=lambda r: r[0]):
        result = fn(selection, *extra)
        results[label] = {
            "recipe": result.name,
            "version": result.version,
            "n_positives": result.n_positives,
            "n_unique_compounds": len(result.deduped),
            "positive_rate": result.n_positives / len(result.deduped),
        }
        print(f"  {label} {result.name}: {result.n_positives:,} positives / {len(result.deduped):,} unique compounds")

    # Dedup-order diagnostic: raw-row thresholding vs dedup-first.
    clean_raw = filter_valid_smiles(selection[COLS])
    order = {}
    for label, threshold_fn, recipe_fn in [
        ("L0", R.wiki_recipe_threshold, R.wiki_recipe),
        ("L1", R.strict_recipe_threshold, R.strict_recipe),
    ]:
        n_raw_first = int(threshold_fn(clean_raw).sum())  # rows, not compounds: the wrong unit and order
        result = recipe_fn(selection)
        summing_only = check_positives_only_from_summing(clean_raw, set(result.positives["SMILES"]), threshold_fn)
        order[label] = {
            "n_rows_passing_when_thresholded_before_dedup": n_raw_first,
            "n_positives_dedup_first": result.n_positives,
            "positives_that_exist_only_because_reads_were_summed": summing_only,
        }
    write_report("recipe-yields", {"recipes": results, "dedup_order_diagnostic": order})


if __name__ == "__main__":
    main()
