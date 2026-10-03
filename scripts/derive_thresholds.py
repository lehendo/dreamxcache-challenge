#!/usr/bin/env python3
"""Derive the data-dependent parameters that recipes L3, L6, L7 and L8 take.

    python scripts/derive_thresholds.py l3-l6                 # matched-size z cutoff + per-library floors
    python scripts/derive_thresholds.py l7 --sizes 1000 5000  # z cutoffs at several target set sizes
    python scripts/derive_thresholds.py l8                    # disynthon-threshold -> positive-count menu

l3-l6
    L3's ``zscore_PGK2`` cutoff is the one whose positive count (other L0
    criteria held fixed) is closest to L0's, for a clean count-vs-z comparison
    at matched set size. L6's per-library ``count_PGK2`` floor is, for each
    library, the integer floor whose positive *rate* within that library is
    closest to L0's global rate (a single global floor gives per-library rates
    that differ by orders of magnitude, driven by sampling depth rather than
    biology).
l7
    A *size sweep*: z cutoffs giving several target positive-set sizes, since
    adding a z criterion to the count criteria can only shrink the set and the
    right size is itself the thing being explored.
l8
    Disynthon threshold -> compound-level positive count is not analytically
    invertible (OR-propagation across three disynthons per compound), so it is
    swept directly, reusing one aggregation.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from dreamxcache.ingest.load import load_selection, parse_compound_id
from dreamxcache.label.disynthon import (
    aggregate_disynthon_counts,
    classify_competitive_hit_disynthons,
    propagate_disynthon_hits_to_compounds,
)
from dreamxcache.label.recipes import SUM_COLS, dedup_selection, dedup_selection_with_zscore, wiki_recipe
from dreamxcache.provenance import write_report

BASE_COLS = ["compound", "SMILES", "count_PGK2", "count_PGK2_with_inhibitor", "count_NTC", "historic_hits"]


def z_cutoff_for_size(z_sorted_desc: pd.Series, target_n: int) -> dict[str, Any]:
    cutoff = float(z_sorted_desc.min()) if len(z_sorted_desc) <= target_n else float(z_sorted_desc.iloc[target_n - 1])
    achieved = int((z_sorted_desc >= cutoff).sum())
    return {
        "target_n": target_n,
        "n_candidates_passing_other_criteria": len(z_sorted_desc),
        "derived_z_cutoff": cutoff,
        "achieved_n": achieved,
        "delta_from_target": achieved - target_n,
    }


def run_l3_l6(max_floor: int) -> None:
    selection = load_selection(columns=[*BASE_COLS, "zscore_PGK2"])
    l0 = wiki_recipe(selection.drop(columns=["zscore_PGK2"]))
    l0_n, l0_total = l0.n_positives, len(l0.deduped)
    l0_rate = l0_n / l0_total
    print(f"L0: {l0_n:,} positives / {l0_total:,} unique compounds = {l0_rate:.6%}")

    deduped_z = dedup_selection_with_zscore(selection)
    count = deduped_z["count_PGK2"]
    other = (
        (deduped_z["count_PGK2_with_inhibitor"] < 0.1 * count)
        & (deduped_z["count_NTC"] == 0)
        & (deduped_z["historic_hits"] < 5)
    )
    l3 = z_cutoff_for_size(deduped_z.loc[other, "zscore_PGK2"].sort_values(ascending=False), l0_n)
    print(f"L3: {l3}")

    deduped = dedup_selection(selection.drop(columns=["zscore_PGK2"]))
    deduped = deduped.assign(library=parse_compound_id(deduped["compound"])["library"])
    count = deduped["count_PGK2"]
    other = (
        (deduped["count_PGK2_with_inhibitor"] < 0.1 * count)
        & (deduped["count_NTC"] == 0)
        & (deduped["historic_hits"] < 5)
    )
    l6: dict[str, Any] = {}
    for lib, lib_df in deduped.groupby("library"):
        n_lib = len(lib_df)
        lib_count = lib_df["count_PGK2"]
        lib_other = other.loc[lib_df.index]
        best_t, best_rate, best_err = None, None, float("inf")
        for t in range(1, max_floor + 1):
            rate = int(((lib_count >= t) & lib_other).sum()) / n_lib
            if abs(rate - l0_rate) < best_err:
                best_t, best_rate, best_err = t, rate, abs(rate - l0_rate)
        l6[str(lib)] = {"n_unique_in_library": n_lib, "derived_floor": best_t, "achieved_rate": best_rate}
        print(f"  L6 {lib}: floor={best_t} rate={best_rate:.4%}")

    path = write_report(
        "derive-thresholds-l3-l6",
        {
            "l0_n_positives": l0_n,
            "l0_rate": l0_rate,
            "l3": l3,
            "l6": l6,
            "l6_thresholds_json": {lib: r["derived_floor"] for lib, r in l6.items()},
        },
    )
    print(
        f"\nPass --z-cutoff {l3['derived_z_cutoff']!r} and the l6_thresholds_json block of {path} to recipe_yields.py."
    )


def run_l7(sizes: list[int]) -> None:
    selection = load_selection(columns=[*BASE_COLS, "zscore_PGK2"])
    deduped = dedup_selection_with_zscore(selection)
    count = deduped["count_PGK2"]
    # L7 omits the no-target-control criterion (see recipes.py).
    other = (count >= 3) & (deduped["count_PGK2_with_inhibitor"] < 0.1 * count) & (deduped["historic_hits"] < 5)
    z_sorted = deduped.loc[other, "zscore_PGK2"].sort_values(ascending=False)
    print(f"{len(z_sorted):,} compounds pass count/inhibitor/historic_hits (before the z criterion)")
    sweep = {}
    for n in sizes:
        sweep[n] = z_cutoff_for_size(z_sorted, n)
        print(f"  target={n:>7,}  z_cutoff={sweep[n]['derived_z_cutoff']:.4f}  achieved={sweep[n]['achieved_n']:>7,}")
    write_report("derive-thresholds-l7", {"n_candidates": len(z_sorted), "sweep": sweep})


def run_l8(thresholds: list[float]) -> None:
    selection = load_selection(columns=BASE_COLS)
    deduped = dedup_selection(selection)
    deduped = deduped.assign(**parse_compound_id(deduped["compound"]))
    agg = aggregate_disynthon_counts(deduped, sum_cols=list(SUM_COLS))
    print(f"{len(deduped):,} deduplicated compounds -> {len(agg):,} unique disynthons")
    print(
        "disynthon count_PGK2 percentiles: "
        + ", ".join(f"p{p}={np.percentile(agg['count_PGK2'], p):.0f}" for p in [50, 75, 90, 95, 99, 99.9])
    )
    sweep = []
    for t in thresholds:
        hit_mask = classify_competitive_hit_disynthons(agg, target_threshold=t)
        ids = set(agg.loc[hit_mask, "disynthon_id"])
        compound_mask = propagate_disynthon_hits_to_compounds(deduped, ids) & (deduped["historic_hits"] < 5)
        sweep.append(
            {"target_threshold": t, "n_disynthon_hits": int(hit_mask.sum()), "n_positives": int(compound_mask.sum())}
        )
        print(
            f"  threshold={t:>7}  n_disynthon_hits={sweep[-1]['n_disynthon_hits']:>8,}  n_positives={sweep[-1]['n_positives']:>9,}"
        )
    positives = np.array([r["n_positives"] for r in sweep])
    if not np.all(np.diff(positives) <= 0):
        print("WARNING: n_positives is not non-increasing in the threshold; investigate before picking a target.")
    write_report("derive-thresholds-l8", {"sweep": sweep})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p36 = sub.add_parser("l3-l6")
    p36.add_argument("--max-floor", type=int, default=50)
    p7 = sub.add_parser("l7")
    p7.add_argument("--sizes", type=int, nargs="+", default=[1000, 2500, 5000, 10000, 13000])
    p8 = sub.add_parser("l8")
    p8.add_argument(
        "--thresholds", type=float, nargs="+", default=[5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    )
    args = parser.parse_args()
    if args.mode == "l3-l6":
        run_l3_l6(args.max_floor)
    elif args.mode == "l7":
        run_l7(args.sizes)
    else:
        run_l8(args.thresholds)


if __name__ == "__main__":
    main()
