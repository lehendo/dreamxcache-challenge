#!/usr/bin/env python3
"""Permutation null for early-recognition metrics, and percentile re-expression
of observed results. Needs no data: the null depends only on ``n_total`` and
``n_hits``.

    python scripts/empirical_null.py --shortlist LRRK2:1500:9 --shortlist WDR91:15419:19 \
        [--observed observed.json] [--n-permutations 1000] [--seed 42] [--save-draws]

``--observed`` is a JSON file ``{"LRRK2": {"ErG": {"EF@50": 0.0, "EF@1%": 0.0,
"LogAUC": 0.0392}, ...}, ...}`` whose results are re-expressed as percentiles of
the matching shortlist's null. ``--save-draws`` stores the raw per-permutation
arrays (a few hundred KB) beside the summary.

The null for a given seed is deterministic, but the *exact* draws depend on where
the hits sit among the compounds, so a null regenerated from counts agrees with
one drawn on the real label vector to within Monte-Carlo error, not bit for bit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamxcache.eval.null_calibration import empirical_null_from_counts, percentiles_vs_null, summarize_null
from dreamxcache.provenance import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", action="append", required=True, metavar="NAME:N_TOTAL:N_HITS")
    parser.add_argument("--observed", type=Path, default=None)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-draws", action="store_true")
    args = parser.parse_args()

    observed = json.loads(args.observed.read_text()) if args.observed else {}
    report: dict[str, object] = {"n_permutations": args.n_permutations, "seed": args.seed, "shortlists": {}}
    for spec in args.shortlist:
        name, n_total, n_hits = spec.split(":")
        null = empirical_null_from_counts(int(n_total), int(n_hits), args.n_permutations, args.seed)
        entry: dict[str, object] = {
            "n_total": int(n_total),
            "n_hits": int(n_hits),
            "null_summary": {m: summarize_null(d) for m, d in null.items()},
        }
        if args.save_draws:
            entry["null_draws"] = {m: d.round(6).tolist() for m, d in null.items()}
        if name in observed:
            entry["observed_vs_null"] = {
                method: percentiles_vs_null(metrics, null) for method, metrics in observed[name].items()
            }
        report["shortlists"][name] = entry  # type: ignore[index]
        print(f"=== {name}: n_total={n_total}, n_hits={n_hits} ===")
        for m, d in null.items():
            s = summarize_null(d)
            print(
                f"  null {m}: mean={s['mean']:.4f} p50={s['p50']:.4f} p95={s['p95']:.4f} p99={s['p99']:.4f} max={s['max']:.4f}"
            )
        for method, metrics in entry.get("observed_vs_null", {}).items():  # type: ignore[attr-defined]
            for m, r in metrics.items():
                print(f"  {method} {m}={r['value']:.4f} -> percentile {r['null_percentile']:.1f}")
    write_report("empirical-null", report)


if __name__ == "__main__":
    main()
