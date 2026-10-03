"""Shared plumbing for the ranker command-line scripts.

Every ranker script scores a *query library* (a CSV or parquet with an id
column and a SMILES column) and writes a full-length score array aligned to
that file's row order, plus, optionally, a naive top-50 selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dreamxcache.config import cache_dir, submissions_dir
from dreamxcache.provenance import write_report
from dreamxcache.submission import N_SELECT, write_validation_txt


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--query", type=Path, required=True, help="library to score: CSV or parquet with id and SMILES columns"
    )
    parser.add_argument("--id-col", default="CatalogID")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--name", required=True, help="output label, e.g. erg_test -> <cache>/erg_test.npy")
    parser.add_argument(
        "--write-top50", action="store_true", help="also write a naive top-50 selection to <submissions>/<name>.txt"
    )
    parser.add_argument("--n-workers", type=int, default=8)


def load_query(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.query) if args.query.suffix == ".parquet" else pd.read_csv(args.query)
    for col in (args.id_col, args.smiles_col):
        if col not in df.columns:
            raise ValueError(f"{args.query} has no column {col!r}")
    return df


def save_ranking(
    args: argparse.Namespace, query: pd.DataFrame, scores: np.ndarray, label: str, extra: dict[str, object]
) -> None:
    """Persist the score array, optionally the top-50 file, and a report."""
    cache_dir().mkdir(parents=True, exist_ok=True)
    scores_path = cache_dir() / f"{args.name}.npy"
    np.save(scores_path, scores)
    print(f"similarity/score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}], mean={np.nanmean(scores):.4f}")
    print(f"Wrote {scores_path}")

    problems: list[str] = []
    if args.write_top50:
        top = np.argsort(-scores)[:N_SELECT]
        out = submissions_dir() / f"{args.name}.txt"
        problems = write_validation_txt(query.loc[top, args.id_col].astype(str).tolist(), out)
        print(f"Wrote {out} (format problems: {problems or 'none'})")

    write_report(
        label,
        {
            "query": args.query.name,
            "n_query": len(query),
            "scores_file": scores_path.name,
            "top50_format_problems": problems,
            **extra,
        },
    )
