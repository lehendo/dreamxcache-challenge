#!/usr/bin/env python3
"""Descriptive audit of the raw files you supplied: schemas, null rates, split
overlap and table-level measurements. Compares nothing against expected values;
it reports what is in *your* copy of the data.

    python scripts/audit_dataset.py [--raw-dir data/raw]

Expected under the raw directory (see DATA.md): PGK2_selection.parquet,
PGK2_NTC_supplement.parquet (optional), Val-Test-set/PGK2_Validation_split.csv
and PGK2_Test_split.csv (optional).
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

import pandas as pd

from dreamxcache.config import raw_dir
from dreamxcache.ingest.audit import (
    check_wiki_recipe_empty_smiles,
    count_zero_target_rows,
    describe_csv,
    describe_parquet,
    library_row_counts,
    ntc_supplement_overlap,
    read_totals,
    singleton_fraction,
    zscore_count_relationship,
)
from dreamxcache.ingest.load import load_ntc_supplement, load_selection
from dreamxcache.provenance import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.raw_dir or raw_dir()

    selection_path = root / "PGK2_selection.parquet"
    supplement_path = root / "PGK2_NTC_supplement.parquet"
    splits = {
        "validation": root / "Val-Test-set" / "PGK2_Validation_split.csv",
        "test": root / "Val-Test-set" / "PGK2_Test_split.csv",
    }

    print("Loading the selection table...")
    selection = load_selection(path=selection_path)
    supplement = load_ntc_supplement(path=supplement_path) if supplement_path.exists() else None

    report: dict[str, Any] = {
        "package_versions": {
            p: importlib.metadata.version(p) for p in ["pandas", "numpy", "pyarrow", "scipy", "rdkit"]
        },
        "schema": {"PGK2_selection": describe_parquet(selection_path)},
        "target_arm_rows_with_zero_count": count_zero_target_rows(selection),
        "singleton_fraction": singleton_fraction(selection),
        "library_row_counts": library_row_counts(selection),
        "wiki_recipe_rows_with_invalid_smiles": check_wiki_recipe_empty_smiles(selection),
    }
    if "zscore_PGK2" in selection.columns:
        report["zscore_count_relationship"] = zscore_count_relationship(selection)

    if supplement is not None:
        report["schema"]["PGK2_NTC_supplement"] = describe_parquet(supplement_path)
        report["read_totals"] = read_totals(selection, supplement)
        report["ntc_supplement_overlap"] = ntc_supplement_overlap(selection, supplement)
    else:
        report["read_totals"] = read_totals(selection)

    selection_smiles = set(selection["SMILES"])
    for name, path in splits.items():
        if not path.exists():
            continue
        report["schema"][f"{name}_split"] = describe_csv(path)
        df = pd.read_csv(path)
        report[f"{name}_split_smiles_overlap_with_selection"] = {
            "n_overlap": int(df["SMILES"].isin(selection_smiles).sum()),
            "n_total": len(df),
        }

    write_report("data-audit", report)


if __name__ == "__main__":
    main()
