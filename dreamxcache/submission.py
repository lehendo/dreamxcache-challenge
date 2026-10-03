"""Submission file formats for the challenge, and local format validation.

Validation split: a ``.txt`` with exactly 50 lines, one ``CatalogID`` per line.
Test split: a CSV with columns ``CatalogID, Sel_50, Score``: every compound in
the split present, exactly 50 rows with ``Sel_50 == 1``, and a ``Score`` on
every row. These checks cover everything that can be verified without gold
labels; they cannot verify hit or series counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_SELECT = 50


def validate_test_format(df: pd.DataFrame, template: pd.DataFrame) -> list[str]:
    problems = []
    if list(df.columns) != ["CatalogID", "Sel_50", "Score"]:
        problems.append(f"columns are {list(df.columns)}, expected ['CatalogID', 'Sel_50', 'Score']")
        return problems
    if set(df["CatalogID"]) != set(template["CatalogID"]):
        problems.append("CatalogID set does not exactly match the template file")
    if df["CatalogID"].duplicated().any():
        problems.append("duplicate CatalogIDs")
    if df["Sel_50"].sum() != N_SELECT:
        problems.append(f"Sel_50 sums to {df['Sel_50'].sum()}, expected exactly {N_SELECT}")
    if not set(df["Sel_50"].unique()).issubset({0, 1}):
        problems.append("Sel_50 has values other than 0/1")
    if df["Score"].isna().any():
        problems.append(f"{df['Score'].isna().sum()} rows have a missing Score")
    if len(df) != len(template):
        problems.append(f"row count {len(df)} != template row count {len(template)}")
    return problems


def validate_validation_format(lines: list[str], template: pd.DataFrame) -> list[str]:
    problems = []
    if len(lines) != N_SELECT:
        problems.append(f"{len(lines)} lines, expected exactly {N_SELECT}")
    if not set(lines).issubset(set(template["CatalogID"])):
        problems.append("some selected CatalogIDs are not in the template file")
    if len(set(lines)) != len(lines):
        problems.append("duplicate CatalogIDs in the selection")
    return problems


def write_validation_txt(selected_ids: list[str], path: Path) -> list[str]:
    """Write the 50-line validation file; returns any format problems found
    against the list itself (duplicates, wrong count)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(selected_ids) + "\n")
    problems = []
    if len(selected_ids) != N_SELECT:
        problems.append(f"{len(selected_ids)} lines, expected {N_SELECT}")
    if len(set(selected_ids)) != len(selected_ids):
        problems.append("duplicate CatalogIDs")
    return problems


def write_test_csv(catalog_ids: pd.Series, selected_idx: np.ndarray, scores: np.ndarray, path: Path) -> None:
    """Write the test-split CSV: `selected_idx` are row indices flagged
    ``Sel_50 = 1``; `scores` is the full-length ``Score`` column."""
    sel_50 = np.zeros(len(catalog_ids), dtype=int)
    sel_50[selected_idx] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"CatalogID": catalog_ids.to_numpy(), "Sel_50": sel_50, "Score": scores}).to_csv(path, index=False)
