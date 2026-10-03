"""Pure loaders and cleaning primitives for the challenge's raw files.

No network access: the raw files must already be on disk (see DATA.md for
how to obtain them). The expected schema is the one documented by the
challenge organizers: a PGK2 selection table with one row per
compound/arm read aggregate (``compound``, ``SMILES``, ``count_PGK2``,
``count_PGK2_with_inhibitor``, ``count_NTC``, ``historic_hits``,
``zscore_PGK2``, ...) and an optional supplementary no-target-control table.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from dreamxcache.config import raw_dir

# Compound IDs have four hyphen-separated parts: library-bb1-bb2-bb3.
# `parse_compound_id` returns NaN for anything that does not match, rather
# than raising, so callers can decide whether that is a data problem.
_COMPOUND_ID_RE = r"^(?P<library>[A-Za-z]+\d+(?:_\d+)?)-(?P<bb1>\d+)-(?P<bb2>\d+)-(?P<bb3>\d+)$"

_ZSCORE_COLUMN_RE = re.compile(r"zscore", re.IGNORECASE)


def load_selection(columns: list[str] | None = None, path: Path | None = None) -> pd.DataFrame:
    """The PGK2 selection table. Rows are raw (one per compound-arm read
    aggregate) and are *not* deduplicated by SMILES.

    `path` defaults to ``<raw_dir>/PGK2_selection.parquet``.
    """
    return pd.read_parquet(path or raw_dir() / "PGK2_selection.parquet", columns=columns)


def load_ntc_supplement(columns: list[str] | None = None, path: Path | None = None) -> pd.DataFrame:
    """The supplementary no-target-control table (compounds never detected
    with the target present).

    `path` defaults to ``<raw_dir>/PGK2_NTC_supplement.parquet``.
    """
    return pd.read_parquet(path or raw_dir() / "PGK2_NTC_supplement.parquet", columns=columns)


def parse_compound_id(compound: pd.Series) -> pd.DataFrame:
    """Split ``qDOS11-34-486-863`` into library/bb1/bb2/bb3 columns.

    Unparseable IDs come back as all-NaN rows (index preserved) rather than
    raising.
    """
    return compound.str.extract(_COMPOUND_ID_RE)


def is_valid_smiles(smiles: pd.Series) -> pd.Series:
    """Cleaning rule: a valid SMILES contains carbon and has length > 10.
    Both conditions are checked explicitly rather than assuming one implies
    the other.
    """
    s = smiles.fillna("")
    has_carbon = s.str.contains("[Cc]", regex=True)
    long_enough = s.str.len() > 10
    return has_carbon & long_enough


def filter_valid_smiles(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    """Drop rows with invalid SMILES before any SMILES-keyed grouping.

    Separate from `dedup_by_smiles` so the row-count change it causes can be
    asserted directly. pandas' ``groupby`` treats the empty string as a valid
    key, so without this step every empty-SMILES row would collapse into one
    phantom "compound" with summed counts.
    """
    return df.loc[is_valid_smiles(df[smiles_col])]


def dedup_by_smiles(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    sum_cols: tuple[str, ...] = (),
    max_cols: tuple[str, ...] = (),
    first_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Filter invalid SMILES, then group by SMILES: named columns are summed
    or maxed, everything else takes the first value in the group.

    Every non-key column must be named in exactly one of
    ``sum_cols``/``max_cols``/``first_cols``. pandas' ``agg`` silently drops
    any column it is not told about, so this raises instead.

    Refuses to sum or max any column whose name contains "zscore": summing
    or maxing a z-score is not the Stouffer combination
    (``sum(z) / sqrt(n)``), and unlike an obviously wrong result, a summed
    z-score still looks like a plausible z-score. Use
    `stouffer_combine_by_smiles` for z-score columns.
    """
    bad = [c for c in (*sum_cols, *max_cols) if _ZSCORE_COLUMN_RE.search(c)]
    if bad:
        raise ValueError(
            f"refusing to sum/max z-score column(s) {bad!r}: this silently produces a "
            "plausible-looking but wrong number. Use stouffer_combine_by_smiles instead."
        )

    named_cols = {smiles_col, *sum_cols, *max_cols, *first_cols}
    unaccounted = [c for c in df.columns if c not in named_cols]
    if unaccounted:
        raise ValueError(
            f"columns {unaccounted!r} are not in sum_cols/max_cols/first_cols and would be "
            "silently dropped by groupby(...).agg(...); name every column explicitly."
        )

    clean = filter_valid_smiles(df, smiles_col)
    agg: dict[str, str] = {c: "sum" for c in sum_cols}
    agg.update({c: "max" for c in max_cols})
    agg.update({c: "first" for c in first_cols})
    return clean.groupby(smiles_col, as_index=False).agg(agg)


def stouffer_combine_by_smiles(df: pd.DataFrame, z_col: str, smiles_col: str = "SMILES") -> pd.Series:
    """Combine a z-score column across duplicate-SMILES rows with the
    unweighted Stouffer method, ``sum(z) / sqrt(n)``. This is the correct
    way to deduplicate a z-score column; `dedup_by_smiles` refuses to sum or
    max one for exactly that reason.

    `df` should already be filtered to valid SMILES (see
    `filter_valid_smiles`); this function does not filter, so a caller always
    knows which population is being combined.

    Returns a Series indexed by SMILES, one row per unique compound.
    """
    grouped = df.groupby(smiles_col)[z_col]
    result: pd.Series = grouped.sum() / grouped.size().pow(0.5)
    return result
