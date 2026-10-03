"""Descriptive data-audit helpers for the selection and supplement tables.

Every function is a pure function over already-loaded dataframes (or a
parquet path, for the metadata-only schema report), so each one is
independently unit-testable and re-runnable. They *measure*: none of them
compares its result with an expected value, so the same code can be pointed
at any release of the data.

Run via ``scripts/audit_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dreamxcache.ingest.load import is_valid_smiles, parse_compound_id
from dreamxcache.label.recipes import wiki_recipe_threshold


def describe_parquet(path: Path) -> dict[str, Any]:
    """Row count, dtypes, null rate and min/max per column, taken from the
    parquet column statistics rather than a full read. Cheap even for
    GB-scale files."""
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    columns: dict[str, dict[str, Any]] = {
        f.name: {"dtype": str(f.type), "null_count": 0, "min": None, "max": None} for f in pf.schema_arrow
    }
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            info = columns.get(col.path_in_schema)
            stats = col.statistics
            if info is None or stats is None:
                continue
            info["null_count"] += stats.null_count or 0
            if stats.has_min_max:
                info["min"] = stats.min if info["min"] is None else min(info["min"], stats.min)
                info["max"] = stats.max if info["max"] is None else max(info["max"], stats.max)
    for info in columns.values():
        info["null_rate"] = info["null_count"] / n_rows if n_rows else None
    return {
        "file": path.name,
        "n_rows": n_rows,
        "n_row_groups": pf.metadata.num_row_groups,
        "columns": columns,
    }


def describe_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    columns = {
        c: {
            "dtype": str(df[c].dtype),
            "null_count": int(df[c].isna().sum()),
            "null_rate": float(df[c].isna().mean()),
        }
        for c in df.columns
    }
    return {"file": path.name, "n_rows": len(df), "columns": columns}


# --- Table-level measurements ----------------------------------------------


def count_zero_target_rows(selection: pd.DataFrame) -> dict[str, int]:
    """Rows with ``count_PGK2 == 0``. The competitor and no-target-control
    arms are only reported for compounds seen with the target present, so
    this is expected to be zero in the selection table."""
    return {
        "n_rows_count_pgk2_eq_0": int((selection["count_PGK2"] == 0).sum()),
        "n_rows_total": len(selection),
    }


def singleton_fraction(selection: pd.DataFrame) -> float:
    """Fraction of rows with ``count_PGK2 == 1``. A large singleton fraction
    is what makes ratio-based competition criteria nearly vacuous at low
    counts: ``inhibitor < 0.1 * count`` is satisfied by ``inhibitor == 0``
    whenever ``count`` is small, whether or not competition occurred."""
    return float((selection["count_PGK2"] == 1).mean())


def read_totals(selection: pd.DataFrame, ntc_supplement: pd.DataFrame | None = None) -> dict[str, int]:
    out = {
        "count_PGK2": int(selection["count_PGK2"].sum()),
        "count_PGK2_with_inhibitor": int(selection["count_PGK2_with_inhibitor"].sum()),
        "count_NTC_main": int(selection["count_NTC"].sum()),
    }
    if ntc_supplement is not None:
        out["count_NTC_combined"] = out["count_NTC_main"] + int(ntc_supplement["count_NTC"].sum())
    return out


def library_row_counts(selection: pd.DataFrame) -> dict[str, Any]:
    """Selection rows per library, parsed from the compound ID, plus the
    number of IDs that did not parse."""
    parsed = parse_compound_id(selection["compound"])
    return {
        "counts_by_library": {k: int(v) for k, v in parsed["library"].value_counts().to_dict().items()},
        "unparsed_compound_ids": int(parsed["library"].isna().sum()),
    }


def ntc_supplement_overlap(selection: pd.DataFrame, ntc_supplement: pd.DataFrame) -> dict[str, int]:
    """Quality measurements for the supplementary no-target-control table:
    empty SMILES, repeated-SMILES groups, SMILES shared with the main table
    and how many of those disagree on ``count_NTC``. The supplement's counts
    are only trustworthy as *presence* evidence if it is disjoint from the
    main table and internally consistent, which is what this quantifies."""
    smiles = ntc_supplement["SMILES"]
    is_empty = smiles.isna() | (smiles.fillna("").str.strip() == "")
    non_empty = ntc_supplement.loc[~is_empty]
    dup_group_sizes = non_empty.groupby("SMILES").size()

    main_dedup = selection.groupby("SMILES", as_index=False).agg(count_NTC=("count_NTC", "sum"))
    supp_dedup = non_empty.groupby("SMILES", as_index=False).agg(count_NTC=("count_NTC", "sum"))
    overlap = pd.merge(main_dedup, supp_dedup, on="SMILES", suffixes=("_main", "_supp"))
    n_agree = int((overlap["count_NTC_main"] == overlap["count_NTC_supp"]).sum())
    return {
        "empty_smiles_rows": int(is_empty.sum()),
        "repeated_smiles_groups": int((dup_group_sizes > 1).sum()),
        "overlapping_smiles": len(overlap),
        "count_agreements": n_agree,
        "count_disagreements": len(overlap) - n_agree,
    }


def zscore_count_relationship(selection: pd.DataFrame, min_count: int = 10) -> dict[str, Any]:
    """Per-library proportionality between ``zscore_PGK2`` and ``count_PGK2``
    (ordinary least squares through the origin, rows with ``count > min_count``).

    If z-scores are close to proportional to count within a library, but the
    slope differs between libraries, then z-score is mostly a per-library
    rescaling of count rather than an independent signal.
    """
    z = selection["zscore_PGK2"]
    parsed = parse_compound_id(selection["compound"])
    df = selection.assign(library=parsed["library"])
    df_hi = df[df["count_PGK2"] > min_count]

    slopes: dict[str, float] = {}
    r2s: dict[str, float] = {}
    for lib, sub in df_hi.groupby("library"):
        if len(sub) < 10:
            continue
        x = sub["count_PGK2"].to_numpy(dtype=float)
        y = sub["zscore_PGK2"].to_numpy(dtype=float)
        slope = float((x @ y) / (x @ x))
        ss_res = float(np.sum((y - slope * x) ** 2))
        ss_tot = float(np.sum(y**2))
        r2s[str(lib)] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        slopes[str(lib)] = slope

    positive_slopes = [s for s in slopes.values() if s > 0]
    return {
        "n_nonpositive": int((z <= 0).sum()),
        "range": [float(z.min()), float(z.max())],
        "slopes_by_library": slopes,
        "r2_by_library": r2s,
        "slope_ratio_max_over_min": (max(positive_slopes) / min(positive_slopes) if positive_slopes else float("nan")),
        "min_r2": min(r2s.values()) if r2s else float("nan"),
    }


# --- Labeling diagnostics --------------------------------------------------


def check_wiki_recipe_empty_smiles(selection: pd.DataFrame) -> dict[str, Any]:
    """How many *raw* rows passing the wiki recipe's threshold have invalid
    (empty / no-carbon / too-short) SMILES.

    Deliberately applies the threshold to raw, pre-deduplication rows.
    ``dreamxcache.label.recipes`` explains why that ordering is wrong for
    real labeling (it undercounts positives); this function exists to
    quantify that gap, not to compute a recipe's output. Use
    ``wiki_recipe()`` for labels.
    """
    passing = selection.loc[wiki_recipe_threshold(selection)]
    n_passing = len(passing)
    n_invalid = int((~is_valid_smiles(passing["SMILES"])).sum())
    return {
        "n_rows_passing_wiki_recipe": n_passing,
        "n_passing_with_invalid_smiles": n_invalid,
        "n_passing_with_valid_smiles": n_passing - n_invalid,
        "fraction_invalid": n_invalid / n_passing if n_passing else float("nan"),
    }


def missing_from_enumeration(selection_compounds: pd.Series, enum_compounds: pd.Series) -> pd.Series:
    """Boolean mask over `selection_compounds`: which do not appear anywhere
    in `enum_compounds`. Uses pyarrow's ``is_in`` kernel rather than a Python
    set, because enumeration files run to hundreds of millions of rows.

    Explicitly typed ``large_string`` (64-bit offsets): plain ``string``
    caps a single array's total string data at ~2 GB, which the largest
    libraries exceed.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    sel_arr = pa.array(selection_compounds.astype(str), type=pa.large_string())
    enum_arr = pa.array(enum_compounds.astype(str), type=pa.large_string())
    found = pc.is_in(sel_arr, value_set=enum_arr)
    return pd.Series(~np.asarray(found, dtype=bool), index=selection_compounds.index)


def check_enumeration_gap(selection_lib: pd.DataFrame, enum_compounds: pd.Series, library: str) -> dict[str, Any]:
    """For one library: what fraction of unique selection compound IDs are
    absent from its enumeration, and does that rate depend on ``count_PGK2``
    (i.e. are higher-count, more likely positive compounds disproportionately
    missing before labeling even starts)? `selection_lib` must already be
    filtered to this library's rows."""
    unique_sel = selection_lib.drop_duplicates("compound")
    missing = missing_from_enumeration(unique_sel["compound"], enum_compounds)
    n_total = len(unique_sel)
    n_missing = int(missing.sum())

    bands = {
        "count_eq_1": unique_sel["count_PGK2"] == 1,
        "count_ge_3": unique_sel["count_PGK2"] >= 3,
        "count_ge_10": unique_sel["count_PGK2"] >= 10,
    }
    by_band = {
        name: {
            "n": int(mask.sum()),
            "n_missing": int((missing & mask).sum()),
            "missing_rate": float((missing & mask).sum() / mask.sum()) if mask.sum() else None,
        }
        for name, mask in bands.items()
    }
    return {
        "library": library,
        "n_unique_selection_compounds": n_total,
        "n_missing_from_enumeration": n_missing,
        "missing_rate": n_missing / n_total if n_total else float("nan"),
        "missing_rate_by_count_band": by_band,
    }


def group_size_distribution(clean_raw: pd.DataFrame, smiles_col: str = "SMILES") -> pd.Series:
    """Raw-row count per unique SMILES (`clean_raw` should already be
    ``filter_valid_smiles``'d). A value above 1 means the structure was
    encoded by more than one raw row: barcode or protecting-group
    duplication in the library, not necessarily replication."""
    return clean_raw.groupby(smiles_col).size()


def check_group_size_enrichment(group_sizes: pd.Series, positive_smiles: set[str] | pd.Index) -> dict[str, Any]:
    """Are the positives enriched for multi-row (barcode-duplicate) groups
    relative to the full population? If summing counts across duplicates
    did no more work for positives than for a random compound, the two
    multi-row fractions would be close; a large ratio means positives are
    disproportionately compounds that got extra chances to accumulate reads,
    not necessarily higher-affinity ones."""
    positive_smiles_set = set(positive_smiles)
    positive_sizes = group_sizes.loc[group_sizes.index.isin(positive_smiles_set)]
    baseline = float((group_sizes >= 2).mean())
    positive = float((positive_sizes >= 2).mean()) if len(positive_sizes) else float("nan")
    return {
        "n_total_unique_compounds": len(group_sizes),
        "n_positives": len(positive_sizes),
        "baseline_multi_row_fraction": baseline,
        "positive_multi_row_fraction": positive,
        "enrichment_ratio": positive / baseline if baseline > 0 else float("nan"),
        "baseline_group_size_distribution": {
            int(k): int(v)  # type: ignore[call-overload]
            for k, v in group_sizes.value_counts().sort_index().items()
        },
        "positive_group_size_distribution": {
            int(k): int(v)  # type: ignore[call-overload]
            for k, v in positive_sizes.value_counts().sort_index().items()
        },
    }


def check_positives_only_from_summing(
    clean_raw: pd.DataFrame,
    positive_smiles: set[str] | pd.Index,
    recipe_threshold_fn: Any,
    smiles_col: str = "SMILES",
) -> dict[str, Any]:
    """Of `positive_smiles`, how many have *no* single raw row that clears
    `recipe_threshold_fn` on its own (unsummed) counts, i.e. are positives
    only because reads were combined across barcode-duplicate rows. Computed
    per compound rather than inferred from a row-count/compound-count gap."""
    positive_smiles_set = set(positive_smiles)
    candidates = clean_raw.loc[clean_raw[smiles_col].isin(positive_smiles_set)]
    individually_passing = set(candidates.loc[recipe_threshold_fn(candidates), smiles_col])
    only_from_summing = positive_smiles_set - individually_passing
    return {
        "n_positives_checked": len(positive_smiles_set),
        "n_positives_with_no_individually_passing_row": len(only_from_summing),
        "fraction_only_from_summing": (
            len(only_from_summing) / len(positive_smiles_set) if positive_smiles_set else float("nan")
        ),
    }
