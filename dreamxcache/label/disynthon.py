"""Disynthon-level count aggregation and competitive-hit classification, after
McCloskey et al. 2020 (arXiv:2002.02530): counts are summed over every
combination of two building blocks across all cycle combinations. For a
three-cycle library A-B-C that gives the A-B, A-C and B-C disynthons.

For a compound ``library-bb1-bb2-bb3`` the three disynthons are the
(bb1, bb2), (bb1, bb3) and (bb2, bb3) pairs. These are *co-occurrence* pairs,
not statements about bonding: the hyphen order of a compound ID is
reaction/synthesis order, not bond connectivity, so nothing here assumes
which building blocks are bonded to which. Compounds are simply grouped by
the pair of building-block identities they share.

Aggregation runs on SMILES-deduplicated, per-compound counts (never on raw
rows) so the dedup-before-threshold invariant of ``recipes.py`` holds one
level up: deduplicate first to get correct per-compound sums, then pool those
sums to the disynthon level.
"""

from __future__ import annotations

import pandas as pd

_DISYNTHON_PAIR_TAGS = (("bb1", "bb2", "AB"), ("bb1", "bb3", "AC"), ("bb2", "bb3", "BC"))


def melt_to_disynthons(compounds: pd.DataFrame, sum_cols: list[str]) -> pd.DataFrame:
    """Long-format table with three rows per input compound (one per
    disynthon pair), each carrying that compound's own deduplicated counts.

    `compounds` needs ``library``, ``bb1``, ``bb2``, ``bb3`` columns (see
    ``parse_compound_id``) plus `sum_cols`. ``disynthon_id`` is a string key
    unique per (pair tag, library, building-block pair); the AB/AC/BC tag is
    included so the same two numeric building-block values in different
    positions never collide.
    """
    parts = []
    for col_a, col_b, tag in _DISYNTHON_PAIR_TAGS:
        disynthon_id = (
            tag + ":" + compounds["library"] + ":" + compounds[col_a].astype(str) + ":" + compounds[col_b].astype(str)
        )
        part = compounds[sum_cols].copy()
        part["disynthon_id"] = disynthon_id
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def aggregate_disynthon_counts(compounds: pd.DataFrame, sum_cols: list[str]) -> pd.DataFrame:
    """Disynthon-level table, one row per ``disynthon_id``, with `sum_cols`
    summed over every compound sharing that disynthon (a compound contributes
    to all three of its disynthons)."""
    melted = melt_to_disynthons(compounds, sum_cols)
    return melted.groupby("disynthon_id", as_index=False)[sum_cols].sum()


def classify_competitive_hit_disynthons(
    disynthon_agg: pd.DataFrame,
    target_threshold: float,
    count_col: str = "count_PGK2",
    inhibitor_col: str = "count_PGK2_with_inhibitor",
    ntc_col: str = "count_NTC",
) -> pd.Series:
    """Competitive-hit criteria applied at the disynthon level: enriched in
    target AND not enriched in the competitor arm AND not enriched in the
    no-target control. `target_threshold` is the swept axis.

    Competitor and NTC both use the ratio form ``< 0.1 x target count``, not
    the compound-level recipe's ``count_NTC == 0``. A disynthon pools reads
    over potentially hundreds of compounds, so requiring exactly zero summed
    NTC reads is a much stricter, scale-dependent bar than it is for a single
    compound; the ratio form is scale-invariant. This is a deliberate
    deviation from the compound-level recipe.
    """
    count = disynthon_agg[count_col]
    enriched_target = count >= target_threshold
    not_enriched_competitor = disynthon_agg[inhibitor_col] < 0.1 * count
    not_enriched_ntc = disynthon_agg[ntc_col] < 0.1 * count
    return enriched_target & not_enriched_competitor & not_enriched_ntc


def propagate_disynthon_hits_to_compounds(compounds: pd.DataFrame, positive_disynthon_ids: set[str]) -> pd.Series:
    """A compound is positive if ANY of its three disynthons is a competitive
    hit (permissive OR propagation). Chosen because DEL binding is often
    driven by a dominant sub-structure contributed by a subset of the
    building blocks, not by simultaneous enrichment across all three pairwise
    combinations. A stricter AND propagation is a natural variant that is not
    built here.

    Does not apply ``historic_hits``: that is a compound-level signal given
    directly, so the caller re-applies it after propagation.
    """
    ids_ab = "AB:" + compounds["library"] + ":" + compounds["bb1"].astype(str) + ":" + compounds["bb2"].astype(str)
    ids_ac = "AC:" + compounds["library"] + ":" + compounds["bb1"].astype(str) + ":" + compounds["bb3"].astype(str)
    ids_bc = "BC:" + compounds["library"] + ":" + compounds["bb2"].astype(str) + ":" + compounds["bb3"].astype(str)
    return (
        ids_ab.isin(positive_disynthon_ids) | ids_ac.isin(positive_disynthon_ids) | ids_bc.isin(positive_disynthon_ids)
    )
