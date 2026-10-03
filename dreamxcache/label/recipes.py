"""Reference labeling recipes, L0-L8.

The challenge ships no active/inactive labels for its DEL data: deriving
them from read counts is part of the task. Each recipe here is a named,
pure function ``selection -> LabelingResult``.

Every recipe deduplicates by SMILES (summing counts) **and then** applies its
threshold. That ordering is enforced structurally: no public function
accepts already-deduplicated data from elsewhere and re-applies a threshold,
and none applies a threshold to raw rows. Reason: two individually
sub-threshold raw rows for the same structure (barcode or protecting-group
duplicates in the enumeration) can clear a threshold once their reads are
summed, so thresholding first undercounts positives.

The ``*_threshold()`` predicates are exposed only so
``dreamxcache.ingest.audit`` can deliberately apply them to raw rows to
quantify what goes wrong when the order is reversed. Do not call them
directly to produce labels; call the recipe functions.

    L0  wiki_recipe                         the organizers' published recipe
    L1  strict_recipe                       stricter count and ratio criteria
    L2  wiki_recipe_individually_evidenced  L0, but every positive needs one raw
                                            row that clears the threshold alone
    L3  wiki_recipe_zscore                  z-score cutoff replaces the count floor
    L4  wiki_recipe_count_ge_10             count floor raised to 10
    L5  wiki_recipe_no_competition          competition criterion dropped
    L6  wiki_recipe_per_library             per-library count floors
    L7  wiki_recipe_joint_zscore            count AND z-score AND competition
    L8  wiki_recipe_disynthon_competitive_hit
                                            disynthon-pooled competitive hits

L3, L6 and L7 take their data-dependent cutoffs as arguments; see
``scripts/derive_thresholds.py`` for how to derive them from your copy of the
data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dreamxcache.ingest.load import (
    dedup_by_smiles,
    filter_valid_smiles,
    parse_compound_id,
    stouffer_combine_by_smiles,
)

SUM_COLS = ("count_PGK2", "count_PGK2_with_inhibitor", "count_NTC")
_MAX_COLS = ("historic_hits",)
_FIRST_COLS = ("compound",)
_NEEDED_COLS = ["compound", "SMILES", *SUM_COLS, *_MAX_COLS]
_Z_COL = "zscore_PGK2"


@dataclass(frozen=True)
class LabelingResult:
    name: str
    version: str
    deduped: pd.DataFrame  # the unique-SMILES table this recipe scored
    positive_mask: pd.Series  # boolean, aligned to `deduped`

    @property
    def positives(self) -> pd.DataFrame:
        return self.deduped.loc[self.positive_mask]

    @property
    def n_positives(self) -> int:
        return int(self.positive_mask.sum())


def dedup_selection(selection: pd.DataFrame) -> pd.DataFrame:
    """Unique-SMILES table: counts summed, ``historic_hits`` maxed."""
    return dedup_by_smiles(selection[_NEEDED_COLS], sum_cols=SUM_COLS, max_cols=_MAX_COLS, first_cols=_FIRST_COLS)


def dedup_selection_with_zscore(selection: pd.DataFrame) -> pd.DataFrame:
    """`dedup_selection` plus a Stouffer-combined ``zscore_PGK2`` column.

    Summed raw counts grow linearly with barcode multiplicity regardless of
    affinity. The z-score already normalizes observed reads against expected
    library representation, and Stouffer combination (``sum(z)/sqrt(n)``)
    grows as sqrt(n) for independent replicates rather than linearly, so real
    replication survives while mechanical inflation from barcode count does
    not.
    """
    deduped = dedup_selection(selection)
    clean_raw = filter_valid_smiles(selection[[*_NEEDED_COLS, _Z_COL]])
    z_combined = stouffer_combine_by_smiles(clean_raw, z_col=_Z_COL).rename(_Z_COL)
    return deduped.merge(z_combined, on="SMILES", how="left")


# --- L0: the published recipe ----------------------------------------------


def wiki_recipe_threshold(df: pd.DataFrame) -> pd.Series:
    """L0's threshold condition as a bare predicate. See the module docstring
    before calling this directly."""
    count = df["count_PGK2"]
    return (
        (count >= 3)
        & (df["count_PGK2_with_inhibitor"] < 0.1 * count)
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < 5)
    )


def wiki_recipe(selection: pd.DataFrame) -> LabelingResult:
    """L0. Deduplicate by SMILES, then threshold."""
    deduped = dedup_selection(selection)
    return LabelingResult(
        name="wiki_recipe", version="v1", deduped=deduped, positive_mask=wiki_recipe_threshold(deduped)
    )


# --- L1: stricter count and ratio criteria ----------------------------------


def strict_recipe_threshold(df: pd.DataFrame) -> pd.Series:
    """L1's threshold condition as a bare predicate: count strictly above 3;
    competitor and no-target-control reads each either zero or below 10% of
    the target count; ``historic_hits`` below 5."""
    count = df["count_PGK2"]
    inhibitor = df["count_PGK2_with_inhibitor"]
    ntc = df["count_NTC"]
    return (
        (count > 3)
        & ((inhibitor == 0) | (inhibitor < 0.1 * count))
        & ((ntc == 0) | (ntc < 0.1 * count))
        & (df["historic_hits"] < 5)
    )


def strict_recipe(selection: pd.DataFrame) -> LabelingResult:
    """L1. Deduplicate by SMILES, then threshold."""
    deduped = dedup_selection(selection)
    return LabelingResult(
        name="strict_recipe",
        version="v1",
        deduped=deduped,
        positive_mask=strict_recipe_threshold(deduped),
    )


# --- L2: redundancy confound, directly --------------------------------------
#
# A large share of L0's positives can pass only because reads were summed
# across barcode-duplicate rows. This variant keeps L0 as the base but
# additionally requires at least one raw row whose own, unsummed counts clear
# the threshold, removing summing-only positives directly rather than
# reasoning about them statistically.


def wiki_recipe_individually_evidenced(selection: pd.DataFrame) -> LabelingResult:
    """L2. L0, restricted to compounds with at least one raw row that clears
    L0's threshold on its own."""
    deduped = dedup_selection(selection)
    dedup_mask = wiki_recipe_threshold(deduped)
    candidate_smiles = set(deduped.loc[dedup_mask, "SMILES"])

    clean_raw = filter_valid_smiles(selection[_NEEDED_COLS])
    candidates_raw = clean_raw.loc[clean_raw["SMILES"].isin(candidate_smiles)]
    individually_passing = set(candidates_raw.loc[wiki_recipe_threshold(candidates_raw), "SMILES"])

    final_mask = dedup_mask & deduped["SMILES"].isin(individually_passing)
    return LabelingResult(
        name="wiki_recipe_individually_evidenced",
        version="v1",
        deduped=deduped,
        positive_mask=final_mask,
    )


# --- L3: z-score in place of the raw-count floor ------------------------------


def wiki_recipe_zscore_threshold(df: pd.DataFrame, z_cutoff: float) -> pd.Series:
    """L3's threshold condition as a bare predicate. The competition, NTC and
    ``historic_hits`` criteria still use raw counts, unchanged from L0."""
    count = df["count_PGK2"]
    return (
        (df[_Z_COL] >= z_cutoff)
        & (df["count_PGK2_with_inhibitor"] < 0.1 * count)
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < 5)
    )


def wiki_recipe_zscore(selection: pd.DataFrame, z_cutoff: float) -> LabelingResult:
    """L3. L0 with ``count_PGK2 >= 3`` replaced by a Stouffer-combined
    ``zscore_PGK2 >= z_cutoff``. Pick the cutoff with
    ``scripts/derive_thresholds.py`` (matched positive count against L0)."""
    deduped = dedup_selection_with_zscore(selection)
    return LabelingResult(
        name="wiki_recipe_zscore",
        version=f"v1_z{z_cutoff:.4f}",
        deduped=deduped,
        positive_mask=wiki_recipe_zscore_threshold(deduped, z_cutoff),
    )


# --- L4: competition only where it has power --------------------------------
#
# The competitor arm is reported only for compounds with count_PGK2 >= 1, and
# most rows have count_PGK2 == 1. For a low-count row, "inhibitor < 0.1 *
# count" is satisfied by inhibitor == 0 whether or not competition occurred,
# so the criterion is close to vacuous there. Raising the count floor
# restricts labeling to the regime where the competition test has power.


def wiki_recipe_count_ge_10_threshold(df: pd.DataFrame) -> pd.Series:
    """L4's threshold condition as a bare predicate."""
    count = df["count_PGK2"]
    return (
        (count >= 10)
        & (df["count_PGK2_with_inhibitor"] < 0.1 * count)
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < 5)
    )


def wiki_recipe_count_ge_10(selection: pd.DataFrame) -> LabelingResult:
    """L4. L0 with the count floor raised from 3 to 10."""
    deduped = dedup_selection(selection)
    return LabelingResult(
        name="wiki_recipe_count_ge_10",
        version="v1",
        deduped=deduped,
        positive_mask=wiki_recipe_count_ge_10_threshold(deduped),
    )


# --- L5: hedge for the other site class --------------------------------------
#
# A competition criterion built on an ATP-site competitor labels ATP-site
# binders only. Compounds that bind a second site are invisible to it.
# Dropping the criterion hedges for that second class, at the cost of also
# admitting non-competing (allosteric or promiscuous) binders.


def wiki_recipe_no_competition_threshold(df: pd.DataFrame) -> pd.Series:
    """L5's threshold condition as a bare predicate."""
    count = df["count_PGK2"]
    return (count >= 3) & (df["count_NTC"] == 0) & (df["historic_hits"] < 5)


def wiki_recipe_no_competition(selection: pd.DataFrame) -> LabelingResult:
    """L5. L0 minus the competition (inhibitor) criterion."""
    deduped = dedup_selection(selection)
    return LabelingResult(
        name="wiki_recipe_no_competition",
        version="v1",
        deduped=deduped,
        positive_mask=wiki_recipe_no_competition_threshold(deduped),
    )


# --- L6: per-library count floors ---------------------------------------------
#
# The positive rate under a single global threshold can differ by orders of
# magnitude between sub-libraries, driven by sampling depth rather than
# biology. Per-library floors, each calibrated to the same target positive
# *rate*, correct for that.


def wiki_recipe_per_library_threshold(df: pd.DataFrame, thresholds: dict[str, int]) -> pd.Series:
    """L6's threshold condition as a bare predicate. `df` needs a ``library``
    column (see ``parse_compound_id``). Libraries missing from `thresholds`
    are excluded, not silently included via a permissive default."""
    count = df["count_PGK2"]
    per_row_threshold = df["library"].map(thresholds)
    return (
        df["library"].isin(thresholds)
        & (count >= per_row_threshold)
        & (df["count_PGK2_with_inhibitor"] < 0.1 * count)
        & (df["count_NTC"] == 0)
        & (df["historic_hits"] < 5)
    )


def wiki_recipe_per_library(selection: pd.DataFrame, thresholds: dict[str, int]) -> LabelingResult:
    """L6. L0 with ``count_PGK2 >= 3`` replaced by a per-library floor.
    Derive `thresholds` with ``scripts/derive_thresholds.py``."""
    deduped = dedup_selection(selection)
    deduped = deduped.assign(library=parse_compound_id(deduped["compound"])["library"])
    return LabelingResult(
        name="wiki_recipe_per_library",
        version="v1",
        deduped=deduped,
        positive_mask=wiki_recipe_per_library_threshold(deduped, thresholds),
    )


# --- L7: joint count + z-score + competition (a size sweep, not a point) -------
#
# z-score is *added* to the count-based criteria (AND), not substituted for
# them (contrast L3). Adding a criterion can only shrink a positive set, so
# this variant probes small, jointly-thresholded positive sets. The z cutoff
# is swept externally to produce several target set sizes. The no-target
# control criterion is deliberately omitted.


def wiki_recipe_joint_zscore_threshold(df: pd.DataFrame, z_cutoff: float) -> pd.Series:
    """L7's threshold condition as a bare predicate."""
    count = df["count_PGK2"]
    return (
        (count >= 3)
        & (df["count_PGK2_with_inhibitor"] < 0.1 * count)
        & (df["historic_hits"] < 5)
        & (df[_Z_COL] >= z_cutoff)
    )


def wiki_recipe_joint_zscore(selection: pd.DataFrame, z_cutoff: float) -> LabelingResult:
    """L7. ``count_PGK2 >= 3`` AND ``count_PGK2_with_inhibitor < 0.1 * count``
    AND ``historic_hits < 5`` AND ``zscore_PGK2 >= z_cutoff``, all jointly
    required."""
    deduped = dedup_selection_with_zscore(selection)
    return LabelingResult(
        name="wiki_recipe_joint_zscore",
        version=f"v1_z{z_cutoff:.4f}",
        deduped=deduped,
        positive_mask=wiki_recipe_joint_zscore_threshold(deduped, z_cutoff),
    )


# --- L8: competitive-hit labeling at the disynthon level ---------------------
#
# McCloskey et al. 2020 (arXiv:2002.02530) label *disynthons*: counts are
# pooled over every compound that shares a given pair of building blocks
# before classification. With only a small fraction of the enumerated library
# observed, most per-compound counts sit close to Poisson noise, and pooling
# across a disynthon's members is the denoising step. A compound is positive
# if ANY of its three disynthons is a competitive hit (permissive OR
# propagation; see ``disynthon.py``), after which ``historic_hits`` is
# re-applied per compound.


def wiki_recipe_disynthon_competitive_hit(selection: pd.DataFrame, target_threshold: float) -> LabelingResult:
    """L8. Deduplicate by SMILES, pool counts to disynthons, classify each
    disynthon (enriched in target, not in competitor, not in NTC), propagate
    to compounds (any hit disynthon), then require ``historic_hits < 5``.
    `target_threshold` is the disynthon-aggregate ``count_PGK2`` floor, swept
    externally (``scripts/derive_thresholds.py --disynthon-sweep``)."""
    from dreamxcache.label.disynthon import (
        aggregate_disynthon_counts,
        classify_competitive_hit_disynthons,
        propagate_disynthon_hits_to_compounds,
    )

    deduped = dedup_selection(selection)
    deduped = deduped.assign(**parse_compound_id(deduped["compound"]))

    disynthon_agg = aggregate_disynthon_counts(deduped, sum_cols=list(SUM_COLS))
    disynthon_hit_mask = classify_competitive_hit_disynthons(disynthon_agg, target_threshold=target_threshold)
    positive_disynthon_ids = set(disynthon_agg.loc[disynthon_hit_mask, "disynthon_id"])

    compound_hit = propagate_disynthon_hits_to_compounds(deduped, positive_disynthon_ids)
    final_mask = compound_hit & (deduped["historic_hits"] < 5)

    return LabelingResult(
        name="wiki_recipe_disynthon_competitive_hit",
        version=f"v1_t{target_threshold:.1f}",
        deduped=deduped,
        positive_mask=final_mask,
    )
