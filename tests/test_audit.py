"""Hand-checked fixtures for dreamxcache/ingest/audit.py.

Each fixture is small enough that the expected result is computed by hand in
the test itself, not by trusting audit.py's own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dreamxcache.ingest.audit import (
    check_enumeration_gap,
    check_group_size_enrichment,
    check_positives_only_from_summing,
    check_wiki_recipe_empty_smiles,
    count_zero_target_rows,
    group_size_distribution,
    library_row_counts,
    missing_from_enumeration,
    ntc_supplement_overlap,
    read_totals,
    singleton_fraction,
    zscore_count_relationship,
)
from dreamxcache.label.recipes import wiki_recipe_threshold


def test_count_zero_target_rows():
    assert count_zero_target_rows(pd.DataFrame({"count_PGK2": [1, 2, 3]}))["n_rows_count_pgk2_eq_0"] == 0
    assert count_zero_target_rows(pd.DataFrame({"count_PGK2": [0, 1, 0]}))["n_rows_count_pgk2_eq_0"] == 2


def test_singleton_fraction_hand_computed():
    assert singleton_fraction(pd.DataFrame({"count_PGK2": [1, 1, 1, 1, 5]})) == 0.8


def test_read_totals():
    selection = pd.DataFrame({"count_PGK2": [1, 2], "count_PGK2_with_inhibitor": [0, 1], "count_NTC": [3, 4]})
    supplement = pd.DataFrame({"count_NTC": [10]})
    totals = read_totals(selection, supplement)
    assert totals == {"count_PGK2": 3, "count_PGK2_with_inhibitor": 1, "count_NTC_main": 7, "count_NTC_combined": 17}


def test_library_row_counts_parses_and_counts_unparsed():
    selection = pd.DataFrame({"compound": ["qDOS11-1-2-3", "qDOS11-4-5-6", "qDOS18_3-1-1-1", "not-a-valid-id"]})
    result = library_row_counts(selection)
    assert result["counts_by_library"] == {"qDOS11": 2, "qDOS18_3": 1}
    assert result["unparsed_compound_ids"] == 1


def test_ntc_supplement_overlap_hand_checked():
    selection = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC"], "count_NTC": [10, 20, 30]})
    # one empty SMILES, one repeated group (CCF x2), CCO agrees with main (10), CCN disagrees (20 vs 99)
    supplement = pd.DataFrame({"SMILES": ["", "CCF", "CCF", "CCO", "CCN"], "count_NTC": [5, 1, 2, 10, 99]})
    obs = ntc_supplement_overlap(selection, supplement)
    assert obs["empty_smiles_rows"] == 1
    assert obs["repeated_smiles_groups"] == 1
    assert obs["overlapping_smiles"] == 2
    assert obs["count_agreements"] == 1
    assert obs["count_disagreements"] == 1


def test_zscore_count_relationship_hand_checked_slope_and_positivity():
    n = 15
    counts = np.arange(11, 11 + n, dtype=float)  # all > 10
    lib_x = pd.DataFrame(
        {"compound": [f"qDOSx1-{i}-1-1" for i in range(n)], "count_PGK2": counts, "zscore_PGK2": counts * 1.0}
    )
    lib_y = pd.DataFrame(
        {"compound": [f"qDOSy1-{i}-1-1" for i in range(n)], "count_PGK2": counts, "zscore_PGK2": counts * 5.0}
    )
    result = zscore_count_relationship(pd.concat([lib_x, lib_y], ignore_index=True))
    assert result["n_nonpositive"] == 0
    assert result["slopes_by_library"]["qDOSx1"] == 1.0
    assert result["slopes_by_library"]["qDOSy1"] == 5.0
    assert result["slope_ratio_max_over_min"] == 5.0
    assert result["r2_by_library"]["qDOSx1"] == 1.0


def test_check_wiki_recipe_empty_smiles_hand_checked():
    # rows 0-3 pass the recipe on counts (row 4 fails count>=3); row 3 has an empty SMILES.
    selection = pd.DataFrame(
        {
            "SMILES": ["CCCCCCCCCCCC", "c1ccccc1CCCC", "CCCCCCCCCCCC", "", "CCCCCCCCCCCC"],
            "count_PGK2": [3, 10, 100, 5, 2],
            "count_PGK2_with_inhibitor": [0, 0, 5, 0, 0],
            "count_NTC": [0, 0, 0, 0, 0],
            "historic_hits": [0, 0, 0, 0, 0],
        }
    )
    result = check_wiki_recipe_empty_smiles(selection)
    assert result["n_rows_passing_wiki_recipe"] == 4
    assert result["n_passing_with_invalid_smiles"] == 1
    assert result["n_passing_with_valid_smiles"] == 3


def test_missing_from_enumeration_hand_checked():
    selection_compounds = pd.Series(["A-1-1-1", "A-2-2-2", "A-3-3-3"], index=[10, 20, 30])
    enum_compounds = pd.Series(["A-1-1-1", "A-3-3-3", "A-9-9-9"])  # A-2-2-2 absent
    missing = missing_from_enumeration(selection_compounds, enum_compounds)
    assert missing.tolist() == [False, True, False]
    assert list(missing.index) == [10, 20, 30]


def test_check_enumeration_gap_hand_checked_and_stratified_by_count():
    selection_lib = pd.DataFrame(
        {"compound": ["L-1-1-1", "L-2-2-2", "L-3-3-3", "L-4-4-4"], "count_PGK2": [1, 1, 10, 10]}
    )
    enum_compounds = pd.Series(["L-1-1-1", "L-4-4-4"])  # L-2-2-2 and L-3-3-3 missing
    result = check_enumeration_gap(selection_lib, enum_compounds, library="L")
    assert result["n_unique_selection_compounds"] == 4
    assert result["n_missing_from_enumeration"] == 2
    assert result["missing_rate"] == 0.5
    band = result["missing_rate_by_count_band"]
    assert band["count_eq_1"]["n_missing"] == 1  # L-2-2-2
    assert band["count_ge_10"]["n_missing"] == 1  # L-3-3-3


def test_group_size_distribution_hand_checked():
    sizes = group_size_distribution(pd.DataFrame({"SMILES": ["a", "a", "a", "b", "c", "c"]}))
    assert sizes.to_dict() == {"a": 3, "b": 1, "c": 2}


def test_check_group_size_enrichment_hand_checked():
    group_sizes = pd.Series([1, 1, 1, 1, 2, 3], index=["p", "q", "r", "s", "t", "u"])
    result = check_group_size_enrichment(group_sizes, positive_smiles={"t", "u"})
    assert result["n_total_unique_compounds"] == 6
    assert result["n_positives"] == 2
    assert result["baseline_multi_row_fraction"] == pytest.approx(2 / 6)
    assert result["positive_multi_row_fraction"] == 1.0
    assert result["enrichment_ratio"] == pytest.approx(3.0)
    assert result["baseline_group_size_distribution"] == {1: 4, 2: 1, 3: 1}


def test_check_positives_only_from_summing_hand_checked():
    # 'SAME': two raw rows of count 2 (fails >=3 alone, passes summed); 'SOLO': one row of count 5.
    clean_raw = pd.DataFrame(
        {
            "SMILES": ["SAME", "SAME", "SOLO"],
            "count_PGK2": [2, 2, 5],
            "count_PGK2_with_inhibitor": [0, 0, 0],
            "count_NTC": [0, 0, 0],
            "historic_hits": [0, 0, 0],
        }
    )
    result = check_positives_only_from_summing(
        clean_raw, positive_smiles={"SAME", "SOLO"}, recipe_threshold_fn=wiki_recipe_threshold
    )
    assert result["n_positives_with_no_individually_passing_row"] == 1
    assert result["fraction_only_from_summing"] == 0.5
