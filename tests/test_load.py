"""Hand-checked fixtures for dreamxcache/ingest/load.py.

Regression coverage for the dedup phantom-row bug: pandas' groupby treats ""
as a valid group key (unlike NaN, which it drops by default), so without the
fix every empty-SMILES row collapses into ONE phantom row with summed counts,
a structureless "compound" that could pass every labeling threshold. These
tests assert that cannot happen, not just that today's fix works.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dreamxcache.ingest.load import (
    dedup_by_smiles,
    filter_valid_smiles,
    is_valid_smiles,
    parse_compound_id,
    stouffer_combine_by_smiles,
)


def test_is_valid_smiles_hand_checked():
    smiles = pd.Series(
        [
            "",  # empty
            "CC",  # has carbon, length 2 <= 10 -> invalid
            "NNNNNNNNNNNN",  # length 12 > 10, no carbon -> invalid
            "CCCCCCCCCCCC",  # has carbon, length 12 > 10 -> valid
            None,  # null -> invalid
        ]
    )
    assert is_valid_smiles(smiles).tolist() == [False, False, False, True, False]


def test_filter_valid_smiles_drops_exactly_the_invalid_rows():
    df = pd.DataFrame({"SMILES": ["", "CCCCCCCCCCCC", "", "c1ccccc1CCCCC"], "x": [1, 2, 3, 4]})
    result = filter_valid_smiles(df)
    assert len(df) - len(result) == 2
    assert set(result["SMILES"]) == {"CCCCCCCCCCCC", "c1ccccc1CCCCC"}


def test_dedup_by_smiles_does_not_collapse_empty_smiles_into_a_phantom_row():
    df = pd.DataFrame(
        {
            "SMILES": ["", "", "", "CCCCCCCCCCCC", "CCCCCCCCCCCC", "c1ccccc1CCCCC"],
            "count_PGK2": [1000, 2000, 3000, 5, 3, 1],
        }
    )
    result = dedup_by_smiles(df, sum_cols=("count_PGK2",))

    assert "" not in set(result["SMILES"])
    assert len(result) == 2
    real = result.set_index("SMILES")["count_PGK2"]
    assert real["CCCCCCCCCCCC"] == 8  # 5 + 3, correctly summed
    assert real["c1ccccc1CCCCC"] == 1
    assert result["count_PGK2"].sum() == 9  # the 6000 empty-SMILES reads appear nowhere


def test_dedup_by_smiles_raises_on_unaccounted_column():
    df = pd.DataFrame({"SMILES": ["CCCCCCCCCCCC"], "count_PGK2": [1], "historic_hits": [0]})
    with pytest.raises(ValueError, match="historic_hits"):
        dedup_by_smiles(df, sum_cols=("count_PGK2",))


def test_dedup_by_smiles_rejects_zscore_in_sum_cols():
    df = pd.DataFrame({"SMILES": ["CCCCCCCCCCCC", "CCCCCCCCCCCC"], "zscore_PGK2": [0.1, 0.2]})
    with pytest.raises(ValueError, match="zscore_PGK2"):
        dedup_by_smiles(df, sum_cols=("zscore_PGK2",))


def test_dedup_by_smiles_rejects_zscore_in_max_cols():
    df = pd.DataFrame({"SMILES": ["CCCCCCCCCCCC", "CCCCCCCCCCCC"], "zscore_NTC": [0.1, 0.2]})
    with pytest.raises(ValueError, match="zscore_NTC"):
        dedup_by_smiles(df, max_cols=("zscore_NTC",))


def test_parse_compound_id_hand_checked():
    compounds = pd.Series(["qDOS11-34-486-863", "qDOS18_3-1-2-3", "not-a-valid-id-at-all-x"])
    parsed = parse_compound_id(compounds)
    assert parsed.loc[0, "library"] == "qDOS11"
    assert parsed.loc[0, ["bb1", "bb2", "bb3"]].tolist() == ["34", "486", "863"]
    assert parsed.loc[1, "library"] == "qDOS18_3"
    assert parsed.loc[2].isna().all()  # unparseable -> NaN, not a raise


def test_stouffer_combine_by_smiles_hand_checked():
    # "AAA": z = [1, 3] -> sum 4 / sqrt(2); "BBB": single row -> unchanged
    df = pd.DataFrame({"SMILES": ["AAA", "AAA", "BBB"], "zscore_PGK2": [1.0, 3.0, 5.0]})
    result = stouffer_combine_by_smiles(df, z_col="zscore_PGK2")
    assert result["AAA"] == pytest.approx(4.0 / (2**0.5))
    assert result["BBB"] == pytest.approx(5.0)
