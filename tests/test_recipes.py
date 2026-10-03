"""Hand-checked fixtures for dreamxcache/label/recipes.py.

The core regression case: two raw rows for the same SMILES can each fail a recipe's threshold
individually, yet the compound is a genuine positive once its reads are
summed (dedup-first is correct; threshold-first is not). Every recipe here
must get this right by construction, not by caller discipline.
"""

from __future__ import annotations

import pandas as pd

from dreamxcache.label.recipes import (
    strict_recipe,
    strict_recipe_threshold,
    wiki_recipe,
    wiki_recipe_count_ge_10_threshold,
    wiki_recipe_disynthon_competitive_hit,
    wiki_recipe_individually_evidenced,
    wiki_recipe_joint_zscore_threshold,
    wiki_recipe_no_competition_threshold,
    wiki_recipe_per_library_threshold,
    wiki_recipe_threshold,
    wiki_recipe_zscore_threshold,
)


def _row(compound, smiles, count_pgk2, inhibitor, ntc, hh):
    return {
        "compound": compound,
        "SMILES": smiles,
        "count_PGK2": count_pgk2,
        "count_PGK2_with_inhibitor": inhibitor,
        "count_NTC": ntc,
        "historic_hits": hh,
    }


def test_wiki_recipe_threshold_hand_checked():
    df = pd.DataFrame(
        [
            # passes: count>=3, inhibitor<0.1*3=0.3 -> 0 ok, NTC==0, hh<5
            _row("A-1-1-1", "s1", 3, 0, 0, 0),
            # fails: count=2 < 3
            _row("A-1-1-2", "s2", 2, 0, 0, 0),
            # fails: inhibitor=1 >= 0.1*10=1.0
            _row("A-1-1-3", "s3", 10, 1, 0, 0),
            # fails: NTC!=0
            _row("A-1-1-4", "s4", 5, 0, 1, 0),
            # fails: historic_hits>=5
            _row("A-1-1-5", "s5", 5, 0, 0, 5),
        ]
    )
    mask = wiki_recipe_threshold(df)
    assert mask.tolist() == [True, False, False, False, False]


def test_wiki_recipe_dedup_before_threshold_regression():
    # Two raw rows for the SAME SMILES, each with count_PGK2=2 (fails >=3
    # alone), but summed to 4 they pass. dedup-first must catch this;
    # threshold-first (the historical bug) would not.
    selection = pd.DataFrame(
        [
            _row("A-1-1-1", "SAME", 2, 0, 0, 0),
            _row("A-1-1-2", "SAME", 2, 0, 0, 0),
            _row("A-1-1-3", "OTHER_VALID_SMILES_XYZ", 1, 0, 0, 0),  # fails: count=1<3
        ]
    )
    # give SMILES enough length/carbon content to pass is_valid_smiles
    selection["SMILES"] = selection["SMILES"].replace(
        {"SAME": "CCCCCCCCCCCCCC", "OTHER_VALID_SMILES_XYZ": "CCCCCCCCCCCCCC1"}
    )

    result = wiki_recipe(selection)
    assert result.n_positives == 1
    assert result.positives["SMILES"].tolist() == ["CCCCCCCCCCCCCC"]
    assert result.positives["count_PGK2"].iloc[0] == 4  # 2 + 2, correctly summed
    assert len(result.deduped) == 2  # two unique SMILES after dedup


def test_strict_recipe_threshold_hand_checked_or_clauses():
    df = pd.DataFrame(
        [
            # passes: count>3(=10), inhibitor==0 (OR branch), NTC<0.1*10=1.0 (0 passes), hh<5
            _row("A-1-1-1", "s1", 10, 0, 0, 0),
            # passes: count>3(=10), inhibitor=0.5<0.1*10=1.0 (second OR branch), NTC==0, hh<5
            _row("A-1-1-2", "s2", 10, 0.5, 0, 0),
            # fails: count=3, not >3
            _row("A-1-1-3", "s3", 3, 0, 0, 0),
            # fails: inhibitor=2, neither ==0 nor <1.0 (0.1*10)
            _row("A-1-1-4", "s4", 10, 2, 0, 0),
            # fails: NTC=2, neither ==0 nor <1.0
            _row("A-1-1-5", "s5", 10, 0, 2, 0),
        ]
    )
    mask = strict_recipe_threshold(df)
    assert mask.tolist() == [True, True, False, False, False]


def test_strict_recipe_dedup_before_threshold_regression():
    # Same structure as the wiki-recipe regression: two raw rows summing to
    # clear the strict recipe's count>3 threshold only after dedup.
    selection = pd.DataFrame(
        [
            _row("B-1-1-1", "CCCCCCCCCCCCCC", 2, 0, 0, 0),
            _row("B-1-1-2", "CCCCCCCCCCCCCC", 2, 0, 0, 0),  # summed count_PGK2 = 4 > 3
        ]
    )
    result = strict_recipe(selection)
    assert result.n_positives == 1
    assert result.positives["count_PGK2"].iloc[0] == 4


def test_wiki_recipe_individually_evidenced_l2_hand_checked():
    # "SUMMED": two raw rows count_PGK2=2 each -> passes wiki_recipe only
    # after summing (2+2=4>=3); no single row individually clears >=3.
    # "SOLO": one raw row count_PGK2=5 -> passes wiki_recipe AND
    # individually (5>=3 on its own).
    selection = pd.DataFrame(
        [
            _row("A-1-1-1", "CCCCCCCCCCCCCC", 2, 0, 0, 0),
            _row("A-1-1-2", "CCCCCCCCCCCCCC", 2, 0, 0, 0),
            _row("A-1-1-3", "CCCCCCCCCCCCCC1", 5, 0, 0, 0),
        ]
    )
    wiki_result = wiki_recipe(selection)
    assert wiki_result.n_positives == 2  # both SUMMED and SOLO pass wiki_recipe

    l2_result = wiki_recipe_individually_evidenced(selection)
    assert l2_result.n_positives == 1  # only SOLO survives L2
    assert l2_result.positives["SMILES"].tolist() == ["CCCCCCCCCCCCCC1"]


def test_wiki_recipe_count_ge_10_threshold_hand_checked():
    df = pd.DataFrame(
        [
            _row("A-1-1-1", "s1", 10, 0, 0, 0),  # passes: count>=10
            _row("A-1-1-2", "s2", 9, 0, 0, 0),  # fails: count=9<10 (would pass wiki_recipe's >=3)
        ]
    )
    mask = wiki_recipe_count_ge_10_threshold(df)
    assert mask.tolist() == [True, False]


def test_wiki_recipe_no_competition_threshold_hand_checked():
    df = pd.DataFrame(
        [
            # passes: count>=3, NTC==0, hh<5 -- inhibitor NOT checked (huge inhibitor still passes)
            _row("A-1-1-1", "s1", 10, 999, 0, 0),
            _row("A-1-1-2", "s2", 2, 0, 0, 0),  # fails: count=2<3
        ]
    )
    mask = wiki_recipe_no_competition_threshold(df)
    assert mask.tolist() == [True, False]


def test_wiki_recipe_zscore_threshold_hand_checked():
    df = pd.DataFrame(
        {
            "zscore_PGK2": [0.5, 0.05, 0.5],
            "count_PGK2": [10, 10, 10],
            "count_PGK2_with_inhibitor": [0, 0, 5],  # row 2 fails: 5 >= 0.1*10=1.0
            "count_NTC": [0, 0, 0],
            "historic_hits": [0, 0, 0],
        }
    )
    mask = wiki_recipe_zscore_threshold(df, z_cutoff=0.3)
    # row0: z=0.5>=0.3, other criteria ok -> pass
    # row1: z=0.05<0.3 -> fail
    # row2: z=0.5>=0.3 but inhibitor fails -> fail
    assert mask.tolist() == [True, False, False]


def test_wiki_recipe_joint_zscore_threshold_hand_checked():
    df = pd.DataFrame(
        {
            "count_PGK2": [10, 10, 10, 2, 10],
            "count_PGK2_with_inhibitor": [0, 5, 0, 0, 0],  # row1 fails: 5 >= 0.1*10=1.0
            "historic_hits": [0, 0, 10, 0, 0],  # row2 fails: promiscuous
            "zscore_PGK2": [0.5, 0.5, 0.5, 0.5, 0.5],
            "count_NTC": [0, 0, 0, 0, 99],  # row4: high NTC, deliberately NOT a criterion here
        }
    )
    mask = wiki_recipe_joint_zscore_threshold(df, z_cutoff=0.3)
    # row0: count=10>=3, inhibitor ok, hh ok, z=0.5>=0.3 -> pass
    # row1: inhibitor fails (5 >= 1.0) -> fail
    # row2: historic_hits fails (10 >= 5) -> fail
    # row3: count=2<3 -> fail
    # row4: NTC=99 but NTC is deliberately NOT part of this recipe -> pass anyway
    assert mask.tolist() == [True, False, False, False, True]


def test_wiki_recipe_per_library_threshold_hand_checked():
    df = pd.DataFrame(
        {
            "library": ["libA", "libA", "libB", "libC"],
            "count_PGK2": [5, 2, 5, 100],
            "count_PGK2_with_inhibitor": [0, 0, 0, 0],
            "count_NTC": [0, 0, 0, 0],
            "historic_hits": [0, 0, 0, 0],
        }
    )
    thresholds = {"libA": 3, "libB": 10}  # libC deliberately unmapped
    mask = wiki_recipe_per_library_threshold(df, thresholds)
    # libA row0: count=5>=3 -> pass; libA row1: count=2<3 -> fail
    # libB row2: count=5<10 -> fail
    # libC row3: unmapped -> excluded regardless of count
    assert mask.tolist() == [True, False, False, False]


def test_wiki_recipe_disynthon_competitive_hit_pools_across_shared_disynthon():
    # compound1 and compound2 share disynthon AB:lib1:1:2 (bb1=1, bb2=2);
    # individually each has count_PGK2=10 (below threshold=15), but pooled
    # they sum to 20 (above) -- both should become positive via the shared
    # disynthon, the denoising effect this recipe exists to capture.
    # compound3 shares no disynthon with either -> stays negative.
    selection = pd.DataFrame(
        [
            _row("lib1-1-2-3", "COMPOUND001", 10, 0, 0, 0),
            _row("lib1-1-2-4", "COMPOUND002", 10, 0, 0, 0),
            _row("lib1-5-6-7", "COMPOUND003", 1, 0, 0, 0),
        ]
    )
    result = wiki_recipe_disynthon_competitive_hit(selection, target_threshold=15)
    mask_by_compound = dict(zip(result.deduped["compound"], result.positive_mask, strict=False))
    assert mask_by_compound["lib1-1-2-3"]
    assert mask_by_compound["lib1-1-2-4"]
    assert not mask_by_compound["lib1-5-6-7"]
    assert result.n_positives == 2
