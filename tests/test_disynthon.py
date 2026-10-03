"""Hand-checked fixtures for dreamxcache/label/disynthon.py.

Core case to get right: two compounds that individually fall well under a
threshold can BOTH become positive once pooled, because they share a
disynthon whose AGGREGATED count clears the bar -- this is the whole point
of disynthon-level denoising (McCloskey et al. 2020).
"""

from __future__ import annotations

import pandas as pd

from dreamxcache.label.disynthon import (
    aggregate_disynthon_counts,
    classify_competitive_hit_disynthons,
    melt_to_disynthons,
    propagate_disynthon_hits_to_compounds,
)


def _compounds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "library": ["libA", "libA", "libA"],
            "bb1": ["1", "1", "5"],
            "bb2": ["2", "2", "6"],
            "bb3": ["3", "4", "7"],
            "count_PGK2": [10, 10, 1],
            "count_PGK2_with_inhibitor": [0, 0, 0],
            "count_NTC": [0, 0, 0],
        }
    )


def test_melt_to_disynthons_produces_3_rows_per_compound():
    compounds = _compounds()
    melted = melt_to_disynthons(compounds, sum_cols=["count_PGK2"])
    assert len(melted) == 9  # 3 compounds x 3 disynthon pairs each, BEFORE dedup of shared IDs
    # compound1 and compound2 share disynthon AB:libA:1:2 -> 8 unique IDs, not 9.
    assert set(melted["disynthon_id"]) == {
        "AB:libA:1:2",  # shared by compound1 and compound2
        "AC:libA:1:3",
        "BC:libA:2:3",
        "AC:libA:1:4",
        "BC:libA:2:4",
        "AB:libA:5:6",
        "AC:libA:5:7",
        "BC:libA:6:7",
    }


def test_aggregate_disynthon_counts_pools_shared_disynthon():
    compounds = _compounds()
    agg = aggregate_disynthon_counts(compounds, sum_cols=["count_PGK2"])
    by_id = agg.set_index("disynthon_id")["count_PGK2"]
    # compound1 (count=10) and compound2 (count=10) SHARE disynthon AB:libA:1:2
    # -> pooled to 20, well above either compound's own individual count.
    assert by_id["AB:libA:1:2"] == 20
    # non-shared disynthons keep their single compound's own count.
    assert by_id["AC:libA:1:3"] == 10
    assert by_id["AC:libA:1:4"] == 10
    assert by_id["AB:libA:5:6"] == 1


def test_classify_competitive_hit_disynthons_hand_checked():
    disynthon_agg = pd.DataFrame(
        {
            "disynthon_id": ["d_hit", "d_low_count", "d_inhibited", "d_ntc"],
            "count_PGK2": [20, 5, 20, 20],
            "count_PGK2_with_inhibitor": [0, 0, 5, 0],  # d_inhibited: 5 >= 0.1*20=2.0 -> fails
            "count_NTC": [0, 0, 0, 5],  # d_ntc: 5 >= 0.1*20=2.0 -> fails
        }
    )
    mask = classify_competitive_hit_disynthons(disynthon_agg, target_threshold=15)
    # d_hit: count=20>=15, no inhibitor/NTC signal -> pass
    # d_low_count: count=5<15 -> fail
    # d_inhibited: count=20>=15 but inhibitor ratio fails -> fail
    # d_ntc: count=20>=15 but NTC ratio fails -> fail
    assert mask.tolist() == [True, False, False, False]


def test_propagate_disynthon_hits_pulls_in_both_compounds_sharing_the_hit_disynthon():
    compounds = _compounds()
    # Only AB:libA:1:2 (the pooled, count=20 disynthon) is a competitive hit.
    positive_disynthon_ids = {"AB:libA:1:2"}
    mask = propagate_disynthon_hits_to_compounds(compounds, positive_disynthon_ids)
    # compound1 and compound2 BOTH contain disynthon AB:libA:1:2 -> both positive,
    # even though each individually has count_PGK2=10, well under a threshold of 15.
    # compound3 shares no disynthon with either -> negative.
    assert mask.tolist() == [True, True, False]


def test_propagate_disynthon_hits_empty_set_yields_no_positives():
    compounds = _compounds()
    mask = propagate_disynthon_hits_to_compounds(compounds, positive_disynthon_ids=set())
    assert mask.tolist() == [False, False, False]
