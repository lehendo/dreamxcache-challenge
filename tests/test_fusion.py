import numpy as np
import pandas as pd

from dreamxcache.fusion.diversity import fuse_and_select, greedy_diversity_select
from dreamxcache.fusion.rrf import RRF_K, combined_channel_score, min_max_normalize, reciprocal_rank_fusion


def test_min_max_normalize_ignores_nan_and_preserves_it():
    out = min_max_normalize(np.array([1.0, np.nan, 3.0, 2.0]))
    assert np.isnan(out[1])
    assert out[0] == 0.0 and out[2] == 1.0 and out[3] == 0.5


def test_min_max_normalize_all_nan_and_constant():
    assert np.isnan(min_max_normalize(np.array([np.nan, np.nan]))).all()
    flat = min_max_normalize(np.array([2.0, 2.0, np.nan]))
    assert flat[0] == 0.0 and flat[1] == 0.0 and np.isnan(flat[2])


def test_rrf_hand_checked_two_channels():
    # channel a ranks idx 2,0,1; channel b ranks idx 0,2,1. weights 1 and 2.
    a = np.array([0.5, 0.1, 0.9])
    b = np.array([0.9, 0.1, 0.5])
    idx, scores = reciprocal_rank_fusion({"a": a, "b": b}, {"a": 1.0, "b": 2.0}, n_candidates_per_channel=3)
    expected = {
        2: 1.0 / (RRF_K + 1) + 2.0 / (RRF_K + 2),
        0: 1.0 / (RRF_K + 2) + 2.0 / (RRF_K + 1),
        1: 1.0 / (RRF_K + 3) + 2.0 / (RRF_K + 3),
    }
    assert all(abs(s - expected[int(i)]) < 1e-12 for i, s in zip(idx, scores, strict=True))
    assert list(idx) == sorted(expected, key=lambda k: -expected[k])


def test_rrf_never_credits_nan_even_when_top_n_exceeds_finite_scores():
    # Only two finite scores but top-N = 4: the slice must not run into the NaN compounds.
    scores = np.array([np.nan, 0.2, np.nan, 0.9, np.nan])
    idx, _ = reciprocal_rank_fusion({"c": scores}, {"c": 1.0}, n_candidates_per_channel=4)
    assert sorted(idx.tolist()) == [1, 3]


def test_rrf_channel_with_no_finite_scores_is_skipped():
    idx, _ = reciprocal_rank_fusion(
        {"empty": np.full(3, np.nan), "ok": np.array([0.1, 0.3, 0.2])}, {"empty": 5.0, "ok": 1.0}, 3
    )
    assert list(idx) == [1, 2, 0]


def test_combined_channel_score_uses_fmax_and_zero_for_all_nan():
    a = np.array([0.0, 1.0, np.nan, np.nan])
    b = np.array([1.0, np.nan, 5.0, np.nan])
    combined = combined_channel_score({"a": a, "b": b})
    # a normalized: [0, 1, nan, nan]; b normalized: [0, nan, 1, nan] -> fmax = [0, 1, 1, nan->0]
    assert combined.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_greedy_diversity_takes_one_per_series_first_then_repeats_in_score_order():
    candidate_idx = np.array([10, 11, 12, 13, 14])  # already sorted by score, descending
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    series = np.array([0, 0, 1, 1, 2])
    picked = greedy_diversity_select(candidate_idx, scores, series, n_select=4)
    # pass 1: 10 (series 0), 12 (series 1), 14 (series 2); pass 2: next best unpicked = 11
    assert picked.tolist() == [10, 12, 14, 11]


def test_greedy_diversity_raises_when_too_few_candidates():
    import pytest

    with pytest.raises(RuntimeError, match="need 3"):
        greedy_diversity_select(np.array([0, 1]), np.array([2.0, 1.0]), np.array([0, 1]), n_select=3)


def test_fuse_and_select_spans_distinct_series():
    # Two near-duplicate scaffold families at the top of one channel, plus distinct singletons.
    smiles = pd.Series(
        [
            "CCCCCCCCCCCC",
            "CCCCCCCCCCCCC",  # family A (near-identical)
            "c1ccccc1CCCCCC",
            "c1ccccc1CCCCCCC",  # family B
            "OC1CCCCC1N",
            "BrC1=CC=CN=C1",
        ]
    )
    channel = {"c": np.array([0.99, 0.98, 0.97, 0.96, 0.5, 0.4])}
    result = fuse_and_select(smiles, channel, {"c": 1.0}, n_select=4, n_candidates_per_channel=6)
    assert len(result.selected_idx) == 4
    assert result.n_series_selected >= 3  # greedy-by-series cannot pick all four from two families
