import numpy as np

from dreamxcache.eval.early_recognition import log_auc
from dreamxcache.eval.null_calibration import (
    empirical_null_from_counts,
    labels_from_counts,
    percentile_of,
    percentiles_vs_null,
    summarize_null,
)


def test_null_is_deterministic_given_seed() -> None:
    a = empirical_null_from_counts(500, 5, n_permutations=50, seed=7)
    b = empirical_null_from_counts(500, 5, n_permutations=50, seed=7)
    for metric in a:
        assert np.array_equal(a[metric], b[metric])


def test_null_logauc_mean_matches_the_closed_form() -> None:
    n, n_hits = 1500, 21
    null = empirical_null_from_counts(n, n_hits, n_permutations=300, seed=0)
    min_fpr = 1.0 / (n - n_hits)
    expected = (1.0 / np.log(10)) / (-np.log10(min_fpr))
    assert abs(null["LogAUC"].mean() - expected) < 0.01


def test_null_ef50_has_heavy_mass_at_zero_with_few_hits() -> None:
    # At ~0.1% hit rate the median random draw lands no hit in the top 50.
    null = empirical_null_from_counts(15_000, 19, n_permutations=300, seed=0)
    assert np.median(null["EF@50"]) == 0.0


def test_percentile_of_hand_checked() -> None:
    dist = np.array([0.0, 1.0, 2.0, 3.0])
    assert percentile_of(2.0, dist) == 75.0  # three of four draws are <= 2.0
    assert percentile_of(-1.0, dist) == 0.0
    assert percentile_of(10.0, dist) == 100.0


def test_a_perfect_ranking_sits_at_the_top_of_its_null() -> None:
    n, n_hits = 1000, 10
    labels = labels_from_counts(n, n_hits)
    perfect = np.zeros(n)
    perfect[:n_hits] = 1.0
    null = empirical_null_from_counts(n, n_hits, n_permutations=200, seed=0)
    result = percentiles_vs_null({"LogAUC": log_auc(perfect, labels)}, null)
    assert result["LogAUC"]["null_percentile"] == 100.0


def test_summarize_null_keys() -> None:
    summary = summarize_null(np.arange(101, dtype=float))
    assert summary["p50"] == 50.0 and summary["p95"] == 95.0 and summary["max"] == 100.0
