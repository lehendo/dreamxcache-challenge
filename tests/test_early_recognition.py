import numpy as np

from dreamxcache.eval.early_recognition import enrichment_factor, enrichment_factor_pct, log_auc


def _labels(n: int, n_hits: int) -> np.ndarray:
    labels = np.zeros(n)
    labels[:n_hits] = 1
    return labels


def test_enrichment_factor_perfect_ranking_hits_theoretical_max() -> None:
    n, n_hits, k = 1000, 10, 50
    labels = _labels(n, n_hits)
    scores = np.zeros(n)
    scores[:n_hits] = 1.0  # hits score highest, exactly filling the first n_hits < k slots
    ef = enrichment_factor(scores, labels, k)
    assert ef == (n_hits / k) / (n_hits / n)


def test_enrichment_factor_worst_ranking_is_zero() -> None:
    n, n_hits, k = 1000, 10, 50
    labels = _labels(n, n_hits)
    scores = np.arange(n).astype(float)
    scores[:n_hits] = -1e9  # hits score lowest
    assert enrichment_factor(scores, labels, k) == 0.0


def test_enrichment_factor_pct_matches_enrichment_factor_at_equivalent_k() -> None:
    n, n_hits = 1000, 10
    labels = _labels(n, n_hits)
    scores = np.random.default_rng(0).random(n)
    assert enrichment_factor_pct(scores, labels, 0.05) == enrichment_factor(scores, labels, 50)


def test_log_auc_perfect_ranking_is_one() -> None:
    n, n_hits = 1000, 10
    labels = _labels(n, n_hits)
    scores = np.zeros(n)
    scores[:n_hits] = 1.0
    assert log_auc(scores, labels) == 1.0


def test_log_auc_worst_ranking_is_zero() -> None:
    n, n_hits = 1000, 10
    labels = _labels(n, n_hits)
    scores = np.arange(n).astype(float)
    scores[:n_hits] = -1e9
    assert log_auc(scores, labels) == 0.0


def test_log_auc_orders_perfect_above_random_above_worst() -> None:
    n, n_hits = 2000, 20
    labels = _labels(n, n_hits)
    rng = np.random.default_rng(0)

    perfect = np.zeros(n)
    perfect[:n_hits] = 1.0
    random_scores = rng.random(n)
    worst = np.arange(n).astype(float)
    worst[:n_hits] = -1e9

    assert log_auc(perfect, labels) > log_auc(random_scores, labels) > log_auc(worst, labels)


def test_log_auc_random_ranking_matches_theoretical_expectation() -> None:
    # For TPR == FPR (a random ranking) the closed-form expectation is
    # (1/ln(10)) / log10(1/min_fpr), i.e. NOT ~1.0.
    n, n_hits = 100_000, 50
    labels = _labels(n, n_hits)
    diagonal_scores = -np.arange(n).astype(float)
    labels_shuffled = np.random.default_rng(1).permutation(labels)
    result = log_auc(diagonal_scores, labels_shuffled)
    min_fpr = 1.0 / (n - n_hits)
    expected = (1.0 / np.log(10)) / (-np.log10(min_fpr))
    assert abs(result - expected) < 0.05  # sampling noise from one random label permutation
    assert result < 0.2  # a random ranking is nowhere near a perfect 1.0
