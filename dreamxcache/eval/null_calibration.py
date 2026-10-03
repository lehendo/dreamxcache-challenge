"""Empirical permutation null for early-recognition metrics.

The closed-form LogAUC for a random ranking pins the null *mean* but says
nothing about its spread, and at a handful of known hits the spread is large.
EF@50 and EF@1% are worse: they are coarse and discrete (landing exactly one
hit in the top 50 by chance happens a few percent of the time at ~20 hits),
and the median random draw scores zero (with 19 hits among 15,419 compounds,
landing one hit in the top 50 by chance happens about 6% of the time).

The fix is to draw the null directly. For a given ``(n_total, n_hits)``,
score the compounds with pure noise `n_permutations` times, compute every
metric on each draw, and read any real result as a *percentile of its own
null*. Because the scores are exchangeable noise, the null depends only on
``n_total`` and ``n_hits``, not on which compounds are the hits, so it can be
regenerated without any data.

Percentile convention: the percentage of null draws less than or equal to the
observed value (100.0 = at or above every draw).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dreamxcache.eval.early_recognition import enrichment_factor, enrichment_factor_pct, log_auc

METRICS = ("EF@50", "EF@1%", "LogAUC")


def labels_from_counts(n_total: int, n_hits: int) -> np.ndarray:
    labels = np.zeros(n_total)
    labels[:n_hits] = 1
    return labels


def empirical_null(labels: np.ndarray, n_permutations: int = 1000, seed: int = 42) -> dict[str, np.ndarray]:
    """`n_permutations` noise rankings of `labels`' compounds; one array per
    metric."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    out = {m: np.empty(n_permutations) for m in METRICS}
    for i in range(n_permutations):
        noise = rng.random(n)
        out["EF@50"][i] = enrichment_factor(noise, labels, top_k=50)
        out["EF@1%"][i] = enrichment_factor_pct(noise, labels, pct=0.01)
        out["LogAUC"][i] = log_auc(noise, labels)
    return out


def empirical_null_from_counts(
    n_total: int, n_hits: int, n_permutations: int = 1000, seed: int = 42
) -> dict[str, np.ndarray]:
    return empirical_null(labels_from_counts(n_total, n_hits), n_permutations, seed)


def summarize_null(dist: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(dist)),
        "p50": float(np.percentile(dist, 50)),
        "p95": float(np.percentile(dist, 95)),
        "p99": float(np.percentile(dist, 99)),
        "max": float(np.max(dist)),
    }


def percentile_of(value: float, dist: np.ndarray) -> float:
    """Percentage of null draws less than or equal to `value`."""
    return float(100.0 * np.mean(dist <= value))


def percentiles_vs_null(observed: dict[str, float], null: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Each observed metric value with its percentile against the matching
    null distribution."""
    return {m: {"value": v, "null_percentile": percentile_of(v, null[m])} for m, v in observed.items() if m in null}
