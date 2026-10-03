"""Early-recognition metrics: enrichment factor and LogAUC.

A ranking can have real *global* signal and no *early* recognition at all.
ROC-AUC integrates over the whole ranking, so a method that pushes most hits
somewhat above the median looks good on it even if none of them is near the
top, which is the only part of the ranking a 50-compound submission ever
sees. The signature is a decent ROC-AUC paired with a PR-AUC at the no-skill
level (Truchon and Bayly 2007). The metrics here are the ones to compare
channels with.

LogAUC follows Mysinger and Shoichet (2010): trapezoidal integration of TPR
against log10(FPR) from a floor FPR (default ``1/n_negatives``, which avoids
log(0)) to FPR = 1, normalized by the log-scale range. A *perfect* ranking
scores 1.0. A *random* ranking does **not** score near 1: for TPR = FPR the
expectation is

    (1 / ln 10) / log10(1 / min_fpr)

about 0.10 at 15,000 negatives and about 0.14 at 1,500. That closed form
fixes the null *mean* only, not its spread, which is why every result in this
repository is read against a permutation null
(``dreamxcache.eval.null_calibration``) rather than against the formula.
"""

from __future__ import annotations

import numpy as np


def enrichment_factor(scores: np.ndarray, labels: np.ndarray, top_k: int) -> float:
    """EF@k = (hit rate in the top k) / (hit rate over the whole set).
    `scores` is higher-is-better; `labels` is 0/1, same length and order."""
    order = np.argsort(-scores)
    top_labels = labels[order[:top_k]]
    hit_rate_top = top_labels.sum() / top_k
    hit_rate_overall = labels.sum() / len(labels)
    if hit_rate_overall == 0:
        return float("nan")
    return float(hit_rate_top / hit_rate_overall)


def enrichment_factor_pct(scores: np.ndarray, labels: np.ndarray, pct: float) -> float:
    """EF at the top `pct` fraction (e.g. ``pct=0.01`` for EF@1%)."""
    top_k = max(1, round(len(scores) * pct))
    return enrichment_factor(scores, labels, top_k)


def log_auc(scores: np.ndarray, labels: np.ndarray, min_fpr: float | None = None) -> float:
    """LogAUC (Mysinger and Shoichet 2010): area under TPR vs log10(FPR) from
    `min_fpr` (default ``1/n_negatives``) to FPR = 1, normalized by
    ``log10(1/min_fpr)``. A perfect ranking scores 1.0; a random ranking
    scores roughly ``(1/ln 10) / log10(1/min_fpr)`` (see the module docstring),
    not 1.0. Emphasizes the early part of the ranking (low FPR) the way
    ROC-AUC's linear FPR axis does not."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp_cumsum = np.cumsum(sorted_labels)
    fp_cumsum = np.cumsum(1 - sorted_labels)
    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    if min_fpr is None:
        min_fpr = 1.0 / n_neg

    log_fpr = np.log10(np.clip(fpr, min_fpr, 1.0))
    log_min_fpr = np.log10(min_fpr)

    valid = fpr >= min_fpr
    if not valid.any():
        return float("nan")
    x = log_fpr[valid]
    y = tpr[valid]
    if x[0] > log_min_fpr:
        x = np.concatenate([[log_min_fpr], x])
        y = np.concatenate([[0.0], y])

    area = float(np.trapezoid(y, x))
    normalization = -log_min_fpr  # log10(1) - log10(min_fpr)
    return area / normalization if normalization > 0 else float("nan")


def summarize(scores: np.ndarray, labels: np.ndarray, name: str = "") -> dict[str, float]:
    """The standard trio: EF@50 (the submission budget), EF@1%, LogAUC."""
    n = len(scores)
    result = {
        "n_total": float(n),
        "n_hits": float(labels.sum()),
        "EF@50": enrichment_factor(scores, labels, min(50, n)),
        "EF@1%": enrichment_factor_pct(scores, labels, 0.01),
        "LogAUC": log_auc(scores, labels),
    }
    if name:
        print(f"=== {name} ===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if k not in ("n_total", "n_hits") else f"  {k}: {int(v)}")
    return result
