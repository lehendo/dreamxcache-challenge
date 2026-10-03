"""Reciprocal rank fusion of heterogeneous scoring channels.

Channels score on incomparable scales (co-folding affinity probabilities,
pharmacophore alignment scores, similarity, a classifier's decision value).
Ranks are comparable across them where raw scores are not, so channels are
combined by rank:

    rrf(c) = sum over channels of  weight / (RRF_K + rank_in_channel(c))

with ``RRF_K = 60`` (Cormack, Clarke and Buettcher 2009). A compound that is
not in a channel's own top-N contributes 0 for that channel: RRF is credit for
showing up near the top *somewhere*, not a penalty for every channel that
missed it.

Channels may contain NaN (compounds a channel never scored: a shortlist-only
channel, or a job that did not finish). NaN is excluded from that channel's
ranking *before* taking its top-N. NumPy sorts NaN last in ascending order, so
``np.argsort(-scores)`` ranks NaN compounds at the bottom, but a top-N slice
that runs past the channel's finite scores would still reach them and credit
compounds the channel never scored. (``np.argsort(scores)[::-1]``, the other
common way to sort descending, puts NaN *first*.)
"""

from __future__ import annotations

import numpy as np

RRF_K = 60


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1], NaN-aware: NaN stays NaN and does not corrupt
    the min and max (plain ``.min()``/``.max()`` propagate NaN)."""
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.full_like(scores, np.nan)
    lo, hi = finite.min(), finite.max()
    if hi <= lo:
        flat: np.ndarray = np.where(np.isfinite(scores), 0.0, np.nan)
        return flat
    scaled: np.ndarray = (scores - lo) / (hi - lo)
    return scaled


def reciprocal_rank_fusion(
    channel_scores: dict[str, np.ndarray],
    channel_weights: dict[str, float],
    n_candidates_per_channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(candidate_row_indices, rrf_scores)``, aligned and sorted by
    ``rrf_scores`` descending. Only each channel's top `n_candidates_per_channel`
    (among its non-NaN scores) can receive credit."""
    rrf: dict[int, float] = {}
    for name, scores in channel_scores.items():
        weight = channel_weights.get(name, 1.0)
        finite_idx = np.flatnonzero(np.isfinite(scores))
        if len(finite_idx) == 0:
            continue
        order = np.argsort(-scores[finite_idx])[:n_candidates_per_channel]
        for rank, idx in enumerate(finite_idx[order], start=1):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + weight / (RRF_K + rank)

    candidate_idx = np.array(sorted(rrf, key=lambda i: -rrf[i]), dtype=int)
    rrf_scores = np.array([rrf[i] for i in candidate_idx])
    return candidate_idx, rrf_scores


def combined_channel_score(channel_scores: dict[str, np.ndarray]) -> np.ndarray:
    """A full-length score for every compound: the max over channels of each
    channel's own min-max-normalized score. ``np.fmax`` ignores NaN per element
    rather than propagating it; a compound NaN in every channel gets 0.

    RRF is only defined over each channel's own top-N slice, so this is what
    populates the ``Score`` column of a submission that must score every
    compound. The 50 *selected* compounds are decided by RRF plus diversity
    selection, never by thresholding this score."""
    normalized = [min_max_normalize(s) for s in channel_scores.values()]
    combined: np.ndarray = np.nan_to_num(np.fmax.reduce(normalized), nan=0.0)
    return combined
