"""Stratified negative sampling for DEL classifiers.

Negatives are drawn at a tunable ratio to the positives, a configurable
fraction (default 80%) from a "similar to the positives" stratum and the rest
from the remainder of the pool. Two stratification sources are supported,
distinguished by the ``binary_stratum`` flag of `sample_negatives`: a real
multi-cluster assignment (e.g. BitBIRCH) or a 2-valued in/out-of-stratum flag
(e.g. nearest-positive similarity, ``scripts/similarity_stratify.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NegativeSamplingResult:
    negative_indices: np.ndarray
    n_requested: int
    n_from_similar_requested: int
    n_from_similar_actual: int
    n_from_elsewhere_requested: int
    n_from_elsewhere_actual: int

    @property
    def shortfall(self) -> int:
        return self.n_requested - len(self.negative_indices)


def sample_negatives(
    cluster_id: np.ndarray,
    positive_mask: np.ndarray,
    ratio: float,
    fraction_from_similar: float = 0.8,
    seed: int = 0,
    binary_stratum: bool = False,
) -> NegativeSamplingResult:
    """Sample negative row indices (into the pool that `cluster_id` and
    `positive_mask` are aligned to) at ``ratio * n_positives``, split
    `fraction_from_similar` from a "similar to positives" pool and the rest
    from elsewhere. Sampling is without replacement within each pool. If a
    pool is smaller than its requested share, everything available is taken
    and the shortfall is reported on the result rather than silently absorbed.

    `binary_stratum` selects how "similar" is derived from `cluster_id`:

    - False (default): `cluster_id` is a real multi-cluster assignment.
      "Similar" is any cluster containing at least one of *this call's own*
      positives. Correct when clusters are compact and numerous.
    - True: `cluster_id` is a 2-valued flag (1 = in the similar stratum,
      0 = not). "Similar" is ``cluster_id == 1`` directly. Deriving
      "similar" from the caller's own positives with a 2-valued array is a
      bug, not just an imprecision: if the positive set includes even one row
      with ``cluster_id == 0`` (true whenever the positives are not a subset
      of the reference set the stratum was built from), the derived cluster
      set becomes {0, 1}, every row counts as "similar", the "elsewhere"
      allocation silently collapses to zero, and the "similar" negatives are
      drawn from the whole pool instead of the intended stratum. Use
      ``binary_stratum=True`` whenever `cluster_id` is a flag.
    """
    rng = np.random.default_rng(seed)
    n_positives = int(positive_mask.sum())
    n_total = round(ratio * n_positives)
    n_from_similar_requested = round(fraction_from_similar * n_total)
    n_from_elsewhere_requested = n_total - n_from_similar_requested

    if binary_stratum:
        is_similar_cluster = cluster_id.astype(bool)
    else:
        positive_clusters = set(cluster_id[positive_mask].tolist())
        is_similar_cluster = np.isin(cluster_id, list(positive_clusters))
    candidate_mask = ~positive_mask  # never sample a known positive as a negative

    similar_pool = np.where(candidate_mask & is_similar_cluster)[0]
    elsewhere_pool = np.where(candidate_mask & ~is_similar_cluster)[0]

    n_from_similar_actual = min(n_from_similar_requested, len(similar_pool))
    n_from_elsewhere_actual = min(n_from_elsewhere_requested, len(elsewhere_pool))

    from_similar = rng.choice(similar_pool, size=n_from_similar_actual, replace=False)
    from_elsewhere = rng.choice(elsewhere_pool, size=n_from_elsewhere_actual, replace=False)
    negative_indices = np.concatenate([from_similar, from_elsewhere])

    return NegativeSamplingResult(
        negative_indices=negative_indices,
        n_requested=n_total,
        n_from_similar_requested=n_from_similar_requested,
        n_from_similar_actual=n_from_similar_actual,
        n_from_elsewhere_requested=n_from_elsewhere_requested,
        n_from_elsewhere_actual=n_from_elsewhere_actual,
    )


def sample_random_negatives(positive_mask: np.ndarray, ratio: float, seed: int = 0) -> NegativeSamplingResult:
    """Uniform-random negative sampling with no similarity stratification: the
    control used by McCloskey et al. 2020 (arXiv:2002.02530). Reuses
    `NegativeSamplingResult` with the "similar" fields zeroed so it is a
    drop-in comparison against `sample_negatives`."""
    rng = np.random.default_rng(seed)
    n_positives = int(positive_mask.sum())
    n_total = round(ratio * n_positives)
    candidate_pool = np.where(~positive_mask)[0]
    n_actual = min(n_total, len(candidate_pool))
    negative_indices = rng.choice(candidate_pool, size=n_actual, replace=False)
    return NegativeSamplingResult(
        negative_indices=negative_indices,
        n_requested=n_total,
        n_from_similar_requested=0,
        n_from_similar_actual=0,
        n_from_elsewhere_requested=n_total,
        n_from_elsewhere_actual=n_actual,
    )
