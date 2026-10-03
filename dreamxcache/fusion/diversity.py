"""Diversity-maximizing selection of the final compounds.

The primary scored quantity is the number of *distinct chemical series* among
the selected hits, not their count. A model that concentrates its confidence on
one scaffold family returns one series by construction under naive top-N
selection, however good the underlying ranking. Selection should therefore
optimize series coverage directly.

- `greedy_diversity_select` (the one used by the fusion pipeline): repeatedly
  take the highest-scoring candidate whose proxy series is not yet
  represented; only when the pool's distinct series are exhausted may a series
  get a second pick, still in score order. After Swamidass et al. 2011
  (diversity-oriented prioritization).
- `murcko_maxmin_select`: best compound per Murcko scaffold among the top-N,
  then greedy MaxMin picking by ECFP4 Tanimoto distance, seeded with the
  top-scoring compound.
- `butina_select`: Butina-cluster the top-N at a Tanimoto cutoff and take the
  best compound per cluster.

Series labels for the fusion pipeline come from ``eval.proxy_clustering``,
applied only to the *candidate pool* (a few thousand compounds), never to a
whole library, where full pairwise clustering is infeasible and unnecessary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dreamxcache.eval.proxy_clustering import DEFAULT_TANIMOTO_DISTANCE_THRESHOLD, assign_proxy_clusters
from dreamxcache.fusion.rrf import combined_channel_score, reciprocal_rank_fusion


def greedy_diversity_select(
    candidate_idx: np.ndarray, scores: np.ndarray, series_labels: np.ndarray, n_select: int
) -> np.ndarray:
    """`candidate_idx` / `scores` are sorted by score descending and aligned;
    ``series_labels[i]`` is the series of ``candidate_idx[i]``. Pass 1 takes
    one pick per distinct series in score order; later passes allow repeats,
    always taking the next-highest-scoring unpicked candidate. Returns the
    selected entries of `candidate_idx`."""
    selected: list[int] = []
    picked_series: set[int] = set()
    n = len(candidate_idx)

    for i in range(n):
        if len(selected) >= n_select:
            break
        series = int(series_labels[i])
        if series not in picked_series:
            selected.append(i)
            picked_series.add(series)

    if len(selected) < n_select:
        selected_set = set(selected)
        for i in range(n):
            if len(selected) >= n_select:
                break
            if i not in selected_set:
                selected.append(i)
                selected_set.add(i)

    if len(selected) < n_select:
        raise RuntimeError(
            f"only {len(selected)} candidates available across all channels, need {n_select}; "
            "increase n_candidates_per_channel"
        )
    result: np.ndarray = candidate_idx[np.array(selected[:n_select])]
    return result


@dataclass(frozen=True)
class FusionResult:
    selected_idx: np.ndarray  # row indices into the input arrays, the n_select picks
    combined_score: np.ndarray  # full-length Score column for a submission
    n_candidates: int
    n_series_in_pool: int
    n_series_selected: int


def fuse_and_select(
    smiles: pd.Series,
    channel_scores: dict[str, np.ndarray],
    channel_weights: dict[str, float],
    n_select: int = 50,
    n_candidates_per_channel: int = 500,
    cluster_threshold: float = DEFAULT_TANIMOTO_DISTANCE_THRESHOLD,
) -> FusionResult:
    """Reciprocal rank fusion of the channels, proxy-series labeling of the
    candidate pool, then greedy per-series diversity selection."""
    for name, s in channel_scores.items():
        if len(s) != len(smiles):
            raise ValueError(f"channel {name!r}: {len(s)} scores != {len(smiles)} compounds")

    candidate_idx, rrf_scores = reciprocal_rank_fusion(channel_scores, channel_weights, n_candidates_per_channel)
    series_labels = assign_proxy_clusters(smiles.iloc[candidate_idx], threshold=cluster_threshold).to_numpy()
    selected_idx = greedy_diversity_select(candidate_idx, rrf_scores, series_labels, n_select)
    selected_series = {int(s) for s in series_labels[np.isin(candidate_idx, selected_idx)]}
    return FusionResult(
        selected_idx=selected_idx,
        combined_score=combined_channel_score(channel_scores),
        n_candidates=len(candidate_idx),
        n_series_in_pool=len(set(series_labels.tolist())),
        n_series_selected=len(selected_series),
    )


# --- Alternative selectors over a single score array -------------------------


def _tanimoto_distance_matrix(fps: np.ndarray) -> np.ndarray:
    fps_f = fps.astype(np.float32)
    popcount = fps_f.sum(axis=1)
    intersection = fps_f @ fps_f.T
    union = popcount[:, None] + popcount[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    dist: np.ndarray = 1.0 - similarity
    return dist


def butina_select(
    scores: np.ndarray, fps: np.ndarray, n_candidates: int, n_select: int, similarity_cutoff: float = 0.625
) -> np.ndarray:
    """Butina-cluster the top `n_candidates` by score at Tanimoto
    `similarity_cutoff`, take the best compound per cluster, order by that
    compound's score, return the top `n_select` (row indices into `scores`)."""
    from rdkit.ML.Cluster import Butina

    top_idx = np.argsort(-scores)[:n_candidates]
    top_scores = scores[top_idx]
    dist = _tanimoto_distance_matrix(fps[top_idx])
    n = len(top_idx)
    # Butina expects the flattened lower triangle in row-major (i=1..n-1, j<i) order.
    flat = np.concatenate([dist[i, :i] for i in range(1, n)]).tolist()
    clusters = Butina.ClusterData(flat, n, 1.0 - similarity_cutoff, isDistData=True)  # type: ignore[no-untyped-call]

    representatives = [int(np.array(c)[np.argmax(top_scores[np.array(c)])]) for c in clusters]
    representatives.sort(key=lambda local: -top_scores[local])
    if len(representatives) < n_select:
        raise RuntimeError(f"only {len(representatives)} clusters among the top {n_candidates}, need {n_select}")
    result: np.ndarray = top_idx[representatives[:n_select]]
    return result


def murcko_maxmin_select(
    scores: np.ndarray, fps: np.ndarray, smiles: list[str], n_candidates: int, n_select: int
) -> np.ndarray:
    """Best-scoring compound per unique Murcko scaffold among the top
    `n_candidates`, then greedy MaxMin picking (ECFP4 Tanimoto distance) over
    that scaffold-deduplicated pool, seeded with the top-scoring compound so
    the best bet is never discarded."""
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    top_idx = np.argsort(-scores)[:n_candidates]
    top_fps = fps[top_idx].astype(np.float32)
    top_scores = scores[top_idx]

    best_local_per_scaffold: dict[str, int] = {}
    for local_idx, global_idx in enumerate(top_idx):
        smi = smiles[global_idx]
        try:
            scaffold = MurckoScaffoldSmiles(smi) or smi  # type: ignore[no-untyped-call]
        except Exception:  # noqa: BLE001 -- unparseable SMILES is its own singleton "scaffold"
            scaffold = smi
        current = best_local_per_scaffold.get(scaffold)
        if current is None or top_scores[local_idx] > top_scores[current]:
            best_local_per_scaffold[scaffold] = local_idx

    candidate_local = np.array(sorted(best_local_per_scaffold.values(), key=lambda i: -top_scores[i]))
    if len(candidate_local) < n_select:
        raise RuntimeError(
            f"only {len(candidate_local)} unique Murcko scaffolds among the top {n_candidates}, need {n_select}"
        )

    cand_fps = top_fps[candidate_local]
    popcount = cand_fps.sum(axis=1)

    def distance_to(i: int) -> np.ndarray:
        intersection = cand_fps @ cand_fps[i]
        union = popcount + popcount[i] - intersection
        sim = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        d: np.ndarray = 1.0 - sim
        return d

    selected = [0]  # candidate_local is sorted by score, so 0 is the top scorer
    min_dist = distance_to(0)
    min_dist[0] = -1.0
    while len(selected) < n_select:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, distance_to(nxt))
        min_dist[nxt] = -1.0

    result: np.ndarray = top_idx[candidate_local[selected]]
    return result
