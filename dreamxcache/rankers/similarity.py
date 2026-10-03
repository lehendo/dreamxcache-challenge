"""Similarity primitives shared by the anchor-based rankers, and the anchor
table loader.

An *anchor* is a known ligand (a co-crystal ligand, a literature active, ...).
Anchor-based rankers score each library compound by its maximum similarity to
any anchor. Anchors are user-supplied: a CSV with a ``smiles`` column and
optional ``source`` and ``detail`` columns used for filtering. See DATA.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def continuous_tanimoto_max_similarity(
    query_fps: np.ndarray, anchor_fps: np.ndarray, batch_size: int = 5000
) -> np.ndarray:
    """Maximum generalized-Tanimoto similarity from each query row to any
    anchor row.

    Generalized Tanimoto, ``dot(a, b) / (|a|^2 + |b|^2 - dot(a, b))``, is the
    standard extension of Tanimoto to non-negative real-valued or count
    vectors such as ErG or pharmacophore-pair counts. Batched over queries to
    bound the ``batch_size x n_anchors`` intermediate matrix.
    """
    n_queries = query_fps.shape[0]
    anchor_norm_sq = np.sum(anchor_fps**2, axis=1)  # (n_anchors,)
    max_sim = np.zeros(n_queries, dtype=np.float32)
    for start in range(0, n_queries, batch_size):
        batch = query_fps[start : start + batch_size]
        batch_norm_sq = np.sum(batch**2, axis=1, keepdims=True)  # (batch, 1)
        dot = batch @ anchor_fps.T  # (batch, n_anchors)
        denom = batch_norm_sq + anchor_norm_sq[None, :] - dot
        sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        max_sim[start : start + batch.shape[0]] = sim.max(axis=1)
    return max_sim


def load_anchors(path: Path, sources: list[str] | None = None, details: list[str] | None = None) -> pd.DataFrame:
    """Read an anchor table (columns ``smiles``; optional ``source``,
    ``detail``) and optionally filter on exact ``source`` values, then on exact
    ``detail`` values. Raises if a requested filter value is absent, so a typo
    never silently yields an empty or wrong anchor set."""
    anchor_df = pd.read_csv(path)
    if "smiles" not in anchor_df.columns:
        raise ValueError(f"{path} must have a 'smiles' column")
    if sources is not None:
        if "source" not in anchor_df.columns:
            raise ValueError(f"{path} has no 'source' column to filter on")
        missing = set(sources) - set(anchor_df["source"].unique())
        if missing:
            raise ValueError(f"sources not present in the anchor table: {missing}")
        anchor_df = anchor_df[anchor_df["source"].isin(sources)]
    if details is not None:
        if "detail" not in anchor_df.columns:
            raise ValueError(f"{path} has no 'detail' column to filter on")
        missing_d = set(details) - set(anchor_df["detail"].unique())
        if missing_d:
            raise ValueError(f"detail values not present (after the source filter): {missing_d}")
        anchor_df = anchor_df[anchor_df["detail"].isin(details)]
    return anchor_df
