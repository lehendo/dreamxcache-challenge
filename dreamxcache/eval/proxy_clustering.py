"""ECFP4 Tanimoto agglomerative clustering (average linkage) as a *proxy* for
chemical-series assignment.

The number of distinct chemical series among the selected hits is a primary
scoring quantity in the challenge, but the official series assignment is
MCS-based and needs every true hit across the held-out splits, so it cannot be
reproduced locally. This proxy is therefore used for two things only: ranking
selection strategies against each other, and labeling the series structure of
a candidate pool during diversity selection. It must never be reported as the
official series count.

The default threshold, 0.32 Tanimoto *distance* (1 - similarity), is the
silhouette optimum for average-linkage ECFP4 clustering of confirmed hits from
the prior challenge round.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

_LinkageMethod = Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]

PROXY_LABELING_VERSION = "proxy_ecfp4_avg_linkage_v1"
DEFAULT_TANIMOTO_DISTANCE_THRESHOLD = 0.32  # 1 - Tanimoto similarity
ECFP4_RADIUS = 2
ECFP4_N_BITS = 2048


def _ecfp4(smiles: str) -> DataStructs.ExplicitBitVect | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined,no-any-return]
        mol, ECFP4_RADIUS, nBits=ECFP4_N_BITS
    )


def assign_proxy_clusters(
    hit_smiles: pd.Series,
    threshold: float = DEFAULT_TANIMOTO_DISTANCE_THRESHOLD,
    linkage_method: _LinkageMethod = "average",
) -> pd.Series:
    """Assign a proxy cluster id to each entry of `hit_smiles` (index preserved).

    Unparseable SMILES each get their own singleton cluster id rather than
    being dropped or silently merged, so callers never lose a row.
    """
    smiles_list = list(hit_smiles)
    fps = [_ecfp4(s) for s in smiles_list]
    n = len(fps)
    labels = np.zeros(n, dtype=int)

    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_idx) >= 2:
        valid_fps = [fps[i] for i in valid_idx]
        m = len(valid_fps)
        dist = np.zeros((m, m))
        for i in range(m):
            sims = np.array(DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps))  # type: ignore[arg-type]
            dist[i, :] = 1.0 - sims
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0  # guard against float asymmetry before squareform
        condensed = squareform(dist, checks=False)
        z = linkage(condensed, method=linkage_method)
        valid_labels = fcluster(z, t=threshold, criterion="distance")
        for local_i, global_i in enumerate(valid_idx):
            labels[global_i] = valid_labels[local_i]
    elif len(valid_idx) == 1:
        labels[valid_idx[0]] = 1

    next_id = int(labels[valid_idx].max()) + 1 if valid_idx else 1
    valid_set = set(valid_idx)
    for i in range(n):
        if i not in valid_set:
            labels[i] = next_id
            next_id += 1

    return pd.Series(labels, index=hit_smiles.index, name="Cluster_proxy")
