"""ECFP4 / FCFP4 Morgan fingerprints and max-Tanimoto-to-anchors ranking.

These are the control against the pharmacophore-style rankers: nearest-
neighbour Tanimoto similarity to the anchors in a plain circular-fingerprint
space, with no ML and no DEL data. FCFP4 (``useFeatures=True``) types atoms by
functional class (donor/acceptor/aromatic/...) instead of exact atomic number,
a step towards pharmacophore generalization without leaving the Morgan family.
Similarity is the standard *binary* Tanimoto, not the generalized variant used
for ErG and CATS.
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

RADIUS = 2
N_BITS = 2048


def _morgan_bits(smiles: str, use_features: bool) -> DataStructs.ExplicitBitVect | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp: DataStructs.ExplicitBitVect = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
        mol, RADIUS, nBits=N_BITS, useFeatures=use_features
    )
    return fp


def _fp_chunk(args: tuple[int, list[str], bool]) -> tuple[int, list[DataStructs.ExplicitBitVect | None]]:
    start_offset, smiles_chunk, use_features = args
    return start_offset, [_morgan_bits(s, use_features) for s in smiles_chunk]


def compute_morgan_fps(
    smiles: list[str], use_features: bool, n_workers: int = 1, chunk_size: int = 10000
) -> list[DataStructs.ExplicitBitVect | None]:
    """Morgan bit vectors; ``None`` for unparseable SMILES."""
    n = len(smiles)
    out: list[DataStructs.ExplicitBitVect | None] = [None] * n
    chunks = [(i, smiles[i : i + chunk_size], use_features) for i in range(0, n, chunk_size)]
    if n_workers <= 1:
        for chunk in chunks:
            start_offset, fps = _fp_chunk(chunk)
            out[start_offset : start_offset + len(fps)] = fps
        return out
    with Pool(n_workers) as pool:
        for start_offset, fps in pool.imap_unordered(_fp_chunk, chunks):
            out[start_offset : start_offset + len(fps)] = fps
    return out


def max_tanimoto_similarity(
    query_fps: list[DataStructs.ExplicitBitVect | None], anchor_fps: list[DataStructs.ExplicitBitVect]
) -> np.ndarray:
    """Maximum binary Tanimoto from each query to any anchor; 0 for queries
    whose SMILES did not parse."""
    max_sim = np.zeros(len(query_fps), dtype=np.float32)
    for i, fp in enumerate(query_fps):
        if fp is None:
            continue
        max_sim[i] = max(DataStructs.BulkTanimotoSimilarity(fp, anchor_fps))
    return max_sim
