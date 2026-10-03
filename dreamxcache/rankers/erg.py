"""ErG (extended reduced graph) fingerprints, Stiefl et al. 2006, via RDKit's
``rdReducedGraphs``.

ErG encodes pharmacophoric atom types (H-bond donor/acceptor, aromatic, ...)
over reduced-graph topological distances rather than exact atom and bond
environments, so it can match chemically dissimilar scaffolds that share a
pharmacophore. It is a real-valued, 315-dimensional vector, compared with the
generalized Tanimoto in ``rankers.similarity``.
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdReducedGraphs

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

ERG_DIM = 315


def _erg_chunk(args: tuple[int, list[str]]) -> tuple[int, np.ndarray, int]:
    start_offset, smiles_chunk = args
    out = np.zeros((len(smiles_chunk), ERG_DIM), dtype=np.float32)
    n_invalid = 0
    for i, smi in enumerate(smiles_chunk):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        out[i] = np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)
    return start_offset, out, n_invalid


def compute_erg_fingerprints(smiles: list[str], n_workers: int = 1, chunk_size: int = 5000) -> tuple[np.ndarray, int]:
    """ErG vectors for `smiles`, shape ``(n, 315)``. Unparseable SMILES get an
    all-zero row (which scores 0 similarity to everything) and are counted in
    the returned ``n_invalid``."""
    n = len(smiles)
    out = np.zeros((n, ERG_DIM), dtype=np.float32)
    chunks = [(i, smiles[i : i + chunk_size]) for i in range(0, n, chunk_size)]
    total_invalid = 0
    if n_workers <= 1:
        results = map(_erg_chunk, chunks)
        for start_offset, fps, n_invalid in results:
            out[start_offset : start_offset + len(fps)] = fps
            total_invalid += n_invalid
        return out, total_invalid
    with Pool(n_workers) as pool:
        for start_offset, fps, n_invalid in pool.imap_unordered(_erg_chunk, chunks):
            out[start_offset : start_offset + len(fps)] = fps
            total_invalid += n_invalid
    return out, total_invalid
