"""USRCAT 3D shape + pharmacophore similarity (Schreyer and Blundell 2012).

USRCAT (Ultrafast Shape Recognition with CREDO Atom Types) is a 60-dimensional
vector of atom-type-specific shape moments: genuinely 3D, not a 2D
fingerprint. It is computed with scikit-fingerprints' ``USRCATFingerprint``.

Similarity is the standard USR-family similarity (Ballester and Richards
2007), ``1 / (1 + mean|a_i - b_i|)``, which lies in (0, 1] and is 1.0 for
identical vectors. It is *not* Tanimoto: these are real-valued moments, not
fingerprint counts.

Query conformers matter. When a co-crystal structure is available, compare
candidates against the real bound pose
(``dreamxcache.structure.ligand_extraction.bound_ligand_mol``), not a
re-embedded low-energy conformer. Without one, `embed_conformer` (ETKDG
embedding followed by MMFF optimization) is the standard fallback.

3D embedding is the expensive step (~10 molecules/s per core), so library
fingerprints are computed in chunked worker processes.
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

USRCAT_DIM = 60


def embed_conformer(smiles: str, seed: int = 0) -> Chem.Mol | None:
    """ETKDG-embedded, MMFF-optimized 3D conformer; ``None`` on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    cid = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True, maxAttempts=10)  # type: ignore[attr-defined]
    if cid < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001, S110 -- a failed optimization still leaves a usable embedded conformer
        pass
    return mol


def usr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """``1 / (1 + mean(|a_i - b_i|))``. Range (0, 1]; 1.0 for identical vectors."""
    return float(1.0 / (1.0 + np.mean(np.abs(a - b))))


def usrcat_vector(mol: Chem.Mol) -> np.ndarray:
    """USRCAT vector of a molecule that already carries a 3D conformer."""
    from skfp.fingerprints import USRCATFingerprint  # type: ignore[import-untyped]

    vec: np.ndarray = USRCATFingerprint().transform([mol])[0]
    return vec


def _usrcat_chunk(args: tuple[int, list[str]]) -> tuple[int, np.ndarray, np.ndarray, int]:
    from skfp.fingerprints import USRCATFingerprint

    start_offset, smiles_chunk = args
    fps = np.zeros((len(smiles_chunk), USRCAT_DIM), dtype=np.float64)
    valid = np.zeros(len(smiles_chunk), dtype=bool)
    fp_calc = USRCATFingerprint()
    for i, smi in enumerate(smiles_chunk):
        mol = embed_conformer(smi)
        if mol is None:
            continue
        try:
            fp = fp_calc.transform([mol])
        except Exception:  # noqa: BLE001, S112 -- USRCAT can fail on unusual topologies; skip, do not crash the batch
            continue
        fps[i] = fp[0]
        valid[i] = True
    return start_offset, fps, valid, int((~valid).sum())


def compute_usrcat_fingerprints(
    smiles: list[str], n_workers: int = 1, chunk_size: int = 500
) -> tuple[np.ndarray, np.ndarray, int]:
    """USRCAT vectors for `smiles`: ``(vectors, valid_mask, n_failed)``.
    Compounds that fail to embed or fingerprint have ``valid_mask == False``
    and an all-zero row."""
    n = len(smiles)
    out = np.zeros((n, USRCAT_DIM), dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)
    chunks = [(i, smiles[i : i + chunk_size]) for i in range(0, n, chunk_size)]
    total_invalid = 0
    if n_workers <= 1:
        results = map(_usrcat_chunk, chunks)
        for start_offset, fps, valid, n_invalid in results:
            out[start_offset : start_offset + len(fps)] = fps
            valid_mask[start_offset : start_offset + len(valid)] = valid
            total_invalid += n_invalid
        return out, valid_mask, total_invalid
    n_done = 0
    with Pool(n_workers) as pool:
        for start_offset, fps, valid, n_invalid in pool.imap_unordered(_usrcat_chunk, chunks):
            out[start_offset : start_offset + len(fps)] = fps
            valid_mask[start_offset : start_offset + len(valid)] = valid
            total_invalid += n_invalid
            n_done += len(fps)
            if n_done % (chunk_size * 20) < chunk_size:
                print(f"  [progress] {n_done:,}/{n:,} processed")
    return out, valid_mask, total_invalid


def max_usr_similarity(library_vectors: np.ndarray, valid_mask: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
    """Maximum USR similarity from each library compound to any query;
    0 for compounds with ``valid_mask == False``."""
    max_sim = np.zeros(len(library_vectors))
    for i in range(len(library_vectors)):
        if not valid_mask[i]:
            continue
        max_sim[i] = max(usr_similarity(library_vectors[i], q) for q in query_vectors)
    return max_sim
