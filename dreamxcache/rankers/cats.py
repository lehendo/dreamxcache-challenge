"""CATS-like pharmacophore-pair fingerprints.

Neither RDKit nor scikit-fingerprints implements CATS (Reutlinger et al. 2013)
literally. This uses scikit-fingerprints' ``PharmacophoreFingerprint``
(``variant="folded"``, ``max_points=2``, ``count=True``): the same underlying
idea, pharmacophoric point *pairs* typed by topological distance, but not
CATS's exact type/distance scheme. ``max_points=2`` (pairs, as CATS uses, not
triples) is also much faster: three-point pharmacophores are by far the
slowest fingerprint in the library.

The fingerprint is a count vector, compared with the generalized Tanimoto in
``rankers.similarity``.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


def compute_cats_fingerprints(smiles: list[str], n_jobs: int = 1) -> tuple[np.ndarray, int]:
    """Folded 2-point pharmacophore count vectors, shape ``(n, 2048)``.
    Unparseable SMILES get an all-zero row and are counted in ``n_invalid``."""
    from skfp.fingerprints import PharmacophoreFingerprint  # type: ignore[import-untyped]

    mols_or_none = [Chem.MolFromSmiles(s) for s in smiles]
    valid_idx = [i for i, m in enumerate(mols_or_none) if m is not None]
    n_invalid = len(smiles) - len(valid_idx)
    fp = PharmacophoreFingerprint(variant="folded", max_points=2, fp_size=2048, count=True, n_jobs=n_jobs)
    valid_fps = fp.transform([mols_or_none[i] for i in valid_idx]).astype(np.float32)

    out = np.zeros((len(smiles), valid_fps.shape[1]), dtype=np.float32)
    out[valid_idx] = valid_fps
    return out, n_invalid
