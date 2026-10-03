"""Binary ECFP4 (Morgan, radius 2, 2048 bits), computed with the official
extraction code from ``aircheck_utils`` (``HitGenBinaryECFP4``) unchanged.

Using the organizers' own extraction code means any gap against a published
baseline is attributable to labeling or model choice rather than to a
fingerprint implementation difference. The code is not redistributed here; see
``scripts/third_party.sh``.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
from rdkit import RDLogger

from dreamxcache.third_party import add_to_sys_path, require_dir

# The extraction code calls GetMorganFingerprintAsBitVect once per molecule,
# which logs an RDKit deprecation notice every call. Harmless, but at millions
# of molecules it produces hundreds of MB of log output.
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

ECFP4_RADIUS = 2
ECFP4_N_BITS = 2048

_hitgen_cls: type | None = None


def _hitgen_binary_ecfp4() -> type:
    global _hitgen_cls
    if _hitgen_cls is None:
        # fingerprints.py does `from utils import ...` (flat, not package
        # qualified), so the directory itself must be on sys.path.
        add_to_sys_path(require_dir("aircheck_utils") / "FingerPrintExtraction")
        import fingerprints  # type: ignore[import-not-found]

        assert isinstance(fingerprints, types.ModuleType)
        _hitgen_cls = fingerprints.HitGenBinaryECFP4
    return _hitgen_cls


def compute_ecfp4(smiles: pd.Series | list[str], use_tqdm: bool = False) -> np.ndarray:
    """SMILES -> (n, 2048) array of binary ECFP4 bits. Invalid SMILES yield a
    row of NaN (the extraction code's own behaviour) rather than raising or
    being dropped, so callers must handle NaN rows themselves."""
    fp_func = _hitgen_binary_ecfp4()()
    result: np.ndarray = fp_func(list(smiles), use_tqdm=use_tqdm)
    return result
