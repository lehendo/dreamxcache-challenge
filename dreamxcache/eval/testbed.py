"""Known-answer testbeds: score a channel on a library where the true hits are
known, and read the result as a percentile of its own permutation null.

A testbed library is a table with ``SMILES`` and a 0/1 ``LABEL`` column (plus an
id column). The point is to find out, on a target the channel was never tuned
against, whether it enriches hits at all, *before* trusting it inside a fusion.
Two cautions this repository learned the hard way:

* A negative control can be a poisoned testbed. An anchor ligand that is
  *anti*-correlated with the true hits (``anchor_similarity_test.py``) makes any
  anchor-based ranking fail by construction, so a bad result tells you nothing
  about the method. Check the anchor against the hits before reading the control.
* Report percentiles against the empirical null, and trust LogAUC and EF@1% over
  EF@50 at low hit counts (``eval/null_calibration.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from dreamxcache.eval.early_recognition import enrichment_factor, enrichment_factor_pct, log_auc
from dreamxcache.eval.null_calibration import empirical_null, percentiles_vs_null, summarize_null


def load_labeled_library(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for col in ("SMILES", "LABEL"):
        if col not in df.columns:
            raise ValueError(f"{path} must have a {col!r} column")
    if int(df["LABEL"].sum()) == 0:
        raise ValueError(f"{path} has no hits (LABEL == 1)")
    return df


def anchor_smiles_from(smiles: str | None, pdb_spec: str | None) -> str:
    """An anchor's SMILES, given directly or extracted from a PDB ligand
    (``file:resname:chain[:altloc]``, no explicit hydrogens assumed; see
    ``structure/ligand_extraction.py`` for the extraction and its pitfalls)."""
    if smiles:
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"anchor SMILES does not parse: {smiles!r}")
        return smiles
    if not pdb_spec:
        raise ValueError("give an anchor SMILES or a PDB ligand spec")
    from dreamxcache.structure.ligand_extraction import bound_ligand_mol

    parts = pdb_spec.split(":")
    mol = bound_ligand_mol(Path(parts[0]), parts[1], parts[2], parts[3] if len(parts) > 3 else "A", False)
    return str(Chem.MolToSmiles(mol))


def evaluate_channel(
    scores: np.ndarray, labels: np.ndarray, n_permutations: int = 1000, seed: int = 42
) -> dict[str, Any]:
    """EF@50, EF@1%, LogAUC of `scores` against `labels`, each with its
    percentile against that library's own permutation null (NaN scores are
    excluded, i.e. only scored compounds enter the metric and the null)."""
    mask = np.isfinite(scores)
    s, y = scores[mask], labels[mask]
    observed = {
        "EF@50": enrichment_factor(s, y, top_k=50),
        "EF@1%": enrichment_factor_pct(s, y, pct=0.01),
        "LogAUC": log_auc(s, y),
    }
    null = empirical_null(y, n_permutations, seed)
    return {
        "n_scored": int(mask.sum()),
        "n_hits_scored": int(y.sum()),
        "metrics": percentiles_vs_null(observed, null),
        "null_summary": {m: summarize_null(d) for m, d in null.items()},
    }
