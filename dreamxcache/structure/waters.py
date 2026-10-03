"""Locating bridging waters in a protein-ligand crystal structure.

A *bridging* water is one that touches both sides: within a distance cutoff of
a ligand polar atom (N or O) **and** of a protein polar atom (N or O). A water
that contacts only one side is not bridging. These positions are natural
reference points for a water-mediated pharmacophore (``rankers.pharmacophore``):
a candidate that places an H-bond partner near them can reproduce the
water-mediated contacts of the reference ligand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD"}
_POLAR_ELEMENTS = {"N", "O"}


def _element(line: str) -> str:
    element = line[76:78].strip()
    if element:
        return element.upper()
    name = line[12:16].strip()
    return name[:1].upper() if name else ""


def find_bridging_waters(
    pdb_path: Path,
    ligand_resname: str,
    ligand_chain: str,
    protein_chain: str | None = None,
    cutoff_angstrom: float = 3.5,
) -> list[tuple[float, float, float]]:
    """Coordinates of waters within `cutoff_angstrom` of both a polar atom of
    the ligand (`ligand_resname`, `ligand_chain`) and a polar protein atom
    (``ATOM`` records of `protein_chain`, default the ligand's chain)."""
    protein_chain = protein_chain or ligand_chain
    ligand_polar: list[np.ndarray] = []
    protein_polar: list[np.ndarray] = []
    waters: list[np.ndarray] = []

    with open(pdb_path) as f:
        for line in f:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            resname = line[17:20].strip()
            chain = line[21]
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            element = _element(line)
            if record == "HETATM" and resname in _WATER_RESNAMES:
                if element == "O" and chain == ligand_chain:
                    waters.append(xyz)
            elif record == "HETATM" and resname == ligand_resname and chain == ligand_chain:
                if element in _POLAR_ELEMENTS:
                    ligand_polar.append(xyz)
            elif record == "ATOM" and chain == protein_chain and element in _POLAR_ELEMENTS:
                protein_polar.append(xyz)

    if not (waters and ligand_polar and protein_polar):
        return []
    lig = np.array(ligand_polar)
    prot = np.array(protein_polar)
    bridging = []
    for w in waters:
        near_ligand = float(np.min(np.linalg.norm(lig - w, axis=1))) <= cutoff_angstrom
        near_protein = float(np.min(np.linalg.norm(prot - w, axis=1))) <= cutoff_angstrom
        if near_ligand and near_protein:
            bridging.append((float(w[0]), float(w[1]), float(w[2])))
    return bridging
