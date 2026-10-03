"""Extracting a bound ligand from a PDB file as a chemically correct molecule.

A PDB file carries coordinates but no bond orders, so the ligand's chemistry
has to be *perceived*. Two traps produce silently wrong results:

1. **Several copies or alternate conformations.** Extracting every ``HETATM``
   record of a residue name at once lets a single distance-based bonding pass
   create spurious bonds between copies (multiple chains) or between altLoc
   duplicates. Isolate one chain, and one altLoc plus atoms with no altLoc.
2. **Bond-order perception can be wrong without looking wrong.** OpenBabel's
   ``PerceiveBondOrders`` can assign a fully aromatic ring as a non-aromatic
   dihydro form with a spurious stereocentre. The molecule parses, the SMILES
   is valid, and it passes an eyeball check. The reliable test is geometric:
   a ring atom assigned sp3 whose own crystallographic coordinates are
   planar, with short bonds to its neighbours, is not sp3.
   `check_ring_geometry_consistency` automates that test.

Extraction method, per ligand:

- explicit hydrogens in the record -> ``rdDetermineBonds`` (geometry-based,
  valence is disambiguated by the hydrogens), bounded by a timeout because the
  combinatorial search can hang on ambiguous input;
- no explicit hydrogens -> OpenBabel ``PerceiveBondOrders``
  (``rdDetermineBonds`` is underdetermined without hydrogens and returns
  chemically absurd structures).

Either way, run `check_ring_geometry_consistency` on the result.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

DETERMINE_BONDS_TIMEOUT_SECONDS = 90

# A tetrahedral centre has bond-angle sum ~328 deg; a planar sp2 centre has
# ~360 deg. Anything above this is planar.
PLANAR_ANGLE_SUM_DEG = 355.0
# sp3 C-N / C-O single bonds are ~1.43-1.47 A; aromatic or amide-like C-N/C-O
# are ~1.32-1.38 A.
SHORT_SINGLE_BOND_ANGSTROM = 1.40


class DetermineBondsTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: object) -> None:
    raise DetermineBondsTimeout()


def extract_ligand_pdb_block(pdb_path: Path, resname: str, chain: str, altloc: str = "A") -> str:
    """The ``HETATM`` records of one ligand copy: residue name `resname`, chain
    `chain`, atoms with altLoc `altloc` or with no altLoc."""
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM") or line[17:20].strip() != resname or line[21] != chain:
                continue
            if line[16] not in (" ", altloc):
                continue
            lines.append(line)
    if not lines:
        raise RuntimeError(f"No matching HETATM lines in {pdb_path} (resname={resname}, chain={chain})")
    return "".join(lines) + "END\n"


def mol_from_block_geometry(block: str, timeout_seconds: int = DETERMINE_BONDS_TIMEOUT_SECONDS) -> Chem.Mol:
    """Heavy-atom molecule with its original 3D conformer, bonds from
    ``rdDetermineBonds``. Only reliable when the block includes explicit
    hydrogens. Uses ``SIGALRM`` for the timeout, so it is Unix-only and must
    run in the main thread."""
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True)
    if mol is None:
        raise RuntimeError("MolFromPDBBlock failed to parse the ligand block")
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_seconds)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    finally:
        signal.alarm(0)
    Chem.SanitizeMol(mol)
    return mol


def mol_from_block_openbabel(block: str) -> Chem.Mol:
    """Heavy-atom molecule with its original 3D conformer, bond orders from
    OpenBabel, round-tripped through an MDL molblock so atom order and bonds
    stay consistent with the coordinates."""
    from openbabel import pybel  # type: ignore[import-untyped]

    ob_mol = pybel.readstring("pdb", block)
    ob_mol.removeh()
    molblock = ob_mol.write("mol")
    mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=True)
    if mol is None:
        raise RuntimeError("Failed to round-trip the ligand through OpenBabel and RDKit")
    return mol


def bound_ligand_mol(
    pdb_path: Path, resname: str, chain: str, altloc: str = "A", has_explicit_h: bool = False
) -> Chem.Mol:
    """The bound ligand as an RDKit molecule with its crystallographic 3D
    conformer. `has_explicit_h` selects the perception method (see the module
    docstring)."""
    block = extract_ligand_pdb_block(pdb_path, resname, chain, altloc)
    return mol_from_block_geometry(block) if has_explicit_h else mol_from_block_openbabel(block)


@dataclass(frozen=True)
class RingGeometryFlag:
    atom_index: int
    angle_sum_deg: float
    shortest_c_heteroatom_bond: float | None

    @property
    def has_short_heteroatom_bond(self) -> bool:
        """A C-N/C-O bond shorter than an sp3 single bond: independent
        evidence of partial double-bond (aromatic) character."""
        return (
            self.shortest_c_heteroatom_bond is not None and self.shortest_c_heteroatom_bond < SHORT_SINGLE_BOND_ANGSTROM
        )

    def message(self, name: str) -> str:
        bond = (
            f", shortest C-N/C-O bond {self.shortest_c_heteroatom_bond:.2f} A"
            f"{' (short: aromatic-like)' if self.has_short_heteroatom_bond else ''}"
            if self.shortest_c_heteroatom_bond is not None
            else ""
        )
        return (
            f"[{name}] ring atom {self.atom_index} is assigned sp3 (non-aromatic, no double bond) but its "
            f"substituent-angle sum is {self.angle_sum_deg:.1f} deg (>{PLANAR_ANGLE_SUM_DEG:.0f} = planar){bond}: "
            "possible mis-assigned aromaticity, verify by hand."
        )


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def check_ring_geometry_consistency(mol: Chem.Mol) -> list[RingGeometryFlag]:
    """Flag ring carbons that the perceived chemistry calls sp3 but whose own
    crystallographic geometry is planar.

    A flagged atom is a non-aromatic ring carbon with no double bond and three
    heavy-atom neighbours (so one implicit hydrogen, i.e. a stereocentre as
    assigned) whose three bond angles sum to more than 355 degrees. Atoms that
    carry a double bond (e.g. lactam carbonyl carbons) are sp2 by assignment
    and consistent with planar geometry, so they are not flagged. `mol` must
    carry a 3D conformer in the same atom order as its bonds. Returns flags;
    it does not raise, because a flag means "verify", not "wrong".
    """
    if mol is None or mol.GetNumConformers() == 0:
        return []
    conf = mol.GetConformer()
    flags: list[RingGeometryFlag] = []
    for atom in mol.GetAtoms():  # type: ignore[no-untyped-call]
        if atom.GetSymbol() != "C" or not atom.IsInRing() or atom.GetIsAromatic():
            continue
        if any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 3:
            continue
        pos_c = np.array(conf.GetAtomPosition(atom.GetIdx()))
        vecs = [np.array(conf.GetAtomPosition(n.GetIdx())) - pos_c for n in neighbors]
        angle_sum = _angle(vecs[0], vecs[1]) + _angle(vecs[1], vecs[2]) + _angle(vecs[0], vecs[2])
        if angle_sum <= PLANAR_ANGLE_SUM_DEG:
            continue
        hetero = [float(np.linalg.norm(v)) for n, v in zip(neighbors, vecs, strict=True) if n.GetSymbol() in ("N", "O")]
        flags.append(
            RingGeometryFlag(
                atom_index=atom.GetIdx(),
                angle_sum_deg=angle_sum,
                shortest_c_heteroatom_bond=min(hetero) if hetero else None,
            )
        )
    return flags
