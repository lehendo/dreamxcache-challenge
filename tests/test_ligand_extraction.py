import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from dreamxcache.structure.ligand_extraction import (
    PLANAR_ANGLE_SUM_DEG,
    check_ring_geometry_consistency,
    extract_ligand_pdb_block,
)


def _hetatm(serial, name, resname, chain, altloc, x):
    return (
        f"HETATM{serial:>5} {name:<4}{altloc}{resname:>3} {chain}   1    "
        f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00 20.00           C\n"
    )


def test_extract_isolates_one_chain_and_one_altloc(tmp_path):
    lines = [
        _hetatm(1, "C1", "LIG", "A", " ", 0.0),  # no altloc: kept
        _hetatm(2, "C2", "LIG", "A", "A", 1.0),  # altloc A: kept
        _hetatm(3, "C2", "LIG", "A", "B", 1.1),  # altloc B: dropped
        _hetatm(4, "C1", "LIG", "B", " ", 5.0),  # other chain: dropped
        _hetatm(5, "C1", "XYZ", "A", " ", 9.0),  # other residue: dropped
    ]
    pdb = tmp_path / "x.pdb"
    pdb.write_text("".join(lines))
    block = extract_ligand_pdb_block(pdb, "LIG", "A", "A")
    assert block.count("HETATM") == 2
    assert block.endswith("END\n")


def test_extract_raises_when_nothing_matches(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(_hetatm(1, "C1", "LIG", "A", " ", 0.0))
    with pytest.raises(RuntimeError, match="No matching HETATM"):
        extract_ligand_pdb_block(pdb, "NOPE", "A")


def _pyrrolidine_with_geometry(planar: bool, bond_length: float = 1.45) -> Chem.Mol:
    """2-methylpyrrolidine; ring atom C2 (index 4 of "C1CNC(C)C1": atoms are
    0:C 1:C 2:N 3:C(ring, 3 heavy neighbours) 4:C(methyl) 5:C) given either a
    planar or a tetrahedral arrangement of its three heavy neighbours."""
    mol = Chem.MolFromSmiles("C1CNC(C)C1")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(i) * 3.0, 10.0, 10.0))  # far apart, irrelevant
    centre = 3
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()]
    conf.SetAtomPosition(centre, Point3D(0.0, 0.0, 0.0))
    r = bond_length
    if planar:
        angles = np.radians([0.0, 120.0, 240.0])  # in-plane, 120 deg apart: angle sum 360
        for n, a in zip(nbrs, angles, strict=True):
            conf.SetAtomPosition(n, Point3D(r * float(np.cos(a)), r * float(np.sin(a)), 0.0))
    else:
        t = 1.0 / np.sqrt(3.0)
        for n, (sx, sy, sz) in zip(nbrs, [(1, 1, 1), (1, -1, -1), (-1, 1, -1)], strict=True):
            conf.SetAtomPosition(n, Point3D(r * t * sx, r * t * sy, r * t * sz))  # tetrahedral: angle sum ~328
    mol.AddConformer(conf)
    return mol


def test_planar_sp3_assigned_ring_carbon_is_flagged():
    flags = check_ring_geometry_consistency(_pyrrolidine_with_geometry(planar=True))
    assert [f.atom_index for f in flags] == [3]
    assert flags[0].angle_sum_deg > PLANAR_ANGLE_SUM_DEG
    assert not flags[0].has_short_heteroatom_bond  # 1.45 A is an ordinary sp3 C-N length


def test_short_cn_bond_adds_independent_evidence_of_aromatic_character():
    flags = check_ring_geometry_consistency(_pyrrolidine_with_geometry(planar=True, bond_length=1.34))
    assert len(flags) == 1
    assert flags[0].has_short_heteroatom_bond  # 1.34 A: aromatic-like, not an sp3 single bond
    assert "short" in flags[0].message("demo")


def test_tetrahedral_sp3_assigned_ring_carbon_is_not_flagged():
    assert check_ring_geometry_consistency(_pyrrolidine_with_geometry(planar=False)) == []


def test_aromatic_and_double_bonded_ring_atoms_are_never_flagged():
    mol = Chem.MolFromSmiles("c1ccccc1")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(6):
        a = np.radians(60.0 * i)
        conf.SetAtomPosition(i, Point3D(1.4 * float(np.cos(a)), 1.4 * float(np.sin(a)), 0.0))
    mol.AddConformer(conf)
    assert check_ring_geometry_consistency(mol) == []

    lactam = Chem.MolFromSmiles("O=C1CCCN1")  # carbonyl carbon: planar by assignment (has a C=O)
    lconf = Chem.Conformer(lactam.GetNumAtoms())
    for i in range(lactam.GetNumAtoms()):
        lconf.SetAtomPosition(i, Point3D(float(i), float(i % 2), 0.0))
    lactam.AddConformer(lconf)
    assert check_ring_geometry_consistency(lactam) == []


def test_mol_without_a_conformer_returns_no_flags():
    assert check_ring_geometry_consistency(Chem.MolFromSmiles("C1CCNC1")) == []
