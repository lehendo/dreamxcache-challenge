"""Structure-based pharmacophore ranker.

A *reference ligand* (a bound ligand from a co-crystal structure) plus a set of
*reference points* define the pharmacophore. Each library compound is scored by

    score = O3A alignment score + sum over reference points of
            max(0, 1 - d / cutoff)

where:

1. the candidate is embedded in 3D (ETKDG, then MMFF) and aligned to the
   reference ligand with RDKit's O3A (shape plus MMFF-partial-charge-weighted
   chemical similarity);
2. in the *aligned* frame, for each reference point ``d`` is the distance to
   the candidate's closest feature of the matching type: an aromatic or
   hydrophobic feature for ``"aromatic"`` points, any H-bond donor or acceptor
   for ``"hbond"`` points (a candidate can H-bond in either direction); and
3. ``cutoff`` is per point (e.g. 2.5 A for direct contacts, 3.5 A for
   water-mediated contacts, which allows one extra hop).

The score therefore rewards both overall shape and chemistry match *and*
occupancy of specific, mechanistically important positions. Reference points
are plain data (``load_reference_points``): derive them from your own
structure, e.g. ring centroids and polar atoms of the reference ligand, bridging
waters from ``dreamxcache.structure.waters.find_bridging_waters``, or a key
residue's side-chain atom. A target without waters or a selectivity residue
simply has fewer point types; nothing in the scoring mechanism is
target-specific.

Reference-point JSON schema (coordinates in the reference structure's frame)::

    [{"name": "ring_centroid", "type": "aromatic", "xyz": [x, y, z], "cutoff": 2.5}, ...]

``examples/wdr91_8SHJ_reference_points.json`` is a worked example derived from
the public PDB entry 8SHJ.
"""

from __future__ import annotations

import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

from dreamxcache.structure.ligand_extraction import bound_ligand_mol

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

ReferencePoint = tuple[str, str, tuple[float, float, float], float]  # name, type, (x, y, z), cutoff in A


def load_reference_points(path: Path) -> list[ReferencePoint]:
    points: list[ReferencePoint] = []
    for entry in json.loads(path.read_text()):
        if entry["type"] not in ("aromatic", "hbond"):
            raise ValueError(f"reference point {entry['name']!r}: type must be 'aromatic' or 'hbond'")
        x, y, z = entry["xyz"]
        points.append((entry["name"], entry["type"], (float(x), float(y), float(z)), float(entry["cutoff"])))
    return points


def load_reference_mol(
    pdb_path: Path, resname: str, chain: str, altloc: str = "A", has_explicit_h: bool = False
) -> Chem.Mol:
    """The reference ligand's *original* 3D bound conformer, with hydrogens
    added and MMFF-optimized (O3A needs MMFF94 atom typing on both molecules).
    See ``dreamxcache.structure.ligand_extraction`` for how the chemistry is
    perceived and why `has_explicit_h` matters."""
    mol = bound_ligand_mol(pdb_path, resname, chain, altloc, has_explicit_h)
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)  # type: ignore[attr-defined]
    return mol


def _score_chunk(args: tuple[int, list[str], bytes, list[ReferencePoint]]) -> tuple[int, np.ndarray, int]:
    start_offset, smiles_chunk, ref_pickle, reference_points = args

    ref_mol = pickle.loads(ref_pickle)
    ref_props = AllChem.MMFFGetMoleculeProperties(ref_mol)  # type: ignore[attr-defined]
    feature_factory = AllChem.BuildFeatureFactory(  # type: ignore[attr-defined]
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )

    out = np.zeros(len(smiles_chunk), dtype=np.float32)
    n_failed = 0

    for i, smi in enumerate(smiles_chunk):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_failed += 1
                continue
            mol = Chem.AddHs(mol)
            cid = AllChem.EmbedMolecule(mol, randomSeed=0, useRandomCoords=True, maxAttempts=10)  # type: ignore[attr-defined]
            if cid < 0:
                n_failed += 1
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001, S110 -- a failed optimization still leaves a usable embedded conformer
                pass

            prb_props = AllChem.MMFFGetMoleculeProperties(mol)  # type: ignore[attr-defined]
            if prb_props is None or ref_props is None:
                n_failed += 1
                continue
            o3a = rdMolAlign.GetO3A(mol, ref_mol, prb_props, ref_props)
            o3a.Align()
            base_score = o3a.Score()

            feats = feature_factory.GetFeaturesForMol(mol)
            conf = mol.GetConformer()
            aromatic_positions = [
                np.array(conf.GetAtomPosition(idx))
                for f in feats
                if f.GetFamily() in ("Aromatic", "Hydrophobe")
                for idx in f.GetAtomIds()
            ]
            hbond_positions = [
                np.array(conf.GetAtomPosition(idx))
                for f in feats
                if f.GetFamily() in ("Donor", "Acceptor")
                for idx in f.GetAtomIds()
            ]

            bonus = 0.0
            for _, ptype, coord, cutoff in reference_points:
                candidates = aromatic_positions if ptype == "aromatic" else hbond_positions
                if not candidates:
                    continue
                d = min(float(np.linalg.norm(np.array(coord) - p)) for p in candidates)
                bonus += max(0.0, 1.0 - d / cutoff)

            out[i] = base_score + bonus
        except Exception:  # noqa: BLE001 -- any per-molecule failure scores 0 and never crashes the batch
            n_failed += 1
            continue

    return start_offset, out, n_failed


def compute_pharmacophore_scores(
    smiles: list[str],
    ref_mol: Chem.Mol,
    reference_points: list[ReferencePoint],
    n_workers: int = 1,
    chunk_size: int = 500,
) -> tuple[np.ndarray, int]:
    """Pharmacophore score per SMILES; compounds that fail to embed, align or
    parse score 0 and are counted in the returned ``n_failed``."""
    ref_pickle = pickle.dumps(ref_mol)
    n = len(smiles)
    out = np.zeros(n, dtype=np.float32)
    chunks = [(i, smiles[i : i + chunk_size], ref_pickle, reference_points) for i in range(0, n, chunk_size)]
    total_failed = 0
    if n_workers <= 1:
        for chunk in chunks:
            start_offset, scores, n_failed = _score_chunk(chunk)
            out[start_offset : start_offset + len(scores)] = scores
            total_failed += n_failed
        return out, total_failed
    n_done = 0
    with Pool(n_workers) as pool:
        for start_offset, scores, n_failed in pool.imap_unordered(_score_chunk, chunks):
            out[start_offset : start_offset + len(scores)] = scores
            total_failed += n_failed
            n_done += len(scores)
            if n_done % (chunk_size * 10) < chunk_size:
                print(f"  [progress] {n_done:,}/{n:,} processed")
    return out, total_failed
