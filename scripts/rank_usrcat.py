#!/usr/bin/env python3
"""Rank a library by USRCAT 3D shape + pharmacophore similarity to query ligands.

    python scripts/rank_usrcat.py --query library.csv --name usrcat_test \
        [--bound-ligand structure.pdb:LIG:A[:altloc[:h]]] ... \
        [--anchors anchors.csv]

Queries are any mix of
  * ``--bound-ligand file:resname:chain[:altloc][:h]``: the *real bound conformer*
    from a co-crystal structure (append ``:h`` if the record has explicit
    hydrogens; see ``structure/ligand_extraction.py``), and
  * ``--anchors``: SMILES, embedded with ETKDG then MMFF-optimized (the fallback
    when no bound structure exists).
Score = the maximum USR similarity to any query (``rankers/usrcat.py``).
Library 3D embedding is the slow step (~10 molecules/s per core).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dreamxcache.rankers.cli import add_common_args, load_query, save_ranking
from dreamxcache.rankers.similarity import load_anchors
from dreamxcache.rankers.usrcat import compute_usrcat_fingerprints, embed_conformer, max_usr_similarity, usrcat_vector
from dreamxcache.structure.ligand_extraction import bound_ligand_mol, check_ring_geometry_consistency


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--bound-ligand", action="append", default=[], metavar="FILE:RESNAME:CHAIN[:ALTLOC[:h]]")
    parser.add_argument("--anchors", type=Path, default=None)
    args = parser.parse_args()
    if not args.bound_ligand and args.anchors is None:
        parser.error("give at least one --bound-ligand or --anchors")

    query_vectors, query_names = [], []
    for spec in args.bound_ligand:
        parts = spec.split(":")
        path, resname, chain = Path(parts[0]), parts[1], parts[2]
        altloc = parts[3] if len(parts) > 3 and parts[3] else "A"
        has_h = len(parts) > 4 and parts[4] == "h"
        mol = bound_ligand_mol(path, resname, chain, altloc, has_h)
        for flag in check_ring_geometry_consistency(mol):
            print("  WARNING " + flag.message(f"{path.name}:{resname}"))
        query_vectors.append(usrcat_vector(mol))
        query_names.append(f"{path.name}:{resname}:{chain}")
    if args.anchors is not None:
        for smi in load_anchors(args.anchors)["smiles"].unique().tolist():
            mol = embed_conformer(smi)
            if mol is None:
                print(f"  {smi}: failed to embed, skipping")
                continue
            query_vectors.append(usrcat_vector(mol))
            query_names.append(smi)
    print(f"{len(query_vectors)} query conformers")

    query = load_query(args)
    vectors, valid, n_failed = compute_usrcat_fingerprints(query[args.smiles_col].tolist(), n_workers=args.n_workers)
    scores = max_usr_similarity(vectors, valid, np.array(query_vectors))
    save_ranking(args, query, scores, "rank-usrcat", {"queries": query_names, "n_failed_to_embed": n_failed})


if __name__ == "__main__":
    main()
