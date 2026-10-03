#!/usr/bin/env python3
"""Write Boltz-2 input YAMLs (one per compound), split them into small chunk
directories, and refuse to proceed on a corrupt MSA.

    python scripts/boltz2_generate_yamls.py \
        --compounds shortlist.csv --id-col CatalogID --smiles-col SMILES \
        --sequence-file target.txt --msa target.a3m \
        --out-dir data/boltz2/yamls --chunks-dir data/boltz2/chunks --n-chunks 16

The MSA is computed once per target (for example with ``boltz predict
--use_msa_server`` on a machine with internet access, then copying the ``.a3m``)
and referenced by every YAML. **Every MSA is checked for NUL bytes first**: a
single trailing NUL makes Boltz skip every input and still exit 0
(``dreamxcache.rankers.boltz2``). Pass ``--fix-msa`` to strip them in place.
Use ``.csv`` or ``.parquet`` for `--compounds`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dreamxcache.rankers.boltz2 import NulByteInMsaError, check_a3m, split_into_chunks, strip_nul_bytes, write_yamls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compounds", type=Path, required=True)
    parser.add_argument("--id-col", default="CatalogID")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--sequence-file", type=Path, required=True, help="plain text, one line, the protein sequence")
    parser.add_argument("--msa", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, default=None)
    parser.add_argument("--n-chunks", type=int, default=16)
    parser.add_argument("--fix-msa", action="store_true", help="strip NUL bytes from the MSA in place before using it")
    args = parser.parse_args()

    if args.fix_msa:
        n = strip_nul_bytes(args.msa)
        print(f"removed {n} NUL byte(s) from {args.msa}")
    try:
        check_a3m(args.msa)
    except NulByteInMsaError as e:
        raise SystemExit(str(e)) from e

    compounds = pd.read_parquet(args.compounds) if args.compounds.suffix == ".parquet" else pd.read_csv(args.compounds)
    sequence = args.sequence_file.read_text().strip()
    print(f"{len(compounds)} compounds, sequence length {len(sequence)}")

    n_written, skipped = write_yamls(compounds, args.id_col, args.smiles_col, sequence, args.msa, args.out_dir)
    print(f"wrote {n_written} YAMLs to {args.out_dir}; skipped {len(skipped)}")
    if skipped:
        skipped_path = args.out_dir.parent / f"{args.out_dir.name}_skipped.csv"
        pd.DataFrame(skipped, columns=[args.id_col, "reason"]).to_csv(skipped_path, index=False)
        print(f"skipped compounds listed in {skipped_path}")

    if args.chunks_dir is not None:
        dirs = split_into_chunks(args.out_dir, args.chunks_dir, args.n_chunks)
        print(f"split into {len(dirs)} chunks under {args.chunks_dir}")


if __name__ == "__main__":
    main()
