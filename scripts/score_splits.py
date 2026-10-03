#!/usr/bin/env python3
"""Score the challenge validation and test splits with a trained model and write
submission-format files.

    python scripts/score_splits.py --model-path data/cache/lightgbm_baseline_l0.pkl \
        [--splits validation test] [--prefix Team_baseline] [--save-score-cache]

Validation: a 50-line ``.txt``. Test: a CSV ``CatalogID, Sel_50, Score``. Format
is validated locally (columns, exactly 50 flagged, every CatalogID present, no
missing scores): everything that can be checked without gold labels. Nothing is
submitted. The selection here is naive top-50-by-score; see
``scripts/fuse_channels.py`` for diversity-aware selection.

``--save-score-cache`` also stores the raw per-compound scores as
``<cache>/model_score_<split>.npy``, the channel format ``fuse_channels.py``
reads.
"""

from __future__ import annotations

import argparse
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dreamxcache.config import cache_dir, raw_dir, submissions_dir
from dreamxcache.models.lightgbm_baseline import score as lgb_score
from dreamxcache.provenance import write_report
from dreamxcache.submission import (
    N_SELECT,
    validate_test_format,
    validate_validation_format,
    write_test_csv,
    write_validation_txt,
)

ECFP4_N_BITS = 2048
SPLIT_FILES = {"validation": "PGK2_Validation_split.csv", "test": "PGK2_Test_split.csv"}


def _fingerprint_chunk(args: tuple[int, list[str]]) -> tuple[int, np.ndarray, int]:
    from dreamxcache.features.ecfp4 import compute_ecfp4

    start_offset, smiles_chunk = args
    fps = compute_ecfp4(smiles_chunk, use_tqdm=False)
    return start_offset, np.nan_to_num(fps, nan=0.0).astype(np.int8), int(np.isnan(fps).any(axis=1).sum())


def fingerprint_split(smiles: list[str], n_workers: int, chunk_size: int = 5000) -> tuple[np.ndarray, int]:
    out = np.zeros((len(smiles), ECFP4_N_BITS), dtype=np.int8)
    chunks = [(i, smiles[i : i + chunk_size]) for i in range(0, len(smiles), chunk_size)]
    total_invalid = 0
    with Pool(n_workers) as pool:
        for start_offset, fps, n_invalid in pool.imap_unordered(_fingerprint_chunk, chunks):
            out[start_offset : start_offset + len(fps)] = fps
            total_invalid += n_invalid
    return out, total_invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=cache_dir() / "lightgbm_baseline_l0.pkl")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=list(SPLIT_FILES))
    parser.add_argument("--prefix", default="baseline", help="output filename prefix")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--save-score-cache", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    report: dict[str, object] = {"model_file": args.model_path.name}

    for split in args.splits:
        template = pd.read_csv(raw_dir() / "Val-Test-set" / SPLIT_FILES[split])
        print(f"--- {split}: {len(template):,} compounds ---")
        fps, n_invalid = fingerprint_split(template["SMILES"].tolist(), n_workers=args.n_workers)
        scores = lgb_score(model, fps)
        if args.save_score_cache:
            np.save(cache_dir() / f"model_score_{split}.npy", scores)

        top_idx = np.argsort(-scores)[:N_SELECT]
        if split == "validation":
            selected = template.loc[top_idx, "CatalogID"].tolist()
            out = submissions_dir() / f"{args.prefix}_validation.txt"
            problems = write_validation_txt(selected, out) + validate_validation_format(selected, template)
        else:
            out = submissions_dir() / f"{args.prefix}_test.csv"
            write_test_csv(template["CatalogID"], top_idx, scores, out)
            problems = validate_test_format(pd.read_csv(out), template)
        print(f"  wrote {out.name}: {'VALID' if not problems else problems}")
        report[split] = {
            "n_compounds": len(template),
            "n_invalid_smiles": n_invalid,
            "output": out.name,
            "format_problems": problems,
        }

    report["elapsed_seconds"] = time.time() - t0
    write_report("score-splits", report)


if __name__ == "__main__":
    main()
