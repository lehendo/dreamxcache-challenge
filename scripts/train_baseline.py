#!/usr/bin/env python3
"""Train the L0 baseline: stratified negative sampling (2x positives, 80% from
the similar stratum, 20% elsewhere) plus LightGBM over ECFP4.

    python scripts/train_baseline.py [--cluster-id-path data/cache/similarity_cluster_id_full.npy] [--skip-cv]

Requires ``compute_pool_fingerprints.py`` and one stratification source
(``similarity_stratify.py``, or ``cluster_pool.py`` for BitBIRCH).

Internal cross-validation is grouped on the (library, bb1, bb2, bb3) tuple
parsed from the compound ID. Random CV is close to useless for DEL data,
because near-duplicate building-block combinations would leak across folds.
Grouped CV is the best held-out signal available without labels for the
challenge splits.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]

from dreamxcache.config import cache_dir
from dreamxcache.ingest.load import load_selection, parse_compound_id
from dreamxcache.label.recipes import wiki_recipe
from dreamxcache.models.lightgbm_baseline import score as lgb_score
from dreamxcache.models.lightgbm_baseline import train as lgb_train
from dreamxcache.negatives.sampling import sample_negatives
from dreamxcache.provenance import write_report

NEGATIVE_RATIO = 2.0
FRACTION_FROM_SIMILAR = 0.8
N_CV_FOLDS = 5
SEED = 0
COLS = ["compound", "SMILES", "count_PGK2", "count_PGK2_with_inhibitor", "count_NTC", "historic_hits"]


def load_pool_with_labels(cluster_id_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """``(pool_df with a bb_group column, cluster_id, positive_mask)``, all
    aligned to the cached fingerprint array's row order. L0 is recomputed
    rather than trusting a stale cache, and the SMILES order is verified
    against the cached pool, so misalignment is a checked invariant rather than
    a silent assumption."""
    cached_pool = pd.read_parquet(cache_dir() / "ecfp4_assayed_pool_full_smiles.parquet")
    wiki_result = wiki_recipe(load_selection(columns=COLS))
    if not (cached_pool["SMILES"].to_numpy() == wiki_result.deduped["SMILES"].to_numpy()).all():
        raise RuntimeError(
            "cached fingerprint pool's SMILES order does not match a fresh L0 deduplication; "
            "fingerprints and labels would silently misalign. Recompute the fingerprints and strata."
        )
    pool_df = wiki_result.deduped.reset_index(drop=True)
    bb = parse_compound_id(pool_df["compound"])
    if bb["library"].isna().any():
        raise RuntimeError("some compound IDs failed to parse")
    bb_group = bb["library"] + "-" + bb["bb1"] + "-" + bb["bb2"] + "-" + bb["bb3"]
    cluster_id = np.load(cluster_id_path)
    if (cluster_id == -1).any():
        raise RuntimeError(f"{(cluster_id == -1).sum()} pool rows have no cluster assignment")
    return pool_df.assign(bb_group=bb_group.to_numpy()), cluster_id, wiki_result.positive_mask.to_numpy()


def run_grouped_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=N_CV_FOLDS).split(x, y, groups)):
        t0 = time.time()
        model = lgb_train(x[train_idx], y[train_idx], x[test_idx], y[test_idx])
        preds = lgb_score(model, x[test_idx])
        rocauc, prauc = roc_auc_score(y[test_idx], preds), average_precision_score(y[test_idx], preds)
        fold_metrics.append({"fold": fold, "n_test": len(test_idx), "rocauc": rocauc, "prauc": prauc})
        print(f"  fold {fold + 1}/{N_CV_FOLDS} in {time.time() - t0:.1f}s: ROC-AUC={rocauc:.4f} PR-AUC={prauc:.4f}")
    return {
        "folds": fold_metrics,
        "mean_rocauc": float(np.mean([f["rocauc"] for f in fold_metrics])),
        "mean_prauc": float(np.mean([f["prauc"] for f in fold_metrics])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--cluster-id-path", type=Path, default=cache_dir() / "similarity_cluster_id_full.npy")
    parser.add_argument("--out-model", type=Path, default=cache_dir() / "lightgbm_baseline_l0.pkl")
    args = parser.parse_args()

    t0 = time.time()
    pool_df, cluster_id, positive_mask = load_pool_with_labels(args.cluster_id_path)
    n_positives = int(positive_mask.sum())
    print(f"L0 positives: {n_positives:,} / {len(pool_df):,} pool compounds")

    # The strata are 2-valued flags, so binary_stratum=True is required (see sample_negatives).
    neg = sample_negatives(
        cluster_id,
        positive_mask,
        ratio=NEGATIVE_RATIO,
        fraction_from_similar=FRACTION_FROM_SIMILAR,
        seed=SEED,
        binary_stratum=True,
    )
    print(
        f"negatives: requested {neg.n_requested:,}, got {len(neg.negative_indices):,} "
        f"(similar {neg.n_from_similar_actual:,}/{neg.n_from_similar_requested:,}, "
        f"elsewhere {neg.n_from_elsewhere_actual:,}/{neg.n_from_elsewhere_requested:,}, shortfall {neg.shortfall})"
    )

    positive_indices = np.where(positive_mask)[0]
    train_indices = np.concatenate([positive_indices, neg.negative_indices])
    y = np.concatenate([np.ones(len(positive_indices)), np.zeros(len(neg.negative_indices))])
    groups = pool_df["bb_group"].to_numpy()[train_indices]
    x = np.array(np.load(cache_dir() / "ecfp4_assayed_pool_full.npy", mmap_mode="r")[train_indices])

    cv = None
    if not args.skip_cv:
        cv = run_grouped_cv(x, y, groups)
        print(f"mean ROC-AUC={cv['mean_rocauc']:.4f} mean PR-AUC={cv['mean_prauc']:.4f}")

    model = lgb_train(x, y, num_boost_round=200)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "wb") as f:
        pickle.dump(model, f)
    print(f"Wrote {args.out_model}")

    write_report(
        "train-baseline-l0",
        {
            "cluster_id_source": args.cluster_id_path.name,
            "n_positives": n_positives,
            "n_pool": len(pool_df),
            "negative_sampling": {
                "ratio": NEGATIVE_RATIO,
                "fraction_from_similar": FRACTION_FROM_SIMILAR,
                "n_requested": neg.n_requested,
                "n_actual": len(neg.negative_indices),
                "shortfall": neg.shortfall,
                "seed": SEED,
            },
            "n_train_total": len(train_indices),
            "cv": cv,
            "model_file": args.out_model.name,
            "elapsed_seconds": time.time() - t0,
        },
    )


if __name__ == "__main__":
    main()
