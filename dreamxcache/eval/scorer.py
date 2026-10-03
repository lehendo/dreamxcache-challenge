"""Wrapper over the official Target2035 evaluator (``aircheck_utils``).

The evaluator is referenced, not copied (see ``scripts/third_party.sh``): it
is loaded unmodified from ``<third_party>/aircheck_utils/EvaluationCode/
evaluation_function.py`` via importlib, so the objective function can never
drift out of sync with the source.

The official ``evaluate_team_model`` needs a ``Cluster`` column on the gold
dataframe (organizer-assigned chemical series, hit molecules only). Where that
column is unavailable, two explicit modes are provided:

- ``"hits_only"``: exact hit count and exact global ROC-AUC / PR-AUC. Every
  hit is given its own singleton cluster id purely to satisfy the evaluator's
  API; the resulting cluster and p-value fields are meaningless and are
  dropped from the return value.
- ``"proxy"``: additionally reports a chemical-series count and p-value using
  ECFP4 Tanimoto agglomerative clustering (``proxy_clustering.py``) as a
  stand-in for the official MCS-based assignment. All such fields are prefixed
  ``PROXY_``. Use them only to rank variants against each other, never as the
  official p-value.

If you have a real ``Cluster`` column, call `evaluate_team_model` directly.

`score` takes a required ``dataset`` argument (``"wdr91"`` or ``"pgk2"``) and
hard-asserts that ``id_col`` (and, for PGK2, ``sel_cols``) match what that
dataset's files use: the prior round's files key on ``RandomID``, the PGK2
submission template keys on ``CatalogID`` and requires exactly ``Sel_50``.
Scoring against the wrong ID column silently produces a near-empty merge and a
meaningless number that looks plausible, so this is an assertion, not a
docstring warning.
"""

from __future__ import annotations

import types
from typing import Any, Literal

import pandas as pd

from dreamxcache.eval.proxy_clustering import (
    DEFAULT_TANIMOTO_DISTANCE_THRESHOLD,
    PROXY_LABELING_VERSION,
    assign_proxy_clusters,
)
from dreamxcache.third_party import load_module_from_file, require_dir

_vendored: types.ModuleType | None = None


def _evaluator() -> types.ModuleType:
    global _vendored
    if _vendored is None:
        path = require_dir("aircheck_utils") / "EvaluationCode" / "evaluation_function.py"
        _vendored = load_module_from_file("_aircheck_evaluation_function", path)
    return _vendored


def evaluate_team_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """The official evaluator, for use with a real ``Cluster`` column."""
    result: dict[str, Any] = _evaluator().evaluate_team_model(*args, **kwargs)
    return result


_DATASET_ID_COLUMNS: dict[str, str] = {"wdr91": "RandomID", "pgk2": "CatalogID"}
_DATASET_LABEL_COLUMNS: dict[str, str] = {"wdr91": "LABEL"}
_PGK2_REQUIRED_SEL_COLS = ["Sel_50"]


def _assert_dataset_columns(
    dataset: Literal["wdr91", "pgk2"], id_col: str, sel_cols: list[str], label_col: str
) -> None:
    expected_id_col = _DATASET_ID_COLUMNS[dataset]
    if id_col != expected_id_col:
        raise AssertionError(
            f"dataset={dataset!r} requires id_col={expected_id_col!r}, got {id_col!r}. "
            "Scoring against the wrong ID column merges almost nothing and produces a "
            "number that looks plausible but is meaningless; refusing rather than warning."
        )
    expected_label_col = _DATASET_LABEL_COLUMNS.get(dataset)
    if expected_label_col is not None and label_col != expected_label_col:
        raise AssertionError(f"dataset={dataset!r} requires label_col={expected_label_col!r}, got {label_col!r}.")
    if dataset == "pgk2" and sel_cols != _PGK2_REQUIRED_SEL_COLS:
        raise AssertionError(
            f"dataset='pgk2' requires sel_cols=={_PGK2_REQUIRED_SEL_COLS!r} (exactly 50 flagged), got {sel_cols!r}."
        )


def score(
    gold_df: pd.DataFrame,
    team_df: pd.DataFrame,
    mode: Literal["hits_only", "proxy"],
    dataset: Literal["wdr91", "pgk2"],
    sel_cols: list[str],
    id_col: str,
    label_col: str,
    score_col: str = "Score",
    smiles_col: str = "SMILES",
    proxy_threshold: float = DEFAULT_TANIMOTO_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    _assert_dataset_columns(dataset, id_col, sel_cols, label_col)

    if "Cluster" in gold_df.columns and gold_df["Cluster"].notna().any():
        raise ValueError(
            "gold_df already has a real Cluster column: call evaluate_team_model directly; "
            "this wrapper's modes are only for when it is absent."
        )

    gold_df = gold_df.copy()
    hit_mask = gold_df[label_col] == 1

    if mode == "hits_only":
        gold_df["Cluster"] = pd.NA
        gold_df.loc[hit_mask, "Cluster"] = range(1, int(hit_mask.sum()) + 1)
        raw = evaluate_team_model(
            gold_df,
            team_df,
            label_gold=label_col,
            score=score_col,
            labels_team=sel_cols,
            cluster="Cluster",
            random_id=id_col,
        )
        result: dict[str, Any] = {
            "mode": "hits_only",
            "dataset": dataset,
            "ROCAUC": raw["ROCAUC"],
            "PRAUC": raw["PRAUC"],
        }
        for sel in sel_cols:
            result[f"Hits_{sel}"] = raw[f"Hits_{sel}"]
        return result

    if mode == "proxy":
        gold_df["Cluster"] = pd.NA
        proxy_labels = assign_proxy_clusters(gold_df.loc[hit_mask, smiles_col], threshold=proxy_threshold)
        gold_df.loc[hit_mask, "Cluster"] = proxy_labels.to_numpy()
        raw = evaluate_team_model(
            gold_df,
            team_df,
            label_gold=label_col,
            score=score_col,
            labels_team=sel_cols,
            cluster="Cluster",
            random_id=id_col,
        )
        result = {
            "mode": "proxy",
            "dataset": dataset,
            "proxy_labeling_version": PROXY_LABELING_VERSION,
            "proxy_threshold": proxy_threshold,
            "ROCAUC": raw["ROCAUC"],
            "PRAUC": raw["PRAUC"],
        }
        for sel in sel_cols:
            result[f"Hits_{sel}"] = raw[f"Hits_{sel}"]
            result[f"PROXY_Clusters_{sel}"] = raw[f"Clusters_{sel}"]
            result[f"PROXY_P-value_{sel}"] = raw[f"P-value_{sel}"]
            result[f"PROXY_ClusterPRAUC_{sel}"] = raw[f"ClusterPRAUC_{sel}"]
        return result

    raise ValueError(f"unknown mode {mode!r}, must be 'hits_only' or 'proxy'")
