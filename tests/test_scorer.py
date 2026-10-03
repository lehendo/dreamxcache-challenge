"""Hand-checked fixture for dreamxcache/eval/scorer.py, the objective function wrapper.

Tests that call the official evaluator need the third-party checkout
(scripts/third_party.sh) and are skipped without it.

Fixture: 10 gold molecules, 3 hits (RandomID 0,1,2), 7 non-hits (3..9).
Team predictions score the 3 hits above all non-hits and select exactly
them (Sel_3), so ROC-AUC / PR-AUC / hit count are hand-verifiable as
perfect (1.0 / 1.0 / 3) independent of any clustering.

Hit SMILES are chosen so the proxy clustering result is unambiguous without
needing to compute a Tanimoto value by hand: RandomID 0 and 1 are the exact
same molecule (ethanol, duplicated) so their fingerprints are identical and
must land in the same cluster at any positive distance threshold; RandomID 2
is benzene, structurally unrelated, and must land in its own cluster. So the
hand-checked expectation is exactly 2 proxy clusters among the 3 hits.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import requires_aircheck_utils

from dreamxcache.eval.proxy_clustering import assign_proxy_clusters
from dreamxcache.eval.scorer import score

HIT_SMILES = ["CCO", "CCO", "c1ccccc1"]  # ethanol, ethanol (dup), benzene
NON_HIT_SMILES = [
    "CCN",
    "CCC",
    "CCCl",
    "CCBr",
    "CCF",
    "CC(C)C",
    "CC(=O)O",
]


@pytest.fixture
def gold_df() -> pd.DataFrame:
    # "LABEL" (all caps) matches the prior round's gold files; dataset="wdr91"
    # hard-asserts on it (the evaluator's own default is "Label", mixed case).
    smiles = HIT_SMILES + NON_HIT_SMILES
    labels = [1, 1, 1] + [0] * len(NON_HIT_SMILES)
    return pd.DataFrame(
        {
            "RandomID": [f"ID{i}" for i in range(len(smiles))],
            "SMILES": smiles,
            "LABEL": labels,
        }
    )


@pytest.fixture
def team_df(gold_df: pd.DataFrame) -> pd.DataFrame:
    n = len(gold_df)
    # Hits (first 3 rows) score highest, guaranteeing perfect separation.
    scores = [1.0, 0.9, 0.8] + [0.5 - 0.01 * i for i in range(n - 3)]
    sel_3 = [1, 1, 1] + [0] * (n - 3)
    return pd.DataFrame({"RandomID": gold_df["RandomID"], "Score": scores, "Sel_3": sel_3})


def test_assign_proxy_clusters_hand_checked():
    hit_smiles = pd.Series(HIT_SMILES, index=["ID0", "ID1", "ID2"])
    labels = assign_proxy_clusters(hit_smiles)

    assert labels["ID0"] == labels["ID1"], "identical molecules must share a proxy cluster"
    assert labels["ID2"] != labels["ID0"], "benzene must not share ethanol's proxy cluster"
    assert labels.nunique() == 2


@requires_aircheck_utils
def test_score_hits_only_mode(gold_df, team_df):
    result = score(
        gold_df,
        team_df,
        mode="hits_only",
        dataset="wdr91",
        sel_cols=["Sel_3"],
        id_col="RandomID",
        label_col="LABEL",
    )

    assert result["mode"] == "hits_only"
    assert result["ROCAUC"] == pytest.approx(1.0)
    assert result["PRAUC"] == pytest.approx(1.0)
    assert result["Hits_Sel_3"] == 3
    # hits_only must never leak a cluster/p-value field under any name
    assert not any("luster" in k or "P-value" in k for k in result)


@requires_aircheck_utils
def test_score_proxy_mode(gold_df, team_df):
    result = score(
        gold_df,
        team_df,
        mode="proxy",
        dataset="wdr91",
        sel_cols=["Sel_3"],
        id_col="RandomID",
        label_col="LABEL",
    )

    assert result["mode"] == "proxy"
    assert result["ROCAUC"] == pytest.approx(1.0)
    assert result["PRAUC"] == pytest.approx(1.0)
    assert result["Hits_Sel_3"] == 3
    # 3 hits selected, but ID0/ID1 (identical molecules) collapse to one
    # cluster -> exactly 2 distinct proxy chemical series, hand-verified above.
    assert result["PROXY_Clusters_Sel_3"] == 2
    assert result["PROXY_P-value_Sel_3"] is not None
    # every proxy field must be unmistakably prefixed, never bare "Clusters_"/"P-value_"
    assert all(k.startswith("PROXY_") for k in result if "luster" in k or "P-value" in k)


def test_score_rejects_real_cluster_column(gold_df, team_df):
    gold_with_cluster = gold_df.copy()
    gold_with_cluster["Cluster"] = [1, 1, 2] + [pd.NA] * 7

    with pytest.raises(ValueError, match="already has a real Cluster column"):
        score(
            gold_with_cluster,
            team_df,
            mode="hits_only",
            dataset="wdr91",
            sel_cols=["Sel_3"],
            id_col="RandomID",
            label_col="LABEL",
        )


def test_score_rejects_wrong_id_col_for_dataset(gold_df, team_df):
    # WDR91 keys on RandomID; passing CatalogID must hard-fail, not silently
    # merge almost nothing.
    with pytest.raises(AssertionError, match="requires id_col='RandomID'"):
        score(
            gold_df,
            team_df,
            mode="hits_only",
            dataset="wdr91",
            sel_cols=["Sel_3"],
            id_col="CatalogID",
            label_col="LABEL",
        )


def test_score_rejects_wrong_label_col_for_dataset(gold_df, team_df):
    # The gold column is "LABEL" (all caps); the evaluator's own default is
    # "Label" (mixed case). Must hard-fail, not silently KeyError deep inside
    # the evaluator.
    with pytest.raises(AssertionError, match="requires label_col='LABEL'"):
        score(
            gold_df,
            team_df,
            mode="hits_only",
            dataset="wdr91",
            sel_cols=["Sel_3"],
            id_col="RandomID",
            label_col="Label",
        )


def test_score_rejects_wrong_id_col_for_pgk2(gold_df, team_df):
    # PGK2's test-submission template keys on CatalogID; passing RandomID
    # (the prior round's column) must hard-fail, not silently merge almost
    # nothing and produce a plausible-looking garbage number.
    with pytest.raises(AssertionError, match="requires id_col='CatalogID'"):
        score(
            gold_df,
            team_df,
            mode="hits_only",
            dataset="pgk2",
            sel_cols=["Sel_50"],
            id_col="RandomID",
            label_col="LABEL",
        )


def test_score_rejects_wrong_sel_cols_for_pgk2(gold_df, team_df):
    # PGK2 test-split submissions require exactly Sel_50; any other
    # selection-column name must hard-fail.
    renamed_team_df = team_df.rename(columns={"RandomID": "CatalogID"})
    with pytest.raises(AssertionError, match="requires sel_cols"):
        score(
            gold_df.rename(columns={"RandomID": "CatalogID"}),
            renamed_team_df,
            mode="hits_only",
            dataset="pgk2",
            sel_cols=["Sel_3"],
            id_col="CatalogID",
            label_col="LABEL",
        )
