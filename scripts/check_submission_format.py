#!/usr/bin/env python3
"""Prove the test-split submission path end to end at real scale, without any
real labels.

    python scripts/check_submission_format.py --test-csv data/submissions/fusion_test.csv

Checks (1) the CSV's own format (columns, exactly 50 flagged, every CatalogID of
the real template, a Score on every row); (2) that the official evaluator's hard
assertions on ``id_col`` / ``sel_cols`` fire at full scale; and (3) that the full
path (load CSV, hand to the evaluator) runs without error on the exact file shape
that would be uploaded. A synthetic placeholder gold set (an arbitrary 50 rows
marked as hits) exercises the plumbing, so **the resulting ROC-AUC / PR-AUC /
hit count are meaningless and must never be reported as performance.** Needs the
official evaluator (``scripts/third_party.sh``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dreamxcache.config import raw_dir
from dreamxcache.eval.scorer import score
from dreamxcache.submission import validate_test_format

N_SYNTHETIC_HITS = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, required=True)
    args = parser.parse_args()

    team_df = pd.read_csv(args.test_csv)
    template = pd.read_csv(raw_dir() / "Val-Test-set" / "PGK2_Test_split.csv")
    problems = validate_test_format(team_df, template)
    if problems:
        raise SystemExit(f"format problems: {problems}")
    print(f"format OK: {len(team_df):,} rows, exactly 50 flagged")

    gold_df = template.copy()
    gold_df["LABEL"] = 0
    gold_df.loc[gold_df.index[:N_SYNTHETIC_HITS], "LABEL"] = 1

    for desc, kwargs in [
        ("wrong id_col", {"sel_cols": ["Sel_50"], "id_col": "RandomID"}),
        ("wrong sel_cols", {"sel_cols": ["Sel_3"], "id_col": "CatalogID"}),
    ]:
        try:
            score(gold_df, team_df, mode="hits_only", dataset="pgk2", label_col="LABEL", **kwargs)
        except AssertionError as e:
            print(f"{desc} correctly rejected: {e}")
        else:
            raise SystemExit(f"expected an AssertionError for {desc}")

    result = score(
        gold_df, team_df, mode="hits_only", dataset="pgk2", sel_cols=["Sel_50"], id_col="CatalogID", label_col="LABEL"
    )
    assert {"ROCAUC", "PRAUC", "Hits_Sel_50"} <= set(result), result
    print(f"evaluator path OK (result keys {sorted(result)}). The numbers are from SYNTHETIC labels and mean nothing.")


if __name__ == "__main__":
    main()
