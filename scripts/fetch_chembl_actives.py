#!/usr/bin/env python3
"""Fetch IC50 activities for one ChEMBL target from the public ChEMBL REST API
into an anchor-style CSV.

    python scripts/fetch_chembl_actives.py --target-chembl-id CHEMBL2886 --out data/external/chembl_actives.csv

A network call: run it once, keep the CSV, and do not call it from analysis code
(fetch, then cache, then analyze, as separate steps). Output columns:
``molecule_chembl_id, smiles, standard_value, standard_units, standard_relation,
target_chembl_id``. Add ``source`` / ``detail`` columns yourself if you want to
filter anchors by them (``rankers/similarity.load_anchors``).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
FIELDS = ["molecule_chembl_id", "smiles", "standard_value", "standard_units", "standard_relation", "target_chembl_id"]


def fetch_all_activities(target_chembl_id: str, standard_type: str = "IC50") -> list[dict[str, object]]:
    activities: list[dict[str, object]] = []
    offset, limit = 0, 1000
    while True:
        url = f"{BASE_URL}?target_chembl_id={target_chembl_id}&standard_type={standard_type}&limit={limit}&offset={offset}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        batch = data.get("activities", [])
        activities.extend(batch)
        total = data.get("page_meta", {}).get("total_count", len(batch))
        print(f"  fetched {len(activities)}/{total}")
        if len(activities) >= total or not batch:
            break
        offset += limit
        time.sleep(0.5)  # be polite to the public API
    return activities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-chembl-id", required=True)
    parser.add_argument("--standard-type", default="IC50")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows, n_no_smiles = [], 0
    for a in fetch_all_activities(args.target_chembl_id, args.standard_type):
        smiles = a.get("canonical_smiles")
        if not smiles:
            n_no_smiles += 1
            continue
        rows.append(
            {
                "molecule_chembl_id": a.get("molecule_chembl_id"),
                "smiles": smiles,
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "standard_relation": a.get("standard_relation"),
                "target_chembl_id": args.target_chembl_id,
            }
        )
    print(f"{len(rows)} activities with SMILES, {n_no_smiles} without")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
