import json

import numpy as np
import pandas as pd
import pytest

from dreamxcache.rankers.boltz2 import (
    MAX_HEAVY_ATOMS,
    NulByteInMsaError,
    aggregate_affinities,
    check_a3m,
    split_into_chunks,
    strip_nul_bytes,
    write_yamls,
)


def _a3m(tmp_path, nul: bool):
    p = tmp_path / "msa.a3m"
    p.write_bytes(b">101\nACDEFG\n>hit\nACDEFA\n" + (b"\x00" if nul else b""))
    return p


def test_check_a3m_passes_a_clean_file(tmp_path):
    check_a3m(_a3m(tmp_path, nul=False))


def test_check_a3m_rejects_a_trailing_nul_byte(tmp_path):
    with pytest.raises(NulByteInMsaError, match="1 NUL byte"):
        check_a3m(_a3m(tmp_path, nul=True))


def test_strip_nul_bytes_fixes_the_file_and_reports_the_count(tmp_path):
    p = _a3m(tmp_path, nul=True)
    assert strip_nul_bytes(p) == 1
    check_a3m(p)
    assert p.read_bytes() == b">101\nACDEFG\n>hit\nACDEFA\n"
    assert strip_nul_bytes(p) == 0


def test_write_yamls_refuses_a_corrupt_msa(tmp_path):
    compounds = pd.DataFrame({"id": ["c1"], "smiles": ["CCO"]})
    with pytest.raises(NulByteInMsaError):
        write_yamls(compounds, "id", "smiles", "ACDEFG", _a3m(tmp_path, nul=True), tmp_path / "yamls")
    assert not (tmp_path / "yamls").exists() or not any((tmp_path / "yamls").iterdir())


def test_write_yamls_writes_valid_and_skips_unparseable_and_oversized(tmp_path):
    big = "C" * (MAX_HEAVY_ATOMS + 1)
    compounds = pd.DataFrame({"id": ["ok", "bad", "huge"], "smiles": ["CCO", "not_a_smiles(((", big]})
    n, skipped = write_yamls(compounds, "id", "smiles", "ACDEFG", _a3m(tmp_path, nul=False), tmp_path / "yamls")
    assert n == 1
    assert {s[0] for s in skipped} == {"bad", "huge"}
    text = (tmp_path / "yamls" / "ok.yaml").read_text()
    assert 'sequence: "ACDEFG"' in text and 'smiles: "CCO"' in text and "binder: B" in text


def test_split_into_chunks_is_a_partition(tmp_path):
    ydir = tmp_path / "y"
    ydir.mkdir()
    for i in range(10):
        (ydir / f"c{i}.yaml").write_text("x")
    dirs = split_into_chunks(ydir, tmp_path / "chunks", n_chunks=3)
    names = [f.name for d in dirs for f in d.iterdir()]
    assert sorted(names) == sorted(f"c{i}.yaml" for i in range(10))
    assert len(names) == len(set(names))


def test_aggregate_affinities_is_nan_safe_and_pools_redundant_dirs(tmp_path):
    for d, cid, val in [("a", "c0", 0.2), ("b", "c0", 0.3), ("b", "c2", 0.9)]:
        out = tmp_path / d / "boltz_results" / "predictions" / cid
        out.mkdir(parents=True, exist_ok=True)
        (out / f"affinity_{cid}.json").write_text(json.dumps({"affinity_probability_binary": val}))
    # a result with the field missing, and one for a compound that is not tracked
    bad = tmp_path / "a" / "predictions" / "c1"
    bad.mkdir(parents=True)
    (bad / "affinity_c1.json").write_text(json.dumps({"other": 1}))
    stray = tmp_path / "a" / "predictions" / "zzz"
    stray.mkdir(parents=True)
    (stray / "affinity_zzz.json").write_text(json.dumps({"affinity_probability_binary": 0.5}))

    scores, stats = aggregate_affinities([tmp_path / "a", tmp_path / "b"], ["c0", "c1", "c2", "c3"])
    assert scores[0] == pytest.approx(0.3)  # redundant copies: last directory wins
    assert np.isnan(scores[1]) and np.isnan(scores[3])  # missing field / never ran -> NaN, not 0
    assert scores[2] == pytest.approx(0.9)
    assert stats == {"n_found": 3, "n_missing_field": 1, "n_unmatched": 1}
