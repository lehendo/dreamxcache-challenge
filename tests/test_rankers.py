import json

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from dreamxcache.rankers.cats import compute_cats_fingerprints
from dreamxcache.rankers.erg import ERG_DIM, compute_erg_fingerprints
from dreamxcache.rankers.morgan import compute_morgan_fps, max_tanimoto_similarity
from dreamxcache.rankers.pharmacophore import compute_pharmacophore_scores, load_reference_points
from dreamxcache.rankers.similarity import continuous_tanimoto_max_similarity, load_anchors
from dreamxcache.rankers.usrcat import USRCAT_DIM, compute_usrcat_fingerprints, max_usr_similarity, usr_similarity

LIBRARY = ["CCO", "c1ccccc1CCO", "CC(=O)Nc1ccc(O)cc1", "CCCCCCCC"]


def test_continuous_tanimoto_hand_checked():
    q = np.array([[1.0, 0.0], [1.0, 1.0]])
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    sim = continuous_tanimoto_max_similarity(q, a)
    # q0 vs a0: dot 1, |q|^2 1, |a|^2 1 -> 1/(1+1-1) = 1.0
    assert sim[0] == pytest.approx(1.0)
    # q1 vs a0: 1/(2+1-1) = 0.5 ; q1 vs a1: 2/(2+4-2) = 0.5 -> max 0.5
    assert sim[1] == pytest.approx(0.5)


def test_continuous_tanimoto_zero_vectors_score_zero_not_nan():
    sim = continuous_tanimoto_max_similarity(np.zeros((1, 3)), np.zeros((1, 3)))
    assert sim[0] == 0.0


def test_load_anchors_filters_and_rejects_typos(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame(
        {"smiles": ["CCO", "CCN", "CCC"], "source": ["x", "x", "y"], "detail": ["one", "two", "three"]}
    ).to_csv(p, index=False)
    assert len(load_anchors(p)) == 3
    assert load_anchors(p, sources=["x"])["smiles"].tolist() == ["CCO", "CCN"]
    assert load_anchors(p, sources=["x"], details=["two"])["smiles"].tolist() == ["CCN"]
    with pytest.raises(ValueError, match="sources not present"):
        load_anchors(p, sources=["nope"])
    with pytest.raises(ValueError, match="detail values not present"):
        load_anchors(p, sources=["x"], details=["three"])


def test_load_anchors_requires_smiles_column(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="smiles"):
        load_anchors(p)


def test_erg_shape_invalid_rows_and_self_similarity():
    fps, n_invalid = compute_erg_fingerprints(LIBRARY + ["not_a_smiles((("], n_workers=1)
    assert fps.shape == (5, ERG_DIM)
    assert n_invalid == 1 and not fps[-1].any()
    sim = continuous_tanimoto_max_similarity(fps, fps[[1]])
    assert sim[1] == pytest.approx(1.0)
    assert sim[4] == 0.0  # the unparseable row scores zero against everything
    assert int(np.argmax(sim)) == 1


def test_morgan_max_tanimoto_self_is_one_and_invalid_is_zero():
    fps = compute_morgan_fps(LIBRARY + ["not_a_smiles((("], use_features=False)
    anchors = [fps[2]]
    sim = max_tanimoto_similarity(fps, anchors)
    assert sim[2] == pytest.approx(1.0)
    assert sim[4] == 0.0
    assert (sim[[0, 1, 3]] < 1.0).all()


def test_fcfp_differs_from_ecfp():
    ecfp = compute_morgan_fps(["CC(=O)Nc1ccc(O)cc1"], use_features=False)[0]
    fcfp = compute_morgan_fps(["CC(=O)Nc1ccc(O)cc1"], use_features=True)[0]
    assert list(ecfp.GetOnBits()) != list(fcfp.GetOnBits())


def test_usr_similarity_hand_checked():
    a = np.array([0.0, 0.0])
    assert usr_similarity(a, a) == 1.0
    assert usr_similarity(a, np.array([1.0, 3.0])) == pytest.approx(1.0 / (1.0 + 2.0))


def test_usrcat_vectors_and_max_similarity_ranks_self_first():
    vecs, valid, n_failed = compute_usrcat_fingerprints(LIBRARY + ["not_a_smiles((("], n_workers=1)
    assert vecs.shape == (5, USRCAT_DIM)
    assert n_failed == 1 and not valid[-1] and valid[:4].all()
    sim = max_usr_similarity(vecs, valid, vecs[[2]])
    assert sim[2] == pytest.approx(1.0)
    assert sim[4] == 0.0
    assert int(np.argmax(sim)) == 2


def test_cats_like_fingerprints_shape_and_invalid_rows():
    fps, n_invalid = compute_cats_fingerprints(LIBRARY + ["not_a_smiles((("], n_jobs=1)
    assert fps.shape[0] == 5 and n_invalid == 1 and not fps[-1].any()
    assert fps[1:4].any(axis=1).all()  # ethanol has too few pharmacophore points to form a pair


def test_reference_point_loader_validates_types(tmp_path):
    good = tmp_path / "p.json"
    good.write_text(json.dumps([{"name": "r", "type": "aromatic", "xyz": [1, 2, 3], "cutoff": 2.5}]))
    assert load_reference_points(good) == [("r", "aromatic", (1.0, 2.0, 3.0), 2.5)]
    bad = tmp_path / "q.json"
    bad.write_text(json.dumps([{"name": "r", "type": "ionic", "xyz": [1, 2, 3], "cutoff": 2.5}]))
    with pytest.raises(ValueError, match="aromatic' or 'hbond"):
        load_reference_points(bad)


def test_pharmacophore_scores_failures_score_zero_and_are_counted():
    ref = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CCO"))
    AllChem.EmbedMolecule(ref, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(ref, maxIters=200)
    points = [("ring", "aromatic", (0.0, 0.0, 0.0), 2.5), ("oh", "hbond", (3.0, 0.0, 0.0), 2.5)]
    scores, n_failed = compute_pharmacophore_scores(["c1ccccc1CCO", "CCCCCC", "not_a_smiles((("], ref, points)
    assert scores.shape == (3,)
    assert n_failed == 1 and scores[2] == 0.0
    assert scores[0] > 0 and scores[1] > 0
