"""Hand-checked fixtures for dreamxcache/negatives/. The BitBIRCH tests need the
third-party checkout (scripts/third_party.sh) and are skipped without it."""

from __future__ import annotations

import numpy as np
from conftest import requires_bitbirch

from dreamxcache.negatives.bitbirch_clustering import cluster_fingerprints
from dreamxcache.negatives.sampling import sample_negatives, sample_random_negatives


@requires_bitbirch
def test_cluster_fingerprints_separates_obviously_distinct_blobs():
    rng = np.random.default_rng(0)
    blob1 = (rng.random((30, 64)) < 0.05).astype(np.int8)
    blob2 = (rng.random((30, 64)) < 0.05).astype(np.int8)
    blob2[:, :32] = 1  # force blob2 far from blob1 in Tanimoto space
    fps = np.vstack([blob1, blob2])

    clusters, cluster_id = cluster_fingerprints(fps, threshold=0.5)

    assert cluster_id.shape == (60,)
    assert cluster_id.min() >= 0
    # every row assigned to exactly the cluster its list membership implies
    for cid, members in enumerate(clusters):
        assert set(np.where(cluster_id == cid)[0].tolist()) == set(members)
    # blob1 (rows 0-29) and blob2 (rows 30-59) must not share any cluster
    blob1_clusters = set(cluster_id[:30].tolist())
    blob2_clusters = set(cluster_id[30:].tolist())
    assert blob1_clusters.isdisjoint(blob2_clusters)


@requires_bitbirch
def test_cluster_fingerprints_upcasts_int8_to_avoid_dot_product_overflow():
    # Regression test for a real bug: np.dot on int8 arrays does not widen the accumulator, so a fingerprint
    # dot product (shared-bit count) over ~127 silently wraps around and
    # corrupts every similarity BitBirch computes. Confirmed directly:
    # np.dot(np.ones(2048, dtype=np.int8), np.ones(2048, dtype=np.int8)) == 0
    # (true value 2048, wraps mod 256). This dimensionality (2048, matching
    # real ECFP4 fingerprints) is exactly what's needed to trigger it --
    # the other test in this file uses 64-dim vectors, too short to overflow,
    # which is exactly why it didn't catch this bug originally.
    dim = 2048
    n_shared_bits = 200  # > int8's 127 max; dot product of two such vectors is 200, not representable in int8
    fp_identical_a = np.zeros(dim, dtype=np.int8)
    fp_identical_a[:n_shared_bits] = 1
    fp_identical_b = fp_identical_a.copy()  # Tanimoto similarity 1.0 with fp_identical_a
    fp_distant = np.zeros(dim, dtype=np.int8)
    fp_distant[n_shared_bits : 2 * n_shared_bits] = 1  # no overlap with the first two at all

    fps = np.vstack([fp_identical_a, fp_identical_b, fp_distant])
    _, cluster_id = cluster_fingerprints(fps, threshold=0.5)

    assert cluster_id[0] == cluster_id[1], "identical high-set-bit-count fingerprints must merge"
    assert cluster_id[2] != cluster_id[0], "a fingerprint with zero overlap must not merge in"


def test_sample_negatives_hand_checked_split_and_exclusion():
    # 10 compounds, clusters: {0,1,2} in cluster A, {3,4,5} in cluster B,
    # {6,7,8,9} in cluster C. Positive is index 1 (cluster A).
    cluster_id = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    positive_mask = np.zeros(10, dtype=bool)
    positive_mask[1] = True

    # ratio=2 -> n_total=2; round(0.8*2)=round(1.6)=2, elsewhere=0
    result = sample_negatives(cluster_id, positive_mask, ratio=2.0, fraction_from_similar=0.8, seed=0)

    assert result.n_requested == 2
    assert result.n_from_similar_requested == 2  # round(0.8*2)=2 (banker's rounding on .6 -> 2)
    assert result.n_from_elsewhere_requested == 0
    assert len(result.negative_indices) == 2
    assert 1 not in result.negative_indices  # the positive itself never sampled
    # both sampled from cluster A (the only "similar" cluster), i.e. from {0, 2}
    assert set(result.negative_indices.tolist()).issubset({0, 2})
    assert result.shortfall == 0


def test_sample_negatives_reports_shortfall_when_pool_too_small():
    # Only 1 non-positive compound total in the "similar" cluster, but the
    # caller asks for far more than that -> must report the shortfall, not silently
    # under-deliver without a trace.
    cluster_id = np.array([0, 0, 1, 1, 1, 1])
    positive_mask = np.zeros(6, dtype=bool)
    positive_mask[0] = True  # cluster 0 has only 1 other member (index 1)

    result = sample_negatives(cluster_id, positive_mask, ratio=10.0, fraction_from_similar=1.0, seed=0)

    assert result.n_from_similar_requested == 10
    assert result.n_from_similar_actual == 1  # only index 1 is available
    assert result.shortfall == 9
    assert result.negative_indices.tolist() == [1]


def test_sample_negatives_binary_stratum_true_vs_false_diverge_when_positives_span_both_values():
    # Regression test for a real bug: with a 2-valued cluster_id (the GPU similarity-stratum flag, not real
    # BitBirch clusters), the default (binary_stratum=False) derivation of
    # "similar" from `positive_clusters` breaks the instant a caller's own
    # positives include a row from BOTH values -- positive_clusters becomes
    # {0, 1}, which covers every possible cluster_id value, so "similar"
    # silently means "the whole pool" and "elsewhere" collapses to empty.
    # cluster_id: 3 in-stratum rows (idx 0-2), 5 elsewhere rows (idx 3-7).
    cluster_id = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    positive_mask = np.zeros(8, dtype=bool)
    positive_mask[0] = True  # a positive inside the stratum
    positive_mask[3] = True  # a positive outside the stratum (e.g. a labeling
    # variant whose positives aren't a subset of the reference set the
    # stratum was built from)

    # ratio=2 -> n_total=4; from_similar_requested=round(0.8*4)=3, elsewhere=1.
    buggy = sample_negatives(cluster_id, positive_mask, ratio=2.0, fraction_from_similar=0.8, seed=0)
    fixed = sample_negatives(
        cluster_id, positive_mask, ratio=2.0, fraction_from_similar=0.8, seed=0, binary_stratum=True
    )

    # Buggy (default) path: positive_clusters={0,1} -> is_similar_cluster is
    # True everywhere -> elsewhere pool is empty -> the whole 20% "elsewhere"
    # share silently goes unfilled, and "similar" negatives are drawn from
    # the entire non-positive pool (all 6 remaining rows), not just the
    # genuine 2-row in-stratum pool.
    assert buggy.n_from_elsewhere_actual == 0
    assert buggy.shortfall == 1
    assert buggy.n_from_similar_actual == 3

    # Fixed (binary_stratum=True) path: "similar" is cluster_id==1 directly,
    # independent of which cluster this call's own positives happen to fall
    # in. Only 2 genuine in-stratum rows are available (idx 1, 2; idx 0 is
    # the positive, excluded) against a request for 3, so the shortfall
    # correctly lands on the similar side instead of silently vanishing
    # from the elsewhere side.
    assert set(fixed.negative_indices[: fixed.n_from_similar_actual].tolist()).issubset({1, 2})
    assert fixed.n_from_similar_actual == 2
    assert fixed.n_from_elsewhere_actual == 1
    assert fixed.shortfall == 1


def test_sample_negatives_is_deterministic_given_seed():
    cluster_id = np.array([0] * 5 + [1] * 5)
    positive_mask = np.zeros(10, dtype=bool)
    positive_mask[0] = True

    r1 = sample_negatives(cluster_id, positive_mask, ratio=2.0, seed=42)
    r2 = sample_negatives(cluster_id, positive_mask, ratio=2.0, seed=42)
    assert r1.negative_indices.tolist() == r2.negative_indices.tolist()


def test_sample_random_negatives_hand_checked():
    positive_mask = np.zeros(10, dtype=bool)
    positive_mask[0] = True
    positive_mask[5] = True

    result = sample_random_negatives(positive_mask, ratio=2.0, seed=0)

    assert result.n_requested == 4  # ratio=2.0 * n_positives=2
    assert len(result.negative_indices) == 4
    assert 0 not in result.negative_indices  # never sample a known positive
    assert 5 not in result.negative_indices
    assert result.shortfall == 0
    assert result.n_from_similar_actual == 0  # no stratification at all


def test_sample_random_negatives_reports_shortfall_when_pool_too_small():
    positive_mask = np.zeros(5, dtype=bool)
    positive_mask[0] = True  # 4 non-positive rows available, but ratio asks for 20

    result = sample_random_negatives(positive_mask, ratio=20.0, seed=0)

    assert result.n_requested == 20
    assert len(result.negative_indices) == 4
    assert result.shortfall == 16


def test_sample_random_negatives_is_deterministic_given_seed():
    positive_mask = np.zeros(20, dtype=bool)
    positive_mask[0] = True

    r1 = sample_random_negatives(positive_mask, ratio=2.0, seed=7)
    r2 = sample_random_negatives(positive_mask, ratio=2.0, seed=7)
    assert r1.negative_indices.tolist() == r2.negative_indices.tolist()
