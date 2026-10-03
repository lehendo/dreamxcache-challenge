"""Hand-checked fixture for dreamxcache/models/lightgbm_baseline.py: a trivially
separable synthetic dataset (positive class has feature 0 set, negative
doesn't), so a working classifier must rank held-out positives above
held-out negatives, not just avoid crashing.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from dreamxcache.models.lightgbm_baseline import score, train


def test_train_and_score_on_trivially_separable_data():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    x = rng.random((n, 16)).astype(np.float32)
    x[:, 0] = np.where(y == 1, x[:, 0] + 5.0, x[:, 0])  # feature 0 perfectly separates classes

    split = n // 2
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = train(x_train, y_train, x_val, y_val, num_boost_round=50, early_stopping_rounds=10)
    scores = score(model, x_val)

    assert scores.shape == (n - split,)
    assert roc_auc_score(y_val, scores) > 0.95  # near-perfect on a trivially separable task


def test_train_seed_is_overridable_and_threaded_into_params():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    x = rng.random((n, 16)).astype(np.float32)
    x[:, 0] = np.where(y == 1, x[:, 0] + 5.0, x[:, 0])

    model_seed1 = train(x, y, num_boost_round=50, seed=1)
    model_seed2 = train(x, y, num_boost_round=50, seed=2)
    assert model_seed1.params["seed"] == 1
    assert model_seed2.params["seed"] == 2

    # With bagging_fraction/feature_fraction at LightGBM's defaults (1.0, no
    # subsampling), boosting on a FIXED (x, y) is deterministic regardless of
    # `seed`, so scores1 == scores2 here, correctly. `seed` only matters when it
    # also changes what (x, y) IS, e.g. a different negative-sampling seed
    # upstream drawing a different training set.
    scores1 = score(model_seed1, x)
    scores2 = score(model_seed2, x)
    assert np.allclose(scores1, scores2)
