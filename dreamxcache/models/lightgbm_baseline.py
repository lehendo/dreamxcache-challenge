"""LightGBM classifier over ECFP4 features.

Reference model for the labeling variants. LightGBM's own defaults are used
(no hyperparameter search), and class imbalance is handled upstream by
negative sampling rather than class weighting, so ``is_unbalance`` stays off
to avoid correcting for the same imbalance twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

RANDOM_STATE = 0


@dataclass(frozen=True)
class TrainedModel:
    booster: lgb.Booster
    params: dict[str, object]
    num_boost_round: int
    best_iteration: int | None


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    seed: int = RANDOM_STATE,
) -> TrainedModel:
    """Train a LightGBM binary classifier. With ``x_val``/``y_val`` it early-
    stops against them; otherwise it trains for the full ``num_boost_round``
    (e.g. a final fit on all labeled data after model selection).

    `seed` is passed to LightGBM. With bagging and feature fractions left at
    their defaults (1.0), boosting on a fixed ``(x, y)`` is deterministic
    whatever the seed; seed variance enters through *what* ``(x, y)`` is,
    e.g. a different negative-sampling seed upstream.
    """
    params: dict[str, object] = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
    }
    train_set = lgb.Dataset(x_train, label=y_train)
    callbacks = []
    valid_sets = None
    if x_val is not None and y_val is not None:
        valid_sets = [lgb.Dataset(x_val, label=y_val, reference=train_set)]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks or None,  # type: ignore[arg-type]
    )
    return TrainedModel(
        booster=booster,
        params=params,
        num_boost_round=num_boost_round,
        best_iteration=booster.best_iteration if valid_sets else None,
    )


def score(model: TrainedModel, x: np.ndarray) -> np.ndarray:
    """Predicted probability of the positive class for each row of `x`."""
    num_iteration = model.best_iteration if model.best_iteration else None
    predicted = model.booster.predict(x, num_iteration=num_iteration)
    assert isinstance(predicted, np.ndarray)
    return predicted
