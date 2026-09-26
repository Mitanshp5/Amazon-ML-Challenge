"""Grouped LightGBM training with schema-asserted inference (F05 Phase E & Step 1: D1-00).

Fixes applied:
- Explicit feature_name=list(feature_names) attached to all lgb.Dataset instances,
  preventing automatic Column_0..N renaming and assert_schema failure.
- Explicit num_boost_round (default 1000) and early_stopping (default 100)
  preventing silent LightGBM 100-iteration default cap.
- Bagging frequency enabled whenever subsample < 1.0.
- Aggregated split and gain feature importance tracking.
- Pre-split fold indices support for exact inner_train_folds reproduction.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def train_grouped_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    params: dict,
    n_splits: int = 3,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
    fold_assignments: np.ndarray | None = None,
) -> tuple[np.ndarray, list[lgb.Booster], list[int], dict[str, float]]:
    """Train grouped cross-validation with schema-asserted LightGBM models."""
    feat_names = list(feature_names)
    oof = np.zeros(len(y), dtype=np.float32)
    models: list[lgb.Booster] = []
    best_iters: list[int] = []
    gains: dict[str, float] = {f: 0.0 for f in feat_names}

    if fold_assignments is not None:
        unique_folds = sorted(np.unique(fold_assignments))
        splits = [
            (np.where(fold_assignments != f)[0], np.where(fold_assignments == f)[0])
            for f in unique_folds
        ]
    else:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))

    for tr, va in splits:
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=feat_names, free_raw_data=False)
        dva = lgb.Dataset(X[va], label=y[va], feature_name=feat_names, reference=dtr, free_raw_data=False)

        p = dict(params)
        p.setdefault("verbose", -1)
        p.setdefault("seed", seed)
        if p.get("subsample", 1.0) < 1.0 and not p.get("bagging_freq"):
            p["bagging_freq"] = 1

        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        bst = lgb.train(
            p,
            dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dva],
            callbacks=callbacks,
        )
        oof[va] = bst.predict(X[va])
        models.append(bst)
        best_iters.append(bst.best_iteration)

        f_gains = bst.feature_importance(importance_type="gain")
        for fn, g in zip(feat_names, f_gains):
            gains[fn] += float(g) / len(splits)

    return oof, models, best_iters, gains


def refit_full(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    params: dict,
    num_boost_round: int,
    seed: int = 42,
) -> lgb.Booster:
    """Refit LightGBM on the full dataset with exact schema names."""
    feat_names = list(feature_names)
    d = lgb.Dataset(X, label=y, feature_name=feat_names, free_raw_data=False)
    p = dict(params)
    p.setdefault("verbose", -1)
    p.setdefault("seed", seed)
    if p.get("subsample", 1.0) < 1.0 and not p.get("bagging_freq"):
        p["bagging_freq"] = 1
    return lgb.train(p, d, num_boost_round=num_boost_round)


def assert_schema(feature_names: list[str], model: lgb.Booster) -> None:
    """Verify that model input schema matches expected feature names in order."""
    names = model.feature_name()
    if list(names) != list(feature_names):
        raise ValueError(f"Feature schema mismatch! Model={names}, Expected={feature_names}")
