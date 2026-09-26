"""Grouped LightGBM training with schema-asserted inference (F05 Phase E)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold


def train_grouped_oof(X, y, groups, feature_names, params, n_splits=3, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    models, iters = [], []
    for tr, va in gkf.split(X, y, groups):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va])
        p = dict(params)
        p.setdefault("verbose", -1)
        # P1 fix: subsample without bagging_freq does no row bagging; enable it
        if p.get("subsample", 1.0) < 1.0 and not p.get("bagging_freq"):
            p["bagging_freq"] = 1
        bst = lgb.train(p, dtr, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = bst.predict(X[va])
        models.append(bst)
        iters.append(bst.best_iteration)
    return oof, models, iters


def refit_full(X_mask, y_mask, feature_names, params, num_boost_round: int):
    import pandas as pd
    df = pd.DataFrame(X_mask, columns=feature_names)
    d = lgb.Dataset(df, label=y_mask)
    p = dict(params)
    p.setdefault("verbose", -1)
    if p.get("subsample", 1.0) < 1.0 and not p.get("bagging_freq"):
        p["bagging_freq"] = 1
    return lgb.train(p, d, num_boost_round=num_boost_round)


def assert_schema(feature_names: list, model) -> None:
    names = model.feature_name()
    if list(names) != list(feature_names):
        raise ValueError(f"feature schema mismatch: model={names} code={feature_names}")
