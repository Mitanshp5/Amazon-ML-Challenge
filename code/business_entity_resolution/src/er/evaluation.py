"""Evaluation tiers + slice reporting (F05 8.2/8.3). Uses exact macro scorer."""
from __future__ import annotations

import numpy as np

from .metrics import f05_single


def per_s1_scores(truth: dict, pred: dict) -> dict:
    return {k: f05_single(set(truth[k]), set(pred.get(k, ()))) for k in truth}


def summarize(truth, pred, meta: dict | None = None) -> dict:
    meta = meta or {}
    scores = per_s1_scores(truth, pred)
    vals = np.array([scores[k] for k in truth])
    out = {"macro_f05": float(vals.mean()), "n": len(truth),
           "empty_rate": float(np.mean([len(pred.get(k, ())) == 0 for k in truth]))}
    by_country: dict = {}
    for k, s in scores.items():
        c = meta.get(k, {}).get("country", "?")
        by_country.setdefault(c, []).append(s)
    out["by_country"] = {c: float(np.mean(v)) for c, v in by_country.items()}
    return out


def paired_bootstrap_ci(truth, pred_a, pred_b, n_boot=1000, seed=0) -> dict:
    rng = np.random.RandomState(seed)
    keys = list(truth)
    sa = np.array([f05_single(set(truth[k]), set(pred_a.get(k, ()))) for k in keys])
    sb = np.array([f05_single(set(truth[k]), set(pred_b.get(k, ()))) for k in keys])
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(keys), len(keys))
        diffs.append(float((sb[idx] - sa[idx]).mean()))
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    return {"mean_diff": float((sb - sa).mean()), "ci95": [lo, hi]}
