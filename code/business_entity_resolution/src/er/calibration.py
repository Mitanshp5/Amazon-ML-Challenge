"""Threshold/decision surface on unsampled candidate lists (F05 Phase F)."""
from __future__ import annotations

import numpy as np

from .metrics import macro_f05


def decide_two_threshold(scores_by_q: dict, t_singleton: float, t_match: float) -> dict:
    out = {}
    for qid, pairs in scores_by_q.items():
        if not pairs or max(s for _, s in pairs) < t_singleton:
            out[qid] = []
        else:
            out[qid] = [pid for pid, s in pairs if s >= t_match]
    return out


def sweep_thresholds(scores_by_q: dict, truth: dict, grid=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9)):
    rows = []
    for t in grid:
        pred = decide_two_threshold(scores_by_q, t, t)
        # include required queries missing from scores as empty
        full = {k: pred.get(k, []) for k in truth}
        rows.append((t, t, macro_f05(truth, full)))
    return sorted(rows, key=lambda r: -r[2])
