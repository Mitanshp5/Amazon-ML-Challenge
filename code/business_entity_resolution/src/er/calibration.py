"""Threshold/decision surface on unsampled natural candidate lists (Step 1: D1-00 & F05 Phase F).

Provides:
- decide_two_threshold(scores_by_q, t_singleton, t_match): applies dual thresholds
  where t_singleton governs non-empty prediction, and t_match filters individual candidates.
- sweep_two_thresholds(scores_by_q, truth, ...): 2D grid search maximizing exact entity macro F0.5.
- save_threshold_policy(path, ...): exports calibrated decision parameters to JSON artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from er.metrics import macro_f05


def decide_two_threshold(
    scores_by_q: Mapping[str, Sequence[tuple[str, float]]],
    t_singleton: float,
    t_match: float,
) -> dict[str, list[str]]:
    """Predict matches for each query using dual thresholds.

    If candidate list is empty or max candidate score < t_singleton, predict empty list (singleton).
    Otherwise, predict all candidates with score >= t_match.
    """
    out: dict[str, list[str]] = {}
    for qid, pairs in scores_by_q.items():
        if not pairs:
            out[qid] = []
            continue
        max_score = max(s for _, s in pairs)
        if max_score < t_singleton:
            out[qid] = []
        else:
            out[qid] = [pid for pid, s in pairs if s >= t_match]
    return out


def sweep_thresholds(
    scores_by_q: Mapping[str, Sequence[tuple[str, float]]],
    truth: Mapping[str, Sequence[str]],
    grid: Sequence[float] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
) -> list[tuple[float, float, float]]:
    """Diagonal 1D sweep: t_singleton == t_match."""
    rows = []
    for t in grid:
        pred = decide_two_threshold(scores_by_q, t, t)
        full = {k: pred.get(k, []) for k in truth}
        score = macro_f05(truth, full)
        rows.append((t, t, score))
    return sorted(rows, key=lambda r: -r[2])


def sweep_two_thresholds(
    scores_by_q: Mapping[str, Sequence[tuple[str, float]]],
    truth: Mapping[str, Sequence[str]],
    t_singleton_grid: Sequence[float] = (0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.88, 0.9, 0.92, 0.95),
    t_match_grid: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9),
) -> tuple[float, float, float, list[dict]]:
    """Full 2D grid sweep optimizing exact macro F0.5.

    Returns:
        best_t_singleton, best_t_match, best_score, all_results_sorted
    """
    results = []
    best_t_sing = 0.85
    best_t_match = 0.85
    best_score = -1.0

    for t_sing in t_singleton_grid:
        for t_m in t_match_grid:
            if t_m > t_sing:
                continue
            pred = decide_two_threshold(scores_by_q, t_sing, t_m)
            full = {k: pred.get(k, []) for k in truth}
            f05 = macro_f05(truth, full)
            entry = {"t_singleton": float(t_sing), "t_match": float(t_m), "macro_f05": float(f05)}
            results.append(entry)
            if f05 > best_score:
                best_score = f05
                best_t_sing = t_sing
                best_t_match = t_m

    results.sort(key=lambda x: -x["macro_f05"])
    return best_t_sing, best_t_match, best_score, results


def save_threshold_policy(
    path: Path,
    t_singleton: float,
    t_match: float,
    calibration_score: float,
    metadata: dict | None = None,
) -> None:
    """Save calibrated threshold policy to JSON."""
    payload = {
        "t_singleton": float(t_singleton),
        "t_match": float(t_match),
        "calibration_macro_f05": float(calibration_score),
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
