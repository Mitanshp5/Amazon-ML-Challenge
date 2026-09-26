"""Exact per-S1 macro F0.5 scorer (challenge contract, PDF p.6).

Formula per entity i with truth G_i and prediction P_i:
  F_i = 5*TP / (|G| + 4*|P|)  for non-empty truth
  G empty, P empty -> 1.0
  G empty, P nonempty -> 0.0
Overall = mean over ALL required S1 (singletons included).

Do NOT substitute sklearn fbeta / micro averages here.
"""
from __future__ import annotations


def f05_single(y_true: set, y_pred: set) -> float:
    y_true = set(y_true)
    y_pred = set(y_pred)
    if not y_true:
        return 1.0 if not y_pred else 0.0
    tp = len(y_true & y_pred)
    if not y_pred:
        return 0.0
    # 5*TP / (|G| + 4*|P|)
    return 5.0 * tp / (len(y_true) + 4.0 * len(y_pred))


def macro_f05(truth: dict, pred: dict) -> float:
    if not truth:
        raise ValueError("truth manifest is empty")
    return sum(f05_single(set(truth[k]), set(pred.get(k, ()))) for k in truth) / len(truth)


def oracle_predictions(truth_sets: dict, candidate_sets: dict) -> dict:
    """Best achievable predictions restricted to candidates: G cap C."""
    return {qid: set(truth_sets.get(qid, ())) & set(candidate_sets.get(qid, ())) for qid in truth_sets}


def oracle_macro_f05(truth_sets: dict, candidate_sets: dict) -> float:
    return macro_f05(truth_sets, oracle_predictions(truth_sets, candidate_sets))
