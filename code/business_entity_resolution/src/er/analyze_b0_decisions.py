"""Calibration-only B0 policy selection; separate screening and error diagnostics.

Run from the repository root with the src directory on PYTHONPATH:
python -m er.analyze_b0_decisions
Does not retrain models, change production thresholds, or access holdout labels.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from er.io import load_record_text_provenance

def exact_scores(g, tp, count):
    denom = g + 4 * count
    out = np.divide(5.0 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return np.where(g == 0, (count == 0).astype(float), out)


def prepare(frame, truth, records):
    ids = sorted(truth)
    index = {q: i for i, q in enumerate(ids)}
    assert set(frame.query_id) <= set(ids)
    assert not frame.duplicated(['query_id', 'target_id']).any()
    qi = frame.query_id.map(index).to_numpy(dtype=int)
    y = frame.is_match.to_numpy(dtype=int)
    expected = np.array([int(p in set(truth[q])) for q, p in zip(frame.query_id, frame.target_id)])
    assert np.array_equal(y, expected)
    prob = frame.probability.to_numpy()
    assert np.isfinite(prob).all()
    g = np.array([len(set(truth[q])) for q in ids])
    mx = np.full(len(ids), -np.inf)
    np.maximum.at(mx, qi, prob)
    return dict(ids=ids, qi=qi, y=y, prob=prob, g=g, mx=mx,
                country=np.array([records[q]['country'] for q in ids]),
                retrieved=np.bincount(qi, weights=y, minlength=len(ids)))


def counts(data, ts, tm):
    accept = (data['prob'] >= tm) & (data['mx'][data['qi']] >= ts)
    n = len(data['ids'])
    return (np.bincount(data['qi'][accept], weights=data['y'][accept], minlength=n),
            np.bincount(data['qi'][accept], minlength=n))


def select_policies(cal):
    grid = np.round(np.arange(0.05, 1.0, 0.005), 3)
    score_matrix = np.array([exact_scores(cal['g'], *counts(cal, t, t)) for t in grid])
    # At tm alone every query with any accepted pair passes; a larger ts can empty it.
    empty_scores = (cal['g'] == 0).astype(float)
    masks = {'global': np.ones(len(cal['ids']), dtype=bool)}
    masks.update({c: cal['country'] == c for c in sorted(set(cal['country']))})
    base = exact_scores(cal['g'], *counts(cal, .7, .7))
    chosen = {}
    for group, mask in masks.items():
        best = float(base[mask].mean())
        choice = {'ts': .7, 'tm': .7, 'calibration_f05': best}
        for ts in grid:
            passed = cal['mx'][mask] >= ts
            means = np.where(passed, score_matrix[:, mask], empty_scores[mask]).mean(axis=1)
            means[grid > ts] = -np.inf
            j = int(np.argmax(means))
            if means[j] > best + 1e-12:
                best = float(means[j])
                choice = {'ts': float(ts), 'tm': float(grid[j]), 'calibration_f05': best}
        chosen[group] = choice
    j = int(np.argmax(score_matrix.mean(axis=1)))
    return {
        'baseline': {'global': {'ts': .7, 'tm': .7}},
        'fine_global': {'global': {'ts': float(grid[j]), 'tm': float(grid[j])}},
        'fine_dual': {'global': chosen['global']},
        'country_dual': {**chosen},  # global fallback for countries without labels
    }


def apply_policy(data, policy):
    tp, count = counts(data, policy['global']['ts'], policy['global']['tm'])
    for country, params in policy.items():
        if country == 'global':
            continue
        mask = data['country'] == country
        ctp, cn = counts(data, params['ts'], params['tm'])
        tp[mask], count[mask] = ctp[mask], cn[mask]
    return exact_scores(data['g'], tp, count), tp, count


def main():
    root = Path('runs/parallel-v1/d1/b0_baseline')
    dest = Path('runs/parallel-v1/d1/decision_audit_v1')
    dest.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(root / 'b0_eval_features.joblib')
    records = load_record_text_provenance(root)['queries']
    data = {}
    for role, prefix, filename in [('calibration', 'calib', 'calibration_predictions.parquet'),
                                    ('screen', 'eval', 'screen_predictions.parquet')]:
        block = bundle[prefix + '_data']
        truth = block[prefix + '_truth_by_q']
        data[role] = prepare(pd.read_parquet(root / filename), truth, records)
    assert not set(data['calibration']['ids']) & set(data['screen']['ids'])
    # Lock all policies using calibration only, before looking at screening scores.
    policies = select_policies(data['calibration'])
    (dest / 'calibration_selected_policies.json').write_text(json.dumps(policies, indent=2))
    screen = data['screen']
    baseline, btp, bn = apply_policy(screen, policies['baseline'])
    assert abs(baseline.mean() - 0.9045857751216922) < 1e-10
    output = {'scope': 'fixed B0 probabilities; calibration-only selection; 2k screen is development',
              'policies': policies, 'results': {}, 'input_sha256': {}}
    per_query = pd.DataFrame({'query_id': screen['ids'], 'country': screen['country'],
                              'truth_count': screen['g'], 'retrieved_tp': screen['retrieved']})
    for name, policy in policies.items():
        score, tp, n = apply_policy(screen, policy)
        cal_score, _, _ = apply_policy(data['calibration'], policy)
        delta = score - baseline
        rng = np.random.default_rng(42)
        strata = [np.flatnonzero(screen['country'] == c) for c in sorted(set(screen['country']))]
        boot = np.array([np.concatenate([delta[rng.choice(s, len(s), replace=True)] for s in strata]).mean()
                         for _ in range(2000)])
        output['results'][name] = {
            'calibration_f05': float(cal_score.mean()), 'screen_f05': float(score.mean()),
            'delta_vs_b0': float(delta.mean()), 'paired_95_ci': np.quantile(boot, [.025, .975]).tolist(),
            'singleton_false_merges': int(np.sum((screen['g'] == 0) & (n > 0))),
            'false_empty_non_singletons': int(np.sum((screen['g'] > 0) & (n == 0))),
            'pair_precision': float(tp.sum()/max(n.sum(), 1)),
            'pair_recall': float(tp.sum()/max(screen['g'].sum(), 1)),
            'by_country': {c: float(score[screen['country'] == c].mean()) for c in sorted(set(screen['country']))},
        }
        per_query[name] = score
    g, retrieved = screen['g'], screen['retrieved']
    fp, fn = bn - btp, g - btp
    oracle = exact_scores(g, retrieved, retrieved)
    output['baseline_diagnostics'] = {
        'oracle': float(oracle.mean()), 'retrieval_loss': float(1-oracle.mean()),
        'matcher_loss': float(oracle.mean()-baseline.mean()),
        'query_error_categories': {
            'perfect': int(np.sum((fp == 0) & (fn == 0))),
            'fp_only': int(np.sum((fp > 0) & (fn == 0))),
            'fn_only': int(np.sum((fp == 0) & (fn > 0))),
            'fp_and_fn': int(np.sum((fp > 0) & (fn > 0))),
        },
        'loss_contributions': {
            'fp_only': float(np.sum((1-baseline)[(fp > 0) & (fn == 0)])/len(g)),
            'fn_only': float(np.sum((1-baseline)[(fp == 0) & (fn > 0)])/len(g)),
            'fp_and_fn': float(np.sum((1-baseline)[(fp > 0) & (fn > 0)])/len(g)),
        },
        'counterfactual_remove_all_fp_gain': float(exact_scores(g, btp, btp).mean()-baseline.mean()),
        'counterfactual_add_all_retrieved_fn_gain': float(exact_scores(g, retrieved, bn+retrieved-btp).mean()-baseline.mean()),
        'counterfactual_caveat': 'Answer-key diagnostics only; not implementable policies; gains are not additive.',
    }
    for filename in ['b0_eval_features.joblib', 'calibration_predictions.parquet', 'screen_predictions.parquet']:
        output['input_sha256'][filename] = hashlib.sha256((root/filename).read_bytes()).hexdigest()
    per_query.to_parquet(dest/'per_query_scores.parquet', index=False)
    report = Path('reports/dev_probe/B0_decision_audit.json')
    report.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
