import numpy as np
import pandas as pd

from er.analyze_b0_decisions import apply_policy, prepare
from er.calibration import decide_two_threshold
from er.metrics import f05_single


def test_vectorized_policies_match_exact_metric_with_missing_candidates():
    truth = {'a': [], 'b': ['t1', 'missing'], 'c': ['unretrieved'], 'd': [], 'e': ['t5']}
    frame = pd.DataFrame([
        ('a', 'bad', .6, 0), ('b', 't1', .8, 1), ('b', 'bad2', .68, 0),
        ('e', 't5', .65, 1),
    ], columns=['query_id', 'target_id', 'probability', 'is_match'])
    records = {q: {'country': 'India' if q in ['a', 'b'] else 'US'} for q in truth}
    data = prepare(frame, truth, records)
    policy = {'global': {'ts': .7, 'tm': .67}, 'US': {'ts': .6, 'tm': .6}}
    scores, tp, count = apply_policy(data, policy)
    for i, q in enumerate(data['ids']):
        params = policy.get(records[q]['country'], policy['global'])
        rows = [(r.target_id, r.probability) for r in frame.itertuples() if r.query_id == q]
        pred = decide_two_threshold({q: rows}, params['ts'], params['tm'])[q]
        assert np.isclose(scores[i], f05_single(set(truth[q]), set(pred)))
        assert tp[i] == len(set(truth[q]) & set(pred))
        assert count[i] == len(pred)
    # A completely unretrieved positive must remain in the false-negative denominator.
    assert scores[data['ids'].index('c')] == 0
    assert scores[data['ids'].index('d')] == 1
