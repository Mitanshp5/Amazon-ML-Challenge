"""Automated contract verification test suite (Step 1: D1-00).

Tests:
1. Label-blind candidate generation: API takes no GT, hashes match deterministically.
2. Exact entity-macro F0.5 metric: singletons, false merges, and non-empty matches.
3. Strict manifest disjointness: train_12k, calibration_5k, comparison_15k, holdout, dev-exposed.
4. Schema assertion: 23 features, exact names and order, no Column_0..N fallback.
5. Normalization adapter integration: name_core, house/unit/postal numbers extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from er.candidate_generation import (
    compute_rrf_rankings,
    expand_duplicate_candidates,
    hash_candidate_sets,
)
from er.features import FEATURES, pair_feature_row, rows_to_matrix
from er.metrics import f05_single, macro_f05, oracle_macro_f05
from er.normalized_adapter import NormalizedRecordAdapter, get_normalized_record
from er.training.matcher import assert_schema, train_grouped_oof


def test_label_invariance_and_hash():
    """Candidate generation must produce identical deterministic results."""
    # Synthetic channels
    ch_results = {
        "joint": (
            {"Q1": ["T1", "T2"], "Q2": ["T3"]},
            {"Q1": [0.9, 0.8], "Q2": [0.7]},
        ),
        "name_only": (
            {"Q1": ["T2", "T4"], "Q2": ["T3"]},
            {"Q1": [0.85, 0.6], "Q2": [0.65]},
        ),
    }
    rrf1 = compute_rrf_rankings(ch_results)
    rrf2 = compute_rrf_rankings(ch_results)

    hash1 = hash_candidate_sets(rrf1)
    hash2 = hash_candidate_sets(rrf2)

    assert hash1 == hash2, "Candidate hash is not deterministic!"
    assert "Q1" in rrf1 and "Q2" in rrf1

    # Verify no sentinel 0.001 injection
    for qid, pairs in rrf1.items():
        for pid, sc in pairs:
            assert sc > 0.005, f"Suspiciously low sentinel score {sc} for {pid}"


def test_duplicate_expansion_no_gt():
    """Duplicate expansion expands pool clones without touching ground truth."""
    rrf = {"Q1": [("T1", 0.03)]}
    dupe_map = {"T1": ["T1_clone"]}
    expanded = expand_duplicate_candidates(rrf, dupe_map)
    cands = [p for p, _ in expanded["Q1"]]
    assert "T1" in cands and "T1_clone" in cands
    assert len(cands) == 2


def test_exact_macro_f05_contract():
    """Verify exact challenge scoring contract (PDF p.6)."""
    # 1. Singleton correctly predicted empty -> 1.0
    assert f05_single(set(), set()) == 1.0

    # 2. Singleton falsely predicted non-empty -> 0.0
    assert f05_single(set(), {"T1"}) == 0.0

    # 3. Non-singleton predicted empty -> 0.0
    assert f05_single({"T1"}, set()) == 0.0

    # 4. Partial match: 1 TP out of 1 GT, 1 FP (2 preds)
    # formula: 5*TP / (|G| + 4*|P|) = 5*1 / (1 + 4*2) = 5/9
    assert pytest.approx(f05_single({"T1"}, {"T1", "T2"})) == 5.0 / 9.0

    # 5. Perfect match: 2 TP out of 2 GT, 0 FP (2 preds)
    # 5*2 / (2 + 4*2) = 10/10 = 1.0
    assert f05_single({"T1", "T2"}, {"T1", "T2"}) == 1.0

    # Macro average over diverse entities
    truth = {"Q1": ["T1"], "Q2": [], "Q3": ["T2", "T3"]}
    pred_perfect = {"Q1": ["T1"], "Q2": [], "Q3": ["T2", "T3"]}
    assert macro_f05(truth, pred_perfect) == 1.0

    # False merge on singleton hurts overall score
    pred_with_fm = {"Q1": ["T1"], "Q2": ["T99"], "Q3": ["T2", "T3"]}
    assert pytest.approx(macro_f05(truth, pred_with_fm)) == 2.0 / 3.0


def test_oracle_macro_f05():
    """Oracle calculation computes exact best possible score within candidates."""
    truth = {"Q1": ["T1", "T2"], "Q2": []}
    # Only T1 was retrieved for Q1
    candidates = {"Q1": ["T1", "T99"], "Q2": []}
    score = oracle_macro_f05(truth, candidates)
    # Q1: 5*1 / (2 + 4*1) = 5/6, Q2: 1.0 -> avg = (5/6 + 1)/2 = 11/12
    assert pytest.approx(score) == (5.0 / 6.0 + 1.0) / 2.0


def test_manifest_disjointness():
    """Verify that parallel-v1 manifests are strictly disjoint and respect holdout."""
    manifest_dir = Path("splits/f05-v1/parallel-v1")
    if not (manifest_dir / "train_12k.json").exists():
        pytest.skip("Manifests not yet generated")

    train_12k = set(json.loads((manifest_dir / "train_12k.json").read_text(encoding="utf-8"))["query_ids"])
    calib_5k = set(json.loads((manifest_dir / "calibration_5k.json").read_text(encoding="utf-8"))["query_ids"])
    comp_15k = set(json.loads((manifest_dir / "comparison_15k.json").read_text(encoding="utf-8"))["query_ids"])
    screen_2k = set(json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))["query_ids"])

    # Disjointness
    assert len(train_12k & calib_5k) == 0, "train_12k and calib_5k overlap!"
    assert len(train_12k & comp_15k) == 0, "train_12k and comp_15k overlap!"
    assert len(calib_5k & comp_15k) == 0, "calib_5k and comp_15k overlap!"
    assert screen_2k.issubset(comp_15k), "screen_2k is not a subset of comp_15k!"

    # Verify against splits.json holdout and dev-exposed
    with open("splits/f05-v1/splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    for qid in train_12k:
        assert splits[qid] == "train", f"{qid} in train_12k is not from train split!"
    for qid in calib_5k:
        assert splits[qid] == "dev", f"{qid} in calib_5k is not from dev split!"
    for qid in comp_15k:
        assert splits[qid] == "dev", f"{qid} in comp_15k is not from dev split!"


def test_normalized_adapter_and_features():
    """Verify normalization adapter passes real components to feature matrix."""
    adapter = NormalizedRecordAdapter()
    q = adapter.normalize("Q1", "Starbucks Coffee LLC", "123 Main St Ste 400", "US")
    t = adapter.normalize("T1", "Starbucks Coffee", "123 Main St Suite 400", "US")

    assert q["name_core"] == "starbucks coffee", f"Unexpected name_core: {q['name_core']}"
    assert "123" in q["numbers"]["house_tokens"]
    assert len(q["numbers"]["unit_tokens"]) > 0 or "400" in q["numbers"]["house_tokens"]

    chmap = {"joint": (0.95, 1)}
    row = pair_feature_row(q, t, chmap, rrf=0.03, src_is_s2=True)
    assert len(row) == len(FEATURES), f"Row length {len(row)} != {len(FEATURES)}"


def test_lightgbm_schema_assertion():
    """Verify LightGBM trains and asserts schema with explicit feature names."""
    rng = np.random.RandomState(42)
    n_samples = 100
    n_feats = len(FEATURES)
    X = rng.randn(n_samples, n_feats).astype(np.float32)
    y = (rng.rand(n_samples) > 0.5).astype(np.int32)
    groups = np.repeat(np.arange(10), 10)

    params = {"objective": "binary", "learning_rate": 0.1, "verbosity": -1}
    oof, models, iters, gains = train_grouped_oof(
        X, y, groups, FEATURES, params, n_splits=2, num_boost_round=10, early_stopping_rounds=5
    )

    assert len(models) == 2
    for m in models:
        # Schema assertion must pass
        assert_schema(FEATURES, m)
        assert list(m.feature_name()) == list(FEATURES)
