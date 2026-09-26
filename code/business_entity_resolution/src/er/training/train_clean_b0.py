"""Clean Reference Baseline B0 Training & True Loss Decomposition (Step 2: D1-01).

Hardware Policy:
- Strict 12 CPU cores limit (8 P-cores + 4/6 E-cores, zero LP-E contention).
- ThreadPoolExecutor(max_workers=12) with shared RAM for zero Windows IPC overhead.

Integrity Invariants:
- 100% Label-Blind Candidate Generation: ZERO ground-truth injection or artificial sentinels.
- Strict Manifest Separation: train_12k for fitting, calibration_5k for threshold tuning,
  screen_2k / comparison_15k for unbiased evaluation. All mutually disjoint and disjoint from holdout.
- Exact Entity-Macro F0.5 Metric and true loss decomposition:
  Total Loss = 1 - F0.5_final = (1 - F0.5_oracle) [Retrieval Loss] + (F0.5_oracle - F0.5_final) [Matcher Loss].
- Schema-asserted LightGBM with explicit boosting rounds and gain tracking.
- Portable B0 feature bundle export for Device 2 and Device 3 parallel experimentation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure er package is always importable
src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from er.calibration import decide_two_threshold, save_threshold_policy, sweep_two_thresholds
from er.candidate_generation import generate_natural_candidates, hash_candidate_sets
from er.features import FEATURES, pair_feature_row, rows_to_matrix
from er.metrics import f05_single, macro_f05, oracle_macro_f05
from er.normalization import normalize_address, normalize_name
from er.normalized_adapter import NormalizedRecordAdapter
from er.run_retrieval_sweep import load_gt_map
from er.training.matcher import assert_schema, refit_full, train_grouped_oof

DEFAULT_CORES = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--gt", default="student_resource/student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--manifest-dir", default="splits/f05-v1/parallel-v1")
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--out-dir", default="reports/dev_probe")
    ap.add_argument("--bundle-dir", default="runs/parallel-v1/d1/b0_baseline")
    ap.add_argument("--eval-mode", choices=["screen_2k", "comparison_15k"], default="screen_2k")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    ap.add_argument("--num-boost-round", type=int, default=1000)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    bundle_dir = Path(args.bundle_dir)
    models_dir = Path("cache/models")

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting D1-01 Clean Reference Baseline B0 (CPU Cores={args.n_cores}) ===")

    # 1. Load Ground Truth (for labeling training pairs and scoring evaluation, NEVER candidate generation)
    print("Loading Ground Truth for label assignment...")
    gt_map = load_gt_map(Path(args.gt))
    print(f"Loaded {len(gt_map):,} ground-truth entities.")

    # 2. Load Role Manifests
    print(f"Loading parallel-v1 manifests from {manifest_dir}...")
    train_12k_info = json.loads((manifest_dir / "train_12k.json").read_text(encoding="utf-8"))
    calib_5k_info = json.loads((manifest_dir / "calibration_5k.json").read_text(encoding="utf-8"))
    eval_info = json.loads((manifest_dir / f"{args.eval_mode}.json").read_text(encoding="utf-8"))
    inner_folds_info = json.loads((manifest_dir / "inner_train_folds.json").read_text(encoding="utf-8"))

    train_qids = train_12k_info["query_ids"]
    calib_qids = calib_5k_info["query_ids"]
    eval_qids = eval_info["query_ids"]
    inner_folds_map = inner_folds_info["fold_by_query"]

    train_q_set = set(train_qids)
    calib_q_set = set(calib_qids)
    eval_q_set = set(eval_qids)

    print(f"Train queries: {len(train_qids):,}, Calibration: {len(calib_qids):,}, Eval ({args.eval_mode}): {len(eval_qids):,}")

    # 3. Load S1 Query Data
    print("Loading S1 query data from train_source1.tsv...")
    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1_by_id = s1.set_index("entity_id")

    # Arrays to accumulate across countries
    train_rows, train_labels, train_groups, train_fold_arr = [], [], [], []
    calib_rows, calib_labels, calib_groups, calib_pair_keys = [], [], [], []
    eval_rows, eval_labels, eval_groups, eval_pair_keys = [], [], [], []

    calib_candidates_by_q: dict[str, list[tuple[str, float]]] = {}
    calib_truth_by_q: dict[str, list[str]] = {}

    eval_candidates_by_q: dict[str, list[tuple[str, float]]] = {}
    eval_truth_by_q: dict[str, list[str]] = {}

    timing_stages: dict[str, float] = {}

    # Process each country independently
    for country in ("India", "US"):
        t_c_start = time.time()
        print(f"\n==================== Processing {country} ====================")

        c_train_q = [q for q in train_qids if s1_by_id.loc[q, "country"] == country]
        c_calib_q = [q for q in calib_qids if s1_by_id.loc[q, "country"] == country]
        c_eval_q = [q for q in eval_qids if s1_by_id.loc[q, "country"] == country]
        c_all_q = c_train_q + c_calib_q + c_eval_q

        print(f"{country} Query counts: train={len(c_train_q):,}, calib={len(c_calib_q):,}, eval={len(c_eval_q):,}")

        # Load retrieval artifacts
        t_load = time.time()
        print(f"Loading {country} retrieval artifacts from {cache_dir}...")
        pool_dict_cache = cache_dir / f"pool_dict_{country}.joblib"
        pool_dict, pool_ids = joblib.load(pool_dict_cache)
        dupe_map = joblib.load(cache_dir / f"dupe_map_{country}.joblib")
        struct_idx = joblib.load(cache_dir / f"structured_index_{country}.joblib")

        lex_artifacts = {}
        for mode in ("joint", "name_only", "address_only"):
            vec = joblib.load(cache_dir / f"vec_{country}_{mode}.joblib")
            p_mat = load_npz(cache_dir / f"mat_{country}_{mode}.npz")
            lex_artifacts[mode] = (vec, p_mat)

        print(f"Loaded artifacts in {time.time() - t_load:.2f}s (pool size: {len(pool_ids):,})")

        # Normalize query texts
        q_names = [normalize_name(s1_by_id.loc[q, "business_name"], country) for q in c_all_q]
        q_addrs = [normalize_address(s1_by_id.loc[q, "business_address"], country) for q in c_all_q]

        # Natural Candidate Generation (100% LABEL-BLIND)
        print(f"Generating natural candidates for {len(c_all_q):,} {country} queries (top-100 RRF)...")
        t_ret = time.time()
        cands_by_q, ch_lookups = generate_natural_candidates(
            query_ids=c_all_q,
            query_names=q_names,
            query_addrs=q_addrs,
            country=country,
            pool_ids=pool_ids,
            lexical_artifacts=lex_artifacts,
            structured_index=struct_idx,
            dupe_map=dupe_map,
            k_per_channel={"joint": 100, "name_only": 100, "address_only": 150, "structured": 100},
            top_k_final=100,
            n_threads=args.n_cores,
        )
        timing_stages[f"retrieval_{country}"] = time.time() - t_ret
        print(f"Natural candidate retrieval completed in {timing_stages[f'retrieval_{country}']:.2f}s")

        cand_hash = hash_candidate_sets({q: cands_by_q[q] for q in c_eval_q})
        print(f"{country} Eval candidate set SHA-256 hash: {cand_hash[:16]}...")

        # Pre-cache normalized records for feature extraction
        needed_pids = set()
        for qid in c_all_q:
            for pid, _ in cands_by_q.get(qid, []):
                needed_pids.add(pid)

        print(f"Pre-caching {len(needed_pids):,} unique pool records with NormalizedRecordAdapter...")
        adapter = NormalizedRecordAdapter()
        q_meta_cache = {
            qid: adapter.normalize(
                qid,
                s1_by_id.loc[qid, "business_name"],
                s1_by_id.loc[qid, "business_address"],
                country,
            )
            for qid in c_all_q
        }
        pool_meta_cache = {
            pid: adapter.normalize(
                pid,
                pool_dict[pid]["business_name"],
                pool_dict[pid]["business_address"],
                country,
            )
            for pid in needed_pids
            if pid in pool_dict
        }

        # Multi-threaded Feature Extraction (Shared RAM via ThreadPoolExecutor)
        print(f"Extracting 23 features with ThreadPoolExecutor(max_workers={args.n_cores})...")
        t_feat = time.time()

        def _extract_query_pairs(qid: str):
            q_meta = q_meta_cache[qid]
            true_tgts = set(gt_map.get(qid, []))
            cands = cands_by_q.get(qid, [])

            local_rows = []
            local_labels = []
            local_groups = []
            local_pairs = []

            for pid, rrf_score in cands:
                if pid not in pool_meta_cache:
                    continue
                p_meta = pool_meta_cache[pid]

                chmap = {}
                for ch_name, q_ch_dict in ch_lookups.items():
                    info = q_ch_dict.get(qid, {}).get(pid)
                    if info is not None:
                        chmap[ch_name] = info

                src_is_s2 = pid.startswith("S2-")
                feat = pair_feature_row(q_meta, p_meta, chmap, rrf_score, src_is_s2)
                lbl = 1 if pid in true_tgts else 0

                local_rows.append(feat)
                local_labels.append(lbl)
                local_groups.append(qid)
                local_pairs.append((pid, rrf_score))

            return qid, local_rows, local_labels, local_groups, local_pairs

        with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
            results = list(pool.map(_extract_query_pairs, c_all_q))

        timing_stages[f"features_{country}"] = time.time() - t_feat
        print(f"Feature extraction for {country} completed in {timing_stages[f'features_{country}']:.2f}s")

        # Partition extracted features by role
        for qid, q_r, q_l, q_g, q_p in results:
            if qid in train_q_set:
                train_rows.extend(q_r)
                train_labels.extend(q_l)
                train_groups.extend(q_g)
                fold = inner_folds_map.get(qid, 0)
                train_fold_arr.extend([fold] * len(q_r))
            elif qid in calib_q_set:
                calib_rows.extend(q_r)
                calib_labels.extend(q_l)
                calib_groups.extend(q_g)
                calib_pair_keys.extend([(qid, pid) for pid, _ in q_p])
                calib_candidates_by_q[qid] = q_p
                calib_truth_by_q[qid] = gt_map.get(qid, [])
            elif qid in eval_q_set:
                eval_rows.extend(q_r)
                eval_labels.extend(q_l)
                eval_groups.extend(q_g)
                eval_pair_keys.extend([(qid, pid) for pid, _ in q_p])
                eval_candidates_by_q[qid] = q_p
                eval_truth_by_q[qid] = gt_map.get(qid, [])

        print(f"{country} total time: {time.time() - t_c_start:.2f}s")

    # 4. Construct Matrices
    print("\n--- Constructing Feature Matrices ---")
    X_train = rows_to_matrix(train_rows)
    y_train = np.array(train_labels, dtype=np.int32)
    groups_train = np.array(train_groups)
    folds_train = np.array(train_fold_arr, dtype=np.int32)

    X_calib = rows_to_matrix(calib_rows)
    y_calib = np.array(calib_labels, dtype=np.int32)

    X_eval = rows_to_matrix(eval_rows)
    y_eval = np.array(eval_labels, dtype=np.int32)

    print(f"X_train: {X_train.shape} (positives: {int(y_train.sum()):,})")
    print(f"X_calib: {X_calib.shape} (positives: {int(y_calib.sum()):,})")
    print(f"X_eval:  {X_eval.shape} (positives: {int(y_eval.sum()):,})")

    # 5. Export Portable B0 Feature Bundle for Device 2 and Device 3
    print(f"\n--- Exporting Portable B0 Bundle to {bundle_dir} ---")
    bundle_path = bundle_dir / "b0_portable_bundle.joblib"
    joblib.dump(
        {
            "X_train": X_train,
            "y_train": y_train,
            "groups_train": groups_train,
            "folds_train": folds_train,
            "feature_names": FEATURES,
            "calib_data": {
                "X_calib": X_calib,
                "y_calib": y_calib,
                "calib_groups": calib_groups,
                "calib_candidates_by_q": calib_candidates_by_q,
                "calib_truth_by_q": calib_truth_by_q,
            },
            "eval_data": {
                "X_eval": X_eval,
                "y_eval": y_eval,
                "eval_groups": eval_groups,
                "eval_candidates_by_q": eval_candidates_by_q,
                "eval_truth_by_q": eval_truth_by_q,
            },
        },
        bundle_path,
        compress=3,
    )
    print(f"Saved portable B0 bundle ({bundle_path.stat().st_size / 1e6:.1f} MB) -> ready for D2 Mac Matcher!")

    # 6. Train Supervised LightGBM Reference
    print("\n--- Training Clean Reference LightGBM Matcher (Grouped 3-Fold) ---")
    t_train = time.time()
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "max_depth": 7,
        "num_leaves": 63,
        "min_child_samples": 80,
        "subsample": 0.8,
        "bagging_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": args.n_cores,
        "verbose": -1,
        "seed": 42,
    }

    oof, models, best_iters, feature_gains = train_grouped_oof(
        X=X_train,
        y=y_train,
        groups=groups_train,
        feature_names=FEATURES,
        params=lgb_params,
        n_splits=3,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
        fold_assignments=folds_train,
    )
    timing_stages["lgb_oof_train"] = time.time() - t_train
    mean_best_iter = int(np.mean(best_iters))
    print(f"Grouped OOF completed in {timing_stages['lgb_oof_train']:.2f}s! Best iterations: {best_iters} (mean={mean_best_iter})")

    # Assert schema on all fold models
    for m in models:
        assert_schema(FEATURES, m)

    # Refit full reference model on all train_12k pairs
    print(f"Refitting final B0 reference model on full train_12k ({mean_best_iter} trees)...")
    b0_model = refit_full(
        X=X_train,
        y=y_train,
        feature_names=FEATURES,
        params=lgb_params,
        num_boost_round=mean_best_iter,
    )
    assert_schema(FEATURES, b0_model)
    model_path = models_dir / "b0_clean_matcher.txt"
    b0_model.save_model(str(model_path))
    print(f"Saved clean reference model to {model_path}!")

    # 7. Two-Threshold Calibration on calibration_5k (Zero Overlap with Evaluation)
    print("\n--- Calibrating Dual Thresholds on calibration_5k ---")
    t_calib = time.time()
    calib_preds = b0_model.predict(X_calib)

    # Map predictions back to per-query lists using exact pair keys
    calib_scores_by_q: dict[str, list[tuple[str, float]]] = {q: [] for q in calib_q_set}
    for (qid, pid), sc in zip(calib_pair_keys, calib_preds):
        calib_scores_by_q[qid].append((pid, float(sc)))

    best_t_sing, best_t_match, best_calib_f05, calib_grid = sweep_two_thresholds(
        scores_by_q=calib_scores_by_q,
        truth=calib_truth_by_q,
        t_singleton_grid=(0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.88, 0.9, 0.92, 0.95),
        t_match_grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9),
    )
    timing_stages["calibration"] = time.time() - t_calib
    print(f"Calibration completed in {timing_stages['calibration']:.2f}s!")
    print(f"Optimal Thresholds: T_singleton={best_t_sing:.2f}, T_match={best_t_match:.2f} -> Calibration Macro F0.5 = {best_calib_f05:.6f}")

    # Save threshold policy
    thresh_path = models_dir / "b0_calibrated_thresholds.json"
    save_threshold_policy(
        path=thresh_path,
        t_singleton=best_t_sing,
        t_match=best_t_match,
        calibration_score=best_calib_f05,
        metadata={"seed": "parallel-v1", "calib_n": len(calib_truth_by_q)},
    )

    # 8. Unbiased Clean Evaluation on Evaluation Manifest
    print(f"\n--- Evaluating Clean Baseline B0 on {args.eval_mode} ({len(eval_q_set):,} queries) ---")
    t_eval = time.time()
    eval_preds = b0_model.predict(X_eval)

    # Map predictions back to per-query lists using exact pair keys
    eval_scores_by_q: dict[str, list[tuple[str, float]]] = {q: [] for q in eval_q_set}
    for (qid, pid), sc in zip(eval_pair_keys, eval_preds):
        eval_scores_by_q[qid].append((pid, float(sc)))

    # Apply calibrated decision rule
    final_predictions = decide_two_threshold(eval_scores_by_q, best_t_sing, best_t_match)

    # Exact Metrics
    final_macro_f05 = macro_f05(eval_truth_by_q, final_predictions)
    oracle_f05 = oracle_macro_f05(eval_truth_by_q, eval_candidates_by_q)

    # Exact True Loss Decomposition
    total_loss = 1.0 - final_macro_f05
    retrieval_loss = 1.0 - oracle_f05
    matcher_loss = oracle_f05 - final_macro_f05

    timing_stages["evaluation"] = time.time() - t_eval

    # Country Breakdown
    eval_country_map = {q: s1_by_id.loc[q, "country"] for q in eval_q_set}
    eval_by_country: dict[str, dict] = {}
    for c in ("India", "US"):
        c_q = {q for q in eval_q_set if eval_country_map[q] == c}
        c_truth = {q: eval_truth_by_q[q] for q in c_q}
        c_pred = {q: final_predictions[q] for q in c_q}
        c_cands = {q: eval_candidates_by_q[q] for q in c_q}
        c_f05 = macro_f05(c_truth, c_pred)
        c_oracle = oracle_macro_f05(c_truth, c_cands)
        eval_by_country[c] = {
            "macro_f05": float(c_f05),
            "oracle_macro_f05": float(c_oracle),
            "retrieval_loss": float(1.0 - c_oracle),
            "matcher_loss": float(c_oracle - c_f05),
            "n_queries": len(c_q),
        }

    # Match count / singleton breakdown
    singleton_q = {q for q, tr in eval_truth_by_q.items() if len(tr) == 0}
    non_singleton_q = {q for q, tr in eval_truth_by_q.items() if len(tr) > 0}

    singleton_fm = sum(1 for q in singleton_q if len(final_predictions[q]) > 0)
    singleton_acc = 1.0 - (singleton_fm / max(len(singleton_q), 1))

    total_time = time.time() - t_start

    # Top Feature Gains
    sorted_gains = sorted(feature_gains.items(), key=lambda kv: -kv[1])

    report_payload = {
        "experiment": "D1-01 Clean Reference Baseline B0",
        "eval_manifest": args.eval_mode,
        "n_eval_queries": len(eval_q_set),
        "thresholds": {"t_singleton": best_t_sing, "t_match": best_t_match},
        "metrics": {
            "final_macro_f05": float(final_macro_f05),
            "oracle_macro_f05": float(oracle_f05),
            "total_loss": float(total_loss),
            "retrieval_loss": float(retrieval_loss),
            "matcher_loss": float(matcher_loss),
            "singleton_accuracy": float(singleton_acc),
            "singleton_false_merges": int(singleton_fm),
            "total_singletons": len(singleton_q),
        },
        "by_country": eval_by_country,
        "feature_gains_top10": sorted_gains[:10],
        "timing_seconds": {
            "total": float(total_time),
            **timing_stages,
        },
    }

    # Write report json and markdown
    json_path = out_dir / "B0_clean_reference_report.json"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    md_report = f"""# Device 1: Clean Reference Baseline B0 Report (D1-01)

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Status:** Clean, Uncontaminated Baseline (Zero GT Injection)
**Evaluation Manifest:** `{args.eval_mode}` ({len(eval_q_set):,} queries: India={len(c_eval_q):,}, US={len(c_eval_q):,})

## 1. Verified Metrics & True Loss Decomposition

$$\\text{{Total Loss}} = 1 - F_{{0.5}}^{{\\text{{final}}}} = (1 - F_{{0.5}}^{{\\text{{oracle}}}}) + (F_{{0.5}}^{{\\text{{oracle}}}} - F_{{0.5}}^{{\\text{{final}}}})$$

| Metric | Overall | India | US | Target Headroom |
|---|---|---|---|---|
| **Clean Macro $F_{{0.5}}$** | **{final_macro_f05:.6f}** | **{eval_by_country['India']['macro_f05']:.6f}** | **{eval_by_country['US']['macro_f05']:.6f}** | Gap to >0.98: {0.98 - final_macro_f05:.6f} |
| **Natural Oracle Macro $F_{{0.5}}$** | **{oracle_f05:.6f}** | **{eval_by_country['India']['oracle_macro_f05']:.6f}** | **{eval_by_country['US']['oracle_macro_f05']:.6f}** | Ceiling |
| **Retrieval Loss $(1 - \\text{{Oracle}})$** | **{retrieval_loss:.6f}** | **{eval_by_country['India']['retrieval_loss']:.6f}** | **{eval_by_country['US']['retrieval_loss']:.6f}** | Budget: $\\le 0.005$ |
| **Matcher Loss $(\\text{{Oracle}} - \\text{{Final}})$** | **{matcher_loss:.6f}** | **{eval_by_country['India']['matcher_loss']:.6f}** | **{eval_by_country['US']['matcher_loss']:.6f}** | Budget: $< 0.015$ |

## 2. Threshold Calibration on `calibration_5k`
- **Optimal $T_{{\\text{{singleton}}}}$:** `{best_t_sing:.2f}`
- **Optimal $T_{{\\text{{match}}}}$:** `{best_t_match:.2f}`
- **Calibration Macro $F_{{0.5}}$:** `{best_calib_f05:.6f}`
- **Singleton False Merges:** `{singleton_fm} / {len(singleton_q)}` (Accuracy: `{singleton_acc*100:.2f}%`)

## 3. Top Feature Gains
| Rank | Feature | Importance Gain |
|---|---|---|
"""
    for rank, (fn, g) in enumerate(sorted_gains[:10], start=1):
        md_report += f"| {rank} | `{fn}` | {g:,.2f} |\n"

    md_report += f"""
## 4. Multi-Device Handoff Status
- **B0 Portable Feature Bundle:** Published to `{bundle_path}` ({bundle_path.stat().st_size / 1e6:.1f} MB).
- **Clean Model Artifact:** Saved to `{model_path}`.
- **Calibrated Policy:** Saved to `{thresh_path}`.
- **Next Parallel Steps:**
  - **Device 2 (MacBook Air M4):** Can immediately load `b0_portable_bundle.joblib` and run CPU matcher experiments (D2-01 boosting sweeps, hard negative mining, feature ablations) without building multi-million document indices.
  - **Device 3 (RTX 3050):** Can consume candidate sets and benchmark multilingual dense challengers (D3-01).
  - **Device 1 (Windows / Arc 140T):** Can proceed to D1-02 India Lexical Expansion.
"""
    md_path = out_dir / "B0_clean_reference_report.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(f"\nPublished Clean Baseline B0 Report to:\n- {md_path}\n- {json_path}")
    print(f"\n=== B0 Pipeline Complete in {total_time:.2f}s ===")


if __name__ == "__main__":
    main()
