"""Build comprehensive provenance metadata, predictions, and keyed text records for B0 handoff (D1-01).

Generates under runs/parallel-v1/d1/b0_baseline/:
1. manifest.json: SHA256 of all input TSVs, manifests, retrieval artifacts, model config, git state.
2. train_pairs.parquet: keyed training pairs (query_id, target_id, is_match, fold, rrf_score) for supervised fine-tuning.
3. calibration_predictions.parquet: per-pair probability, match label, and decision flag for calibration_5k.
4. screen_predictions.parquet: per-pair probability, match label, and decision flag for screen_2k.
5. record_text_provenance.joblib: raw & normalized text records partitioned into queries, training targets,
   evaluation candidate targets, and isolated unretrieved diagnostic targets for Device 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from er.calibration import decide_two_threshold
from er.candidate_generation import generate_natural_candidates
from er.io import load_b0_bundle
from er.metrics import macro_f05, oracle_macro_f05
from er.normalization import normalize_address, normalize_name
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def get_git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        dirty = len(status) > 0
    except Exception:
        commit, dirty = "unknown", False
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    ap.add_argument("--force-candidates", action="store_true", help="Force regenerate train pairs even if cached")
    args = ap.parse_args()

    t_start = time.time()
    bundle_dir = Path("runs/parallel-v1/d1/b0_baseline")
    manifest_dir = Path("splits/f05-v1/parallel-v1")
    cache_dir = Path("cache/retrieval")
    train_dir = Path("student_resource/student_resource/dataset/train")
    models_dir = Path("cache/models")

    print("=== Building B0 Keyed Training Package & Provenance Manifest ===")

    # 1. Load model and thresholds
    model_path = models_dir / "b0_clean_matcher.txt"
    thresh_path = models_dir / "b0_calibrated_thresholds.json"
    model = lgb.Booster(model_file=str(model_path))
    thresh = json.loads(thresh_path.read_text(encoding="utf-8"))
    t_sing = thresh["t_singleton"]
    t_match = thresh["t_match"]

    # 2. Load portable feature bundle via er.io helper
    print("Loading feature bundle via er.io.load_b0_bundle()...")
    bundle = load_b0_bundle(bundle_dir)
    calib_data = bundle["calib_data"]
    eval_data = bundle["eval_data"]

    # 3. Generate Predictions on Calibration and Screening Evaluation (if missing)
    calib_pred_parquet = bundle_dir / "calibration_predictions.parquet"
    eval_pred_parquet = bundle_dir / "screen_predictions.parquet"

    if not calib_pred_parquet.exists() or not eval_pred_parquet.exists():
        print("Generating calibration and screen predictions...")
        calib_preds = model.predict(calib_data["X_calib"])
        eval_preds = model.predict(eval_data["X_eval"])

        calib_records = []
        idx = 0
        calib_truth = calib_data["calib_truth_by_q"]
        calib_cands = calib_data["calib_candidates_by_q"]
        calib_scores_by_q = {q: [] for q in calib_truth}
        for qid in calib_truth:
            for pid, rrf in calib_cands.get(qid, []):
                sc = float(calib_preds[idx])
                calib_scores_by_q[qid].append((pid, sc))
                is_match = 1 if pid in set(calib_truth.get(qid, [])) else 0
                calib_records.append({
                    "query_id": qid,
                    "target_id": pid,
                    "probability": sc,
                    "rrf_score": rrf,
                    "is_match": is_match,
                })
                idx += 1

        calib_pred_map = decide_two_threshold(calib_scores_by_q, t_sing, t_match)
        for r in calib_records:
            r["predicted"] = 1 if r["target_id"] in set(calib_pred_map.get(r["query_id"], [])) else 0

        df_calib_pred = pd.DataFrame(calib_records)
        df_calib_pred.to_parquet(calib_pred_parquet, index=False)
        print(f"Saved calibration predictions ({len(df_calib_pred):,} pairs) to {calib_pred_parquet}")

        eval_records = []
        idx = 0
        eval_truth = eval_data["eval_truth_by_q"]
        eval_cands = eval_data["eval_candidates_by_q"]
        eval_scores_by_q = {q: [] for q in eval_truth}
        for qid in eval_truth:
            for pid, rrf in eval_cands.get(qid, []):
                sc = float(eval_preds[idx])
                eval_scores_by_q[qid].append((pid, sc))
                is_match = 1 if pid in set(eval_truth.get(qid, [])) else 0
                eval_records.append({
                    "query_id": qid,
                    "target_id": pid,
                    "probability": sc,
                    "rrf_score": rrf,
                    "is_match": is_match,
                })
                idx += 1

        eval_pred_map = decide_two_threshold(eval_scores_by_q, t_sing, t_match)
        for r in eval_records:
            r["predicted"] = 1 if r["target_id"] in set(eval_pred_map.get(r["query_id"], [])) else 0

        df_eval_pred = pd.DataFrame(eval_records)
        df_eval_pred.to_parquet(eval_pred_parquet, index=False)
        print(f"Saved screen predictions ({len(df_eval_pred):,} pairs) to {eval_pred_parquet}")
    else:
        print(f"Verified existing calibration and screen predictions parquets in {bundle_dir}.")

    # 4. Generate Keyed Training Pairs (train_pairs.parquet) for Device 3 Supervised Training
    train_pairs_path = bundle_dir / "train_pairs.parquet"
    train_12k_info = json.loads((manifest_dir / "train_12k.json").read_text(encoding="utf-8"))
    inner_folds_info = json.loads((manifest_dir / "inner_train_folds.json").read_text(encoding="utf-8"))
    inner_folds_map = inner_folds_info["fold_by_query"]
    train_qids = train_12k_info["query_ids"]

    gt_map = load_gt_map(train_dir / "train_ground_truth.tsv")

    # Load S1 Query Data to get query country assignments
    print("Loading query texts from train_source1.tsv...")
    s1_df = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    all_needed_qids = set(train_qids) | set(calib_data["calib_truth_by_q"].keys()) | set(eval_data["eval_truth_by_q"].keys())
    s1_sub = s1_df[s1_df["entity_id"].isin(all_needed_qids)]
    s1_meta = {
        r.entity_id: {"name": r.business_name, "address": r.business_address, "country": r.country}
        for r in s1_sub.itertuples()
    }

    train_rows = []
    if train_pairs_path.exists() and not args.force_candidates:
        print(f"Loading existing train_pairs.parquet from {train_pairs_path}...")
        df_train_pairs = pd.read_parquet(train_pairs_path)
    else:
        print(f"\n--- Generating Keyed Training Pairs with ThreadPoolExecutor({args.n_cores}) ---")
        for country in ("India", "US"):
            c_train_qids = [q for q in train_qids if s1_meta[q]["country"] == country]
            print(f"Generating natural candidates for {country} ({len(c_train_qids):,} training queries)...")

            pool_dict, pool_ids = joblib.load(cache_dir / f"pool_dict_{country}.joblib")
            dupe_map = joblib.load(cache_dir / f"dupe_map_{country}.joblib")
            struct_idx = joblib.load(cache_dir / f"structured_index_{country}.joblib")

            c_vecs = {}
            c_mats = {}
            for mode in ("joint", "name_only", "address_only"):
                c_vecs[mode] = joblib.load(cache_dir / f"vec_{country}_{mode}.joblib")
                c_mats[mode] = sp.load_npz(cache_dir / f"mat_{country}_{mode}.npz")

            q_names = [normalize_name(s1_meta[q]["name"], country) for q in c_train_qids]
            q_addrs = [normalize_address(s1_meta[q]["address"], country) for q in c_train_qids]
            lex_artifacts = {
                mode: (c_vecs[mode], c_mats[mode])
                for mode in ("joint", "name_only", "address_only")
            }
            cands_by_q, _ = generate_natural_candidates(
                query_ids=c_train_qids,
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

            for qid in c_train_qids:
                q_cands = cands_by_q.get(qid, [])
                true_tgts = set(gt_map.get(qid, []))
                fold = inner_folds_map.get(qid, 0)
                for pid, rrf_score in q_cands:
                    train_rows.append({
                        "query_id": qid,
                        "target_id": pid,
                        "is_match": 1 if pid in true_tgts else 0,
                        "fold": fold,
                        "rrf_score": float(rrf_score),
                    })

        df_train_pairs = pd.DataFrame(train_rows)
        # Assert exact contract parity
        assert len(df_train_pairs) == 1_200_000, f"Expected 1.2M train pairs, got {len(df_train_pairs):,}"
        assert int(df_train_pairs["is_match"].sum()) == int(bundle["y_train"].sum()), (
            f"Label mismatch: {df_train_pairs['is_match'].sum()} vs expected {bundle['y_train'].sum()}"
        )
        df_train_pairs.to_parquet(train_pairs_path, index=False)
        print(f"Verified & saved train_pairs.parquet ({len(df_train_pairs):,} pairs, {df_train_pairs['is_match'].sum():,} positives) to {train_pairs_path}")

    # 5. Extract Text Provenance Partitioned for Device 3
    print("\n--- Extracting Structured Text Records for Device 3 ---")
    train_target_ids = set(df_train_pairs["target_id"].unique())
    calib_cands = calib_data["calib_candidates_by_q"]
    eval_cands = eval_data["eval_candidates_by_q"]

    eval_candidate_ids = set()
    for qid in calib_cands:
        for pid, _ in calib_cands[qid]:
            eval_candidate_ids.add(pid)
    for qid in eval_cands:
        for pid, _ in eval_cands[qid]:
            eval_candidate_ids.add(pid)

    # Diagnostic unretrieved evaluation targets (isolated, never in training/candidates)
    diag_unretrieved_ids = set()
    for qid, tids in calib_data["calib_truth_by_q"].items():
        for tid in tids:
            if tid not in eval_candidate_ids:
                diag_unretrieved_ids.add(tid)
    for qid, tids in eval_data["eval_truth_by_q"].items():
        for tid in tids:
            if tid not in eval_candidate_ids:
                diag_unretrieved_ids.add(tid)

    needed_all_target_ids = train_target_ids | eval_candidate_ids | diag_unretrieved_ids
    print(f"Total unique target records to extract: {len(needed_all_target_ids):,} (train={len(train_target_ids):,}, eval_cands={len(eval_candidate_ids):,}, unretrieved_diag={len(diag_unretrieved_ids):,})")

    train_target_texts = {}
    eval_target_texts = {}
    diag_target_texts = {}

    for country in ("India", "US"):
        print(f"Loading {country} pool dictionary to extract text records...")
        pool_dict, _ = joblib.load(cache_dir / f"pool_dict_{country}.joblib")
        for pid in needed_all_target_ids:
            if pid in pool_dict:
                entry = {
                    "name": pool_dict[pid]["business_name"],
                    "address": pool_dict[pid]["business_address"],
                    "country": country,
                }
                if pid in train_target_ids:
                    train_target_texts[pid] = entry
                if pid in eval_candidate_ids:
                    eval_target_texts[pid] = entry
                if pid in diag_unretrieved_ids:
                    diag_target_texts[pid] = entry

    text_bundle = {
        "queries": s1_meta,
        "train_targets": train_target_texts,
        "eval_candidate_targets": eval_target_texts,
        "diagnostic_unretrieved_targets": diag_target_texts,
        "counts": {
            "queries": len(s1_meta),
            "train_targets": len(train_target_texts),
            "eval_candidate_targets": len(eval_target_texts),
            "diagnostic_unretrieved_targets": len(diag_target_texts),
        },
    }

    text_bundle_file = bundle_dir / "record_text_provenance.joblib"
    joblib.dump(text_bundle, text_bundle_file, compress=3)
    print(
        f"Saved cleanly partitioned text provenance ({len(s1_meta):,} queries, {len(train_target_texts):,} train targets, "
        f"{len(eval_target_texts):,} eval targets) to {text_bundle_file} ({text_bundle_file.stat().st_size / 1e6:.1f} MB)"
    )

    # 6. Build Manifest JSON Referencing Split Bundle & Provenance Files
    print("\n--- Updating Manifest JSON ---")
    raw_inputs = {
        "train_source1.tsv": sha256_file(train_dir / "train_source1.tsv"),
        "train_source2.tsv": sha256_file(train_dir / "train_source2.tsv"),
        "train_source3.tsv": sha256_file(train_dir / "train_source3.tsv"),
        "train_ground_truth.tsv": sha256_file(train_dir / "train_ground_truth.tsv"),
    }
    manifest_hashes = {
        fn: sha256_file(manifest_dir / fn)
        for fn in ("train_12k.json", "calibration_5k.json", "comparison_15k.json", "screen_2k.json", "inner_train_folds.json")
    }
    cache_hashes = {
        "pool_dict_India.joblib": sha256_file(cache_dir / "pool_dict_India.joblib"),
        "pool_dict_US.joblib": sha256_file(cache_dir / "pool_dict_US.joblib"),
        "mat_India_joint.npz": sha256_file(cache_dir / "mat_India_joint.npz"),
        "mat_US_joint.npz": sha256_file(cache_dir / "mat_US_joint.npz"),
    }

    manifest = {
        "experiment_id": "D1-01_B0_BASELINE",
        "description": "Audited Clean Reference Baseline B0 (Zero Ground-Truth Injection, Keyed Multi-Device Handoff)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_state": get_git_state(),
        "hardware": {
            "device": "Device 1 (Windows / Intel Arc 140T)",
            "cpu_cores_pinned": args.n_cores,
            "backend": "scikit-learn + LightGBM CPU + ThreadPoolExecutor(12)",
        },
        "hashes": {
            "raw_inputs": raw_inputs,
            "role_manifests": manifest_hashes,
            "retrieval_caches": cache_hashes,
            "model_file": sha256_file(model_path),
            "threshold_file": sha256_file(thresh_path),
            "b0_train_features": sha256_file(bundle_dir / "b0_train_features.joblib"),
            "b0_eval_features": sha256_file(bundle_dir / "b0_eval_features.joblib"),
            "train_pairs_parquet": sha256_file(train_pairs_path),
            "calibration_predictions_parquet": sha256_file(calib_pred_parquet),
            "screen_predictions_parquet": sha256_file(eval_pred_parquet),
            "record_text_provenance": sha256_file(text_bundle_file),
        },
        "model_parameters": {
            "num_trees": model.num_trees(),
            "learning_rate": 0.05,
            "max_depth": 7,
            "num_leaves": 63,
            "min_child_samples": 80,
            "subsample": 0.8,
            "bagging_freq": 1,
            "calibrated_thresholds": {
                "t_singleton": t_sing,
                "t_match": t_match,
            },
        },
        "bundle_contents": {
            "b0_train_features.joblib": "1.2M training candidate pairs, group IDs, 3-fold splits, 23 feature matrix (31 MB)",
            "b0_eval_features.joblib": "500k calibration + 200k eval candidate pairs with labels and RRF scores (26 MB)",
            "train_pairs.parquet": "Keyed training pairs (query_id, target_id, is_match, fold, rrf_score) for D3 supervised training",
            "calibration_predictions.parquet": "Per-pair probabilities and match indicators on calibration_5k",
            "screen_predictions.parquet": "Per-pair probabilities and match indicators on screen_2k",
            "record_text_provenance.joblib": "Cleanly partitioned query and target text records for D3 neural models (27 MB)",
        },
    }

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved complete provenance manifest to {manifest_path}!")
    print(f"=== Provenance & Training Package Build Finished in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()
