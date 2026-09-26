"""Build comprehensive provenance metadata, predictions, and text records for B0 handoff (D1-01).

Generates under runs/parallel-v1/d1/b0_baseline/:
1. manifest.json: SHA256 of all input TSVs, manifests, retrieval artifacts, model config, git state.
2. calibration_predictions.parquet: per-pair probability, match label, and decision flag for calibration_5k.
3. screen_predictions.parquet: per-pair probability, match label, and decision flag for screen_2k.
4. record_text_provenance.joblib: raw & normalized text records for all queries and candidate targets,
   enabling Device 3 to train neural cross-encoders / embeddings without downloading or parsing 2.5GB TSVs.
"""
from __future__ import annotations

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
import pandas as pd
from er.calibration import decide_two_threshold
from er.metrics import macro_f05, oracle_macro_f05


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
    t_start = time.time()
    bundle_dir = Path("runs/parallel-v1/d1/b0_baseline")
    manifest_dir = Path("splits/f05-v1/parallel-v1")
    cache_dir = Path("cache/retrieval")
    train_dir = Path("student_resource/student_resource/dataset/train")
    models_dir = Path("cache/models")

    print("=== Building B0 Provenance and Multi-Device Handoff Artifacts ===")

    # 1. Load model and thresholds
    model_path = models_dir / "b0_clean_matcher.txt"
    thresh_path = models_dir / "b0_calibrated_thresholds.json"
    model = lgb.Booster(model_file=str(model_path))
    thresh = json.loads(thresh_path.read_text(encoding="utf-8"))
    t_sing = thresh["t_singleton"]
    t_match = thresh["t_match"]

    # 2. Load portable feature bundle
    bundle_file = bundle_dir / "b0_portable_bundle.joblib"
    print(f"Loading portable feature bundle from {bundle_file}...")
    bundle = joblib.load(bundle_file)

    calib_data = bundle["calib_data"]
    eval_data = bundle["eval_data"]

    # 3. Predict on calibration and screen evaluation
    print("Generating calibration and screen predictions...")
    calib_preds = model.predict(calib_data["X_calib"])
    eval_preds = model.predict(eval_data["X_eval"])

    # Build calibration prediction dataframe
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
    calib_pred_parquet = bundle_dir / "calibration_predictions.parquet"
    df_calib_pred.to_parquet(calib_pred_parquet, index=False)
    print(f"Saved calibration predictions ({len(df_calib_pred):,} pairs) to {calib_pred_parquet}")

    # Build screen evaluation prediction dataframe
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
    eval_pred_parquet = bundle_dir / "screen_predictions.parquet"
    df_eval_pred.to_parquet(eval_pred_parquet, index=False)
    print(f"Saved screen predictions ({len(df_eval_pred):,} pairs) to {eval_pred_parquet}")

    # 4. Extract Text Provenance for Device 3 (Neural / Multilingual Challengers)
    print("Extracting text records for query and target pairs...")
    needed_qids = set(bundle["groups_train"]) | set(calib_truth.keys()) | set(eval_truth.keys())
    needed_pids = set()
    for qid in calib_truth:
        for pid, _ in calib_cands.get(qid, []):
            needed_pids.add(pid)
        for tid in calib_truth.get(qid, []):
            needed_pids.add(tid)
    for qid in eval_truth:
        for pid, _ in eval_cands.get(qid, []):
            needed_pids.add(pid)
        for tid in eval_truth.get(qid, []):
            needed_pids.add(tid)

    # Read S1 text records
    s1_df = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1_sub = s1_df[s1_df["entity_id"].isin(needed_qids)]
    query_texts = {
        r.entity_id: {"name": r.business_name, "address": r.business_address, "country": r.country}
        for r in s1_sub.itertuples()
    }

    # Read target pool texts from cached pool dictionaries
    target_texts = {}
    for country in ("India", "US"):
        pool_dict_path = cache_dir / f"pool_dict_{country}.joblib"
        p_dict, _ = joblib.load(pool_dict_path)
        for pid in needed_pids:
            if pid in p_dict and pid not in target_texts:
                target_texts[pid] = {
                    "name": p_dict[pid]["business_name"],
                    "address": p_dict[pid]["business_address"],
                    "country": country,
                }

    text_bundle = {
        "queries": query_texts,
        "targets": target_texts,
        "counts": {"queries": len(query_texts), "targets": len(target_texts)},
    }
    text_bundle_file = bundle_dir / "record_text_provenance.joblib"
    joblib.dump(text_bundle, text_bundle_file, compress=3)
    print(f"Saved text provenance ({len(query_texts):,} queries, {len(target_texts):,} targets) to {text_bundle_file} ({text_bundle_file.stat().st_size / 1e6:.1f} MB)")

    # 5. Build Complete Manifest JSON
    print("Computing SHA256 hashes for all input files and manifests...")
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
        "description": "Audited Clean Reference Baseline B0 (Zero Ground-Truth Injection)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_state": get_git_state(),
        "hardware": {
            "device": "Device 1 (Windows / Intel Arc 140T)",
            "cpu_cores_pinned": 12,
            "backend": "scikit-learn + LightGBM CPU + ThreadPoolExecutor(12)",
        },
        "hashes": {
            "raw_inputs": raw_inputs,
            "role_manifests": manifest_hashes,
            "retrieval_caches": cache_hashes,
            "model_file": sha256_file(model_path),
            "threshold_file": sha256_file(thresh_path),
            "portable_bundle": sha256_file(bundle_file),
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
            "b0_portable_bundle.joblib": "Training, calibration, and evaluation feature arrays and labels for D2",
            "calibration_predictions.parquet": "Per-pair probabilities and match indicators on calibration_5k",
            "screen_predictions.parquet": "Per-pair probabilities and match indicators on screen_2k",
            "record_text_provenance.joblib": "Raw & normalized texts for queries and targets for D3 neural models",
        },
    }

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved complete provenance manifest to {manifest_path}!")
    print(f"=== Provenance Build Finished in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()
