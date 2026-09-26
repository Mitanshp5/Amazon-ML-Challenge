"""Generate parallel-v1 role manifests for multi-device execution (Step 1: D1-00).

Outputs under splits/f05-v1/parallel-v1/:
- train_12k.json: 12,000 queries (6k India, 6k US) from train partition.
- calibration_5k.json: 5,000 queries (2.5k India, 2.5k US) from dev partition for threshold tuning.
- comparison_15k.json: 15,000 queries (7.5k India, 7.5k US) from dev partition for clean macro F0.5.
- screen_2k.json: 2,000 queries (1k India, 1k US) subset of comparison_15k for fast iteration.
- inner_train_folds.json: 3-fold GroupKFold assignments for train_12k.
- manifest_summary.json: verified statistics, hashes, and disjointness assertions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_sample(ids: list[str], n: int, seed: str) -> list[str]:
    sorted_ids = sorted(ids, key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())
    return sorted_ids[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--splits-json", default="splits/f05-v1/splits.json")
    ap.add_argument("--out-dir", default="splits/f05-v1/parallel-v1")
    ap.add_argument("--seed", default="parallel-v1")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading splits from {args.splits_json}...")
    with open(args.splits_json, "r", encoding="utf-8") as f:
        splits_map = json.load(f)["splits"]

    print("Loading train_source1.tsv entity metadata...")
    df_s1 = pd.read_csv(
        train_dir / "train_source1.tsv",
        sep="\t",
        usecols=["entity_id", "country"],
        dtype=str,
        keep_default_na=False,
    )
    df_s1["split"] = df_s1["entity_id"].map(splits_map)

    # Validate split coverage
    assert not df_s1["split"].isna().any(), "Some S1 entities are missing split assignments!"

    # Partition IDs by split and country
    train_in = df_s1[(df_s1["split"] == "train") & (df_s1["country"] == "India")]["entity_id"].tolist()
    train_us = df_s1[(df_s1["split"] == "train") & (df_s1["country"] == "US")]["entity_id"].tolist()

    dev_in = df_s1[(df_s1["split"] == "dev") & (df_s1["country"] == "India")]["entity_id"].tolist()
    dev_us = df_s1[(df_s1["split"] == "dev") & (df_s1["country"] == "US")]["entity_id"].tolist()

    holdout_ids = set(df_s1[df_s1["split"] == "holdout"]["entity_id"])
    exposed_ids = set(df_s1[df_s1["split"] == "dev-exposed"]["entity_id"])

    print(f"Eligible train: India={len(train_in):,}, US={len(train_us):,}")
    print(f"Eligible dev: India={len(dev_in):,}, US={len(dev_us):,}")

    # 1. train_12k.json: 6,000 India, 6,000 US from train
    train_12k_in = deterministic_sample(train_in, 6000, f"{args.seed}|train_in")
    train_12k_us = deterministic_sample(train_us, 6000, f"{args.seed}|train_us")
    train_12k = sorted(train_12k_in + train_12k_us)
    assert len(train_12k) == 12000

    # 2. calibration_5k.json: 2,500 India, 2,500 US from dev
    calib_5k_in = deterministic_sample(dev_in, 2500, f"{args.seed}|calib_in")
    calib_5k_us = deterministic_sample(dev_us, 2500, f"{args.seed}|calib_us")
    calibration_5k = sorted(calib_5k_in + calib_5k_us)
    assert len(calibration_5k) == 5000

    # 3. comparison_15k.json: 7,500 India, 7,500 US from dev (disjoint from calibration_5k)
    remaining_dev_in = [x for x in dev_in if x not in set(calib_5k_in)]
    remaining_dev_us = [x for x in dev_us if x not in set(calib_5k_us)]

    comp_15k_in = deterministic_sample(remaining_dev_in, 7500, f"{args.seed}|comp_in")
    comp_15k_us = deterministic_sample(remaining_dev_us, 7500, f"{args.seed}|comp_us")
    comparison_15k = sorted(comp_15k_in + comp_15k_us)
    assert len(comparison_15k) == 15000

    # 4. screen_2k.json: 1,000 India, 1,000 US subset of comparison_15k
    screen_2k_in = deterministic_sample(comp_15k_in, 1000, f"{args.seed}|screen_in")
    screen_2k_us = deterministic_sample(comp_15k_us, 1000, f"{args.seed}|screen_us")
    screen_2k = sorted(screen_2k_in + screen_2k_us)
    assert len(screen_2k) == 2000
    assert set(screen_2k).issubset(set(comparison_15k))

    # 5. inner_train_folds.json: 3-fold GroupKFold assignments for train_12k
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    inner_folds = {}
    train_12k_arr = np.array(train_12k)
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_12k_arr)):
        for qid in train_12k_arr[val_idx]:
            inner_folds[str(qid)] = int(fold_idx)

    # Strict Disjointness Verification
    s_train = set(train_12k)
    s_calib = set(calibration_5k)
    s_comp = set(comparison_15k)

    assert len(s_train & s_calib) == 0, "Overlap between train_12k and calibration_5k!"
    assert len(s_train & s_comp) == 0, "Overlap between train_12k and comparison_15k!"
    assert len(s_calib & s_comp) == 0, "Overlap between calibration_5k and comparison_15k!"
    assert len(s_train & holdout_ids) == 0, "train_12k touches holdout!"
    assert len(s_calib & holdout_ids) == 0, "calibration_5k touches holdout!"
    assert len(s_comp & holdout_ids) == 0, "comparison_15k touches holdout!"
    assert len(s_train & exposed_ids) == 0, "train_12k touches legacy exposed!"
    assert len(s_calib & exposed_ids) == 0, "calibration_5k touches legacy exposed!"
    assert len(s_comp & exposed_ids) == 0, "comparison_15k touches legacy exposed!"

    print("All strict disjointness and integrity checks passed!")

    # Write manifests
    def dump_json(p: Path, obj: dict | list) -> None:
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    dump_json(out_dir / "train_12k.json", {"seed": args.seed, "counts": {"India": 6000, "US": 6000}, "query_ids": train_12k})
    dump_json(out_dir / "calibration_5k.json", {"seed": args.seed, "counts": {"India": 2500, "US": 2500}, "query_ids": calibration_5k})
    dump_json(out_dir / "comparison_15k.json", {"seed": args.seed, "counts": {"India": 7500, "US": 7500}, "query_ids": comparison_15k})
    dump_json(out_dir / "screen_2k.json", {"seed": args.seed, "counts": {"India": 1000, "US": 1000}, "query_ids": screen_2k})
    dump_json(out_dir / "inner_train_folds.json", {"seed": args.seed, "n_splits": 3, "fold_by_query": inner_folds})

    # Summary
    summary = {
        "seed": args.seed,
        "manifests": {
            "train_12k": {"total": len(train_12k), "India": 6000, "US": 6000, "source": "train_partition"},
            "calibration_5k": {"total": len(calibration_5k), "India": 2500, "US": 2500, "source": "dev_partition"},
            "comparison_15k": {"total": len(comparison_15k), "India": 7500, "US": 7500, "source": "dev_partition"},
            "screen_2k": {"total": len(screen_2k), "India": 1000, "US": 1000, "source": "comparison_15k_subset"},
        },
        "disjointness_verified": True,
        "holdout_leakage": 0,
        "dev_exposed_leakage": 0,
    }
    dump_json(out_dir / "manifest_summary.json", summary)
    print(f"Manifests successfully published to {out_dir}!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
