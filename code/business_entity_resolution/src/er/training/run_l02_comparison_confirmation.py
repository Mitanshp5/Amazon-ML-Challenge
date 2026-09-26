"""L02: Build comparison_15k natural features and confirm contenders on full 15k and unexposed 13k.

Authority: LOCAL_COLAB_IMPLEMENTATION_PLAN.md Section 5 (L02)
Hardware: Windows / 12 CPU threads
Evaluates:
  1. Control B0 reference model
  2. Capacity 127 model (L01 Winner)
  3. Regularized 127 model (L01 Regularized Winner)
Policies:
  - Baseline (0.70 / 0.70)
  - Country Dual (India: 0.72/0.70, US: 0.585/0.585, Global: 0.715/0.685)
Evaluated across:
  - comparison_15k (all 15,000 queries)
  - unexposed_13k (13,000 queries strictly outside screen_2k)
  - screen_2k (2,000 queries for parity check)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from er.analyze_b0_decisions import apply_policy, counts, exact_scores, prepare
from er.candidate_generation import generate_natural_candidates, hash_candidate_sets
from er.features import pair_feature_row, rows_to_matrix
from er.metrics import macro_f05, oracle_macro_f05
from er.normalization import normalize_address, normalize_name
from er.normalized_adapter import NormalizedRecordAdapter
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


def bootstrap_delta(delta: np.ndarray, countries: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    unique_c = sorted(set(countries))
    strata = [np.flatnonzero(countries == c) for c in unique_c]
    boot_means = np.array([
        np.concatenate([delta[rng.choice(s, len(s), replace=True)] for s in strata]).mean()
        for _ in range(n_boot)
    ])
    ci = [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]
    return float(delta.mean()), ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--gt", default="student_resource/student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--manifest-dir", default="splits/f05-v1/parallel-v1")
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--out-dir", default="runs/local-v2/L02_comparison_15k")
    ap.add_argument("--reports-dir", default="reports/dev_probe")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" L02: Comparison 15k Feature Construction & Matcher Confirmation")
    print(f" CPU Cores: {args.n_cores} | Output Dir: {out_dir}")
    print("=" * 70)

    # 1. Load Ground Truth (strictly for evaluation, NEVER candidate generation)
    print("Loading Ground Truth for evaluation scoring...")
    gt_map = load_gt_map(Path(args.gt))
    print(f"Loaded ground truth for {len(gt_map):,} queries.")

    # 2. Load Role Manifests
    print("Loading role manifests...")
    c15k_info = json.loads((manifest_dir / "comparison_15k.json").read_text(encoding="utf-8"))
    s2k_info = json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))

    comp_qids = c15k_info["query_ids"]
    screen_qids_set = set(s2k_info["query_ids"])
    unexp_qids = [q for q in comp_qids if q not in screen_qids_set]

    print(f"Total Comparison Queries: {len(comp_qids):,}")
    print(f"  - Screen Subset:        {len(screen_qids_set):,}")
    print(f"  - Unexposed Subset:     {len(unexp_qids):,}")

    # 3. Load S1 Query Data
    print("Loading S1 query metadata...")
    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1_by_id = s1.set_index("entity_id")

    query_records = {}
    for qid in comp_qids:
        query_records[qid] = {
            "entity_id": qid,
            "business_name": s1_by_id.loc[qid, "business_name"],
            "business_address": s1_by_id.loc[qid, "business_address"],
            "country": s1_by_id.loc[qid, "country"],
        }

    # Data structures across countries
    all_rows = []
    all_labels = []
    all_groups = []
    all_pair_keys = []
    cands_by_q_all = {}
    truth_by_q_all = {}

    timing_stages = {}

    for country in ("India", "US"):
        t_c_start = time.time()
        print(f"\n==================== Processing {country} ====================")
        c_qids = [q for q in comp_qids if s1_by_id.loc[q, "country"] == country]
        c_screen_qids = [q for q in c_qids if q in screen_qids_set]
        print(f"{country} queries: Total={len(c_qids):,} (Screen={len(c_screen_qids):,}, Unexposed={len(c_qids)-len(c_screen_qids):,})")

        # Load retrieval artifacts
        t_load = time.time()
        print(f"Loading {country} retrieval artifacts from {cache_dir}...")
        pool_dict, pool_ids = joblib.load(cache_dir / f"pool_dict_{country}.joblib")
        dupe_map = joblib.load(cache_dir / f"dupe_map_{country}.joblib")
        struct_idx = joblib.load(cache_dir / f"structured_index_{country}.joblib")

        lex_artifacts = {}
        for mode in ("joint", "name_only", "address_only"):
            vec = joblib.load(cache_dir / f"vec_{country}_{mode}.joblib")
            p_mat = load_npz(cache_dir / f"mat_{country}_{mode}.npz")
            lex_artifacts[mode] = (vec, p_mat)

        print(f"Loaded {country} artifacts in {time.time() - t_load:.2f}s (pool size: {len(pool_ids):,})")

        # Normalize query texts
        q_names = [normalize_name(s1_by_id.loc[q, "business_name"], country) for q in c_qids]
        q_addrs = [normalize_address(s1_by_id.loc[q, "business_address"], country) for q in c_qids]

        # Natural candidate generation (top-100 RRF, 100% label blind)
        print(f"Generating natural candidates for {len(c_qids):,} {country} queries...")
        t_ret = time.time()
        cands_by_q, ch_lookups = generate_natural_candidates(
            query_ids=c_qids,
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
        print(f"Retrieval for {country} completed in {timing_stages[f'retrieval_{country}']:.2f}s")

        # Check candidate hash on screen queries
        screen_cands = {q: cands_by_q[q] for q in c_screen_qids}
        screen_hash = hash_candidate_sets(screen_cands)
        print(f"{country} Screen candidate SHA-256 hash: {screen_hash[:16]}...")

        # Pre-cache normalized records
        needed_pids = set()
        for qid in c_qids:
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
            for qid in c_qids
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

        # Multi-threaded feature extraction
        print(f"Extracting 23 features with ThreadPoolExecutor({args.n_cores})...")
        t_feat = time.time()

        def _extract_query(qid: str):
            q_meta = q_meta_cache[qid]
            true_tgts = set(gt_map.get(qid, []))
            cands = cands_by_q.get(qid, [])

            l_rows, l_lbls, l_grps, l_pairs = [], [], [], []
            for pid, rrf_score in cands:
                if pid not in pool_meta_cache:
                    continue
                p_meta = pool_meta_cache[pid]
                chmap = {ch: d[pid] for ch, qdict in ch_lookups.items() if (d := qdict.get(qid)) and pid in d}
                feat = pair_feature_row(q_meta, p_meta, chmap, rrf_score, pid.startswith("S2-"))
                l_rows.append(feat)
                l_lbls.append(1 if pid in true_tgts else 0)
                l_grps.append(qid)
                l_pairs.append((pid, rrf_score))
            return qid, l_rows, l_lbls, l_grps, l_pairs

        with ThreadPoolExecutor(max_workers=args.n_cores) as pool:
            results = list(pool.map(_extract_query, c_qids))

        timing_stages[f"features_{country}"] = time.time() - t_feat
        print(f"Features for {country} completed in {timing_stages[f'features_{country}']:.2f}s")

        for qid, q_r, q_l, q_g, q_p in results:
            all_rows.extend(q_r)
            all_labels.extend(q_l)
            all_groups.extend(q_g)
            all_pair_keys.extend([(qid, pid) for pid, _ in q_p])
            cands_by_q_all[qid] = q_p
            truth_by_q_all[qid] = gt_map.get(qid, [])

        # Clean memory for this country
        del pool_dict, pool_ids, dupe_map, struct_idx, lex_artifacts, pool_meta_cache, q_meta_cache
        gc.collect()
        print(f"{country} finished in {time.time() - t_c_start:.2f}s. Memory cleared.")

    # 4. Assemble matrix
    print("\nAssembling full 15k feature matrix...")
    X_comp, feature_names = rows_to_matrix(all_rows)
    print(f"X_comp matrix shape: {X_comp.shape}, Labels: {len(all_labels):,}")

    pair_query_ids = [k[0] for k in all_pair_keys]
    pair_target_ids = [k[1] for k in all_pair_keys]

    # Models to evaluate
    models_to_test = {
        "control_b0": Path("cache/models/b0_clean_matcher.txt"),
        "capacity_127": Path("runs/local-v2/L01_matcher_screen/capacity_127/matcher_model.txt"),
        "regularized_127": Path("runs/local-v2/L01_matcher_screen/regularized_127/matcher_model.txt"),
    }

    # Policies
    policies = {
        "baseline": {"global": {"ts": 0.70, "tm": 0.70}},
        "country_dual": {
            "global": {"ts": 0.715, "tm": 0.685},
            "India": {"ts": 0.720, "tm": 0.700},
            "US": {"ts": 0.585, "tm": 0.585},
        },
    }

    results = {}
    b0_baseline_15k_scores = None
    b0_baseline_13k_scores = None
    b0_baseline_2k_scores = None

    for m_name, m_path in models_to_test.items():
        print(f"\n==================== Scoring Model: {m_name} ====================")
        assert m_path.exists(), f"Model file not found: {m_path}"
        booster = lgb.Booster(model_file=str(m_path))
        probs = booster.predict(X_comp)

        df_preds = pd.DataFrame({
            "query_id": pair_query_ids,
            "target_id": pair_target_ids,
            "is_match": all_labels,
            "probability": probs,
        })

        parquet_out = out_dir / f"predictions_{m_name}.parquet"
        df_preds.to_parquet(parquet_out, index=False)
        print(f"Saved predictions to {parquet_out} ({len(df_preds):,} rows)")

        # Prepare evaluation blocks
        prepared_15k = prepare(df_preds, truth_by_q_all, query_records)

        # Slice for 13k unexposed
        mask_13k = np.isin(prepared_15k["ids"], unexp_qids)
        mask_2k = np.isin(prepared_15k["ids"], list(screen_qids_set))

        m_results = {}
        for p_name, pol in policies.items():
            scores_15k, tp_15k, count_15k = apply_policy(prepared_15k, pol)
            f05_15k = float(scores_15k.mean())

            scores_13k = scores_15k[mask_13k]
            f05_13k = float(scores_13k.mean())

            scores_2k = scores_15k[mask_2k]
            f05_2k = float(scores_2k.mean())

            # Country slices on 15k
            c_15k = prepared_15k["country"]
            india_15k = float(scores_15k[c_15k == "India"].mean())
            us_15k = float(scores_15k[c_15k == "US"].mean())

            # Country slices on 13k
            c_13k = c_15k[mask_13k]
            india_13k = float(scores_13k[c_13k == "India"].mean())
            us_13k = float(scores_13k[c_13k == "US"].mean())

            if m_name == "control_b0" and p_name == "baseline":
                b0_baseline_15k_scores = scores_15k
                b0_baseline_13k_scores = scores_13k
                b0_baseline_2k_scores = scores_2k

            delta_15k_val, ci_15k = bootstrap_delta(scores_15k - b0_baseline_15k_scores, c_15k)
            delta_13k_val, ci_13k = bootstrap_delta(scores_13k - b0_baseline_13k_scores, c_13k)

            print(f"[{m_name}] Policy: {p_name}")
            print(f"  Full 15k F0.5:     {f05_15k:.6f} (India: {india_15k:.6f}, US: {us_15k:.6f}) | Delta: {delta_15k_val:+.6f} CI: [{ci_15k[0]:+.6f}, {ci_15k[1]:+.6f}]")
            print(f"  Unexposed 13k F0.5:{f05_13k:.6f} (India: {india_13k:.6f}, US: {us_13k:.6f}) | Delta: {delta_13k_val:+.6f} CI: [{ci_13k[0]:+.6f}, {ci_13k[1]:+.6f}]")
            print(f"  Screen 2k F0.5:    {f05_2k:.6f}")

            m_results[p_name] = {
                "f05_15k": f05_15k,
                "india_15k": india_15k,
                "us_15k": us_15k,
                "delta_15k": delta_15k_val,
                "ci_15k": ci_15k,
                "f05_13k": f05_13k,
                "india_13k": india_13k,
                "us_13k": us_13k,
                "delta_13k": delta_13k_val,
                "ci_13k": ci_13k,
                "f05_2k": f05_2k,
            }

        results[m_name] = m_results

    # Summary table
    print("\n" + "=" * 70)
    print(" L02: Comparison 15k & Unexposed 13k Final Summary")
    print("=" * 70)
    print(f"{'Model':<16} {'Policy':<14} {'Full 15k':<10} {'Delta 15k':<11} {'Unexp 13k':<10} {'Delta 13k':<11} {'Screen 2k':<10}")
    print("-" * 85)
    for m_name, p_dict in results.items():
        for p_name, r in p_dict.items():
            print(f"{m_name:<16} {p_name:<14} {r['f05_15k']:<10.6f} {r['delta_15k']:<+11.6f} {r['f05_13k']:<10.6f} {r['delta_13k']:<+11.6f} {r['f05_2k']:<10.6f}")

    # Write summary JSON
    summary_path = out_dir / "L02_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote summary JSON to {summary_path}")

    # Generate Markdown Report
    report_md = f"""# L02: Comparison 15k Confirmation Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Hardware:** Windows / {args.n_cores} CPU threads
**Candidate Set:** Natural K100 candidates (100% label-blind)
**Total Queries Evaluated:** 15,000 (`comparison_15k`)
**Partitions:**
- `screen_2k`: 2,000 queries (screening subset)
- `unexposed_13k`: 13,000 queries (strictly out-of-screen development)

---

## 1. Full Comparison (15,000 Queries) Results

| Model | Policy | Full 15k F0.5 | India F0.5 | US F0.5 | Delta vs B0 Ref | 95% Bootstrap CI |
|---|---|---|---|---|---|---|
| **control_b0** | baseline (0.70/0.70) | {results['control_b0']['baseline']['f05_15k']:.6f} | {results['control_b0']['baseline']['india_15k']:.6f} | {results['control_b0']['baseline']['us_15k']:.6f} | Ref | - |
| **control_b0** | country_dual | **{results['control_b0']['country_dual']['f05_15k']:.6f}** | {results['control_b0']['country_dual']['india_15k']:.6f} | {results['control_b0']['country_dual']['us_15k']:.6f} | **{results['control_b0']['country_dual']['delta_15k']:+.6f}** | [{results['control_b0']['country_dual']['ci_15k'][0]:+.6f}, {results['control_b0']['country_dual']['ci_15k'][1]:+.6f}] |
| **capacity_127** | baseline (0.70/0.70) | {results['capacity_127']['baseline']['f05_15k']:.6f} | {results['capacity_127']['baseline']['india_15k']:.6f} | {results['capacity_127']['baseline']['us_15k']:.6f} | {results['capacity_127']['baseline']['delta_15k']:+.6f} | [{results['capacity_127']['baseline']['ci_15k'][0]:+.6f}, {results['capacity_127']['baseline']['ci_15k'][1]:+.6f}] |
| **capacity_127** | country_dual | **{results['capacity_127']['country_dual']['f05_15k']:.6f}** | {results['capacity_127']['country_dual']['india_15k']:.6f} | {results['capacity_127']['country_dual']['us_15k']:.6f} | **{results['capacity_127']['country_dual']['delta_15k']:+.6f}** | [{results['capacity_127']['country_dual']['ci_15k'][0]:+.6f}, {results['capacity_127']['country_dual']['ci_15k'][1]:+.6f}] |
| **regularized_127** | baseline (0.70/0.70) | {results['regularized_127']['baseline']['f05_15k']:.6f} | {results['regularized_127']['baseline']['india_15k']:.6f} | {results['regularized_127']['baseline']['us_15k']:.6f} | {results['regularized_127']['baseline']['delta_15k']:+.6f} | [{results['regularized_127']['baseline']['ci_15k'][0]:+.6f}, {results['regularized_127']['baseline']['ci_15k'][1]:+.6f}] |
| **regularized_127** | country_dual | **{results['regularized_127']['country_dual']['f05_15k']:.6f}** | {results['regularized_127']['country_dual']['india_15k']:.6f} | {results['regularized_127']['country_dual']['us_15k']:.6f} | **{results['regularized_127']['country_dual']['delta_15k']:+.6f}** | [{results['regularized_127']['country_dual']['ci_15k'][0]:+.6f}, {results['regularized_127']['country_dual']['ci_15k'][1]:+.6f}] |

---

## 2. Unexposed Development (13,000 Queries) Results

| Model | Policy | Unexp 13k F0.5 | India F0.5 | US F0.5 | Delta vs B0 Ref | 95% Bootstrap CI |
|---|---|---|---|---|---|---|
| **control_b0** | baseline (0.70/0.70) | {results['control_b0']['baseline']['f05_13k']:.6f} | {results['control_b0']['baseline']['india_13k']:.6f} | {results['control_b0']['baseline']['us_13k']:.6f} | Ref | - |
| **control_b0** | country_dual | **{results['control_b0']['country_dual']['f05_13k']:.6f}** | {results['control_b0']['country_dual']['india_13k']:.6f} | {results['control_b0']['country_dual']['us_13k']:.6f} | **{results['control_b0']['country_dual']['delta_13k']:+.6f}** | [{results['control_b0']['country_dual']['ci_13k'][0]:+.6f}, {results['control_b0']['country_dual']['ci_13k'][1]:+.6f}] |
| **capacity_127** | baseline (0.70/0.70) | {results['capacity_127']['baseline']['f05_13k']:.6f} | {results['capacity_127']['baseline']['india_13k']:.6f} | {results['capacity_127']['baseline']['us_13k']:.6f} | {results['capacity_127']['baseline']['delta_13k']:+.6f} | [{results['capacity_127']['baseline']['ci_13k'][0]:+.6f}, {results['capacity_127']['baseline']['ci_13k'][1]:+.6f}] |
| **capacity_127** | country_dual | **{results['capacity_127']['country_dual']['f05_13k']:.6f}** | {results['capacity_127']['country_dual']['india_13k']:.6f} | {results['capacity_127']['country_dual']['us_13k']:.6f} | **{results['capacity_127']['country_dual']['delta_13k']:+.6f}** | [{results['capacity_127']['country_dual']['ci_13k'][0]:+.6f}, {results['capacity_127']['country_dual']['ci_13k'][1]:+.6f}] |
| **regularized_127** | baseline (0.70/0.70) | {results['regularized_127']['baseline']['f05_13k']:.6f} | {results['regularized_127']['baseline']['india_13k']:.6f} | {results['regularized_127']['baseline']['us_13k']:.6f} | {results['regularized_127']['baseline']['delta_13k']:+.6f} | [{results['regularized_127']['baseline']['ci_13k'][0]:+.6f}, {results['regularized_127']['baseline']['ci_13k'][1]:+.6f}] |
| **regularized_127** | country_dual | **{results['regularized_127']['country_dual']['f05_13k']:.6f}** | {results['regularized_127']['country_dual']['india_13k']:.6f} | {results['regularized_127']['country_dual']['us_13k']:.6f} | **{results['regularized_127']['country_dual']['delta_13k']:+.6f}** | [{results['regularized_127']['country_dual']['ci_13k'][0]:+.6f}, {results['regularized_127']['country_dual']['ci_13k'][1]:+.6f}] |

---

## 3. Screening Parity Check (2,000 Queries)
- **control_b0 baseline:** `{results['control_b0']['baseline']['f05_2k']:.6f}` (Expected: 0.904586)
- **control_b0 country_dual:** `{results['control_b0']['country_dual']['f05_2k']:.6f}` (Expected: 0.906774)

Total execution time: {time.time() - t_start:.2f}s.
"""
    report_file = reports_dir / "L02_comparison_confirmation_report.md"
    report_file.write_text(report_md, encoding="utf-8")
    print(f"Saved Markdown report to {report_file}")
    print(f"\n=== L02 Confirmation Completed in {time.time() - t_start:.2f}s ===")


if __name__ == "__main__":
    main()
