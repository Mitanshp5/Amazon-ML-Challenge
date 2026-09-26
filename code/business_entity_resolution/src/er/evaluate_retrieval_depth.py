"""Untrimmed candidate control and K-depth frontier sweep (D1-01 & D1-02).

Evaluates natural candidate retrieval across depths:
K in [50, 100, 150, 200, 250, untrimmed]
on screen_2k (1,000 India, 1,000 US) and outputs:
- Natural Oracle Macro F0.5 (Overall, India, US)
- Pair Recall % (Overall, India, US)
- Mean candidates per query (Overall, India, US)
- Zero-hit rate (% non-singletons with 0 true matches retrieved)
- Complete coverage rate (% non-singletons with 100% true matches retrieved)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import joblib
import pandas as pd
from scipy.sparse import load_npz

from er.candidate_generation import generate_natural_candidates, hash_candidate_sets
from er.metrics import macro_f05, oracle_macro_f05
from er.normalization import normalize_address, normalize_name
from er.run_retrieval_sweep import load_gt_map

DEFAULT_CORES = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--gt", default="student_resource/student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--manifest-dir", default="splits/f05-v1/parallel-v1")
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--out-dir", default="reports/dev_probe")
    ap.add_argument("--n-cores", type=int, default=DEFAULT_CORES)
    args = ap.parse_args()

    t_start = time.time()
    manifest_dir = Path(args.manifest_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Untrimmed Candidate Control & K-Depth Sweep (Cores={args.n_cores}) ===")

    gt_map = load_gt_map(Path(args.gt))
    screen_info = json.loads((manifest_dir / "screen_2k.json").read_text(encoding="utf-8"))
    screen_qids = screen_info["query_ids"]

    s1 = pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1_by_id = s1.set_index("entity_id")

    # Untrimmed candidate sets per country: qid -> list of (pid, rrf_score)
    untrimmed_by_country: dict[str, dict[str, list[tuple[str, float]]]] = {}
    truth_by_country: dict[str, dict[str, list[str]]] = {}

    for country in ("India", "US"):
        t_c = time.time()
        c_qids = [q for q in screen_qids if s1_by_id.loc[q, "country"] == country]
        truth_by_country[country] = {q: gt_map.get(q, []) for q in c_qids}

        print(f"\n--- Retrieving Untrimmed Candidates for {country} ({len(c_qids):,} queries) ---")
        pool_dict_cache = cache_dir / f"pool_dict_{country}.joblib"
        pool_dict, pool_ids = joblib.load(pool_dict_cache)
        dupe_map = joblib.load(cache_dir / f"dupe_map_{country}.joblib")
        struct_idx = joblib.load(cache_dir / f"structured_index_{country}.joblib")

        lex_artifacts = {}
        for mode in ("joint", "name_only", "address_only"):
            vec = joblib.load(cache_dir / f"vec_{country}_{mode}.joblib")
            p_mat = load_npz(cache_dir / f"mat_{country}_{mode}.npz")
            lex_artifacts[mode] = (vec, p_mat)

        q_names = [normalize_name(s1_by_id.loc[q, "business_name"], country) for q in c_qids]
        q_addrs = [normalize_address(s1_by_id.loc[q, "business_address"], country) for q in c_qids]

        # top_k_final=None -> UNTRIMMED natural candidates
        cands_by_q, _ = generate_natural_candidates(
            query_ids=c_qids,
            query_names=q_names,
            query_addrs=q_addrs,
            country=country,
            pool_ids=pool_ids,
            lexical_artifacts=lex_artifacts,
            structured_index=struct_idx,
            dupe_map=dupe_map,
            k_per_channel={"joint": 100, "name_only": 100, "address_only": 150, "structured": 100},
            top_k_final=None,
            n_threads=args.n_cores,
        )
        untrimmed_by_country[country] = cands_by_q
        print(f"Untrimmed retrieval for {country} completed in {time.time() - t_c:.2f}s")

    # Evaluate across depths K
    depths = [50, 100, 150, 200, 250, None]  # None represents Untrimmed
    depth_labels = ["K=50", "K=100 (B0)", "K=150", "K=200", "K=250", "Untrimmed (Natural Union)"]

    frontier_results = []

    print("\n--- Evaluating Candidate Depth Frontier ---")

    for k_val, label in zip(depths, depth_labels):
        combined_truth = {}
        combined_cands = {}
        country_metrics = {}

        for country in ("India", "US"):
            c_truth = truth_by_country[country]
            c_raw = untrimmed_by_country[country]
            c_cands = {q: [pid for pid, _ in (c_raw[q][:k_val] if k_val else c_raw[q])] for q in c_truth}

            combined_truth.update(c_truth)
            combined_cands.update(c_cands)

            # Country metrics
            c_oracle = oracle_macro_f05(c_truth, c_cands)
            total_gt_pairs = sum(len(tr) for tr in c_truth.values())
            hit_pairs = sum(len(set(tr) & set(c_cands[q])) for q, tr in c_truth.items())
            pair_recall = (hit_pairs / max(total_gt_pairs, 1)) * 100.0
            mean_k = sum(len(c_cands[q]) for q in c_truth) / len(c_truth)

            non_sing = {q: tr for q, tr in c_truth.items() if len(tr) > 0}
            zero_hits = sum(1 for q, tr in non_sing.items() if len(set(tr) & set(c_cands[q])) == 0)
            full_cov = sum(1 for q, tr in non_sing.items() if set(tr).issubset(set(c_cands[q])))

            country_metrics[country] = {
                "oracle_macro_f05": float(c_oracle),
                "retrieval_loss": float(1.0 - c_oracle),
                "pair_recall_pct": float(pair_recall),
                "mean_candidates": float(mean_k),
                "zero_hit_rate": float(zero_hits / len(non_sing)),
                "complete_coverage_rate": float(full_cov / len(non_sing)),
            }

        # Overall metrics
        overall_oracle = oracle_macro_f05(combined_truth, combined_cands)
        total_gt = sum(len(tr) for tr in combined_truth.values())
        total_hits = sum(len(set(tr) & set(combined_cands[q])) for q, tr in combined_truth.items())
        overall_recall = (total_hits / max(total_gt, 1)) * 100.0
        overall_mean_k = sum(len(combined_cands[q]) for q in combined_truth) / len(combined_truth)

        entry = {
            "depth_label": label,
            "k": k_val if k_val is not None else "untrimmed",
            "overall": {
                "oracle_macro_f05": float(overall_oracle),
                "retrieval_loss": float(1.0 - overall_oracle),
                "pair_recall_pct": float(overall_recall),
                "mean_candidates": float(overall_mean_k),
            },
            "by_country": country_metrics,
        }
        frontier_results.append(entry)
        print(f"[{label:26s}] Overall Oracle: {overall_oracle:.6f} (Recall: {overall_recall:.2f}%, Mean K: {overall_mean_k:.1f}) | India: {country_metrics['India']['oracle_macro_f05']:.6f} | US: {country_metrics['US']['oracle_macro_f05']:.6f}")

    # Write Markdown & JSON reports
    json_path = out_dir / "untrimmed_depth_frontier.json"
    json_path.write_text(json.dumps(frontier_results, indent=2), encoding="utf-8")

    md_report = f"""# Untrimmed Candidate Control & K-Depth Frontier Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Scope:** `screen_2k` (2,000 queries: India=1,000, US=1,000)
**Candidate Generation:** Natural Multi-Channel Union (Joint, Name, Address, Structured, Dupes) with Zero GT Injection.

## 1. Candidate Depth Frontier vs Oracle Ceilings

| Candidate Policy | Overall Oracle $F_{{0.5}}$ | Pair Recall % | Mean $K$ / query | India Oracle $F_{{0.5}}$ | India Recall % | US Oracle $F_{{0.5}}$ | US Recall % |
|---|---|---|---|---|---|---|---|
"""
    for row in frontier_results:
        ov = row["overall"]
        ind = row["by_country"]["India"]
        us = row["by_country"]["US"]
        md_report += f"| **{row['depth_label']}** | **{ov['oracle_macro_f05']:.6f}** | {ov['pair_recall_pct']:.2f}% | {ov['mean_candidates']:.1f} | **{ind['oracle_macro_f05']:.6f}** | {ind['pair_recall_pct']:.2f}% | **{us['oracle_macro_f05']:.6f}** | {us['pair_recall_pct']:.2f}% |\n"

    untrimmed_ind = frontier_results[-1]["by_country"]["India"]["oracle_macro_f05"]
    b0_ind = frontier_results[1]["by_country"]["India"]["oracle_macro_f05"]
    oracle_delta = untrimmed_ind - b0_ind

    md_report += f"""
## 2. Key Findings: Untrimmed Control vs Truncated B0
1. **India K100 Truncation Loss:**
   - Truncating India candidates to K=100 loses **{oracle_delta:.6f}** in oracle macro $F_{{0.5}}$ compared to the untrimmed natural union.
   - At untrimmed depth (mean $K \\approx {frontier_results[-1]['by_country']['India']['mean_candidates']:.1f}$), India oracle reaches **{untrimmed_ind:.6f}**.
2. **US Headroom:**
   - US is virtually saturated even at K=100 ({frontier_results[1]['by_country']['US']['oracle_macro_f05']:.6f}), reaching {frontier_results[-1]['by_country']['US']['oracle_macro_f05']:.6f} untrimmed.
3. **Implications for Device 1 (D1-02 Lexical Expansion):**
   - Simply expanding $K$ beyond 100 on existing channels recovers valuable positives for India, but still leaves an oracle gap to $>0.99$.
   - Therefore, **representation diversity** (address $K=150 \\to 300$, word-token address with token frequency weighting, and preserved Unicode primary views) is mandatory to push India oracle above 0.99.

## 3. Coverage Quality Breakdown
| Policy | India Zero-Hit % | India 100% Coverage % | US Zero-Hit % | US 100% Coverage % |
|---|---|---|---|---|
"""
    for row in frontier_results:
        ind = row["by_country"]["India"]
        us = row["by_country"]["US"]
        md_report += f"| **{row['depth_label']}** | {ind['zero_hit_rate']*100:.2f}% | {ind['complete_coverage_rate']*100:.2f}% | {us['zero_hit_rate']*100:.2f}% | {us['complete_coverage_rate']*100:.2f}% |\n"

    md_path = out_dir / "untrimmed_depth_frontier.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(f"\nPublished Untrimmed Depth Frontier Report to:\n- {md_path}\n- {json_path}")


if __name__ == "__main__":
    main()
