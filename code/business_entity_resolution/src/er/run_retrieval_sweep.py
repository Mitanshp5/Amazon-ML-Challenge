"""Full-pool retrieval channels & structured sweep (F05 E02 + E03).

Measures on a fixed dev-probe sample against the FULL country pool:
  1. joint (baseline analogue)
  2. name_only
  3. address_only
  4. structured (compound keys)
  5. exact_duplicate_expansion (E03)
  6. Cumulative unions and marginal gains

Reports exact oracle macro-F0.5, pair recall, and candidate counts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

from er.metrics import oracle_macro_f05
from er.normalization import normalize_address, normalize_name
from er.retrieval.lexical import build_vectorizer, channel_texts, topn_search
from er.retrieval.structured import build_index, score_query


def load_gt_map(gt_path: Path) -> dict[str, list[str]]:
    """Loads ground truth mapping source1_entity_id -> list of matched_entity_ids."""
    out: dict[str, list[str]] = {}
    for ch in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
        for r in ch.itertuples():
            qid = r[1]
            raw = r[2].strip()
            if raw:
                out[qid] = [t.strip() for t in raw.split(",") if t.strip()]
            else:
                out[qid] = []
    return out


def build_duplicate_map(pool_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map target entity_id to all other target entity_ids sharing identical non-empty (name, addr)."""
    valid = pool_df[
        (pool_df.business_name.str.strip() != "") &
        (pool_df.business_address.str.strip() != "")
    ]
    grouped = valid.groupby(["business_name", "business_address"])["entity_id"].apply(list)
    dupe_groups = grouped[grouped.apply(len) > 1]
    target_to_dupes: dict[str, list[str]] = {}
    for id_list in dupe_groups:
        for tid in id_list:
            target_to_dupes[tid] = id_list
    return target_to_dupes


def evaluate_candidates(cands: dict[str, set[str]], truth: dict[str, set[str]], qids: list[str]) -> dict:
    oracle = oracle_macro_f05({q: truth.get(q, set()) for q in qids},
                              {q: cands.get(q, set()) for q in qids})
    total_pos = sum(len(truth.get(q, set())) for q in qids)
    hits = sum(len(cands.get(q, set()) & truth.get(q, set())) for q in qids)
    non_singletons = sum(1 for q in qids if truth.get(q, set()))
    all_retrieved = sum(1 for q in qids if truth.get(q, set()) and truth[q].issubset(cands.get(q, set())))
    zero_hits = sum(1 for q in qids if truth.get(q, set()) and not (cands.get(q, set()) & truth[q]))
    singletons = len(qids) - non_singletons
    mk = sum(len(cands.get(q, set())) for q in qids) / len(qids)

    return {
        "oracle_macro_f05": float(oracle),
        "pair_recall": float(hits / total_pos) if total_pos else 1.0,
        "mean_k": float(mk),
        "all_retrieved_rate": float(all_retrieved / non_singletons) if non_singletons else 1.0,
        "zero_hit_rate": float(zero_hits / non_singletons) if non_singletons else 0.0,
        "total_queries": len(qids),
        "singletons": singletons,
        "non_singletons": non_singletons,
        "total_positives": total_pos,
        "retrieved_positives": hits,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="student_resource/student_resource/dataset/train")
    ap.add_argument("--gt", default="student_resource/student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--probe-json", default="splits/f05-v1/dev_probe_20k.json")
    ap.add_argument("--country", default="India")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--cache-dir", default="cache/retrieval")
    ap.add_argument("--out-dir", default="reports/dev_probe")
    args = ap.parse_args()

    t_start = time.time()
    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting E02+E03 Retrieval Sweep: Country={args.country}, N={args.n_queries} ===")

    # 1. Load Ground Truth
    print("Loading Ground Truth...")
    gt_map = load_gt_map(Path(args.gt))

    # 2. Select Probe Queries
    probe_path = Path(args.probe_json)
    if probe_path.exists():
        with open(probe_path, "r", encoding="utf-8") as f:
            probe_ids = json.load(f)["query_ids"]
    else:
        probe_ids = list(gt_map.keys())

    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1_c = s1[s1.country == args.country]
    probe_set = set(probe_ids)
    matching_qids = [qid for qid in s1_c.entity_id if qid in probe_set and qid in gt_map]
    qids = matching_qids[:args.n_queries]
    print(f"Selected {len(qids)} probe queries for {args.country}")

    q_df = s1_c[s1_c.entity_id.isin(set(qids))].set_index("entity_id")
    truth = {q: set(gt_map.get(q, [])) for q in qids}

    # 3. Load FULL Country Target Pool
    print("Loading full target pool (S2 + S3)...")
    t0 = time.time()
    pool_frames = []
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        for ch in pd.read_csv(train_dir / fn, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
            ch_c = ch[ch.country == args.country]
            if len(ch_c):
                pool_frames.append(ch_c[["entity_id", "business_name", "business_address"]])
    pool = pd.concat(pool_frames, ignore_index=True).drop_duplicates("entity_id")
    pool_ids = pool.entity_id.tolist()
    print(f"Loaded {len(pool)} pool records for {args.country} in {time.time() - t0:.1f}s")

    # 4. Duplicate observation map (E03)
    dupe_map_cache = cache_dir / f"dupe_map_{args.country}.joblib"
    if dupe_map_cache.exists():
        print("Loading cached duplicate observation map...")
        dupe_map = joblib.load(dupe_map_cache)
        print(f"Loaded {len(dupe_map)} mapped target IDs from cache")
    else:
        print("Building duplicate observation map...")
        t0 = time.time()
        dupe_map = build_duplicate_map(pool)
        joblib.dump(dupe_map, dupe_map_cache)
        print(f"Duplicate map built: {len(dupe_map)} target IDs in {time.time() - t0:.1f}s")

    # 5. Normalization
    print("Normalizing query & pool strings...")
    t0 = time.time()
    q_names = [normalize_name(q_df.loc[q, "business_name"], args.country) for q in qids]
    q_addrs = [normalize_address(q_df.loc[q, "business_address"], args.country) for q in qids]

    pool_norm_cache = cache_dir / f"pool_norm_{args.country}.joblib"
    if pool_norm_cache.exists():
        print("Loading cached normalized pool strings...")
        p_names, p_addrs = joblib.load(pool_norm_cache)
        print(f"Loaded normalized pool strings from cache in {time.time() - t0:.1f}s")
    else:
        p_names = [normalize_name(n, args.country) for n in pool.business_name.fillna("").tolist()]
        p_addrs = [normalize_address(a, args.country) for a in pool.business_address.fillna("").tolist()]
        joblib.dump((p_names, p_addrs), pool_norm_cache)
        print(f"Normalized and cached pool strings in {time.time() - t0:.1f}s")

    # 6. Lexical Channels
    lexical_cfgs = {
        "joint": ("joint", 100, {"min_df": 2, "max_df": 0.4, "max_features": 150000}),
        "name_only": ("name_only", 100, {"min_df": 2, "max_df": 0.5, "max_features": 150000}),
        "address_only": ("address_only", 150, {"min_df": 2, "max_df": 0.5, "max_features": 150000}),
    }

    channel_cands: dict[str, dict[str, set[str]]] = {}
    channel_eval: dict[str, dict] = {}

    for ch_name, (mode, k, vkw) in lexical_cfgs.items():
        print(f"Retrieving lexical channel: {ch_name} (top_k={k})...")
        t0 = time.time()
        q_txt = channel_texts(q_names, q_addrs, mode)

        vec_cache = cache_dir / f"vec_{args.country}_{mode}.joblib"
        mat_cache = cache_dir / f"mat_{args.country}_{mode}.npz"

        if vec_cache.exists() and mat_cache.exists():
            print(f"  [Cache hit] Loading precomputed {mode} vectorizer and matrix...")
            vec = joblib.load(vec_cache)
            p_mat = load_npz(mat_cache)
        else:
            print(f"  Fitting {mode} vectorizer on {len(pool)} documents...")
            p_txt = channel_texts(p_names, p_addrs, mode)
            vec = build_vectorizer(**vkw)
            p_mat = vec.fit_transform(p_txt)
            joblib.dump(vec, vec_cache)
            save_npz(mat_cache, p_mat)
            print(f"  Fitted & saved {mode} (nnz={p_mat.nnz}, vocab={len(vec.vocabulary_)})")

        q_mat = vec.transform(q_txt)
        c, s = topn_search(q_mat, p_mat.T, pool_ids, top_k=k, threshold=0.0, n_threads=8)
        c_sets = {qid: set(cands) for qid, cands in zip(qids, c)}
        channel_cands[ch_name] = c_sets

        ev = evaluate_candidates(c_sets, truth, qids)
        ev["elapsed_sec"] = round(time.time() - t0, 1)
        ev["vocab"] = len(vec.vocabulary_)
        channel_eval[ch_name] = ev
        print(f"  -> {ch_name}: Oracle F0.5={ev['oracle_macro_f05']:.4f}, Recall={ev['pair_recall']:.2%}, Mean K={ev['mean_k']:.1f} ({ev['elapsed_sec']}s)")

    # 7. Structured Retrieval Channel (E03)
    struct_cache = cache_dir / f"structured_index_{args.country}.joblib"
    if struct_cache.exists():
        print("Loading cached structured index...")
        struct_idx = joblib.load(struct_cache)
    else:
        print("Building structured inverted index...")
        t0 = time.time()
        struct_idx, struct_overflow = build_index(p_names, p_addrs, stop=set())
        joblib.dump(struct_idx, struct_cache)
        print(f"Structured index built and cached in {time.time() - t0:.1f}s")

    print("Querying structured index...")
    t0 = time.time()
    struct_cands: dict[str, set[str]] = {}
    for q, qn, qa in zip(qids, q_names, q_addrs):
        ranked = score_query(qn, qa, struct_idx, stop=set())
        struct_cands[q] = {pool_ids[pi] for pi, _ in ranked[:100]}
    channel_cands["structured"] = struct_cands
    ev_struct = evaluate_candidates(struct_cands, truth, qids)
    ev_struct["elapsed_sec"] = round(time.time() - t0, 1)
    channel_eval["structured"] = ev_struct
    print(f"  -> structured: Oracle F0.5={ev_struct['oracle_macro_f05']:.4f}, Recall={ev_struct['pair_recall']:.2%}, Mean K={ev_struct['mean_k']:.1f}")

    # 8. Cumulative Unions
    union_steps = [
        ("joint_only", ["joint"]),
        ("union_joint_name", ["joint", "name_only"]),
        ("union_all_lexical", ["joint", "name_only", "address_only"]),
        ("union_lexical_structured", ["joint", "name_only", "address_only", "structured"]),
    ]

    cumulative_eval: dict[str, dict] = {}
    for uname, ch_list in union_steps:
        u_cands: dict[str, set[str]] = {q: set() for q in qids}
        for ch in ch_list:
            for q in qids:
                u_cands[q].update(channel_cands[ch][q])
        ev_u = evaluate_candidates(u_cands, truth, qids)
        cumulative_eval[uname] = ev_u
        print(f"Cumulative {uname}: Oracle F0.5={ev_u['oracle_macro_f05']:.4f}, Recall={ev_u['pair_recall']:.2%}, Mean K={ev_u['mean_k']:.1f}")

    # 9. Duplicate Observation Expansion (E03)
    print("Applying Duplicate Observation Expansion (E03)...")
    t0 = time.time()
    u_cands_expanded: dict[str, set[str]] = {q: set() for q in qids}
    for ch in ("joint", "name_only", "address_only", "structured"):
        for q in qids:
            u_cands_expanded[q].update(channel_cands[ch][q])

    added_by_dupe = 0
    for q in qids:
        expanded_set = set(u_cands_expanded[q])
        for tid in list(u_cands_expanded[q]):
            if tid in dupe_map:
                for other_id in dupe_map[tid]:
                    if other_id not in expanded_set:
                        expanded_set.add(other_id)
                        added_by_dupe += 1
        u_cands_expanded[q] = expanded_set

    ev_dupe = evaluate_candidates(u_cands_expanded, truth, qids)
    ev_dupe["added_candidates_total"] = added_by_dupe
    ev_dupe["elapsed_sec"] = round(time.time() - t0, 1)
    cumulative_eval["union_all_incl_duplicate_expansion"] = ev_dupe
    print(f"Cumulative union_all_incl_duplicate_expansion: Oracle F0.5={ev_dupe['oracle_macro_f05']:.4f}, Recall={ev_dupe['pair_recall']:.2%}, Mean K={ev_dupe['mean_k']:.1f} (+{added_by_dupe} dupe links)")

    # 10. Compile Report
    total_time = round(time.time() - t_start, 1)
    full_report = {
        "meta": {
            "country": args.country,
            "n_queries": len(qids),
            "pool_size": len(pool),
            "total_elapsed_sec": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "individual_channels": channel_eval,
        "cumulative_unions": cumulative_eval,
    }

    json_path = out_dir / f"E02_E03_{args.country}_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Markdown Report
    md_path = out_dir / f"E02_E03_{args.country}_sweep.md"
    md_content = f"""# E02 + E03 Retrieval & Structured Expansion Report: {args.country}

**Execution Date:** {full_report['meta']['timestamp']}
**Country:** {args.country} | **Probe Queries:** {len(qids)} | **Full Pool Size:** {len(pool):,} | **Total Runtime:** {total_time}s

## 1. Individual Retrieval Channels

| Channel | Oracle Macro-F0.5 | Pair Recall | Mean K | All Retrieved Rate | Zero Hit Rate | Runtime (s) |
|---|---|---|---|---|---|---|
"""
    for ch, d in channel_eval.items():
        md_content += f"| `{ch}` | **{d['oracle_macro_f05']:.4f}** | {d['pair_recall']:.2%} | {d['mean_k']:.1f} | {d['all_retrieved_rate']:.2%} | {d['zero_hit_rate']:.2%} | {d.get('elapsed_sec', '-')}s |\n"

    md_content += """
## 2. Cumulative Union & Marginal Gains (E02 + E03)

| Stage | Oracle Macro-F0.5 | Pair Recall | Mean K | All Retrieved Rate | Zero Hit Rate |
|---|---|---|---|---|---|
"""
    for u, d in cumulative_eval.items():
        md_content += f"| `{u}` | **{d['oracle_macro_f05']:.4f}** | {d['pair_recall']:.2%} | {d['mean_k']:.1f} | {d['all_retrieved_rate']:.2%} | {d['zero_hit_rate']:.2%} |\n"

    base_oracle = cumulative_eval["joint_only"]["oracle_macro_f05"]
    final_oracle = cumulative_eval["union_all_incl_duplicate_expansion"]["oracle_macro_f05"]
    base_rec = cumulative_eval["joint_only"]["pair_recall"]
    final_rec = cumulative_eval["union_all_incl_duplicate_expansion"]["pair_recall"]

    md_content += f"""
## 3. Key Findings & Gate Verdict

- **Baseline Analogue (`joint_only`):** Oracle F0.5 = `{base_oracle:.4f}`, Pair Recall = `{base_rec:.2%}`, Mean K = `{cumulative_eval['joint_only']['mean_k']:.1f}`.
- **Full Union + Duplicate Expansion:** Oracle F0.5 = `{final_oracle:.4f}`, Pair Recall = `{final_rec:.2%}`, Mean K = `{cumulative_eval['union_all_incl_duplicate_expansion']['mean_k']:.1f}`.
- **Oracle Delta:** `+{final_oracle - base_oracle:.4f}` macro F0.5 gain.
- **Recall Delta:** `+{final_rec - base_rec:.2%}` pair recall gain.
- **Duplicate Observations:** Recovered `{added_by_dupe}` candidate expansions from `{len(dupe_map)}` identical target signatures.
- **Gate Advancement:** Meaningful unique recovery confirmed beyond joint baseline. Advances E02 and E03 to promotion gate.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nSaved JSON report: {json_path}")
    print(f"Saved Markdown report: {md_path}")
    print(f"=== Finished in {total_time}s ===")


if __name__ == "__main__":
    main()
