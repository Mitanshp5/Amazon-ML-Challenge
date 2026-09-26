"""Label-blind natural candidate generation API (Step 1: D1-00).

Contract invariants:
1. Label-blind: accepts NO ground-truth or target labels.
2. Natural retrieval: combines lexical channels (joint, name_only, address_only),
   structured index, and duplicate-observation expansion via RRF fusion.
3. Complete query coverage: every input query is present in the output, even
   if 0 candidates were found (vital for correct singleton macro F0.5 scoring).
4. Provenance preservation: per-channel score and rank sidecars are preserved
   for O(1) feature matrix construction.
5. Deterministic hash verification: hash_candidate_sets() allows asserting
   candidate invariant checksums across runs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from er.retrieval.lexical import channel_texts, topn_search
from er.retrieval.structured import score_query

DEFAULT_WEIGHTS = {
    "joint": 1.2,
    "address_only": 1.1,
    "name_only": 0.8,
    "structured": 0.7,
}

DEFAULT_K_PER_CHANNEL = {
    "joint": 100,
    "name_only": 100,
    "address_only": 150,
    "structured": 100,
}


def compute_rrf_rankings(
    channel_results: dict[str, tuple[dict[str, list[str]], dict[str, list[float]]]],
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> dict[str, list[tuple[str, float]]]:
    """Compute Reciprocal Rank Fusion (RRF) scores across retrieval channels."""
    weights = weights or DEFAULT_WEIGHTS
    q_scores: dict[str, dict[str, float]] = {}
    for ch_name, (cands_map, _) in channel_results.items():
        w = weights.get(ch_name, 1.0)
        for qid, pid_list in cands_map.items():
            slot = q_scores.setdefault(qid, {})
            for rank, pid in enumerate(pid_list, start=1):
                slot[pid] = slot.get(pid, 0.0) + w / (rrf_k + rank)
    sorted_out: dict[str, list[tuple[str, float]]] = {}
    for qid, pmap in q_scores.items():
        sorted_pairs = sorted(pmap.items(), key=lambda kv: (-kv[1], str(kv[0])))
        sorted_out[qid] = sorted_pairs
    return sorted_out


def build_channel_lookups(
    channel_results: dict[str, tuple[dict[str, list[str]], dict[str, list[float]]]],
    query_ids: Sequence[str],
) -> dict[str, dict[str, tuple[float, int]]]:
    """Convert channel results to fast dict lookups: qid -> pid -> (score, rank)."""
    ch_lookups: dict[str, dict[str, dict[str, tuple[float, int]]]] = {}
    for ch_name, (c_map, s_map) in channel_results.items():
        ch_lookups[ch_name] = {
            qid: {pid: (s_map[qid][idx], idx + 1) for idx, pid in enumerate(c_map[qid])}
            for qid in query_ids
            if qid in c_map
        }
    return ch_lookups


def expand_duplicate_candidates(
    rrf_rankings: dict[str, list[tuple[str, float]]],
    dupe_map: Mapping[str, list[str]],
    discount: float = 0.99,
) -> dict[str, list[tuple[str, float]]]:
    """Expand retrieved candidates with identical-text duplicate pool records."""
    out: dict[str, list[tuple[str, float]]] = {}
    for qid, pairs in rrf_rankings.items():
        existing_ids = {pid for pid, _ in pairs}
        extra_dupes = []
        for pid, sc in pairs:
            if pid in dupe_map:
                for other_id in dupe_map[pid]:
                    if other_id not in existing_ids:
                        existing_ids.add(other_id)
                        extra_dupes.append((other_id, sc * discount))
        out[qid] = sorted(pairs + extra_dupes, key=lambda kv: (-kv[1], str(kv[0])))
    return out


def generate_natural_candidates(
    query_ids: Sequence[str],
    query_names: Sequence[str],
    query_addrs: Sequence[str],
    country: str,
    pool_ids: Sequence[str],
    lexical_artifacts: dict[str, tuple[Any, Any]],  # mode -> (vectorizer, p_mat)
    structured_index: Any,
    dupe_map: Mapping[str, list[str]],
    k_per_channel: dict[str, int] | None = None,
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    top_k_final: int = 100,
    n_threads: int = 12,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, dict[str, dict[str, tuple[float, int]]]]]:
    """Retrieve natural candidate sets across all channels without ground-truth.

    Returns:
        candidates_by_q: qid -> list of (cand_id, rrf_score) capped to top_k_final
        ch_lookups: ch_name -> qid -> cand_id -> (channel_score, channel_rank)
    """
    k_cfg = dict(DEFAULT_K_PER_CHANNEL)
    if k_per_channel:
        k_cfg.update(k_per_channel)

    channel_results: dict[str, tuple[dict[str, list[str]], dict[str, list[float]]]] = {}

    # 1. Lexical Channels (sparse matrix top-N)
    for mode in ("joint", "name_only", "address_only"):
        if mode in lexical_artifacts:
            vec, p_mat = lexical_artifacts[mode]
            q_txt = channel_texts(query_names, query_addrs, mode)
            q_mat = vec.transform(q_txt)
            k = k_cfg.get(mode, 100)
            c, s = topn_search(q_mat, p_mat.T, pool_ids, top_k=k, threshold=0.0, n_threads=n_threads)
            channel_results[mode] = (
                {qid: cands for qid, cands in zip(query_ids, c)},
                {qid: scores for qid, scores in zip(query_ids, s)},
            )

    # 2. Structured Channel (exact name, name3+pin)
    if structured_index is not None:
        struct_cands = {}
        struct_scores = {}
        for q, qn, qa in zip(query_ids, query_names, query_addrs):
            ranked = score_query(qn, qa, structured_index, stop=set())
            k = k_cfg.get("structured", 100)
            struct_cands[q] = [pool_ids[pi] for pi, _ in ranked[:k]]
            struct_scores[q] = [sc for _, sc in ranked[:k]]
        channel_results["structured"] = (struct_cands, struct_scores)

    # 3. Channel lookups for O(1) feature retrieval
    ch_lookups = build_channel_lookups(channel_results, query_ids)

    # 4. RRF Multi-Channel Fusion
    rrf_rankings = compute_rrf_rankings(channel_results, weights=weights, rrf_k=rrf_k)

    # 5. Duplicate Observation Expansion
    if dupe_map:
        rrf_rankings = expand_duplicate_candidates(rrf_rankings, dupe_map)

    # 6. Final candidate list formatting (ensure all query_ids exist, cap to top_k_final)
    candidates_by_q: dict[str, list[tuple[str, float]]] = {}
    for qid in query_ids:
        cands = rrf_rankings.get(qid, [])
        candidates_by_q[qid] = cands[:top_k_final]

    return candidates_by_q, ch_lookups


def hash_candidate_sets(candidates_by_q: Mapping[str, Sequence[tuple[str, float]]]) -> str:
    """Compute deterministic SHA-256 fingerprint of candidates for integrity checks."""
    canonical_list = []
    for qid in sorted(candidates_by_q.keys()):
        cands = [(str(pid), round(float(sc), 6)) for pid, sc in candidates_by_q[qid]]
        canonical_list.append((qid, cands))
    s = json.dumps(canonical_list, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
