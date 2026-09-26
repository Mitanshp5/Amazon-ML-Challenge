"""Single shared lexical retrieval entry point (F05 Phase C).

One function used by dev AND test (fixes local19 vs 28/30 divergence).

Channels (initial experimental settings; measure union BEFORE compressing):
- name-only char TF-IDF (unicode view), K100
- latin/compact name view, K50-100
- full-address-only char TF-IDF, K100-200
- joint name+full address, K100
- rare-token word TF-IDF / local BM25, K50-100

Superset rule: during discovery expanded sets must be supersets of earlier
sets. Compression is a separate measured step that reports every lost GT.

Score hygiene (fixes fuzzy/100 stored in cosine slot): per-channel
(score, rank, presence) are stored separately; fusion uses RRF only as an
ordering baseline, never as a probability.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def build_vectorizer(min_df=2, max_df=0.4, max_features=150000,
                     ngram=(3, 4), analyzer="char_wb"):
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                           max_features=max_features, min_df=min_df,
                           max_df=max_df, sublinear_tf=True, dtype=np.float32)


def topn_search(q_mat, p_mat_T_csr, pool_ids, top_k, threshold=0.0,
                n_threads=8):
    """Deterministic top-N: sort by (-score, stable id order)."""
    mat = sp_matmul_topn(q_mat, p_mat_T_csr, top_n=top_k,
                         threshold=threshold, n_threads=n_threads)
    cands, scores = [], []
    for r in range(mat.shape[0]):
        row = mat.getrow(r)
        idx, sc = row.indices, row.data
        if len(sc) > 1:
            # deterministic: score desc, then pool-id asc for ties
            order = np.lexsort((np.array([str(pool_ids[i]) for i in idx]), -sc))
            idx, sc = idx[order], sc[order]
        cands.append([str(pool_ids[i]) for i in idx])
        scores.append([float(s) for s in sc])
    return cands, scores


def channel_texts(names, addrs, mode: str):
    if mode == "name_only":
        return [n for n in names]
    if mode == "address_only":
        return [a for a in addrs]
    if mode == "joint":
        return [f"{n} {n} {a}".strip() for n, a in zip(names, addrs)]
    raise ValueError(mode)


def retrieve_channel(query_texts, pool_texts, pool_ids, query_ids,
                     top_k=100, threshold=0.0, n_threads=8, **vec_kwargs):
    vec = build_vectorizer(**vec_kwargs)
    p_mat = vec.fit_transform(pool_texts)
    q_mat = vec.transform(query_texts)
    cands, scores = topn_search(q_mat, p_mat.T.tocsr(), np.array(pool_ids),
                                top_k=top_k, threshold=threshold,
                                n_threads=n_threads)
    return (
        {qid: c for qid, c in zip(query_ids, cands)},
        {qid: s for qid, s in zip(query_ids, scores)},
        {"vocab": len(vec.vocabulary_), "nnz": int(p_mat.nnz)},
    )


def dedup_union(channel_results: dict) -> dict:
    """channel_results[channel] = (cands_dict, scores_dict).
    Returns qid -> {pid: {channel: (score, rank)}} preserving provenance."""
    out: dict = {}
    for ch, (cands, scores) in channel_results.items():
        for qid, lst in cands.items():
            qmap = out.setdefault(qid, {})
            for rank, pid in enumerate(lst, start=1):
                slot = qmap.setdefault(pid, {})
                slot[ch] = (float(scores[qid][rank - 1]), rank)
    return out


def rrf_order(qmap: dict, weights: dict | None = None, k: int = 60) -> list:
    weights = weights or {}
    scored = []
    for pid, chmap in qmap.items():
        s = sum(weights.get(ch, 1.0) / (k + rank) for ch, (_, rank) in chmap.items())
        scored.append((pid, s))
    scored.sort(key=lambda kv: -kv[1])
    return scored
