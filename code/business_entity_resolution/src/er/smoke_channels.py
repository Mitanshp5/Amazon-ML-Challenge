"""Engineering smoke: channel code paths on a capped pool (NOT a recall claim).

F05 8.2 tier 1: 200 queries, all code paths, pool explicitly capped.
Full-pool quality measurement is a separate step (probe_channels.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from er.normalization import normalize_address, normalize_name
from er.retrieval.lexical import channel_texts, dedup_union, retrieve_channel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--country", default="US")
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--pool-cap", type=int, default=200000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    train_dir = Path(args.train_dir)
    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str,
                     keep_default_na=False, nrows=500000)
    s1 = s1[s1.country == args.country].head(args.n_queries)
    qids = s1.entity_id.tolist()
    q_names = [normalize_name(n, args.country) for n in s1.business_name.fillna("").tolist()]
    q_addrs = [normalize_address(a, args.country) for a in s1.business_address.fillna("").tolist()]
    pool_frames, seen = [], 0
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        for ch in pd.read_csv(train_dir / fn, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=100000):
            ch = ch[ch.country == args.country]
            if len(ch):
                pool_frames.append(ch[["entity_id", "business_name", "business_address"]])
                seen += len(ch)
            if seen >= args.pool_cap:
                break
        if seen >= args.pool_cap:
            break
    pool = pd.concat(pool_frames, ignore_index=True).drop_duplicates("entity_id")
    pool = pool.head(args.pool_cap)
    pool_ids = pool.entity_id.tolist()
    p_names = [normalize_name(n, args.country) for n in pool.business_name.fillna("").tolist()]
    p_addrs = [normalize_address(a, args.country) for a in pool.business_address.fillna("").tolist()]
    out = {}
    for mode, k in (("joint", 20), ("name_only", 20), ("address_only", 20)):
        c, s, info = retrieve_channel(
            channel_texts(q_names, q_addrs, mode),
            channel_texts(p_names, p_addrs, mode),
            pool_ids, qids, top_k=k, threshold=0.0, n_threads=8,
            min_df=2, max_df=0.6, max_features=30000)
        out[mode] = {"mean_k": sum(len(v) for v in c.values()) / len(c),
                     "info": info,
                     "sample_q": qids[0], "sample_n": len(c[qids[0]])}
    channels = {m: ({q: out[m + "_c"][q]} if False else None) for m in ()}
    # union size check via fresh retrieval dicts
    res = {}
    for mode, k in (("joint", 20), ("name_only", 20), ("address_only", 20)):
        c, s, _ = retrieve_channel(
            channel_texts(q_names, q_addrs, mode),
            channel_texts(p_names, p_addrs, mode),
            pool_ids, qids, top_k=k, threshold=0.0, n_threads=8,
            min_df=2, max_df=0.6, max_features=30000)
        res[mode] = (c, s)
    union = dedup_union(res)
    u_sizes = [len(union[q]) for q in qids]
    payload = {"modes": out, "union": {"mean": sum(u_sizes) / len(u_sizes),
                                       "max": max(u_sizes), "min": min(u_sizes)},
               "meta": {"country": args.country, "n_queries": len(qids),
                        "pool": len(pool_ids), "seconds": round(time.time() - t0, 1),
                        "note": "SMOKE ONLY: capped pool, not a recall/oracle measurement"}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
