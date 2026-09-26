"""Dev-probe channel comparison (F05 E02 first cut, small scale).

Compares on a fixed query sample with the FULL country pool:
  A joint (baseline analogue) vs B name-only vs C address-only vs union.
Reports oracle macro-F0.5 + pair recall + mean K. No thresholds, no matcher:
this isolates representation signal before compression.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from er.metrics import oracle_macro_f05
from er.normalization import normalize_address, normalize_name
from er.retrieval.lexical import channel_texts, retrieve_channel


def load_gt_map(gt_path: Path) -> dict:
    out = {}
    for ch in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        for r in ch.itertuples():
            s1, tgt = r[1], r[2]
            out.setdefault(s1, []).append(tgt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--country", default="US")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    train_dir = Path(args.train_dir)
    gt = load_gt_map(Path(args.gt))
    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1 = s1[s1.country == args.country]
    qids = sorted(set(gt) & set(s1.entity_id.tolist()))
    qids = qids[args.seed:args.seed + args.n_queries]
    qmap = {r.entity_id: r for r in s1.itertuples() if r.entity_id in set(qids)}

    pool_frames = []
    for fn in ("train_source2.tsv", "train_source3.tsv"):
        for ch in pd.read_csv(train_dir / fn, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=250000):
            ch = ch[ch.country == args.country]
            if len(ch):
                pool_frames.append(ch[["entity_id", "business_name", "business_address"]])
    pool = pd.concat(pool_frames, ignore_index=True).drop_duplicates("entity_id")
    pool_ids = pool.entity_id.tolist()

    q_names = [normalize_name(qmap[q].business_name, args.country) for q in qids]
    q_addrs = [normalize_address(qmap[q].business_address, args.country) for q in qids]
    p_names = [normalize_name(n, args.country) for n in pool.business_name.fillna("").tolist()]
    p_addrs = [normalize_address(a, args.country) for a in pool.business_address.fillna("").tolist()]

    results = {}
    cfgs = {
        "joint": ("joint", 100, {"min_df": 3, "max_df": 0.35}),
        "name_only": ("name_only", 100, {"min_df": 2, "max_df": 0.5}),
        "address_only": ("address_only", 150, {"min_df": 2, "max_df": 0.5}),
    }
    union = {}
    for ch_name, (mode, k, vkw) in cfgs.items():
        c, s, info = retrieve_channel(
            channel_texts(q_names, q_addrs, mode),
            channel_texts(p_names, p_addrs, mode),
            pool_ids, qids, top_k=k, threshold=0.0, n_threads=8, **vkw)
        truth = {q: gt[q] for q in qids if q in gt}
        oracle = oracle_macro_f05({q: set(v) for q, v in truth.items()},
                                  {q: set(c[q]) for q in qids})
        hits = sum(len(set(c[q]) & set(truth.get(q, ()))) for q in qids)
        tot = sum(len(truth.get(q, ())) for q in qids)
        mk = sum(len(c[q]) for q in qids) / len(qids)
        results[ch_name] = {"oracle_macro_f05": oracle, "pair_recall": hits / tot,
                            "mean_k": mk, "info": info}
        for q in qids:
            union.setdefault(q, set()).update(c[q])
    truth = {q: gt[q] for q in qids if q in gt}
    oracle_u = oracle_macro_f05({q: set(v) for q, v in truth.items()}, union)
    hits_u = sum(len(union[q] & set(truth.get(q, ()))) for q in qids)
    tot = sum(len(truth.get(q, ())) for q in qids)
    results["union_all"] = {"oracle_macro_f05": oracle_u,
                            "pair_recall": hits_u / tot,
                            "mean_k": sum(len(v) for v in union.values()) / len(union)}
    results["meta"] = {"country": args.country, "n_queries": len(qids),
                       "pool": len(pool_ids), "seconds": time.time() - t0}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
