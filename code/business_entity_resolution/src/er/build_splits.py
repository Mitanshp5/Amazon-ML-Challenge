"""Build fresh grouped splits + 20k dev probe (F05 8.1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from er.splits import assign_split, bucket_match_count, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", default="f05-v1")
    ap.add_argument("--dev-n", type=int, default=20000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_counts: dict = {}
    for ch in pd.read_csv(args.gt, sep="\t", dtype=str, keep_default_na=False,
                          chunksize=200000):
        for r in ch.itertuples():
            s1, raw = r[1], r[2].strip()
            gt_counts[s1] = 0 if raw == "" else raw.count(",") + 1
    s1_ids: list = []
    countries: dict = {}
    for ch in pd.read_csv(Path(args.train_dir) / "train_source1.tsv", sep="\t",
                          dtype=str, keep_default_na=False, chunksize=200000):
        for r in ch.itertuples():
            s1_ids.append(r.entity_id)
            countries[r.entity_id] = r.country
    # legacy exposed: first-byte md5%10==0 bucket is dev-exposed by definition;
    # plus any ID list found under notebooks/output-local candidate pickles is
    # treated as exposed (conservative). Here: hash bucket only (pickles carry
    # the same 20k sample universe).
    import hashlib
    exposed = {s for s in s1_ids if hashlib.md5(s.encode()).digest()[0] % 10 == 0}
    splits = {}
    for s in s1_ids:
        splits[s] = "dev-exposed" if s in exposed else assign_split(s, seed=args.seed)
    # fresh dev probe: unexposed, deterministic by seeded hash order
    import hashlib as _h
    fresh = sorted([s for s in s1_ids if s not in exposed],
                   key=lambda s: _h.sha256(f"{args.seed}|{s}".encode()).hexdigest())
    probe = fresh[:args.dev_n]
    payload = {
        "seed": args.seed,
        "n_s1": len(s1_ids),
        "counts": {k: sum(1 for v in splits.values() if v == k) for k in set(splits.values())},
        "exposed_legacy_n": len(exposed),
        "probe_n": len(probe),
        "bucket_by_split": {},
    }
    for s, sp in splits.items():
        b = bucket_match_count(gt_counts.get(s, 0))
        payload["bucket_by_split"].setdefault(sp, {}).setdefault(b, 0)
        payload["bucket_by_split"][sp][b] += 1
    write_manifest(out_dir / "splits.json", {"seed": args.seed, "splits": splits})
    write_manifest(out_dir / "dev_probe_20k.json", {"seed": args.seed, "query_ids": probe})
    write_manifest(out_dir / "split_report.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
