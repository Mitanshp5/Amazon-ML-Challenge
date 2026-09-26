"""Identity-group splits with exposure ledger (F05 Phase A).

Replaces notebook cell 8's first-byte hash bucket (26/256 ~10.16%, sorted-ID
prefix 20k sample) with a full-width seeded hash + stratified manifests.

Rules:
- One S1 + all its GT aliases = one supervised identity group.
- Full GT audit: no shared positive target across S1, so S1 grouping suffices;
  near-duplicate observations are audited separately, not merged silently.
- The 224,776 legacy 'validation' IDs and the 20,000 repeatedly inspected IDs
  are development-exposed and must NEVER become a new unbiased holdout.
- Query subsampling is allowed; corpus must keep full S2/S3 pools incl.
  unassigned distractors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def group_key(s1: str, seed: str = "f05-v1") -> int:
    h = hashlib.sha256(f"{seed}|{s1}".encode()).hexdigest()
    return int(h[:16], 16)


def assign_split(s1: str, n_train: int = 80, n_dev: int = 10, n_hold: int = 10,
                 seed: str = "f05-v1") -> str:
    total = n_train + n_dev + n_hold
    r = group_key(s1, seed) % total
    if r < n_train:
        return "train"
    if r < n_train + n_dev:
        return "dev"
    return "holdout"


def bucket_match_count(n: int) -> str:
    if n == 0:
        return "0-singleton"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    return "6+"


def build_splits(s1_ids: list, gt_counts: dict, exposed_ids: set,
                 seed: str = "f05-v1") -> dict:
    """Assign splits; force every exposed ID into dev (never holdout)."""
    out = {}
    for s1 in s1_ids:
        if s1 in exposed_ids:
            out[s1] = "dev-exposed"
        else:
            out[s1] = assign_split(s1, seed=seed)
    return out


def write_manifest(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path
