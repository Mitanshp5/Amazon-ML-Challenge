"""Run/output manifests: identity, atomicity, resume (F05 16.3).

Fixes notebook P0s:
- same output filenames overwritten at different stages -> run-specific
  immutable output directories; paired files exported from one manifest only.
- resume on line count; merge on count/no-duplicate -> atomic writes,
  checksums, exact chunk ID sets, manifest equality, exact final coverage.
- fixed locked filenames loaded regardless of query membership -> manifests
  carry exact query/pool IDs, source hashes, normalizer + full config; assert.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_state(root: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                             text=True).strip())
    except Exception:
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def verify_chunk_sets(required_ids: set, chunk_id_sets: list) -> None:
    seen: set = set()
    for s in chunk_id_sets:
        dup = seen & set(s)
        if dup:
            raise ValueError(f"duplicate S1 across chunks: {sorted(list(dup))[:5]}")
        seen |= set(s)
    if seen != set(required_ids):
        raise ValueError(
            f"chunk coverage mismatch: missing={len(set(required_ids)-seen)} "
            f"extra={len(seen-set(required_ids))}")
