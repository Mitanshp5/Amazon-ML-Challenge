"""Freeze historical baseline (F05 E00): immutable copy + hashes, no overwrites."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "notebooks" / "output-local"
FREEZE = ROOT / "runs" / "historical_2026-09-26_baseline"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    FREEZE.mkdir(parents=True, exist_ok=True)
    small = ["model_config.json", "model_config_base08626.json"]
    for fn in small:
        src = OUT / fn
        if src.exists():
            shutil.copy2(src, FREEZE / fn)
    manifest = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          text=True).strip(),
        "files": {},
        "notes": ("Immutable historical pointer. Large pickles/models referenced by "
                  "hash, not duplicated. Do not overwrite."),
    }
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            rel = p.relative_to(OUT).as_posix()
            manifest["files"][rel] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    (FREEZE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"freeze: {len(manifest['files'])} files -> {FREEZE}")


if __name__ == "__main__":
    main()
