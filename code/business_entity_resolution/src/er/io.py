"""TSV IO with exact contract (F05 18.1): tab, UTF-8, exact headers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MATCH_COL = "matched_entity_ids"
CAND_COL = "candidate_entity_ids"
S1_COL = "source1_entity_id"


def write_id_lists(path: Path, rows: dict, col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(f"{S1_COL}\t{col}\n")
        for qid in sorted(rows.keys()):
            vals = sorted(set(str(v) for v in rows[qid]))
            f.write(f"{qid}\t{','.join(vals)}\n")
    tmp.replace(path)


def read_id_lists(path: Path, col: str) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(df.columns) == [S1_COL, col], f"bad header {list(df.columns)}"
    out = {}
    for r in df.itertuples():
        qid = r[1]
        raw = r[2].strip()
        out[qid] = [] if raw == "" else raw.split(",")
    return out
