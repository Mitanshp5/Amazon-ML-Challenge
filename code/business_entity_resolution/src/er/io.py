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


def load_b0_bundle(bundle_dir: str | Path = "runs/parallel-v1/d1/b0_baseline") -> dict:
    """Loads the B0 portable bundle seamlessly from split parts (<32MB each, Git-friendly) or single file."""
    import joblib

    bundle_dir = Path(bundle_dir)
    single_path = bundle_dir / "b0_portable_bundle.joblib"
    train_path = bundle_dir / "b0_train_features.joblib"
    eval_path = bundle_dir / "b0_eval_features.joblib"

    if train_path.exists() and eval_path.exists():
        train_data = joblib.load(train_path)
        eval_data = joblib.load(eval_path)
        return {
            "X_train": train_data["X_train"],
            "y_train": train_data["y_train"],
            "groups_train": train_data["groups_train"],
            "folds_train": train_data["folds_train"],
            "feature_names": train_data["feature_names"],
            "calib_data": eval_data["calib_data"],
            "eval_data": eval_data["eval_data"],
        }
    elif single_path.exists():
        return joblib.load(single_path)
    else:
        raise FileNotFoundError(f"Could not find B0 feature bundle in {bundle_dir}")
