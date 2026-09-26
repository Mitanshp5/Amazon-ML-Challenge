# MacBook Setup — Test-Shard Worker (US + France)

Role: this machine runs **test blocking for the US and France shards only**.
India runs on the Windows box; the final merge also happens there.
Do not retrain, re-tune, or re-block the sample set here — frozen artifacts stay untouched.

## 0. Prerequisites (already confirmed present)

- Homebrew + Xcode CLT installed.
- ~15 GB free disk (data ~3.7 GB + working headroom ~10 GB).

## 1. System packages

```zsh
brew install python@3.11 git
```

## 2. Clone (same commit as the merge machine — mandatory)

```zsh
git clone https://github.com/Mitanshp5/ML-Devs.git
cd ML-Devs
git rev-parse --short HEAD   # must match the merge machine's commit (recorded in its test manifest)
```

If the merge machine pushes a newer commit mid-run, **do not pull** until your shards finish —
mixed versions produce inconsistent candidates and the merge gate will reject them.

## 3. Data placement (same relative layout, required)

Inside the clone, recreate exactly:

```
student_resource/student_resource/dataset/test/test_source1.tsv
student_resource/student_resource/dataset/test/test_source2.tsv
student_resource/student_resource/dataset/test/test_source3.tsv
student_resource/student_resource/dataset/train/train_source1.tsv
student_resource/student_resource/dataset/train/train_source2.tsv
student_resource/student_resource/dataset/train/train_source3.tsv
student_resource/student_resource/dataset/train/train_ground_truth.tsv
```

Train files are needed only for the setup chain (val split + skeleton + Phase 2 definitions);
no train compute runs here. Copy via USB drive or Drive download — pick whatever is faster,
then verify row counts open (spot-check file sizes match the Windows box).

## 4. Checkpoints (copy once via Drive — saves a 35-min rebuild)

The `checkpoints/` dir is gitignored, so clone it empty. Copy these 7 files from the
Windows box (`notebooks/output-local/checkpoints/`) into the same relative path here:

```
candidates_v3_india_8065.pkl
candidates_v3_us_11935.pkl
candidates_v3_us_11935_BASELINE_9317.pkl
candidates_v4_india_8065.pkl
candidates_v4_us_11935.pkl
candidates_v4_us_11935_PRE80.pkl
candidates_v5_us_11935.pkl
```

(~140 MB total.) With them present, Phase 2 loads the frozen winners in seconds
(`LOCKED: loading …`); without them it falls back to a full recompute — slow but valid.

## 5. venv + packages (ARM notes)

```zsh
python3.11 -m venv venv
source venv/bin/activate
pip install -q pandas scikit-learn scipy numpy psutil rapidfuzz lightgbm sparse_dot_topn
python -c "import sparse_dot_topn; print('sparse path OK')"
```

- If `sparse_dot_topn` has no arm64 wheel: `brew install libomp`, then
  `pip install --no-binary sparse_dot_topn sparse_dot_topn` (builds from source).
- If the build fails: stop and report back — fallback is batched sklearn matmul
  (slower, needs a notebook edit; do not improvise it locally).
- `faiss-cpu` / `openvino` are **not** needed for the test-blocking path — skip them.
- Jupyter: `pip install -q notebook` (or use VS Code with the Python + Jupyter extensions).

## 6. Cells to run (`notebooks/entity_resolution_local.ipynb`, venv kernel)

In order, top to bottom — stop after smoke:

| # | Cell | Must print |
|---|---|---|
| 1 | `%pip install` | once only |
| 2 | Config (RAM budgets) | `OS=Darwin … MODE=…`, `DATA_ROOT = …/dataset` (must succeed) |
| 3 | Phase 0 scorer + unit test | `pdf-example=0.7142857` |
| 4 | Phase 0 val split | `GT rows=2206821 val_S1=224776` |
| 5 | Phase 1 normalization lib | `v4 … asserts OK` |
| 6 | Phase 0.5 skeleton | `mean K=1.9`, `macro-F0.5 ~0.088` |
| 7 | Phase 2 country blocking | `LOCKED: loading …` for **both** shards (seconds). If it recomputes instead, the checkpoint copy (step 4) is missing/wrongly placed — fix that, don't continue into a 35-min rebuild blind |
| 8 | Recall report | India 92.82 / US 95.03 / overall 94.15 |
| 9 | `p5-smoke` | `SMOKE GATE: PASS` (~10 min). **This is the ARM proof** — it exercises `sparse_dot_topn` on real per-shard pools |

**Do NOT run:** 2c, 3a–3c (matcher already final), FAISS demo (optional), OV cells (self-skip),
`p5-testblock` until step 7 passes, `p5-merge` (merge happens on the Windows box, never here).

## 7. Full shards (after smoke PASS)

In `p5-testblock`, set **only**:

```python
TEST_SHARDS = ["US", "France"]
RUN_TEST = True
```

Leave `TEST_TOPK = {"India": 90, "US": 80, "__default__": 40}`, threshold, and chunk size
untouched — identical config on both devices is a merge requirement, not a suggestion.
Then run the cell and leave the machine: plugged in, ventilated, `caffeinate -i` against
sleep. Expect ~10–11h (fanless throttling margin included). Per-chunk ETA prints as it goes;
any interruption resumes in seconds via `resume-skip` — just re-run the cell.

## 8. Hand the parts back

Copy this machine's `notebooks/output-local/test_parts_full/test_US_*.tsv` and
`test_France_*.tsv` to the Windows box's `notebooks/output-local/test_parts_full/`
(Drive/USB — filenames carry chunk indexes, so nothing collides). The merge + coverage
gate runs there (`p5-merge`). Do not rename files.

## 9. Tripwires (stop and report)

- Smoke prints anything other than `SMOKE GATE: PASS`.
- Phase 2 recomputes instead of `LOCKED: loading` (checkpoints misplaced).
- A chunk fails twice in a row (note the chunk id + error text).
- `git rev-parse --short HEAD` differs from the merge machine's commit.
- Free disk drops under 5 GB mid-run.
