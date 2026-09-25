# Amazon ML Challenge 2026 — Business Entity Resolution

Match every Source 1 business record against Source 2 + Source 3 records.
Scored with macro-averaged **F0.5** (precision-heavy; singletons included).
Full spec: `amazon_ml_challenge_problem_statement.pdf` + `ENTITY_RESOLUTION_PLAN.md`.

## Repo map

| Path | What |
|---|---|
| `ENTITY_RESOLUTION_PLAN.md` | Solution plan (data analysis, blocking, model ladder, thresholds) |
| `notebooks/entity_resolution.ipynb` | SageMaker `ml.t3.medium` — sample-only pipeline |
| `notebooks/entity_resolution_local.ipynb` | Windows PC (Ultra 9, 32 GB, Arc 140T) + OpenVINO inference + Drive sync |
| `notebooks/entity_resolution_a100.ipynb` | Colab A100 — GPU ANN, fine-tuning, rerank, OpenVINO export |
| `student_resource/` | Challenge files (README, validator, ground truth schema). Dataset TSVs live here locally but are **gitignored** |
| `amazon_ml_challenge_problem_statement.pdf` | Original problem statement |

All three notebooks share identical pipeline logic (scorer, val split,
normalization, skeleton); only budgets and hardware stages differ.
Each notebook's 2nd cell is a **run/skip guide** — read it first.

## Setup per machine

| Machine | Dataset placement | Notes |
|---|---|---|
| Local Windows PC | Already present in repo checkout | Outputs → `notebooks/output-local/` (gitignored) |
| Colab A100 | Drive `MyDrive/AmazonMLChallenge/dataset/` → auto-copied to `/content/dataset` by the notebook | Checkpoints/outputs stay on Drive (`output-a100/`, `checkpoints/`, `models/`) |
| SageMaker `t3.medium` | Upload `dataset/` beside the notebook (or run on Colab — Drive cell auto-copies) | Sample mode only (4 GB RAM) |

Model flow: **Colab fine-tunes → exports OpenVINO IR to Drive → local auto-pulls
`models/minilm-ov` + raw `models/minilm-er` → Arc 140T inference.**
Local sharing: `output-local/*.tsv` + local models sync to Drive
`AmazonMLChallenge/shared/` (oplog: offline-first, delta resume via manifest).
Needs Drive for Desktop running; without it the run just stays local.

## Flow (Phase 0 → 5, see plan §3)

```
TSVs → normalize → validate-split + F0.5 scorer → walking skeleton (PIN block)
     → FAISS blocking (recall ≥95% gate) → matcher (LightGBM → MiniLM → rerank)
     → tau threshold → matching_results.tsv + candidate_pairs.tsv
     → validate_submission.py --check-ids → submission zip + methodology doc
```

## Todo

Done:

- [x] Data analysis (row counts, GT distribution, noise samples)
- [x] Solution plan (reviewed; recall-first blocking, FAISS primary, France tables)
- [x] F0.5 scorer + unit test vs PDF worked example (0.714)
- [x] Hash-based val split + normalization lib + walking skeleton (code)
- [x] Three runtime notebooks + run/skip guides + Drive dataset wiring
- [x] Drive sharing layer (local push/pull, offline-first) + safetensors pin

Next (in order — each unblocks the next):

- [ ] **1. Skeleton PASS.** Run the local notebook sample pipeline to a
  validation `PASS` + first real F0.5 number. Fixes integration bugs early.
- [ ] **2. Full blocking + recall gate.** FAISS index over millions of docs
  (per-country shards, chunked queries), rank-merge + adaptive cap; prove
  ≥95% recall ceiling on val, then minimize mean K (§1.6).
- [ ] **3. Matcher.** LightGBM baseline on pairwise features → MiniLM
  bi-encoder fine-tune (A100) → cross-encoder rerank top-10.
- [ ] **4. Threshold.** Tune global `tau` on val macro-F0.5 (≈0.6–0.8);
  `max_score < tau` → empty (singleton).
- [ ] **5. Test inference.** Chunked full-test run → `matching_results.tsv` +
  `candidate_pairs.tsv` for all 1.73M test S1.
- [ ] **6. Submit.** Validator `--check-ids` PASS → leaderboard upload →
  submission zip (`output/`, runnable `code/`, methodology doc).

## Git rules

Never committed (all gitignored): `*.tsv`, `dataset/`, `output*/`,
`checkpoints/`, `models/`, `sync_manifest.json`, notebook execution outputs
(stripped before commit). Commit: code, plan, docs, clean notebooks.
