# F05 Gate-A (partial) + E00–E02 smoke report — 2026-09-26

Commit: `13b3791` + working-tree package (see `runs/historical_2026-09-26_baseline/manifest.json`).
Graphify: `graphify-out/graph.json` (120 nodes / 150 edges) used for navigation
(`query` blocking→matching→outputs, `explain` macro-F0.5, `path` validate→predictions);
report + labels in `graphify-out/GRAPH_REPORT.md`.

## 1. Frozen baseline (recomputed in prior audit, referenced — NOT retrained here)

- Frozen candidates (India K90 / US K80): pair recall **94.1489%** (64,959/68,996),
  exact oracle macro-F0.5 **0.9793097093** on the 20,000-query dev-exposed sample.
  Source: `tmp/audit/candidate_audit.json`.
- Saved matcher config reports grouped OOF macro-F0.5 **0.8886** (18 feats, refit
  769 trees); 11-feat base reports 0.8626. Source: `notebooks/output-local/*.json`,
  `tmp/audit/model_artifact_audit.json`. OOF was not rerun here.
- Implication (plan §1): even perfect classification of frozen candidates cannot
  exceed 0.98 on that sample. Retrieval + matcher must both improve. No new score
  is claimed in this session.

## 2. E00 freeze — DONE

- `runs/historical_2026-09-26_baseline/`: `model_config*.json` copies +
  `manifest.json` with sha256+bytes for all 22 files under `notebooks/output-local/`.
- Old checkpoints untouched; new code lives in `code/business_entity_resolution/`.

## 3. Fresh grouped splits — DONE (`splits/f05-v1/`)

- Seed `f05-v1`, full-width SHA-256 assignment (replaces first-byte bucket).
- Legacy 224,776 md5-bucket IDs forced to `dev-exposed`; fresh `dev` 197,876 /
  `holdout` 198,351 / `train` 1,585,818; 20k unexposed probe `dev_probe_20k.json`.
- Bucket mix per split ~5.5% singleton, rest 1/2-3/4-5/6+ (see `split_report.json`).
- Corpus policy: splits cover queries only; evaluation retrieval must use full
  S2/S3 pools incl. unassigned distractors.

## 4. Normalization fixes + parity — DONE (code + 10 tests pass)

- `src/er/normalization.py`: Unicode primary views (NFKC+casefold, keep L/M/N);
  Latin folding is an extra view; `ste` dispatched by field+country
  (US-addr→suite, FR-name→societe); legal-form vs street tables separated;
  house/unit/postal parsed separately; missingness + normalization-loss flags kept.
- `tests/test_normalization.py` covers Hindi/Tamil preservation, combining-mark
  composition, FR/US `ste`, empty-aware ngrams, French accents.
- `tests/test_metrics.py` pins the exact F0.5 formula incl. singleton semantics.

## 5. Shared retrieval entry point — DONE (smoke, not quality)

- `src/er/retrieval/lexical.py`: one `retrieve_channel` for dev AND test;
  deterministic (-score, id) ordering; per-channel (score,rank,presence) provenance;
  `dedup_union` + RRF-ordering baseline. `structured.py` logs overflow explicitly.
- `src/er/features.py`: schema-named 23 features; empty-empty similarity = 0;
  house/unit/pin equality AND conflict features; `training/matcher.py` asserts
  schema and enables `bagging_freq` when subsampling.
- Smoke (`reports/dev_probe/smoke_{US,India}_200.json`): 200 queries × 200k-capped
  pool, 3 channels × K20, vocab 30k — all paths run; union mean 47.3 (US) / 53.5
  (India) of 60 max. **Capped pool: no recall/oracle claim.**
- Full-pool quality sweep (`probe_channels.py`, 150k vocab, full country pools):
  attempted 200-query US run, exceeded 10-min budget on pool TF-IDF fits (3 × ~6M
  docs). Queued as next step with per-channel single-fit batching; no numbers
  reported until it completes on the fixed 20k probe.

## 6. Not done (explicit)

- No new matcher training, no threshold tuning, no full-test run, no package zip.
- E03–E12 (structured union quality, dense encoder, matcher scale-up, calibration,
  France robustness, production pipeline) pending behind the retrieval gate.
- No claim that >0.98 is attainable; budgets in plan §6 are engineering targets.
