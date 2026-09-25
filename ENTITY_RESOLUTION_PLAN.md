# Business Entity Resolution Challenge — Solution Plan

## 0. Task Summary
Build an ML pipeline that, for each Source 1 record (deduplicated reference),
finds all matching records from Source 2 and Source 3. A Source 1 entity may
match zero, one, or many S2/S3 records. Outputs are scored with macro-averaged
F0.5 (precision-heavy, singletons included).

Reference: `student_resource/student_resource/README.md`, problem statement PDF,
validator `student_resource/student_resource/utils/validate_submission.py`.

**User decisions locked in:**
- Compute: cloud GPU available (blocking/inference may stay CPU; encoder training/rerank on GPU).
- Scope: plan only for now (no implementation yet).
- Validation: stratified S1 split (not country holdout).

---

## 1. Data Analysis (verified by inspection)

### 1.1 Scale (exact row counts)
| split | S1 (queries) | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 |
| test | 1,732,544 | 4,887,273 | 5,082,316 |

- Full Cartesian (~2.2M x 10.3M train) is infeasible. Blocking is mandatory.
- Test has ~1.73M S1 queries against ~10M S2+S3 candidates.
- Files are large: train S1 210 MB, S2 489 MB, S3 504 MB, GT 127 MB;
  test S1 175 MB, S2 510 MB, S3 506 MB. All processing must be chunked/sharded.
- Local box: 16-core CPU, 33.7 GB RAM, torch CPU-only, sklearn 1.9,
  no faiss / sentence-transformers / lightgbm / xgboost preinstalled
  (install on GPU box as needed).

### 1.2 Labels (`train_ground_truth.tsv`, 2,206,821 rows)
- Singletons (empty `matched_entity_ids`): 123,247 = **5.6%**.
- Matched: 94.4%, averaging **3.67 matches** per matched S1.
- Source coverage: ~80.5% span both S2+S3, ~6.5% S2-only, ~7.5% S3-only.
- Match-count distribution peaks at 3-4, tail to 11:
  `0:123247, 1:119157, 2:375212, 3:530841, 4:484115, 5:321957, 6:164868, 7:63968, 8:18680, 9:4205, 10:534, 11:37`.
- Implication: this is **multi-match retrieval**, not 1:1 linking. The decision
  stage must emit a variable-length list per S1, including empty.

### 1.3 Countries
- Train S1: ~60% US (~1.32M), ~40% India (~0.88M).
- Test S1: ~15% France (~259k), remainder US/India.
- `country` must be treated as an **open set of string labels**: do not hard-code,
  filter, or one-hot to {US, India}. Use it as a soft block shard key + a
  match feature, and always emit every test S1 row including France.

### 1.4 Noise patterns (verified samples)
- `Lumay Boral / 1056 Belden Avenue, Akron, OH (US)` matches:
  - `S2: Lumay Boral Inc. / 1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH`
  - `S3: Lumay Boral / 1056c Belden Ave, AKON, Ohio` (typo AKON)
  - `S3: Lumay Bóral / 1056c Belden Ave, AKON, Ohio` (accent + typo)
- `Red Ventures Private Limited / Rajasthan, Jaipur, Banipark, Gokul Apartment,
  E-3A Kanti Chandra Road, G-1 (India)` matches:
  - `S3: Red Ventures Private / Doro No 316 G-1, Gokul Apartment,
    E-3a Kanti Chandra Road, Banipark, Subhash Nagar, RJ` (suffix drop + reorder).
- Confirms: abbreviations, legal-suffix inconsistency, accents/typos,
  address abbreviation + component reorder + missing components.
- Requires normalization + character-level similarity (char n-gram TF-IDF,
  Jaro-Winkler / edit-based features), not exact matching.

### 1.5 Scoring, ranking, and format constraints
- Leaderboard metric: macro `F0.5 = (1.25 * P * R) / (0.25 * P + R)` per S1,
  averaged over all S1. Singleton scoring: 1.0 if correctly empty, 0.0 if any FP
  predicted. F0.5 weights precision ~2x over recall: **prefer high threshold**.
- `matching_results.tsv` is the only file scored on the public/private
  leaderboard. HOWEVER, per the latest update, **final rankings go beyond the
  leaderboard**: reviewers audit `candidate_pairs.tsv` + the code that produces
  it, and **smaller mean candidate set per S1 ranks higher** at equal/close
  F0.5. Blocking efficiency is therefore a first-class objective, not just a
  recall gate.
- `candidate_pairs.tsv` is **part of the final submission** (`output/` folder in
  the zip) and must be the exact pre-model candidate set (audited for recall
  ceiling, reduction ratio, and mean K; `matching` should be a subset of
  `candidates`).
- Both files: tab-separated, `source1_entity_id` + ID-list column, one row per
  test S1, S2-/S3-only IDs, no intra-list duplicates, no duplicate S1 rows.
- Validate locally with `utils/validate_submission.py` (stdlib only), including
  `--check-ids` before submitting.
- Fair play: **no external DB/API/lookup, no geocoding APIs, no internet
  augmentation**. Final model must be MIT/Apache-2.0 licensed, <= 8B params.

---

## 2. Solution Architecture

```
S1/S2/S3 TSVs
  -> Normalization (unicode, abbrev, suffix, PIN/ZIP extraction)
  -> Scalable blocking (country-sharded PIN block + TF-IDF ANN + rank-merge,
     adaptive K, mean K target <=15-20) — must scale to billions: no Cartesian,
     chunked/sharded ANN, logged build time
  -> candidate_pairs.tsv  [FINAL SUBMISSION ARTIFACT — ranked on recall ceiling
     + reduction ratio + mean/median/p90 K; smaller K preferred at equal F0.5]
  -> Pairwise feature engineering (name/addr/country/source signals)
  -> Matcher: LightGBM baseline -> MiniLM bi-encoder -> cross-encoder rerank (top-10)
  -> F0.5 thresholding + singleton rule (max_score < tau -> empty)
  -> matching_results.tsv -> validate_submission.py -> zip package
```

> Ranking rule (update): beyond leaderboard F0.5, reviewers compare
> `candidate_pairs.tsv` efficiency — same recall at smaller mean K wins. Every
> blocking decision below optimizes the recall-per-K Pareto frontier.

---

## 3. Detailed Plan

### Phase 0 — Reproducible validation split
1. Stratified 90/10 split on **train S1 IDs** by
   `(country x match-count bucket x singleton flag)`.
2. Keep the full S2/S3 pool as the retrieval universe (no S1 leakage).
3. Implement an exact macro-F0.5 scorer mirroring the leaderboard formula.
4. France-robustness proxies (since France is test-only):
   - Report per-country F0.5 (US / India) separately.
   - Country-ablation run: drop the `country_match` feature and confirm no collapse.
   - Optional: synthetic noise probe (accent/typo injection) on val names.
5. All threshold and model selection happens on this val split only.

### Phase 1 — Normalization (offline, no external calls)
1. Unicode NFKD + strip accents, lowercase, whitespace/punctuation canonicalization.
2. `&` -> `and`; expand abbreviations via in-repo dict
   (`corp/corporation, pvt/private, ltd/limited, rd/road, st/street, ave/avenue`, ...).
3. Legal-suffix handling: map to canonical form AND keep a separate
   `suffix_match` binary feature (do not silently delete signal).
4. Address parsing (regex only, no geocoder): extract PIN/ZIP (5-6 digit),
   building numbers, and normalized city/state tokens when present.
5. Country: normalize case/whitespace only; compare as strings; never enumerate
   the label set in code.

### Phase 2 — Blocking / candidate generation (SCALABLE + EFFICIENCY-RANKED)
Goal: billion-scale-ready blocking that maximizes **recall per unit K**.
Target: **>=95% recall ceiling @ mean K <=15-20** (median lower; report p90 too),
with logged build time and reduction ratio. A wide `K<=50` union is only a
fallback baseline — the ranked solution must beat it on mean K at equal recall.

1. **Shard by normalized `country` string value (open-set).** France then shards
   automatically at test time with no code change. Sharding is also what makes
   this scale: per-shard ANN indexes, chunked S1 queries (e.g. 50k/chunk),
   streaming writes. Log per-shard index size, build time, and query throughput
   for the methodology doc (evidence of billion-scale readiness).
2. **Primary signal (one strong index, not a wide union):** sparse TF-IDF
   `char_wb 3-5g` on `normalized_name + normalized_address`, cosine retrieval
   via chunked sklearn `NearestNeighbors` (FAISS on GPU box if available).
   This single index should supply the bulk of candidates to keep K small.
3. **Surgical backfill only (not blind union):** exact PIN/ZIP block
   (high precision, especially India) + first-significant-token block ONLY for
   S1 where the ANN top score is low-confidence or PIN is missing. This is what
   keeps mean K down versus unioning everything for every S1.
4. **Rank-merge + adaptive K (the efficiency win):** fuse branch scores
   (ANN cosine first, PIN/token as bonus), sort per S1, and keep top-N with an
   adaptive cap — confident S1 (top-1 score >> top-2, exact PIN+name hit) keep
   K≈5-10; uncertain S1 expand to K≈25. Never emit a fixed 50 for everyone.
   Dedupe per S1, persist the **exact post-merge set** as `candidate_pairs.tsv`.
5. **Gate (must pass before any matcher work):** on val report recall ceiling,
   **mean/median/p90 K, reduction ratio vs Cartesian, and recall@K curve
   (5/10/20/30)** plus blocking runtime. If recall < 95%, improve normalization
   or ANN quality first (better text representation beats raising K — raising K
   hurts the final efficiency ranking). If mean K > 20 at target recall, tighten
   backfill thresholds and rank-merge cutoff, not the matcher.
6. **Scale proof:** no pair matrix ever materialized; chunked ANN; memory bounded
   per shard/chunk. Record peak RAM and wall-clock on train-scale data as the
   billion-scale proxy in the doc.

### Phase 3 — Pairwise features + matching model
Features per (S1, candidate) pair:
- Name: Jaro-Winkler, RapidFuzz/Sequence ratio, TF-IDF cosine, token Jaccard,
  abbreviation-aware token overlap, length difference.
- Address: token overlap, number-match, PIN/ZIP-match, city/state-match,
  TF-IDF cosine.
- Context: `country_match` binary, source indicator (S2 vs S3).

Model ladder (all <= 8B, MIT/Apache-2.0):
1. **Baseline (CPU):** LightGBM/XGBoost classifier on the above features.
2. **Upgrade (GPU):** fine-tune `all-MiniLM-L6-v2` (MIT) bi-encoder on val pairs
   with in-batch negatives; combine embedding cosine with GBM score.
3. **Optional rerank:** MiniLM cross-encoder over top-10 candidates per S1
   for final precision lift.
Rationale: GBM gives a fast strong baseline in hours; bi-encoder + rerank gives
the best F0.5 if GPU budget allows.

### Phase 4 — Precision-tuned decision + singleton handling
1. Tune a global threshold `tau` on val for **macro-F0.5** (expect ~0.6-0.8
   given the precision weight).
2. Rule per S1: if `max_score < tau` -> emit empty list (singleton);
   else emit all candidates with `score >= tau`.
3. Consider per-country `tau` (with unseen-country fallback to global) only if
   val shows clear drift; otherwise keep one global `tau` to avoid overfitting
   to US/India.
4. Enforce output invariants: dedupe IDs, S2/S3-only, one row per S1.

### Phase 5 — Test inference, validation, packaging
1. Chunked inference (e.g. 50k S1 per chunk), country-sharded, CPU-friendly;
   GPU used only for encoder scoring.
2. Write `output/matching_results.tsv` + `output/candidate_pairs.tsv`
   with `df.to_csv(sep="\t", index=False, encoding="utf-8")`.
3. Run `python utils/validate_submission.py --matching output/matching_results.tsv
   --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`.
   Must print PASS (exit 0). Fix all numbered issues before uploading.
4. Final zip `<team_name>_submission.zip`:
   - `output/matching_results.tsv`, `output/candidate_pairs.tsv` (BOTH required;
     candidate file is efficiency-ranked — include a `blocking_report` with mean/
     median/p90 K, recall ceiling, reduction ratio, build + query time)
   - `code/business_entity_resolution/src/` (must include the blocking generator
     code that reproduces `candidate_pairs.tsv` — reviewers read it) +
     `README.md` (exact reproduce steps) + `requirements.txt` (pinned)
   - Filled `Documentation_template.md` (methodology, blocking with K/recall/
     runtime numbers, model, features).

---

## 4. Risks and Mitigations
| Risk | Mitigation |
|---|---|
| France (unseen country) generalization | Open-set country handling; country-ablation check; per-country val reporting |
| Bloated candidate sets hurt final ranking (smaller K preferred) | Rank-merge + adaptive K; report recall@K curve; optimize recall-per-K, not raw recall |
| Blocking doesn't scale to billions | Country sharding + chunked ANN + streaming writes; log build/query time and peak RAM as scale proof |
| Candidate explosion on 1.73M test S1 | Same as above; never materialize full pair matrix; cap p90 K |
| F0.5 punishes false merges incl. singletons | High `tau`, explicit empty-prediction path, macro-F0.5 tuning |
| Memory blowup (500 MB+ source files) | Chunked CSV, sparse TF-IDF, per-shard indexes, streaming joins |
| Format rejection wasting submissions | Local validator gate on every run, including `--check-ids` |
| Fair-play disqualification | No external lookup/geocoding; all normalization in-repo |

---

## 5. Suggested Build Order (when implementation starts)
1. Val split + F0.5 scorer + normalization module.
2. Scalable blocking + `candidate_pairs.tsv` + efficiency report (recall ceiling,
   mean/median/p90 K, recall@5/10/20/30, reduction ratio, runtime/RAM).
3. Feature builder + LightGBM baseline + `tau` tuning.
4. Bi-encoder fine-tune + (optional) cross-encoder rerank.
5. Full test inference + validator + submission zip + methodology doc.
