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
  leaderboard. `candidate_pairs.tsv` ships in the final zip and is used to
  analyse blocking quality (recall ceiling, reduction ratio) and verify the
  pipeline; top teams' packages are reviewed before final rankings are
  confirmed. Per the PDF this is an audit for sanity/fair-play, not a stated
  tie-break on K size — so **recall ceiling is the primary blocking objective**
  and mean K is a sanity bound (keep it under ~40-50, don't minimize it at
  recall's expense). A match never retrieved can never be predicted.
- `candidate_pairs.tsv` must be the exact pre-model candidate set
  (`matching` should be a subset of `candidates`).
- Both files: tab-separated, `source1_entity_id` + ID-list column, one row per
  test S1, S2-/S3-only IDs, no intra-list duplicates, no duplicate S1 rows.
- Validate locally with `utils/validate_submission.py` (stdlib only), including
  `--check-ids` before submitting.
- Fair play: **no external DB/API/lookup, no geocoding APIs, no internet
  augmentation**. Final model must be MIT/Apache-2.0 licensed, <= 8B params.

### 1.6 Organizer guidelines shared in chat (kept verbatim in-plan)
> - Update: `candidate_pairs.tsv` is part of your final submission.
> - Blocking has to scale. Amazon resolves business entities across billions of
>   records, so comparing every record with every other one is not an option.
>   Your blocking / candidate-generation step must cut the search space to a
>   small candidate set per Source 1 entity.
> - Candidate generation counts toward the final ranking. We will review your
>   `candidate_pairs.tsv` and the code that produces it when deciding final
>   rankings, alongside your `matching_results.tsv` score. The approach that
>   generates a smaller candidate set per Source 1 entity will be ranked higher
>   in the final evaluation beyond the public/private leaderboard.
> - Please make sure to go through the problem statement carefully and review
>   all the requirements, guidelines, and submission details before getting
>   started.
>
> How this plan reconciles §1.5 with the above: the PDF frames candidates as a
> quality/verify audit, while this update adds an explicit smaller-K preference
> on top. So the blocking objective is lexicographic — (1) hit the recall
> ceiling target (≥95%; a lost match caps scored F0.5), then (2) minimize mean K
> subject to holding that recall (rank-merge + adaptive caps, recall@K curve to
> prove it). Never trade recall for K.

---

## 2. Solution Architecture

```
S1/S2/S3 TSVs
  -> Normalization (unicode, abbrev incl. French table, suffix, PIN/ZIP extraction)
  -> Blocking (country-sharded FAISS ANN primary + PIN/token backfill +
     rank-merge, adaptive K capped ~25-40) — sized for the real N (low millions
     per shard): chunked queries, no Cartesian, logged build/query time
  -> candidate_pairs.tsv  [recall ceiling >=95% primary; mean/median/p90 K and
     reduction ratio reported as sanity bounds, not minimized at recall's expense]
  -> Pairwise feature engineering (name/addr/country/source signals)
  -> Matcher: LightGBM baseline -> MiniLM bi-encoder -> cross-encoder rerank (top-10)
  -> F0.5 thresholding + singleton rule (max_score < tau -> empty)
  -> matching_results.tsv -> validate_submission.py -> zip package
```

> Blocking rule (§1.6): recall ceiling first, then smallest K that holds it.
> Rank-merge/adaptive caps exist to minimize mean K subject to the recall target
> — never the reverse.

---

## 3. Detailed Plan

### Phase 0 — Reproducible validation split
1. Stratified 90/10 split on **train S1 IDs** by
   `(country x match-count bucket x singleton flag)`.
2. Keep the full S2/S3 pool as the retrieval universe (no S1 leakage).
3. Implement an exact macro-F0.5 scorer mirroring the leaderboard formula, and
   **unit-test it against the PDF's worked example** (P=0.667, R=1.0 -> F0.5=0.714)
   before trusting it for any threshold tuning.
4. France-robustness proxies (since France is test-only):
   - Report per-country F0.5 (US / India) separately.
   - Country-ablation run: drop the `country_match` feature and confirm no collapse.
   - Optional: synthetic noise probe (accent/typo injection) on val names.
5. All threshold and model selection happens on this val split only.

### Phase 0.5 — Walking skeleton (before real blocking)
1. Trivial blocking: exact normalized-name + PIN match only, on a small sample
   (e.g. 5k S1).
2. Push the sample through the *entire* pipeline: feature building -> LightGBM
   stub -> thresholding -> `matching_results.tsv` / `candidate_pairs.tsv` ->
   `validate_submission.py --check-ids`.
3. Confirm PASS end-to-end before investing in the real blocking algorithm.
   Cheap insurance against a late-discovered format/integration bug.

### Phase 1 — Normalization (offline, no external calls)
1. Unicode NFKD + strip accents, lowercase, whitespace/punctuation canonicalization.
2. `&` -> `and`; abbreviation expansion via in-repo dict covering **US/India AND
   France** (`corp/corporation, pvt/private, ltd/limited, rd/road, st/street,
   ave/avenue` plus French `sarl, sas, sa, eurl, rue, bd/boulevard, av/avenue,
   cedex`, etc.). France is 15% of test with 0% train coverage — a US/India-only
   dict would leave exactly the unseen-country slice with weaker normalization.
3. Legal-suffix handling: map to canonical form AND keep a separate
   `suffix_match` binary feature (do not silently delete signal).
4. Address parsing (regex only, no geocoder): extract PIN/ZIP (5-6 digit,
   incl. French 5-digit postal codes), building numbers, and normalized
   city/state/commune tokens where present.
5. Country: normalize case/whitespace only; compare as strings; never enumerate
   the label set in code.

### Phase 2 — Blocking / candidate generation (RECALL-FIRST, EFFICIENCY-SECOND)
Goal: **>=95% recall ceiling** on val; report mean/median/p90 K and reduction
ratio as sanity bounds — do not tune K down at recall's expense. Real N is low
millions per shard (~60% of S1/S2/S3 in the US shard alone), not billions, so
size everything for that.

1. **Shard by normalized `country` string value (open-set).** France then shards
   automatically at test time with no code change. Chunk S1 queries
   (e.g. 50k/chunk); never materialize a pair matrix.
2. **Primary index: FAISS (IVF-PQ or HNSW) from the start** over char 3-5g TF-IDF
   (or a cheap trained embedding) of `normalized_name + normalized_address`,
   built on the GPU box. Exact cosine via chunked sklearn `NearestNeighbors` is
   effectively brute-force sparse matmul at low-millions scale per shard and
   will be too slow — FAISS is the pipeline's bottleneck component, not an
   optional upgrade.
3. **Backfill only where ANN is weak:** exact PIN/ZIP block + first-significant-
   token block for S1 where the ANN top score is low-confidence or PIN is
   missing.
4. **Rank-merge + adaptive cap:** fuse branch scores (ANN cosine first, PIN/token
   as bonus), sort per S1 — confident S1 (top-1 >> top-2, exact PIN+name hit)
   keep K≈5-10, uncertain S1 expand to K≈25-40. The cap bounds compute and avoids
   degenerate K=500 cases; it is not a minimal-K target.
5. **Gate (must pass before any matcher work):** recall ceiling, recall@K curve
   (5/10/20/30), mean/median/p90 K, reduction ratio vs Cartesian, build+query
   time. If recall < 95%, fix normalization/index quality first; raising K is the
   fallback lever, and it stays on the table — recall caps the scored metric.
6. Persist the **exact post-merge set** as `candidate_pairs.tsv`; log build time
   and peak RAM as evidence the pipeline is reasoned about, without
   over-building for scale you don't have.

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
     include a short `blocking_report` with recall ceiling, mean/median/p90 K,
     reduction ratio, build + query time)
   - `code/business_entity_resolution/src/` (must include the blocking generator
     code that reproduces `candidate_pairs.tsv` — reviewers read it) +
     `README.md` (exact reproduce steps) + `requirements.txt` (pinned)
   - Filled `Documentation_template.md` (methodology, blocking with recall/K/
     runtime numbers, model, features).

---

## 4. Risks and Mitigations
| Risk | Mitigation |
|---|---|
| Chasing small K costs recall on the *actually scored* metric | Recall ceiling is primary; K is a sanity bound (~40-50), not a minimization target |
| France (unseen country, 15% of test) underperforms | French normalization table; per-country val reporting; country-ablation check |
| sklearn NN blocking too slow at real N (low millions/shard) | FAISS (IVF-PQ/HNSW) as primary ANN from the start, sized for actual data |
| Late-discovered format/integration bugs | Phase 0.5 walking skeleton (trivial blocking end-to-end + validator PASS) |
| F0.5 punishes false merges incl. singletons | High `tau`, explicit empty-prediction path, macro-F0.5 tuning |
| Memory blowup (500 MB+ source files) | Chunked CSV, sparse TF-IDF, per-shard indexes, streaming joins |
| Format rejection wasting submissions | Local validator gate on every run, including `--check-ids` |
| Fair-play disqualification | No external lookup/geocoding; all normalization in-repo |

---

## 5. Suggested Build Order (when implementation starts)
1. Phase 0 scorer (+ unit test vs PDF worked example) + Phase 0.5 walking
   skeleton (trivial blocking end-to-end, validator PASS).
2. Normalization module (incl. France table) + FAISS-based blocking + recall/K
   report (recall ceiling, recall@5/10/20/30, mean/median/p90 K, reduction ratio,
   runtime/RAM).
3. Feature builder + LightGBM baseline + `tau` tuning.
4. Bi-encoder fine-tune + (optional) cross-encoder rerank.
5. Full test inference + validator + submission zip + methodology doc.
