# Competitor Research — Amazon ML Challenge 2026 (Business Entity Resolution)

Survey date: 2026-09-25 (challenge launch day). Method: `gh search repos`
`--sort updated` over `amazon ml challenge` (50 repos) +
`business-entity-resolution` (20 repos) → 18 file-tree inspections →
README + source-code reads (`blocking.py`, `features.py`, `model.py`,
experiment CSVs, strategy docs) of the 7 strongest. Plus one self-run
verification on our own training data (see §5).

> Everyone converged on blocking → features → classifier → threshold within
> hours. Differentiators are blocking recall, CV hygiene, and threshold metric.

## 1. Verdict

**Base on `purvanshjoshi/business-entity-resolution`, harden with §2 borrows,
avoid §3 traps.** It is the most complete runnable recipe and fits our hardware
(Kaggle-CPU-class ≈ local PC / Colab; MIT; 7 deps).

## 2. What to take

### 2.1 Base: purvanshjoshi/business-entity-resolution
- Blocking: char 3/4-gram TF-IDF (`char_wb`, 150k feats, `min_df=3`,
  `max_df=0.35`, sublinear_tf, fp32), sparse matmul in batches of 2500
  (no `.toarray()` OOM), `top_k=12`, representation = normalized name +
  first-3 address tokens, pickle-checkpointed results.
- Features (7, `tfidf_score` reused free from blocking): Jaro-Winkler,
  token_sort, token_set, address token_sort, 3-level postal
  (exact 1.0 / 3-digit-prefix 0.5 / else 0.0), name length ratio. RapidFuzz SIMD.
- Model: LightGBM (lr 0.05, 63 leaves, depth 7, min_child 80, reg) +
  **GroupKFold split by S1 entity** (leak-free) + macro-F0.5 threshold grid.
- Runnable: `run_pipeline.py --output-dir … --sample-train 100000 --top-k 12`.
- Deps: numpy, pandas, scipy, scikit-learn, lightgbm, rapidfuzz, psutil.

### 2.2 PranjalGoyal06/Amazon-ML-2026 — blocking discipline + numbers
- `experiments/blocking_sweep.csv`: inverted-key blocking recall ceiling only
  **0.04–0.37** (exact-name 0.375; token/soundex/PIN unions worse). Quantitative
  evidence for TF-IDF cosine retrieval over key indexes.
- `str.translate` + dict-lookup normalization (fast), custom Soundex,
  `max_key_freq` pruning, country-partitioned indexes, LightGBM (150 trees) +
  threshold sweep 0.3–0.95 on exact macro-F0.5.

### 2.3 Epic021/amazon-ml-challenge-2026 — competition intel + EDA method
- Ties go to the **earlier submission** → submit early and often.
- FP costs ~2.7× a miss (4-true-match example: 3/3 correct = 0.94,
  4 correct + 1 wrong = 0.83). License rule constrains the *model*, not utils.
- `eda/eda3.py`: IDF inverted-index blocking-recall probe pattern worth copying.
- Confirmed test mix: India 809,986 / US 663,106 / France 259,452.

### 2.4 K-Siddharth06/Null-Variance — token stoplists
- Best generic name/address token lists (the/ltd/inc/road/street/near/opp…)
  for pruning — drop-in for our normalization. Adaptive block benchmarks +
  per-key block profiles worth copying for the blocking report.

## 3. What to avoid

| Repo | Trap |
|---|---|
| purvanshjoshi | "Greedy bipartite 1-to-1 matching" — wrong for multi-match GT (avg 3.67/S1). Description claims "50+ features", code has 7. Take the code, not the marketing. |
| rithishbarathn/amazon-ml-challenge-2026 | Threshold tuned on **pair-level micro-F0.5** (sklearn `fbeta_score` on pairs). Singletons contribute zero pairs → optimizes the wrong metric. Our macro scorer is correct. Otherwise most experiments (dual-source, per-country, global top-k). |
| SmithC05/amazon-ml-challenge-2026 | Clean modular baseline (16 features, threshold 0.75, saved `matcher.pkl`) but LogisticRegression as final model < GBM. |
| MokshFF/amazon-ml-business-entity-resolution | Scaffolding only — `build_index`/`generate_candidates` are TODO stubs. |
| areveeess/amazon-ml-challenge-2026 | Read the critique doc as a failure checklist (OOM `.toarray()`, 40h pandas loops, missing entrypoint), not as a solution. |

## 4. Deltas to our plan/notebooks

1. Blocking: purvanshjoshi's exact config (3/4-gram, 150k, batch-2500 sparse,
   top_k 12, name + 3 addr tokens) **inside country shards**.
2. Features: their 7 as the base; add `country_match`, `source_is_s2/s3`,
   `address_missing` (SmithC05's 16-list is the candidate pool).
3. Training: GroupKFold-by-S1 + macro-F0.5 threshold grid (their `model.py`
   pattern, our scorer).
4. Keep our MiniLM bi-encoder path — **no surveyed repo does neural matching**;
   it is our differentiator, not table stakes.
5. Ops: submit early/often (tie-break + credit dynamics).

## 5. Self-verified on our training data (2026-09-25)

300k sampled GT rows (1,039,380 pairs): **0 cross-country matches.**
Country-sharded blocking is provably safe (~2.5× search-space cut, no recall
cost). Implementation must still shard on the raw string value (open-set:
France) rather than a `{US, India}` enum.

## 6. Landscape notes

- ~70 challenge repos scanned, all 0–1 stars at scrape time, all pushed within
  hours of launch: `updatedAt` order currently measures scaffolding speed, not
  quality.
- Re-audit in ~1 week: stars/forks will have separated real pipelines from
  stubs by then.

## 7. Established ER tooling (star-sorted, all-time — concepts, not code)

| Repo | Stars | Takeaway for us |
|---|---|---|
| `dedupeio/dedupe` | ~4.5k | Active-learning ER + learned blocking predicates. Borrow the predicate-blocking idea; verify license before vendoring anything (challenge constrains the model license). |
| `J535D165/recordlinkage` | ~1k | Modular blocking index (full-index, sorted-neighbourhood, Q-gram). Concept reference for our FAISS/sparse stage. |
| `vintasoftware/entity-embed` | ~160 | Company/product vectors + ANN for linkage — prior art supporting our MiniLM bi-encoder path. |
| `OlivierBinette/er-evaluation` + `Awesome-Entity-Resolution` | ~40/140 | End-to-end ER evaluation framework + reading list. Useful when hardening our blocking report. |
| `vaneseltine/nominally` | ~40 | Name parser for record linkage — people-name oriented, limited business-name value. Skip. |
| `benseverndev-oss/goldenmatch` | ~130 | Grandiose claims (beats Splink, 250M rows/11min, 97 MCP tools), landed same-day. Treat as marketing until code is audited. Skip. |

## 8. Similar-name sweep + owner standings (round 2–3)

- `Nikhil0809/Business-Entity-Solution`, `harshini-1234567890/business-entity-resolution-ml`:
  README-only stubs. Nothing to take.
- `ItsMrZxD/fuzzy-entity-matching` (MIT, CI-tested): generic RapidFuzz+pandas
  pair scorer with 0–100 review scores. Not challenge-shaped, but a clean
  reference for scoring hygiene.
- `chasanth/entity-matching-model`: 5-feature pair classifier + saved `.pkl`.
  Baseline-level; superseded by our LightGBM plan.
- `anushayk70/Amazon-ML-Challenge`, `saimapathankhan/amazon-ml-challenge-2026`:
  empty or removed. Nothing to take.
- Owner totals (sum of stars across each author's repos): `purvanshjoshi` 143,
  `Satwik-2005` 167, everyone else ≈ 0 (`Epic021` 2, `SmithC05` 4).
  No high-star veteran is competing under their main account — the field is
  newcomers, and the community has already picked purvanshjoshi's repo as the
  reference implementation, which independently confirms the §1 verdict.
