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

- ~400 repos scanned in the expanded round (379 unique, 375 owners), all
  pushed within hours of launch: `updatedAt` order currently measures
  scaffolding speed, not quality.
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
## 9. Borrowed ideas — concrete port list (status: planned, not yet ported)

Borrowed = proven elsewhere, fits our stack (CPU-first, MIT/Apache, no
external lookups). Each item maps to a plan/notebook location.

### Blocking (notebooks: Phase 2 cells; plan §Phase 2)
- [ ] purvanshjoshi TF-IDF config verbatim as starting point: `char_wb`,
  3/4-grams, 150k max feats, `min_df=3`, `max_df=0.35`, sublinear_tf, float32,
  sparse matmul in batches of 2500 (never `.toarray()`), `top_k=12`,
  representation = normalized name + first-3 address tokens, pickle
  checkpoints of candidate dicts.
- [ ] PranjalGoyal06 `str.translate` + dict-lookup normalization (replace regex
  cleaning on hot paths), custom Soundex keys as backfill branch,
  `max_key_freq` pruning of ubiquitous keys.
- [ ] K-Siddharth06 generic token stoplists (name + address) for pruning;
  per-key block caps (exact 500 / prefix 250 / token 150) as degeneracy guards;
  per-key block-profile diagnostics for the blocking report.
- [ ] Epic021 `eda3.py` recall-probe pattern (IDF-weighted inverted index,
  R@1/5/10/20/50 by country × source) as the blocking gate harness.
- [ ] Country-shard everything (self-verified safe: 0/1,039,380 cross-country
  pairs) with open-set string sharding for France.

### Features (notebooks: Phase 3 cells; plan §Phase 3)
- [ ] purvanshjoshi 7-feature base: `tfidf_score` (free from blocking),
  Jaro-Winkler, token_sort, token_set, address token_sort, 3-level postal
  (exact 1.0 / 3-digit prefix 0.5 / else 0.0), name length ratio.
- [ ] SmithC05 additions: `country_match`, `source_is_s2/s3`,
  `address_missing`, char-3-gram Jaccard, token-count diffs.
- [ ] RapidFuzz SIMD throughout; never pure-Python Levenshtein on hot paths.

### Training + threshold (notebooks: Phase 3/4 cells)
- [ ] GroupKFold split **by S1 entity** (purvanshjoshi) — leak-free; never split
  by pair.
- [ ] LightGBM starting params: lr 0.05, 63 leaves, depth 7, min_child 80,
  subsample/colsample 0.8, reg_alpha 0.1, reg_lambda 1.0, ≤1200 rounds,
  early stopping 60.
- [ ] Threshold: fine grid on **macro-F0.5** (our scorer), tie-break toward
  higher precision / higher threshold (rithishbarathn's tie-break rule, applied
  to the right metric).
- [ ] Persist `model_config.json` (feature order + threshold) next to the model
  artifact (SmithC05 pattern) so inference can assert feature alignment.

### Ops / packaging
- [ ] `run_pipeline.py --sample-train --top-k` CLI pattern (purvanshjoshi) for
  the final `code/` bundle reproducibility gate.
- [ ] Submit early and often (Epic021: ties → earlier submission).
- [ ] Keep the MiniLM bi-encoder upgrade path — no surveyed repo does neural
  matching; it stays our differentiator.

## 10. Ranking methodology note — total stars, done right

First attempt (sequential, 2.2s pacing) was aborted as too slow; second attempt
returned all zeros due to a `stargazersCount` (GraphQL) vs `stargazers_count`
(REST) field-name bug; both were fixed and the full ranking completed over 377
repos / 373 owners in §11. Totals are first-100-repos sums (orgs with 100+
repos undercounted — immaterial for ranking here). Standing conclusion: the
challenge field is newcomers, so fame-rank surfaces industrial/academic ER
repos, not competitors — mined above for transferable ideas rather than
threats.

## 11. Top-20 owner deep-dive (codebases actually read, 2026-09-25)

377 unique owners ranked by total stars across all their projects (fixed the
`stargazers_count` snake_case bug that zeroed the first attempt). Rank → owner
(total stars) → verdict on their ER-relevant repo (code read, not README-only):

| # | Owner (total ★) | Repo read | Verdict |
|---|---|---|---|
| 1 | ing-bank (5007) | EntityMatchingModel | **Highest-value established codebase.** MIT headers confirmed. Complementary indexers (TF-IDF word+char cosine via `sparse_dot_topn` + sorted neighbourhood + first/first2/first3 char keys), abbreviation finders (merged initials `FC Barcelona`, CamelCase `PetroBras`, punctuated initials), rank-based + legal-entity features, millions-vs-millions distributed. Borrow: abbreviation regexes, complementary-indexer design, `sparse_dot_topn`. |
| 2 | IBM (3277) | table-representation-evals | Tabular-embedding benchmark harness (row-similarity MAP). Infra, not a solution. Skip. |
| 3 | davidmoten (1235) | viem | Java lib for volatile-identifier tracking (vessels/craft). Wrong domain. Skip. |
| 4 | tshu-w (516) | ComEM | COLING-2025 LLM matching paper; blocking via `retriv` SparseRetriever + topK recall print. LLM matchers violate our compute/license posture; the cached-index recall-probe pattern is already covered by Epic eda3. Skip. |
| 5 | os-climate (415) | financial-entity-cleaner | **Gold for normalization.** Per-country legal-form JSONs from GLEIF ISO-20275 (CC0) — `fr_legal_forms.json` EXISTS (SAS/SICAV/SICAF + dotted variants). Named ordered cleaning-rules dict. Repo is archived (borrow data + pattern, not code). Disclosure note: static open-code-list maps = same category as hand-written abbrev maps; document in methodology. |
| 6 | wbsg-uni-mannheim (402) | PyDI / MaDI-Bench / billiger-de / auto-labeling | Academic group (Mannheim DWS). PyDI blocking modules (standard/token/sorted-neighbourhood/embedding) are clean reference implementations; benchmarks are product-domain. Skim PyDI blocking if ours stalls. |
| 7 | machuangtao (342) | CE-RAG4EM | SIGMOD LLM-RAG blocking on Wikidata. External-KG retrieval = fair-play grey zone + wrong stack. Skip. |
| 8 | purvanshjoshi (194) | business-entity-resolution | Base pick (§1). No change. |
| 9 | magicsunday (181) | webtrees-obituary-matcher | PHP people-matcher (GPL-3.0 — license-incompatible anyway). Skip. |
| 10 | GaganB982006Hello (161) | Amazon-ML-Challenge-2026 | Tree fetch failed; no readable code. Skip. |
| 11 | abhishekck31 (158) | Amazon-ML-Challenge | **Best challenge-repo methodology.** Entity-level macro-F0.5 documented as distinct from sklearn macro (names the exact trap), DUAL thresholds (T_singleton, T_match), fully vectorized blocking (`groupby.indices`, no `iterrows`), 1:6 hard negatives from blocking candidates, S1-entity splits, jellyfish, tests. Borrow: dual-threshold scheme, hard-negative ratio, metrics doc wording. |
| 12 | rodrigolourencofarinha (143) | Entity-Matching-Demo | Academic comparison (exact vs fuzzy vs LLM) on firm data. Pedagogy, not pipeline. Skip. |
| 13 | neo4j-field (140) | entity_matching_tool | Electron+Neo4j dedup app. Wrong stack. Skip. |
| 14 | knoxiboy (98) | BUSINESS-ENTITY-RESOLUTION-CHALLENGE | Dotted-variant suffix map (`l.l.c.`, `d/b/a`, `[limited]`), India addr abbrevs (`nagar`, `marg`), `addr_missing` flag, and a real baseline number (54% recall on exact+2-token union — more evidence key-union blocking underperforms). Anti-pattern: `difflib.SequenceMatcher` (slow; RapidFuzz instead). |
| 15 | OnkarNanaware (72) | Amazon-ml-challenge | SageMaker/S3 multi-account ops pattern (account-prefix config). Our sharing layer already covers this via Drive; marginal. Blocking config (tfidf_top_k 50, trigrams) noted. |
| 16 | saketlab (72) | alethia | Exact-match short-circuit + null-preservation at inference (cheap precision win), token_sort identity documented. Borrow the short-circuit pattern. R/Python dual — take pattern only. |
| 17 | Resham1424 (52) | amazon-ml-challenge | Empty/gone. Nothing. |
| 18 | VegirajuMahaveerVarma (41) | amazon-ml-challenge- | NFKC+casefold normalization utils, suffix-strip loop, postal/house-number regexes, token signatures. Dataset via Git LFS (do not copy that practice). Borrow: `longest_token` / `token_signature` helpers. |
| 19 | Hamza-Faarooq (37) | Amazon_ML_Challenge | Zip + README only. Nothing. |
| 20 | AaryaSingh5 (30) | Amazon-ml-challenge | Claims >98% recall (unverified), open-set country symmetry, staged parquet pipeline. Process-heavy; borrow nothing until numbers appear. |

Net vs §1 verdict: base pick unchanged (purvanshjoshi #8 by fame, #1 by challenge-code completeness). abhishekck31 (#11) joins as co-reference for threshold/metrics/training-data discipline; ing-bank EMM (#1) as the industrial reference for indexing + abbreviation handling; os-climate (#5) solves the French legal-form gap with an authoritative table.

## 12. Borrowed ideas — round-2 additions (§9 still stands, these extend it)

- [ ] EMM abbreviation finders (merged-initials / CamelCase / punctuated-initials regexes) as normalization features for DBA/trade-name noise.
- [ ] EMM complementary-indexer design: TF-IDF cosine (via `sparse_dot_topn`) + sorted neighbourhood + short-prefix keys, unioned — with per-branch recall measured (Epic eda3 harness).
- [ ] os-climate `fr_legal_forms.json` (GLEIF/ISO-20275, CC0) as the French suffix table; disclose as open-code-list map in methodology.
- [ ] abhishekck31 dual thresholds (`T_singleton`, `T_match`) tuned on entity-level macro-F0.5; 1:6 hard negatives mined from blocking candidates; S1-entity splits; `metrics.py`-style docstring distinguishing entity-macro from sklearn macro.
- [ ] alethia inference short-circuit: exact normalized-name matches accepted before scoring; null-likes preserved as no-match rows.
- [ ] knoxiboy dotted-variant suffix surfaces (`l.l.c.`, `d/b/a`) + `nagar`/`marg` addr abbrevs; `addr_missing` flag (already planned — confirmed by second source).
- [ ] Vegiraju `token_signature` / `longest_token` helpers as backfill blocking keys.

"Top 10 users by total gathered stars" was attempted (375 unique owners across
379 repos) and abandoned as a ranking: the field is newcomers on fresh accounts
(≈0 stars each), so a total-stars rank surfaces unrelated orgs (IBM, Neo4j…​)
whose stars come from flagship projects, not ER ability. Used instead:
domain-relevant stars (ER/data repos) + code substance (implemented
blocking/model vs TODO stubs, experiment artifacts, trained weights, docs
depth) + activity (push recency, iteration). Under that composite the order is
§1–§2 above, with purvanshjoshi first on both community stars (143) and code
completeness.

