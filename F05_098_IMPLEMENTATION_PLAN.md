# Amazon ML Challenge: implementation plan to maximize verified macro F0.5

> **Active execution plan:** [LOCAL_COLAB_IMPLEMENTATION_PLAN.md](LOCAL_COLAB_IMPLEMENTATION_PLAN.md). All required work now runs on this device; optional GPU-intensive experiments can run in parallel on [Colab A100](reports/experiments/COLAB_A100_RUNBOOK.md). The Mac/RTX assignments and old numeric promotion floors below are superseded. Current artifact verification confirms that keyed training pairs and partitioned texts are complete; that earlier blocker is resolved.

> **Current objective, revised by the user on 26 September 2026:** maximize reliable F0.5 with the available resources; **0.98 is no longer an acceptance gate**. The latest [maximization review and execution order](reports/dev_probe/F05_MAXIMIZATION_REVIEW.md) supersedes the numerical score/oracle floors and priority order in older sections below. Keep this filename for existing links; historical measurements remain preserved.

**Current established reference:** clean B0 **0.904586** on screen_2k. **New provisional challenger:** country-calibrated thresholds **0.906774**, selected on separate calibration_5k; paired 95% delta interval includes zero, so confirm on larger development before promotion. Untrimmed candidate oracle is **0.990452** on screen_2k, but no untrimmed end-to-end matcher result is established yet.

**Next work:** align split-file readers and isolated run outputs; run the bounded cached-feature matcher screen locally; build the remaining comparison features once; confirm the threshold/model challengers; compare actual scores for India K250/untrimmed candidates. Optional A100 frozen-encoder work can run alongside these local stages. B0 already uses 743 trees. Follow L00–L07 and G00–G04 in the active plan. Older audit sections below preserve the historical findings and do not require repeating completed work.

**Original audit date:** 26 September 2026. **Historical repository snapshot:** `13b3791` (`partial run`). **Latest revision:** 26 September 2026, after reviewing E02–E09 reports, current working-tree source and saved feature caches; three-device hardware allocation updated from user-provided specifications.

**Purpose:** maximize competitive performance under the supplied rules, using the actual code, all seven data files, saved experiments, model artifacts, and previous plans. This is an implementation plan, not a claim that a new model has been trained or that a leaderboard result has been achieved.

## 1. Historical audit decision before clean B0

**Current verdict: the reports do not demonstrate that we are on course for >0.98.** Retrieval has improved on small probes, but the new matcher evaluation uses ground-truth-augmented validation candidates. Repair this before interpreting model improvements or spending all three devices on larger versions of the current training scripts.

The original frozen baseline had **94.1489% pair recall**, **0.9793097093 oracle macro F0.5** on an exposed 20,000-query sample, and an older model configuration reporting **0.8886 OOF**. These remain historical evidence, not the current system's scores. Original audit measurements in subsequent sections describe that snapshot unless explicitly updated below.

### 1.1 Latest report assessment and confirmed blockers

| Result | Interpretation after review |
|---|---|
| US full-pool union oracle **0.9984713**, recall **99.5279%**, 300 queries | Promising retrieval evidence; larger fixed-probe confirmation needed |
| India full-pool union oracle **0.9858318**, recall **96.5193%**, 300 queries | Improved retrieval, but only 0.00583 score headroom over the target |
| India K100 oracle **0.9811691** | Reject as an assumed default: loses 0.0046627 oracle and 17 positives versus untrimmed |
| E07 reported **0.9050761**; E09 **0.9037012** | Dense experiment regresses by 0.0013749; neither is a clean end-to-end estimate |
| E08 **0.9224410**, 600 queries | Earlier smaller calibration experiment; not a new comparable high score |
| Cached scaled validation | **283 injected positive pairs across 229 of 3,000 queries**; natural oracle **0.9909795** becomes **1.0** after injection |

Both matcher scripts append missing ground-truth targets with artificial RRF `0.001` before validation scoring. Read-only cache inspection confirmed every sentinel row is positive. Grouped OOF does not remove this answer-key dependency. Removing injected pairs from old scores is insufficient because training also contains the synthetic shortcut; rebuild clean features and retrain.

Further source-confirmed priorities: training adapters bypass real unit/name-core parsing; channel scores are still mixed in `tfidf_max`; OOF boosting omits an explicit round limit; feature names are not attached to OOF datasets despite schema assertions; caches lack complete input/model/schema identity; the calibration entry point is empty. Current report/source disagreement prevents assuming the latest artifacts can be reproduced from the current tree. E09 adds a cosine feature to existing pairs; it does not complete multilingual retrieval or the originally planned ER cross-encoder.

The detailed evidence, exact cache reconstruction method, report corrections and uncertainty assessment are in [the current progress review](reports/dev_probe/F05_PROGRESS_REVIEW_2026-09-26.md). Preserve the original reports as historical records; this review overrides their unsupported promotion/completion claims.

### 1.2 Revised execution priorities

Therefore, the next implementation should improve **both retrieval and the matcher**, with trustworthy evaluation and a working final inference path. A better transformer, a higher threshold, or more LightGBM trees alone cannot overcome the current candidate ceiling. Conversely, 99% blocking recall alone will not produce a winning matching score.

The priority order is:

1. **Repair evaluation first:** label-blind natural candidates, separate fit/calibration/comparison roles, exact all-query scoring, cache/schema identities, reproducible entry points. Keep the existing locked holdout untouched.
2. **Establish clean B0:** wire shared normalization into real feature generation, specify OOF tree limits/names, preserve an untrimmed candidate control, and save per-query/pair scores with natural oracle on the identical population.
3. **Run complementary work across three devices:** current machine owns integrity and India-focused retrieval; the 16 GB M4 Air owns CPU matcher/features/calibration; the 16 GB i5 HX/RTX 3050 machine owns bounded multilingual/neural challengers.
4. **Increase retrieval headroom and matching quality together:** target oracle ≥0.995, with >0.99 as a minimum research floor and ≥99% pair recall per labeled country. Measure actual matcher loss; do not promote K100 or dense features by assumption.
5. **Select on shared evidence:** paired comparisons on the same manifests, separately selected thresholds, uncertainty and resource costs. Recombine winning components, retrain and recalibrate; standalone gains do not add automatically.
6. **Evaluate the locked finalist and production path:** preserve France uncertainty, then verify all test IDs, exact candidate-to-matcher parity, runtime, both outputs and reproducible packaging.

**No evidence can currently establish that >0.98 is attainable on the private leaderboard, or that it is sufficient to win.** France has no training labels, the public/private split is hidden, and the existing 20,000-query sample has been reused for development. The plan below provides measurable gates toward the target rather than promised improvement percentages.

### 1.3 Retired three-device allocation

**Superseded by the user's latest instruction.** The table below is retained for historical traceability only. All CPU/matcher/retrieval/evaluation work is now local; eligible heavy GPU experiments use optional Colab A100. Use the active plan linked at the top of this file.

Read [the shared experiment protocol](reports/experiments/THREE_DEVICE_PROTOCOL.md) before running any device's experiments. It defines disjoint query roles, proposed portable artifacts, resource limits, promotion gates and the dependency order.

| Device | Separate execution plan | Main experiments |
|---|---|---|
| Current Windows / Intel Arc machine | [Device 1](reports/experiments/DEVICE_1_LOCAL_BASELINE.md) | Integrity repair, clean B0, address/lexical/structured retrieval, compression, final integration |
| M4 Air, 10 CPU / 10 GPU cores, 16 GB unified memory | [Device 2](reports/experiments/DEVICE_2_MAC_MATCHER.md) | Boosting curves, field/channel features, training-size/hard-negative experiments, calibration, optional ensembles |
| 13th-gen i5 HX, RTX 3050, 16 GB RAM | [Device 3](reports/experiments/DEVICE_3_CHALLENGERS.md) | Frozen multilingual pair features, full-pool dense retrieval, domain encoder adaptation, conditional small reranker |

These can make progress concurrently, but clean scored experiments depend on the common repaired evaluator. Run one substantial workload per device initially; 16 GB machines should not launch entire grids in parallel. The third device's exact VRAM and all devices' free disk still require local measurement. The plans contain memory fallbacks and stage gates rather than invented runtime promises.

### 1.4 What would justify saying we are approaching the goal?

On the same natural candidate/query population, report `1 - final = (1 - oracle) + (oracle - final)`. Aim for retrieval loss ≤0.005 and downstream loss <0.015. First demonstrate clean, repeatable 0.95 and 0.97 checkpoints; these are progress markers, not predicted outcomes. Then require a selected pipeline exceeding 0.98 on independent locked evaluation, with uncertainty and country slices reported; a lower 95% bound above 0.98 would be stronger evidence on the labeled distribution. None of these establishes private France performance or guarantees winning.

The latest experiments have different sample sizes and contaminated validation; a calendar-rate extrapolation would be misleading. Do not claim that another fixed number of runs or days will close the gap. If the clean oracle or model frontier plateaus below target, preserve the best verified result and report the shortfall honestly.

## 2. Scope, evidence, and limits of this audit

### 2.1 What was inspected

- All eight PDF pages were text-extracted; the scoring and package pages were also rendered and visually inspected. The substantive statement occupies pages 1–7.
- Every source and Markdown cell in all three notebooks: 22 cells in `entity_resolution.ipynb`, 38 in `entity_resolution_local.ipynb`, and 25 in `entity_resolution_a100.ipynb`; saved local outputs were extracted separately.
- Every original root Markdown file, including ignored `RESEARCH_FINDINGS.md`; student README, methodology template, and the entire supplied validator.
- All six source TSVs and the complete ground-truth TSV, using chunked full-file checks. Total input size is **2,520,573,701 bytes**.
- All eight saved candidate pickle checkpoints, current/previous LightGBM model files and configurations, existing sample TSVs, smoke outputs, and three completed India test-blocking parts.
- Recent Git history and the actual installed package versions. Third-party virtual-environment implementation files were not treated as project-authored source.
- Official model cards and library documentation for version-sensitive recommendations. No business record was sent to a search engine or external identity service.

### 2.2 Evidence levels used below

| Label | Meaning |
|---|---|
| Recomputed | Calculated during this audit from local data/artifacts |
| Artifact-reported | Present in a saved config/log/model, but the original training evaluation was not rerun |
| Source-confirmed | Directly evident in code or authoritative challenge documents |
| Historical claim | Recorded in a prior plan/research note; not automatically a current result |
| Proposed | An implementation or experiment to run; no gain is claimed |

The complete TSV integrity checks are full-corpus checks. Script/accent/postal diagnostics use a deterministic approximately 1-in-997 ID-hash sample, explicitly reported as a sample. All 4,037 current blocking misses were joined to their true target records; this is more comprehensive than the earlier 15-example investigation. A human semantic adjudication of every business pair was not performed. No hidden test labels, live leaderboard score, remaining deadline, or current cloud budget are available.

### 2.3 Audit artifacts

The compact supporting evidence is in [`audit/2026-09-26/`](audit/2026-09-26/): data profiles, candidate and model audits, dependency snapshot, and reconciliation of earlier documents. Reproduction scripts and detailed local-only record/miss extracts are under `tmp/audit/`. Original notebooks, datasets, trained models, and previous plans were not changed.

`graphify-out/graph.json`, `GRAPH_REPORT.md`, and `graph.html` provide a navigation graph of the documents and validator. It has **120 nodes and 150 edges**, with no dangling endpoints after checking. Notebook and dataset inspection was performed directly because the graph extractor does not classify `.ipynb` as code. The graph is supplementary, not proof of whole-repository correctness. Extraction token telemetry was unavailable; graph zero counters are placeholders, not a measured zero cost.

## 3. Challenge contract: what the implementation must satisfy

The primary authority is [`amazon_ml_challenge_problem_statement.pdf`](amazon_ml_challenge_problem_statement.pdf), supported by the supplied [`student README`](student_resource/student_resource/README.md).

| Requirement | Source | Implementation consequence |
|---|---|---|
| S1 is the deduplicated reference; zero, one, or many S2/S3 matches | PDF pp.1–2 | Variable-length sets; never force one match, one match per source, or a fixed cardinality |
| Countries form an open set; France appears only in test | pp.1–2 | Generic country fallback; every test country must be processed |
| TSV inputs and outputs | pp.1–4 | Explicit tab separator, UTF-8, exact headers, comma-separated target lists |
| Every test S1 appears exactly once in both outputs | pp.3–5 | Compare exact ID sets and row counts; empty predictions still need rows |
| Only existing test S2/S3 IDs; no duplicates within lists | pp.3–5 | Validate membership, prefixes, list uniqueness and S1 uniqueness |
| Candidate file is the exact final set passed to matching inference | pp.3–4 | Export candidates at the scoring boundary; preserve provenance and scores internally |
| Every final match must be a candidate | p.4 | Enforce a hard internal subset assertion |
| Only `matching_results.tsv` is leaderboard-scored | pp.3,6 | Candidate recall is diagnostic, not the leaderboard metric |
| Entity-macro F0.5, singletons included | p.6 | Use the exact per-S1 scorer, including entities with zero candidates |
| Private leaderboard determines final ranking | p.6 | Avoid adapting repeatedly to public feedback |
| Final model MIT/Apache-2.0, at most 8B parameters | p.5 | Record exact model/checkpoint licenses, parameter counts, revisions and derived-model provenance |
| No external identity lookup or internet data augmentation | p.7 | No business registries, geocoding, ER APIs, external record enrichment or external identity labels |
| Runnable, pinned, self-contained code and methodology package | pp.4–7 | Data-to-output CLI, dependencies, model provenance, both output files and filled template |

### 3.1 Rule ambiguities already present in your notes

`ENTITY_RESOLUTION_PLAN.md` §1.6 preserves a purported organizer chat update saying smaller candidate sets affect final ranking. The supplied PDF says candidates are audited for blocking quality and not leaderboard-scored; it supplies no formal K-ranking formula or maximum K. Preserve the quoted update as supplemental, unverified policy and retain any actual organizer announcement if available. Build a recall/score/cost frontier that works under either interpretation; do not impose an arbitrary K=50 ceiling at the expense of reaching the stated objective.

`RESEARCH_FINDINGS.md` claims an earlier-submission tie-break. That is not established by the supplied PDF. An early valid baseline is operationally useful, but no tie-break, submission quota, deadline or credit rule should be invented.

The proposed GLEIF/ISO-20275 French legal-form download is external reference data. Its CC0 license does not establish challenge permission. Use supplied-data-derived transformations and generic authored normalization rules; do not import external business or reference tables as an assumed exception to the prohibition. Algorithm documentation and license research are different from augmenting the challenge records.

The validator's prose says nonexistent IDs only reduce score, whereas the PDF says invalid IDs are rejected. Follow the stricter PDF contract: emit none. Its candidate subset check is warning-only and candidate input can be skipped; neither behavior removes the final-package requirements.

For a multi-stage matching cascade, save all pairs actually scored by the first identity matcher, including pairs rejected before an optional cross-encoder. Do not relabel that first matcher as a blocker merely to report a smaller candidate count. Keep separate raw-retrieval, pre-matcher and reranker-route counts. If organizer clarification specifies another cascade boundary, retain the full audit trail and apply that definition consistently.

## 4. Dataset: verified facts and their consequences

### 4.1 Complete row counts and country distribution

| File | Total rows | US | India | France |
|---|---:|---:|---:|---:|
| train S1 | 2,206,821 | 1,323,633 | 883,188 | 0 |
| train S2 | 5,034,616 | 3,016,817 | 2,017,799 | 0 |
| train S3 | 5,285,603 | 3,170,056 | 2,115,547 | 0 |
| test S1 | 1,732,544 | 663,106 | 809,986 | 259,452 |
| test S2 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 |
| test S3 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 |

Test reference weights are **38.2735% US, 46.7513% India, 14.9752% France**. The known-country mix differs substantially from the roughly 60/40 US/India training mix. A US-only gain can obscure an important India regression.

Test target-pool sizes are 3,817,031 US, 4,717,565 India and 1,434,993 France. Full test matching involves 1,732,544 queries against 9,969,589 target records. All-pairs matching is infeasible; country sharding and selective retrieval are justified.

### 4.2 Ground-truth integrity: full scan

| Quantity | Recomputed result |
|---|---:|
| Ground-truth S1 rows | 2,206,821 |
| Positive S1-to-target links | 7,638,365 |
| Singletons | 123,247 = 5.5848% |
| Non-singleton S1 | 2,083,574 |
| Mean positives per non-singleton | approximately 3.666 |
| Positive S2 links / distinct labeled S2 targets | 3,693,619 / 3,693,619 |
| Positive S3 links / distinct labeled S3 targets | 3,944,746 / 3,944,746 |
| Targets assigned to more than one S1 | 0 |
| Cross-country GT links | 0 across all 7,638,365 links |
| Missing GT target IDs / missing GT S1 IDs | 0 / 0 |
| Duplicate GT S1 rows / repeated targets within one list | 0 / 0 |
| GT S1 coverage equals train S1 exactly | Yes |
| Duplicate entity IDs within each source file | 0 in every file |
| Train/test exact ID overlap within a source | 0 for S1, S2 and S3 |
| Duplicate `(country, raw name, raw address)` S1 observations | 0 in both train and test |

The same-country invariant is now established for the **entire provided training corpus**, improving upon the research note's 300,000-GT-row sample. It is still an empirical training invariant, not proof about unseen labels. Use country partitions by default, with a generic route for unexpected/missing labels; do not silently drop records.

There are **1,340,997 S2 and 1,340,857 S3 training targets with no S1 assignment**, about 25.99% of the training target pool. These are essential distractors. Do not restrict evaluation retrieval to labeled positives. Do not interpret an unassigned target as universally unrelated in every possible context beyond the supplied task's complete GT semantics.

One S1 has many targets, but each labeled target belongs to at most one S1. This supports testing reverse-candidate ambiguity features or capacity-one-per-target consistency. It does **not** justify a one-to-one S1-to-target assignment or forcing every target to have an owner.

### 4.3 Match-count distribution

| True matches per S1 | S1 count |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

Both sources contain matches for 1,776,047 S1; 143,029 have only S2 matches and 164,498 only S3. The maximum observed count 11 is descriptive, not a valid hard output cap for test.

### 4.4 Missing data and text variation

Full-file blank-address counts:

| File | Empty addresses |
|---|---:|
| train S1 / test S1 | 0 / 0 |
| train S2 / S3 | 168,967 / 175,916 |
| test S2 / S3 | 129,408 / 136,098 |

No raw blank names or countries were observed. A few name cells equal null-like strings such as `na`/`null` under case folding: 6/18 in train S2/S3 and 49/61 in test S2/S3. These are flags for inspection, not proof those strings are missing: short legal/business names can coincide with parser NA tokens. Read with `dtype=str, keep_default_na=False`, then apply explicit field-specific missingness rules. Also handle placeholders embedded inside addresses, such as the observed `null` and `N/A`, without deleting meaningful neighboring information.

Deterministic sample findings:

- Train India S2: **470/2,087 sampled names become devoid of alphanumeric characters under ASCII folding**; S3: **224/2,126**. Test India S2: **511/2,265**; S3: **319/2,431**. This is a material asymmetric source problem, not an occasional accent issue.
- Train India S2/S3 samples contain non-Latin address letters in **482/2,087** and **509/2,126** records. The assertion in the old plans that India addresses are always Roman English is false.
- S1 India sample names and addresses were Roman-script, reinforcing the need for cross-script target matching.
- A generic 5–6 digit regex fired on only **2/911 train India S1** and **4/810 test India S1** samples. France test S1 had **3/242**. These are regex-hit rates, not validated postal-code coverage. Postal blocking cannot be the backbone.
- France test S1 sample had non-ASCII name characters in **40/242** and address characters in **72/242** records. Preserve original accents and add a folded view.

The profiler's general `nonlatin` flag is a Unicode-name heuristic; it can flag symbols such as `º` in French addresses. Do not interpret that flag alone as an accurate language/script classifier. Implement script detection using actual Unicode script/category properties for production diagnostics.

### 4.5 Current blocking misses: all joined, not just anecdotes

The frozen candidates miss **4,037 links across 3,201 S1**:

| Miss characteristic | India: 1,981 missed links | US: 2,056 missed links |
|---|---:|---:|
| Non-Latin target-name letters | 1,108 (55.93%) | 0 |
| Non-Latin target-address letters | 443 (22.36%) | 0 |
| Empty target address | 199 (10.05%) | 204 (9.92%) |

These categories overlap and are not a complete causal partition. They identify experiments; they do not establish that all affected misses are recoverable by one normalization fix.

Concrete patterns from the supplied GT:

- `Mentor Family Partners Inc.` matches `DREXJAX (ID: 38740)` with essentially the same full street address. A name-dominant joint representation suppresses this pair; address-only retrieval is appropriate.
- `Oncology Physicians` matches `Orbikelotavo` at the same address. Semantic name similarity alone is insufficient.
- `Wyatt's Supply` matches `Wyatt's Spupayl` with no target address. Name-only typo retrieval is needed.
- `Blue Constructions Limited` matches a Tamil name with a heavily truncated address. Two target IDs have identical observed target text. Cross-script learning and duplicate-observation retrieval can help; generic address overlap can also be ambiguous.
- `Hyderabad Infrastructure Private Limited` matches `infrastructureprivate.com` with reordered/typo-filled address text and an embedded `null`. Domain compaction and full-address views both matter.

Do not hard-code these individual business examples. Learn transformations and evaluate generalization on unseen identity groups.

## 5. Your process so far: reconstructed state

### 5.1 What is actually implemented

| Component | Current status |
|---|---|
| Exact macro scorer with singleton examples | Implemented; sample tests pass |
| Deterministic hash validation assignment | Implemented, but not a true stratified split |
| PIN walking skeleton | Executed; sample F0.5 0.0880 |
| Local country sparse TF-IDF retrieval | Implemented; cached India K90 / US K80 winners |
| Multi-key v6 challenger | Implemented and cached; measured below |
| 18-feature LightGBM | Trained artifact exists; config reports grouped OOF 0.8886 |
| Dual threshold selection | Implemented; both selected thresholds equal 0.85 |
| Full test blocking | Implemented separately; only 75,000 India rows completed locally |
| Full test LightGBM scoring and final matching export | Missing from the local notebook's Phase 5 path |
| A100 dense retrieval | Sample demo only; CPU FlatIP search after GPU encoding |
| A100 fine-tuning | Skeleton-pair training code; no saved execution evidence or local encoder artifacts found |
| A100 full-scale path | Stub |
| OpenVINO inference | Single-pair demo; not integrated into production matching |
| Shared production package / pinned requirements / final methodology | Not present as a complete submission pipeline |

### 5.2 Recomputed checkpoint history

| Checkpoint/run | Mean K | Hits / GT | Pair recall |
|---|---:|---:|---:|
| India v3 | 50 | 25,239 / 27,605 | 91.4291% |
| US v3 / BASELINE_9317 | 40 | 38,564 / 41,391 | 93.1700% |
| US v4 / PRE80 | 40 | 38,870 / 41,391 | 93.9093% |
| India v4 frozen | 90 | 25,624 / 27,605 | 92.8238% |
| US v5 frozen | 80 | 39,335 / 41,391 | 95.0327% |
| Frozen combined | 84.0325 | 64,959 / 68,996 | 94.1489% |
| v6 capped challenger | 50 | 64,621 / 68,996 | 93.6591% |
| **Frozen union v6, no new trimming** | **97.4934** | **65,679 / 68,996** | **95.1925%** |

The last row is a new **offline candidate-set diagnostic using already saved artifacts**, not a newly trained system. It shows that v6 contains useful complementary evidence. It recovers 720 links absent from frozen, while v6 alone discards 1,058 frozen positives. Do not discard its underlying structured branch; remove the premature cap and assess it properly.

### 5.3 Oracle ceilings and current ranking depth

For each S1, let `G` be true targets and `C` its candidates. The best possible predictions restricted to C are `G ∩ C`. Average their exact F0.5 across **all** S1, including singletons.

| Candidate policy | Pair recall | Oracle macro F0.5 |
|---|---:|---:|
| Frozen ranking top 1 | 24.8884% | 0.629070 |
| Top 3 | 60.9630% | 0.878052 |
| Top 5 | 79.1336% | 0.932781 |
| Top 10 | 88.1529% | 0.957085 |
| Top 20 | 91.0459% | 0.967198 |
| Top 40 | 92.7590% | 0.973618 |
| Top 50 | 93.1895% | 0.975385 |
| Top 60 | 93.5431% | 0.976906 |
| Top 80 | 94.0257% | 0.978775 |
| All frozen: India90 / US80 | 94.1489% | **0.979310** |
| v6 alone, K50 | 93.6591% | 0.977455 |
| Frozen union v6 | 95.1925% | **0.983070** |

These top-K rows truncate the existing saved rankings; they are not independent reruns with changed vectorizers. Scores for the frozen full set:

- India oracle **0.971825**, US **0.984367**.
- 15,710 of 18,911 non-singletons have every true target retrieved: about 83.08% complete coverage.
- 118 non-singletons have zero retrieved true targets; all score zero even with perfect matching.
- There are 1,089 singletons in the sample. A perfect oracle gives all of them 1.

The union improves the oracle to India **0.976420**, US **0.987563**. It still leaves almost no room for classification errors if the target is >0.98. The long-term retrieval goal should be comfortably above 0.98, not just barely cross it.

### 5.4 Model artifacts and output mismatch

`notebooks/output-local/model_config.json` records 18 features, 468,489 training-sampled pairs from 1,680,650 natural candidates, 3 grouped folds, iterations `[769, 811, 674]`, refit 769 trees, thresholds 0.85/0.85 and OOF score **0.8886**. Loading the model confirms 18 inputs and 769 trees. The earlier 11-feature model has 593 trees and reports **0.8626**.

The approximately **0.09071 difference between frozen oracle and reported OOF score** is the scale of remaining classification/decision loss on this development setup. It is not all recoverable, and the rounded reported score is not a fresh holdout result. Save raw OOF predictions to measure the loss exactly by slice.

The existing `matching_results.tsv` has 20,000 training S1 rows and recomputes to **0.0880187942**, the skeleton score. **770 rows contain matches outside the currently overwritten candidate file's frozen set.** Thus the matching and candidate files in the same directory describe different pipeline stages. They are not a valid full-test submission pair.

Three `test_India_c000`–`c002` parts contain 75,000 rows. They are blocking results, not model predictions. The saved smoke output covers 2,000 test S1 (France283, India933, US784), demonstrating only that its blocking format path ran.

### 5.5 Reconcile the earlier documents

| File | Retain | Supersede or correct |
|---|---|---|
| `README.md` | Repository map, machine roles, task metric | TODOs are stale; notebooks no longer share identical logic; full matching path incomplete |
| `ENTITY_RESOLUTION_PLAN.md` | Multi-match semantics, full pools, packaging | FAISS mandatory premise; K12/25–40 versus actual90/80; 95% gate; exact-name autoaccept; training on validation; guessed thresholds |
| `IMPLEMENTATION_PLAN.md` | Full-address India insight | 15 misses do not prove all addresses Roman; relaxed88/92 gates; conflicting US formula; optimistic runtime |
| `PERFECT_PLAN.md` | Country-aware representations and diagnostics | Separate specialists are proposals, not deployed models; name×3 differs from current×2; same unsupported universal-address claim |
| `Recall_99_Implementation_Plan.md` | Most of its evaluation, Unicode, union, hard-negative and reproducibility recommendations | It audited a ZIP without current data/artifacts; update92.47 baseline to94.15 and target macro F0.5 rather than recall alone |
| `RESEARCH_FINDINGS.md` | Complementary indexers, source inspection, metric discipline | Repository stars do not predict winners; external-table permission unproven; exact-name acceptance unsafe; tie-break unverified |
| `MACBOOK_SETUP.md` | Country workload sharing and fixed versions | Claims of manifest enforcement and finished matcher exceed code; 15GB disk budget unsuitable for new artifacts |
| Student README/template | Authoritative contract and final write-up structure | Fill with actual final measurements, not the projected architecture |

## 6. Correct objective and an explicit error budget

For a nonempty true set, the exact score is:

```text
F_i = 5 TP_i / (5 TP_i + 4 FP_i + FN_i)
    = 5 TP_i / (|G_i| + 4 |P_i|)

G_i empty, P_i empty:  F_i = 1
G_i empty, P_i nonempty: F_i = 0
Overall = sum_i F_i / N_S1
```

The formula's FP/FN coefficients are 4:1. The PDF's informal “precision 2×” language describes beta, not a fixed two-to-one exchange rate for mistakes. Do not optimize pair-level `fbeta_score`, label-macro sklearn scores, accuracy, AUC, or average P/R substituted into this formula.

Examples show why entity cardinality matters:

| True count | Correct predictions | Wrong predictions | F0.5 |
|---:|---:|---:|---:|
| 4 | 4 | 0 | 1 |
| 4 | 3 | 0 | 0.9375 |
| 4 | 4 | 1 | 0.8333 |
| 1 | 0 | 0 | 0 |
| 1 | 1 | 1 | 0.5556 |
| 0 | 0 | 0 | 1 |
| 0 | 0 | 1 | 0 |

Define per-entity oracle `O_i = F(G_i, C_i ∩ G_i)` and actual score `A_i`. Then:

```text
1 - mean(A_i) = mean(1 - O_i) + mean(O_i - A_i)
                retrieval loss    matching/decision loss
```

This decomposition is exact when all predictions lie within C. Use it to allocate work. The target requires total loss **less than 0.02**.

Proposed engineering budgets, not known achievable metrics:

| Component | Development objective |
|---|---|
| Oracle retrieval loss | ≤0.005, i.e. oracle ≥0.995; minimum promotion floor >0.99 |
| Matching/decision loss relative to oracle | ≤0.010 |
| Remaining safety margin | approximately0.005 for uncertainty/domain shift, not a guaranteed bound |
| Secondary retrieval targets | ≥99% pair recall globally and per labeled country;99.5% stretch |
| ANN approximation contribution | ≤0.001 absolute pair-recall loss on benchmark queries |

An oracle ≥0.995 is an ambitious planning target. If it proves unattainable, measure the frontier and choose the best achievable final score instead of relaxing gates and continuing to call the result a >0.98 system.

Singletons represent5.5848% of train. A 10% false-merge rate on them alone loses approximately0.005585 macro points if the same mixture holds. Likewise a completely missed one-match entity loses a full point. Prioritize errors by **sum of per-entity score loss**, not only link counts.

For France, on the full test mixture, `F_total = .382735 F_US + .467513 F_India + .149752 F_France`. If both known countries score0.99, France must exceed approximately**0.923223** for overall>0.98. If they score0.985, France must be materially stronger still. These are scenario calculations, not French performance estimates; the private subset's mixture may differ.

## 7. Correctness backlog: fix before interpreting another score

Notebook cell references below use **zero-based JSON cell indices**.

| Priority/location | Verified problem | Required implementation and check |
|---|---|---|
| P0 local10 | `normalize_ascii(...ignore)` deletes Indic names | Raw+Unicode primary views; Latin folding only as an additional view; test actual Hindi/Kannada/Tamil strings |
| P0 local10 | `[^\w\s]` removes combining marks from Indic words | Preserve Unicode categories L/M/N; canonicalize punctuation separately; verify complete word preservation |
| P0 local10 | Merged dictionaries override US `ste` with French `societe`; CO→company, FL→floor | Dispatch by country and field/context; preserve raw and lightly normalized address |
| P0 local19 | US expansion retrieves60 after80, then trims union to60 at the same0.02 floor | Superset-preserving expansion with larger depth; separate measured compression |
| P0 local19/22 | Oversized key buckets deleted; postings sliced in input order | Compound-key refinement/secondary ranking; log overflow; never first-N truncation as relevance |
| P0 local19 | Fuzzy scores/100 stored in same slot as cosine | Separate score/rank/presence per channel; retrain on new schema |
| P0 local22 | K50 RRF challenger discards earlier true candidates | Diagnose uncapped union; preserve baseline during discovery; compression must report lost GT |
| P0 local19 vs28/30 | Validation US fallback absent in test code | One shared retrieval function, same-record parity fixture |
| P0 local19 | Fixed locked filenames load regardless of target query membership | Manifest exact query/pool IDs, source hashes, normalizer and full config; assert equality |
| P0 local24–26 | “val” identities used to train/refit, tune blockers, tune thresholds | Rename roles; fresh locked holdout; learned retrieval and OOF models obey group boundaries |
| P0 local30/32 | Full path produces candidates only, drops scores, never calls saved matcher | Implement chunked features→model→decision→both outputs with score sidecars |
| P0 local12/19 | Same output filenames overwritten at different pipeline stages | Run-specific immutable output directories; export paired files only from one manifest |
| P0 local30/32 | Resume based on line count; merge based on count/no duplicate only | Atomic writes, checksums, exact chunk ID sets, manifest equality and exact final coverage |
| P1 local24 | `_char3('')` returns `{''}`; short names also collapse to same set | Empty/short-aware ngrams, explicit availability; empty equality not positive evidence |
| P1 local24 | Number intersection treats any shared postal/unit/house number alike | Parse components and conflicts separately; preserve alphanumeric units/ranges |
| P1 local10 | `LEGAL_SUFFIX=set(ABBR)` includes road/state-like abbreviations | Separate legal-form dictionaries from address abbreviations; preserve core-name and full-name views |
| P1 local8/11 | First-byte hash bucket is26/256≈10.15625%; sorted-ID prefix20k sample | Full-width seeded hash, stratified sample manifest; report actual split sizes |
| P1 local25 | `subsample=.8` but `bagging_freq=0` in actual model | Enable positive subsample frequency if bagging is intended; seed all randomized components |
| P1 local26 | Printed “gain” is default split importance | Request explicit gain and use grouped permutation/ablation for actual usefulness |
| P1 local3/25 | Thread counts called physical-core pinning; classifier uses os.cpu_count | Benchmark thread counts; affinity is distinct from thread limits; avoid unsupported hardware claims |
| P1 A10016 | Positives only from PIN skeleton; same sample called validation; aliases can become in-batch negatives | All training GT joins, identity-aware negatives, fixed-shape batches and fold boundaries |
| P1 A10014/22 | GPU label on CPU IndexFlatIP; full scale stub | Explicit device implementation and runnable country-scale index path |
| P1 A10018 | Generic MS MARCO reranker, untrained for ER; arbitrary top10 | Train task-specific pair model; route by ambiguity; no universal top10 cap |
| P1 local35/A10020 | Unmasked mean-pool demo and tokenizer/export uncertainty | Save tokenizer/pooling; attention-mask pooling; FP32/export/batch-length parity test |
| P1 sync/local13 | Both raw and OV directories flatten into shared/models | Separate versioned directories; no filename collisions; completed-manifest sync only |

**A non-bug worth preserving:** `eval_X` and `eval_y` are valid in the installed LightGBM **4.7.0** and its official API. Do not “fix” them based on older tutorials. Pin the compatible runtime; older releases require their supported validation interface. Also `subsample_freq=0` disables row bagging, and default feature importance counts splits. [Official LightGBM API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html).

The live run guide's “Run All” is unsuitable: it can overwrite sample artifacts and launch `RUN_TEST=True`. Replace it with explicit CLI commands and thin launch notebooks. Do not delete or overwrite old checkpoints during migration.

## 8. Phase A: rebuild evaluation before optimizing

### 8.1 Persistent identity-group splits

Create train/development/locked-holdout manifests, initially about80/10/10, stratified by country, match-count bucket, singleton status and target-script/missingness where feasible. A single S1 and all its GT aliases belong to one supervised identity group. Full GT audit found no shared positive target across S1, so S1 grouping is sufficient for that verified link structure; still audit near-duplicate observations and unlabeled aliases.

The existing224,776 “validation” IDs and especially the20,000 repeatedly inspected IDs are **development-exposed**. Do not use them as a new unbiased holdout. Select unexamined groups, record past exposure, and train from base weights when prior encoder training included those groups. Inspect cross-split exact/normalized near-duplicate identity patterns; report a stricter family-level stress split when brand families recur, without assuming all same-brand businesses are one entity.

Keep the **full appropriate S2/S3 pool**, including unassigned distractors, for evaluation retrieval. Query subsampling is fine; positive-only or PIN-biased corpus subsampling is not a final quality measurement.

Clarify the unsupervised corpus policy: fitting IDF on the unlabeled retrieval pool is part of corpus indexing, not label supervision. Current code fits separate train/test pool IDF. Reproduce that policy in evaluation, compare score-distribution stability, and document it. Never use holdout labels for normalization dictionaries, encoder examples, mining, feature fitting or candidate injection.

### 8.2 Evaluation tiers

1. **Engineering smoke:** 200–2,000 stratified train/dev queries, all country/code paths, full applicable target pools; catches functional failures.
2. **Development probe:** fixed20,000 random/stratified IDs; every experiment has identical queries and pool. Add a disjoint20,000 stability probe periodically.
3. **Full development:** approximately220,000 IDs; selected candidates/models only, after probe success.
4. **Locked holdout:** evaluate the chosen pipeline after selection. If used to make a decision, it becomes development evidence and must be described accordingly.
5. **Country generalization diagnostics:** train India/evaluate US and train US/evaluate India, keeping full pools. These diagnose transfer but are not French validation.

Do not replace the main stratified evaluation with country holdout; the older plan explicitly preferred stratified validation. Add cross-country tests as supplementary robustness experiments.

### 8.3 Required report per experiment

Persist per-S1 predictions, candidate IDs, all GT IDs, OOF fold, config hash and scores; produce:

- Exact macro F0.5 overall and per country, source coverage, match-count bucket, script pair, missing-address status, common-name frequency, source and candidate-count bucket.
- Candidate oracle macro F0.5, pair recall, non-singleton macro recall, zero-hit rate and all-true-matches-retrieved rate.
- Final micro precision/recall, matcher recall conditional on candidate retrieval, final complete-set accuracy, mean predicted count, empty-list rate and singleton false-merge rate.
- Candidate counts mean/median/p90/p95/p99/max; raw union versus retained versus scored counts; total comparisons and reduction ratio using an explicitly stated global or country-wise denominator.
- Runtime split into preprocessing, index build, retrieval, features, model scoring, writing; peak RSS/VRAM and disk bytes.
- Paired per-S1 score differences and bootstrap confidence intervals. Resample S1/identity groups, not individual correlated links. Use fixed bootstrap seeds; report uncertainty rather than only four-decimal scores.

Target-related report: show whether the lower confidence bound exceeds0.98 on the held-out labeled distribution. This still cannot certify France or private leaderboard performance.

### 8.4 Miss and false-positive taxonomy

For every missed true link, distinguish: not in pool; representation destroyed; zero vector; below retrieval floor; below topK; dropped by posting cap; lost during fusion/compression; dropped by neural routing; scored and rejected. Compute direct GT-pair similarity when it is outside returned results; obtain exact ranks on a diagnostic subset against the full shard.

For each wrong match, record same-name/different-location, same-address/different-business, shared numeric token, short/generic name, empty fields, synthetic alias confusion, accent/script changes, source dominance and multi-S1 conflict. Include record text for **local** review, not external lookup.

Rank error categories by macro loss and estimated tractable fraction. Review at least200 stratified false positives and200 misses, including one-match and singleton entities. Include examples of valid conflicting house numbers so numeric disagreement is not converted prematurely into a hard veto.

**Gate A:** reproducible baseline, exact candidate ceilings, correct all-S1 scoring, persisted splits, and no stale-cache acceptance. Model/threshold tuning cannot substitute for this gate.

## 9. Phase B: preserve information with multiple text views

### 9.1 Record representation

Persist raw columns unchanged. Precompute:

```text
record_idx, entity_id, source, country_raw, country_key
name_raw, address_raw
name_unicode, address_unicode
name_latin_folded, address_latin_folded
name_core, name_compact, name_token_signature
name_alias_segments, acronym_candidates
house_tokens, unit_tokens, number_ranges, postal_candidates
script_flags, field_availability, normalization_loss_flags
```

Use integer record indices internally but retain exact reversible entity IDs. Do not feed identifier digits, file row order, split hash or GT cardinality into a predictive model. Those are bookkeeping or labels, not identity evidence.

### 9.2 Unicode and field-aware normalization

- NFKC and casefold for a stable Unicode view. Preserve letters, combining marks and numbers; collapse separators without breaking graphemes.
- Accent folding is an additional Latin view. It must not erase non-Latin scripts from the primary representation.
- Separate name legal-form handling from street/unit handling. Unknown countries use a generic conservative normalizer.
- Keep both suffix-preserving and suffix-reduced names; suffix mismatch is a feature, not automatic rejection.
- Recognize dotted initialisms, `M/s`, DBA/trading-as segments, domains/compact names, punctuation and token-order variants. Retain original full strings.
- Domain removal and `1→l`/`0→o` corrections are extra views, not irreversible replacement. Test mixed numeric business names.
- Parse address components before removing punctuation: `12A`, `12/3`, `1056-1060`, unit identifiers and postal candidates have different meanings.
- A first five/six-digit number is not automatically a valid postal code. Store confidence/context and missingness; do not equate same postal prefix with identity.

Transliteration is optional complementary retrieval. Prefer a deterministic local algorithm or a character model trained solely from supplied train identity pairs. Learn alignment from name pairs with strong address support; label noise must be controlled. Different scripts do not automatically mean the same name after transliteration. No external translation/geocoding/business service.

### 9.3 Normalization checks

Verify full Hindi/Kannada/Tamil word preservation; French accents; `Ste` in US address versus French name; CO/FL address context; distinct house/unit numbers; empty fields; punctuation-only names; one/two-character names; domain compaction; leading zeros; unexpected country labels; input idempotence and train/test parity. Treat tests as focused checks for observed failure modes, not superficial test-count goals.

**Gate B:** representations preserve information, no empty-empty positive agreement, and old versus corrected normalization has a measured retrieval/score comparison on the fixed development probe.

## 10. Phase C: complementary candidate generation

### 10.1 Begin with a diagnostic union, then optimize cost

Retain the frozen baseline as a historical control. The already measured frozen+v6 union is a useful bridge baseline, but its0.98307 ceiling is insufficient margin. Recompute corrected channels on the new development sample and full country pools.

| Channel | Initial experimental setting | Failure mode targeted |
|---|---|---|
| Unicode name-only char TF-IDF | `char_wb`,3–5grams, K100 | Typos, reordered names, missing target address |
| Latin/accent-folded/compact name | K50–100; retain raw-name channel | French accents, domains, concatenation |
| Full-address-only char TF-IDF | K100–200 | Unrelated trade names, cross-script names, reordered address components |
| Joint name+full address | K100; compare explicit field weights | Preserve useful combined evidence without first-token dependence |
| Rare-token word TF-IDF or local BM25 | K50–100 | Distinctive tokens/addresses diluted by character grams |
| Structured compound keys | frequency-ranked additions | Exact nonempty signatures, house+street, name+locality, postal+name |
| Exact full-observation duplicate expansion | retrieve equivalent target records as candidates | Multiple S2/S3 IDs sharing the same observable record |
| Multilingual dense retrieval | K50/100/200 sweep | Cross-script names/addresses and broader variant matching |

Values are starting points, not promises or official limits. Do not build all variants at full scale simultaneously. First test independent name and full-address channels, then add the channel that recovers the most high-loss misses per unit cost.

### 10.2 Lexical implementation and rank diagnostics

Use sparse matrices throughout. Sweep `min_df=1/2/3`, vocabulary limits150k/300k/unlimited where memory permits, char3–4 versus3–5, word tokens, and K50/100/200/400 on small query probes with the full pool. Rare grams removed by `min_df`/feature caps can carry identity signal; evaluate that loss directly.

Sweep floors0.02/0.005/0 for nonzero similarities. A zero floor cannot recover records with no shared representation. Track empty query vectors and positive pairs with zero overlap; route them to alternative channels.

`sparse_dot_topn` integrates top-N selection during sparse multiplication. Its documented input conversions and default unsorted output matter: benchmark a precomputed `P.T.tocsr()` reused across query batches, sorting deterministically by score then stable ID. This may avoid repeated CSC→CSR conversion, but measure the extra transpose memory and actual runtime rather than assuming a speedup. [Official package documentation](https://github.com/ing-bank/sparse_dot_topn).

Do not densify150,000-dimensional TF-IDF for Faiss. A dense model/SVD index is a separate representation with its own information-loss experiment. The old4096-dimensional hashing demo is a smoke test, not a full-scale design.

### 10.3 Structured retrieval

For each key family, log posting length distribution, overflow, retrieved positives and unique gains. Common-name/house-number keys need intersections with independent evidence or secondary ranking. Preserve exact-name candidates as candidates; never accept exact names as final matches automatically.

Use token frequencies estimated on the permitted corpus. Rare-word overlap and inverse-block-frequency are useful, but an arbitrary first token can be generic or reordered. Test acronym/initialism and sorted-neighborhood routes only if they address observed misses. American Soundex is not a general Indic/French matcher.

For duplicate-observation expansion, map a retrieved target to other target IDs sharing nonempty country+name+address signatures. Audit GT purity and ambiguity first. Additional IDs must be included in the actual candidate set and scored; name-only or empty-address duplicate keys require much stricter treatment. This is not permission for unrestricted transitive matching.

### 10.4 Fusion, source balancing and budgets

Start with deduplicated set union. For each pair retain channel presence, independent raw scores and ranks. Use reciprocal-rank fusion as a simple ordering baseline:

```text
rrf(q,c) = sum_j weight_j / (60 + rank_j(q,c))
# ranks start at 1; missing channel contributes 0
```

RRF is not a probability. It cannot guarantee recall after a cap. Evaluate untrimmed union first, then K50/100/150/200/400 budgets and protected per-channel reservations. If reservations exceed budget, expand explicitly; never silently discard them.

Compare joint S2+S3 retrieval against source-balanced quotas. Both sources can hold several matches, so quota1 is wrong. Track whether one source crowds out the other's positives, especially when source-specific noise differs.

Adaptive depth features can include script/availability, name frequency, channel disagreement, rank-tail density, saturated topK, record length, source balance and score margins. A strong top1 or large top1/top2 gap is insufficient: it says little about the fourth difficult true alias.

During discovery, expanded sets must be supersets of preceding sets. Once a final compression policy is chosen, report every positive it loses and its change in oracle/actual score. A learned pre-matcher pruner must be evaluated out of fold, versioned and disclosed; its scored population must not be hidden in efficiency reporting.

### 10.5 Retrieval promotion gate

- Oracle macro F0.5 above0.99 as a minimum research promotion target, aiming≥0.995.
- Pair recall≥99% overall and each labeled country as a secondary target; all-match and zero-hit metrics materially improved.
- Every channel's marginal gains and pair costs measured; repeated changes do not degrade singleton/matcher performance unnoticed.
- Approximation, score-floor, caps and compression losses reported separately.
- Same implementation used for development, holdout and test.

If a channel union misses these targets, inspect remaining high-macro-loss misses. Do not keep increasing K indiscriminately when the representation has no relevant signal.

## 11. Phase D: task-specific multilingual retrieval

### 11.1 Model choice and eligibility

Start the multilingual experiment with `intfloat/multilingual-e5-small`: its publisher lists MIT licensing, multilingual support and384-dimensional embeddings. Use consistent serialization, attention-mask-aware pooling and L2 normalization; symmetric record similarity can start with `query:` on both sides according to the model card. This is a proposed challenger, not a demonstrated winner. [Publisher model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md).

Keep the existing `all-MiniLM-L6-v2` as a cheap English control if useful. The current model card labels it **Apache-2.0**, not the MIT license claimed in the notebook, and labels its language English. It is eligible under the stated license categories but not a strong default assumption for Indic coverage. [Publisher model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/README.md).

Record immutable revisions, license texts, parameter counts and tokenizer/checkpoint provenance for each actual model. Verify the final combined system remains within the stated parameter rule rather than assuming the limit only applies to one component. Larger multilingual checkpoints should be evaluated only after the compact baseline demonstrates complementary gain and throughput is measured.

### 11.2 Training pairs and negatives

1. Join **all GT positive links in the training partition**, including those missed by every current blocker. The full dataset has7.64M positive links; do not restrict supervision to the handful visible through PIN blocking.
2. Sample identity groups uniformly, rotating among their target aliases. Add S2–S3 positives belonging to the same training S1 as a separate augmentation experiment; avoid overweighting identities with many aliases.
3. Mine negatives from lexical/dense/structured candidates: same name elsewhere, same address different business, short/generic names, near-edit spellings, wrong franchise branches, source-specific distractors and current model false positives.
4. Exclude all known same-identity aliases from negatives. Include random negatives for breadth, but do not let trivial cross-country pairs dominate an in-country inference problem.
5. Use one identity group per in-batch contrastive slot, or explicitly mask all known positives. String-level duplicate removal alone misses distinct aliases of the same business.
6. Ensure each batch has a fixed number of text columns or use a collator that explicitly supports the chosen variable-negative format. The existing `[anchor, positive] + up-to-3-negatives` examples may differ in arity.
7. Keep mining, augmentation dictionaries and training strictly inside train/OOF identity boundaries.

MultipleNegativesRankingLoss provides an initial contrastive objective; its in-batch negatives require care, and Sentence Transformers recommends duplicate-aware batching. Extend that to **identity-aware** batching for this dataset. [Official loss documentation](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.losses.MultipleNegativesRankingLoss).

Suggested first training sweep: learning rate1e-5/2e-5,1–3 epochs, sequence length128/256, measured effective batch size. These are starting values. Measure actual truncation by field and country; try512 only when necessary. Preserve address numbers and enough name/address context. Save periodic optimizer/model/RNG checkpoints and sample manifests.

Use only supplied-data noise patterns for augmentation: punctuation, observed abbreviations, mild typos, case, safe reordering, and selective field dropout. Arbitrary number replacements, severe deletion or unrestricted translations can change identity. Compare pretrained and fine-tuned models to detect loss of cross-language generalization.

### 11.3 Indexing and approximation

Encode target records once per model/view into chunked memmaps or columnar vector shards with explicit row-to-ID maps. Store tokenizer/serialization/pooling hashes. Encode Unicode text; do not feed the ASCII-deleted representation to a multilingual model.

On a fixed development-query subset, compare exact normalized-vector inner-product retrieval over the **full country pool** with candidate ANN configurations. Then evaluate actual GT recall, not only agreement with exact neighbors. Test HNSW or IVF/compressed variants according to memory and measured throughput; tune efSearch/nprobe and compression. [Faiss index guidance](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

Separate encoder representation failure from ANN approximation failure. A dense channel may have mediocre standalone recall and still be valuable through unique recoveries. Promote based on hybrid oracle/final score delta per added pair and GPU hour.

**Gate D:** verified held-out retrieval gain, acceptable approximation loss, no identity leakage, feasible full-pool memory, and inference parity across saved/exported formats.

## 12. Phase E: stronger supervised matching

### 12.1 Use more of the labeled corpus deliberately

The current20,000-query sample is under1% of all training S1. It is adequate for a first model, not evidence that the available supervision is exhausted. Build learning curves at50k/100k/300k/1M training identities, stratified by country, cardinality and difficult target types. Stop expansion when out-of-sample gain flattens relative to feature cost; “train on every possible negative” is not necessary.

Start with pooled LightGBM and a generic country fallback. Country specialists are optional experiments, not assumed better. A Western/US model is not automatically correct for France. Retrain whenever candidate channels, normalization or feature definitions change.

### 12.2 Feature specification

| Feature group | Concrete additions |
|---|---|
| Name lexical | Unicode/folded/core-name edit ratios, char ngram overlap, token order and containment, IDF-weighted token overlap, short-name flags |
| Alias/abbreviation | DBA segment alignment, acronym expansion match, compact/domain-name similarity, suffix present/equal/conflicting |
| Address lexical | Full Unicode/folded address similarities, rare address-token overlap, component coverage, reordered-token similarity |
| Numeric structure | House equality/conflict/range overlap, unit equality/conflict/missing, postal confidence/equality, leading-zero normalization |
| Availability | Raw missingness, normalized-empty flags, normalization loss, name/address lengths, script pair |
| Retrieval | Score, rank, reciprocal rank and presence **per channel**, number of channels agreeing, candidate source, index/view IDs |
| Dense/neural | Name/joint embedding cosine, optional learned transliteration similarity, task-trained cross-encoder score |
| Query context | Candidate count, score tail, source balance, name frequency, field quality, count above score bands |
| Reverse context | Candidate's competing S1 scores/margin, ambiguity count; compute identically on honest evaluation/test corpora |

Do not add raw entity IDs, file order or true match counts as predictors. Country-match is constant inside verified same-country partitions, so it cannot by itself provide transfer. Avoid raw country one-hot training with no unseen-category handling.

Empty equality, high partial similarity for one-character strings, token containment in generic legal suffixes, and common numeric matches are not independent positive evidence. Preserve missingness/conflict flags and let supervised evaluation judge interactions.

### 12.3 Sampling and optimization

- Include all retrieved training positives. Consider explicitly marked GT-positive augmentation for matcher training only, while calibrating on natural candidate sets.
- Mix high-score hard negatives, channel-specific negatives and random candidates. Current1:6 is a reasonable initial budget but current code samples randomly from retrieved negatives; it does not specifically select the hardest negatives.
- Include true singleton queries and non-singletons with no retrieved positive. Preserve realistic candidate-list distributions.
- Compare ordinary pair weighting, inverse-candidate-count/per-S1 weighting, and capped class weights. Macro evaluation values entities equally; no weighting surrogate automatically optimizes it.
- Group OOF by S1. If the encoder or pruner is learned, its folds must also respect the OOF boundary; classifier grouping alone cannot undo encoder leakage.
- Early-stop with a compatible library API. Pair logloss can guide optimization, but promote models using end-to-end entity-macro F0.5 on unsampled candidates.
- Retain real feature names in model artifacts rather than only `Column_0`…; assert names/order/dtype/schema hash at inference.

Use the current63-leaf/depth7 model as baseline. Search modest alternatives such as31/63/127 leaves, regularization/min-leaf changes, controlled row/column subsampling and2–3 seeds. Compare a second model family only after features and data scale improve; ensembles of identical errors add cost without value.

### 12.4 Neural cross-encoder, conditional on classifier error evidence

Train a compact multilingual encoder with a binary pair-classification head on serialized record pairs. This is a different objective from generic web-search relevance. Use positives/negatives from the final retrieval distribution, supplemented with training-only hard positives and targeted confusing negatives. Fine-tune with BCE or a validated ranking+classification objective; calibrate on natural candidate prevalence.

Route ambiguous pairs and high-risk contexts: score near threshold, name/address conflict, cross-script uncertainty, singleton-risk lists, competing owners and disagreement between model channels. Outside the route, retain calibrated LightGBM decisions. Include a random audit sample of unrouted pairs to quantify routing mistakes.

Do **not** keep only the first10 retrieval candidates for final matching. The measured current top10 oracle is0.957085, so that design cannot approach0.98 on this sample. Even after better retrieval, route recall must be measured rather than assumed.

Fuse using held-out/OOF scores: a simple weighted calibrated blend or a small meta-model trained on disjoint folds. Report standalone and incremental gains. Never train a stacker on in-sample base-model scores and call it OOF.

**Gate E:** final macro score improves on independent labeled data, retrieval gains survive matching, and singleton/one-match/India slices have no unexplained regressions.

## 13. Phase F: choose sets, including empty sets, for the real metric

### 13.1 Threshold baseline

Preserve a simple global threshold baseline. Search coarse then fine thresholds over unsampled dev/OOF scores, using the full S1 manifest, including zero-candidate rows. No theoretical rule implies the optimum must lie between0.6 and0.8; the current sampled-training model selected0.85.

Your two-threshold rule is:

```text
if max(pair_scores) < T_singleton: emit empty
else: emit every candidate with score >= T_match
```

If `T_singleton <= T_match`, the singleton gate is redundant: the pair threshold already produces the same empty decision. Both saved thresholds equal0.85, so the current model has effectively one threshold. A stricter `T_singleton > T_match` can change behavior; search that region deliberately rather than presenting equal thresholds as a separate singleton model.

### 13.2 Explicit singleton/no-match modeling

Train a query-level probability of no true target / no safe output using cross-fitted candidate-list features: top scores, score gaps/tail, channel agreement, source balance, missingness and candidate competition. Avoid leaking ground-truth counts into features.

Distinguish a true singleton from a non-singleton whose matches were not retrieved. Both can have no positive candidate, but they have different metric consequences and different recovery actions. A second retrieval expansion may be appropriate before declaring an uncertain query empty.

Tune this query gate jointly with pair threshold on exact macro F0.5. Report singleton false-merge rate and non-singleton false-empty rate. Do not assume the training5.6% singleton rate holds in France/test or force that percentage of empty outputs.

### 13.3 Calibration and cardinality-aware decisions

Negative sampling changes class prevalence. Raw classifier scores are not automatically probabilities. If using probability-based fusion or set decisions, fit calibration on grouped, unsampled held-out candidate scores and compare Brier/reliability as diagnostics. Preserve a global fallback; per-country calibration needs adequate labeled evidence and cannot be French-label tuned.

Advanced optional experiment: evaluate candidate subsets by estimated expected entity F0.5, including the empty set. For fixed known truth cardinality n, a set of k predictions has `F = 1.25 TP/(0.25 n+k)`. In production n is unknown; a calibrated cardinality/uncertainty model or Monte Carlo label sampling is needed. Correlations and unretrieved positives must be considered. Do not plug in the training mean3.67 as a fixed truth count or assume independent scores are exact.

This is a later-stage experiment after trustworthy probabilities, not a prerequisite for the first improved submission. Promote only on a separate validation slice.

### 13.4 Safe consistency experiments

Use reverse-candidate margins to flag one target plausibly assigned to multiple S1. The full training GT supports at most one S1 owner per target, while permitting many targets per S1 and unassigned targets. Compare soft competition penalties and optional capacity-constrained selection against independent thresholding. Leave ambiguous targets unmatched when justified; never force assignment.

Do not use unrestricted connected-components closure. One false positive can connect unrelated businesses and produce many false merges. A bounded one-hop alias expansion can be tested as **additional retrieval**, with every added edge scored and reported. Train-derived alias dictionaries must respect folds; test predictions must not become unverified labels driving uncontrolled propagation.

## 14. France and robustness strategy

France is large enough to decide the target and has no labeled validation. Address it explicitly:

1. Preserve accented raw and Unicode text; add Latin-folded, compact and legal-form views without overwriting originals.
2. Keep full-address and name-only retrieval. Avoid assuming the smaller France pool makes defaultK40 safe.
3. Use a generic country-independent feature schema and pooled fallback. Compare frozen multilingual encoders to fine-tuned ones on cross-country diagnostics for catastrophic transfer loss.
4. Inspect unlabeled country/source distributions: missingness, script, lengths, zero vectors, token frequencies, candidate saturation, score bands and predicted cardinalities. Distribution checks do not provide French accuracy labels.
5. Apply controlled supplied-data-based noise stress tests on held-out US/India identities: accents, punctuation, case, safe word order, observed abbreviations, missing fields and script-supported transformations. Report them separately from natural validation.
6. Evaluate leave-US-out and leave-India-out transfer in addition to main validation. Neither is proof of France quality, but severe degradation should block deployment of a specialized fallback.
7. Do not tune a French threshold to an expected match count or use external business/geographic references. Favor robust global calibration when there is no labeled basis for country-specific tuning.

Public leaderboard feedback can reveal that a whole pipeline improved; without country-specific labels it cannot identify French recall. Maintain a limited, predeclared submission ledger and avoid numerous tiny threshold probes against public scores.

## 15. Scale, hardware and efficient execution

### 15.1 Budget from actual artifacts

Frozen test settings imply at most **136,325,300 pairs** if all query lists fill K90/80/40. This is already substantial. Increasing averageK requires an explicit matching and disk plan.

| Mean candidates/query | Test pairs | Dense40-feature float32 matrix alone |
|---:|---:|---:|
| 50 | 86.63M | 13.86GB |
| 100 | 173.25M | 27.72GB |
| 200 | 346.51M | 55.44GB |
| 400 | 693.02M | 110.88GB |

These exclude IDs, indexes, strings, model memory, temporary allocations and output. Never materialize such a feature matrix as Python lists or one global NumPy array. AtK100, two uint32 row IDs plus one float32 score already require roughly2.08GB before additional channel metadata; string-heavy structures cost much more.

One9,969,589-record384-dimensional float32 embedding view occupies approximately15.31GB. The India test subset alone uses about7.25GB before index/copies. Float16 cache storage does not imply a CPU Flat/HNSW index uses half the memory. Two views and index-build copies can exceed32GB.

Current India blocking logs show25,000-query chunks in1,636–1,712seconds, approximately14.6–15.3queries/second, after index preparation. Those are **blocking-only historical measurements**. They are not a forecast for a hybrid retriever or cross-encoder. The old2.5hour full-pipeline estimate is not supported.

### 15.2 Execution architecture

- Normalize once into country/source-partitioned Parquet with integer row IDs; benchmark compression and disk footprint.
- Cache fitted vectorizers, sparse matrices and completed embedding shards with content-addressed manifests.
- Process one country/view at a time on the32GB PC. Limit simultaneously resident raw object DataFrames, record tuples, duplicate search strings and index copies.
- Score features in bounded pair batches, initially100k–500k pairs adjusted to observed RSS; write predictions and candidate sidecars immediately.
- Precompute per-record tokens, ngrams, numeric components and missingness rather than reconstructing sets for every pair.
- Keep target IDs and channel scores aligned in compact arrays; final TSV is a serialization format, not the primary computation store.
- Benchmark lexical retrieval with several batch/thread settings. Thread count is not P-core affinity. Avoid multiplying BLAS/OpenMP/joblib parallelism.
- A100: use for embeddings, contrastive learning and routed pair scoring; checkpoint to durable storage after locally completed chunks. Confirm GPU availability/VRAM at runtime instead of relying on notebook labels.
- MacBook: run independent country or chunk jobs from the same package/config, not a copied divergent notebook. Benchmark it separately; avoid relying on guessed thermal/runtime margins.
- OpenVINO: adopt only after tokenizer/pooling and FP32-versus-export score parity pass. Test padded batches, not just one unpadded pair.

At averageK100 there are173.25M pair decisions. If a measured cross-encoder runs1,000pairs/second, all-pair scoring is48.1hours; routing5% is2.4hours plus overhead. These are arithmetic examples, not measured hardware throughput. Measure1k–5k representative queries per country before deciding route size.

### 15.3 Disk and deadline discipline

Measure free disk and projected intermediate size before launch. The MacBook runbook's15GB free-space suggestion is inadequate for multiple dense caches, expanded candidates and resumable model outputs. Do not promise a fixed budget without measuring the chosen artifacts. Retain only promoted model/index versions plus reproducible manifests when space is constrained; remove nothing from the historical baseline without an explicit retention decision.

Use a release freeze at least the measured remaining inference time plus a substantial rerun/validation margin before the actual deadline. As no deadline is supplied, the schedule below is relative. Avoid starting an unvalidated all-test run just because a small smoke file has the right number of rows.

## 16. Production design and migration

### 16.1 Shared code layout

```text
code/business_entity_resolution/
  README.md
  requirements.txt
  configs/
    baseline.yaml
    hybrid.yaml
  src/er/
    cli.py
    schema.py
    io.py
    normalization.py
    splits.py
    metrics.py
    evaluation.py
    features.py
    calibration.py
    consistency.py
    manifests.py
    pipeline.py
    retrieval/
      lexical.py
      structured.py
      dense.py
      hybrid.py
    training/
      encoder.py
      matcher.py
      cross_encoder.py
  tests/
  models/                   # final licensed artifacts/manifests as needed
```

Notebooks become launch/report interfaces importing this package. Local cells6/8 migrate to metrics/splits;10 to normalization;19/22 to retrieval;24 to features;25 to training;26 to calibration;28/30/32 to pipeline/manifests. A100 cells14/16/18/20 become explicit dense/train/rerank/export commands. Keep the SageMaker notebook as an optional tiny engineering demo; it should not control production configuration.

### 16.2 Essential API contracts

```python
normalize(record, policy) -> NormalizedRecord
retrieve(query_batch, index_bundle, config) -> CandidateBatch
# CandidateBatch: query_idx, target_idx, channel scores/ranks/presence
build_features(query_batch, candidates, record_store, schema) -> FeatureBatch
score(feature_batch, model_bundle) -> PairScores
decide(query_ids, pair_scores, decision_config) -> MatchSets
evaluate(required_query_ids, truth, candidates, matches) -> MetricReport
write_outputs(required_query_ids, scored_candidates, matches, destination)
```

Same query/record/config must produce identical candidate identity and feature definitions in development and production. Assert no missing required S1 when converting a scored pair table to per-query rows.

### 16.3 Artifact identity and resume

Each run manifest should contain full Git commit and dirty-code/source hash, input file hashes, exact split/query/pool hashes, normalization/serialization version, vectorizer vocabulary/IDF hash, retrieval parameters, encoder revision/checkpoint, index parameters, feature schema/order/dtypes, matcher hash, threshold/calibration hash, software versions and seeds.

Per-chunk manifests add country, source partitions, exact ordered query IDs, row/pair counts, output checksum, completion flag and timing. Write temporary outputs, fsync/close where appropriate, validate, then atomically rename and publish the completion manifest. Resume only when identity matches; matching line counts are insufficient. A partial final row can still leave the expected number of lines.

Merge must reject conflicting manifests, missing/unexpected chunks and duplicate IDs; verify exact required-query set equality. Do not scan arbitrary `test_*.tsv` files and assume all belong to the selected run. Current partial runs do not write a reliable per-device manifest and current merge writes `commit:null`, despite stronger claims in the runbook.

### 16.4 Pin the runtime actually validated

Audit environment includes Python3.11, numpy2.4.6, pandas3.0.5, scipy1.17.1, scikit-learn1.9.1, LightGBM4.7.0, RapidFuzz3.14.6, sparse-dot-topn1.2.0 and faiss-cpu1.15.1. These are an environment snapshot, not proof that every package combination and target platform is production-tested. `sentence-transformers` was not installed in this local interpreter; optional training belongs in its own validated environment.

Pin exact CPU and GPU dependency sets after clean-install verification. Record torch/CUDA, tokenizer/model libraries, OpenVINO/Optimum when used, architecture/OS and model-export commands. Package license notices for included libraries/models and preserve code attribution if borrowing implementations. Deleting attribution is not a fair-play strategy.

## 17. Ordered experiments and promotion decisions

**Current override:** execute [local L00–L07 and optional Colab G00–G04](LOCAL_COLAB_IMPLEMENTATION_PLAN.md). The E-series definitions and former device/session ordering below are reference material. No fixed 0.98 score or 0.995 oracle gate remains in force.

**Current execution order is the three-device revision in §1.2–1.3 and the linked device files.** The E00–E12 table below retains the original experimental intent, not a claim that similarly named reports have completed those gates. In particular, E09's cosine-feature report is not the original E09 cross-encoder experiment.

### Current gate status after the E09 review

| Original milestone | Revised status | Required next evidence |
|---|---|---|
| E00 | Historical freeze/splits exist | Extend provenance to current runs and preserve exposure ledger |
| E01 | Shared functions exist; downstream integration incomplete | Actual training adapter uses parsed fields and guarded missingness |
| E02/E03 | Promising 300-query full-pool measurements | Fixed larger-probe reproduction and per-country coverage/cost |
| E04 | GPU inference integration only; multilingual retrieval unproven | Actual complementary full-pool multilingual channel |
| E05 | Not established by reports | Group-safe domain training versus frozen control |
| E06 | Frontier measured; K100 promotion rejected | Quality-preserving budget, including actual matcher score |
| E07 | Report exists; end-to-end evaluation invalidated | Clean natural-candidate baseline and identity/feature curves |
| E08 | Small contaminated calibration result | Reproducible calibrator, separate calibration/comparison IDs |
| E09 | Cosine challenger regressed; original reranker not completed | Controlled neural ablation and optional task-trained routed reranker |
| E10 | Tiny gains on contaminated evaluations | Lower-priority clean ownership/consistency ablation |
| E11 | Pending | Selected pipeline on independent locked evaluation |
| E12 | Pending | Complete inference, both outputs, validator and reproducible package |

Before returning to any performance experiment, complete **R0 evaluation integrity** in the current review and **D1-00** in the local-device plan. The next full matcher run should not execute the existing training scripts unchanged.

### Original experiment definitions

| ID | Work | Required artifact | Advance when |
|---|---|---|---|
| E00 | Freeze/reproduce historical baseline and new splits | baseline manifest, metric and exposure ledger | all values traceable; no mixed outputs |
| E01 | Unicode/field fixes and train/test parity | normalization comparison and parity checks | correctness holds; gains/regressions explained |
| E02 | Independent name and full-address lexical channels | channel recall/oracle/cost table | meaningful unique recovery beyond joint baseline |
| E03 | Structured/duplicate-observation union without cap | raw-union and marginal-gain report | gain survives proper evaluation |
| E04 | Frozen multilingual dense challenger | exact/ANN and hybrid comparisons | useful unique recovery at feasible cost |
| E05 | Train-domain encoder, group-aware hard negatives | learning curves, multilingual transfer tests | beats frozen challenger without transfer collapse |
| E06 | Candidate compression/adaptive budgets | K/oracle/actual-score/cost frontier | selected budget holds target with measured loss |
| E07 | More identities and stronger LightGBM features | full OOF scores and training-size curve | end-to-end macro improves with slice stability |
| E08 | Calibration, meaningful singleton gate, thresholds | reliability and macro threshold surface | paired improvement on separate dev slice |
| E09 | Routed ER cross-encoder and optional ensemble | incremental gain and full-runtime projection | benefit exceeds added error/compute cost |
| E10 | Reverse ambiguity/limited consistency | independent versus constrained comparison | no harmful forced ownership/false propagation |
| E11 | Full development and locked holdout | final report with uncertainty | candidate and final metric gates honestly met |
| E12 | Production smoke, full inference and package | validator logs, exact output checks, reproducible zip | clean reproduction generates identical outputs |

Do not sweep every parameter at once. Change one conceptual component, or a necessary coupled group such as normalization+retraining, and store all recovered/lost links. Select by paired macro improvement first, then robustness/cost. No fixed improvement is assigned to any row.

### 17.1 First implementation session

**Historical session plan:** much of the shared-package/split scaffolding below is already implemented. Do not restart it blindly. The next session is D1-00/D1-01, with D2/D3 preparing independent work in parallel.

1. Create immutable historical run directory and copy the two model configs, selected candidate manifests and baseline metrics.
2. Export the exact scorer and normalization functions into shared modules; add checks for the observed Unicode/empty/address failures.
3. Persist fresh train/dev/holdout manifests and20k development probe; retire sorted-prefix sampling for model selection.
4. Implement a single lexical retrieval entry point used by both train/test; require score/provenance persistence.
5. Reproduce the baseline on the new probe, then compare name-only, full-address-only and their union against joint retrieval.
6. Produce candidate oracle, loss taxonomy and throughput report before any new full-test run.

### 17.2 If time is very short

Prioritize a coherent, valid end-to-end baseline: complete LightGBM test inference with correct retrieval features, fix output/version mixing, validate every test S1, and assemble the package. Keep a tested rollback artifact. Avoid retraining and deploying an unvalidated neural pipeline hours before the deadline.

If several days of development and GPU time remain, run E01–E08 before committing to expensive cross-encoder deployment. If time is plentiful, E09/E10 and broader transfer experiments are justified after the core error budget improves. Wall-clock estimates should come from the initial full-pool benchmarks, not from the old plan's timing guesses.

### 17.3 Decision tree when progress stalls

```text
Is candidate oracle below target?
  Yes -> inspect high-loss misses; add representation/channel, fix cutoff or approximation.
  No  -> is actual score far below oracle?
           Yes -> fix features, supervision, hard negatives, calibration and singleton decisions.
           No  -> evaluate uncertainty/France risk and optimize candidate/runtime cost.

Does a new route improve pair recall but hurt macro score?
  Inspect new false positives, one-match/singleton losses and threshold calibration.
  Retrain on the new distribution before deciding that retrieval gains are worthless.

Does a new model only improve the repeatedly tuned20k sample?
  Reject the generalization claim; evaluate fresh groups/full development.
```

## 18. Full test export and final package

### 18.1 Output construction

For every required test S1, write exactly one row to each output, including empty lists. Use stable deterministic target ordering, no duplicate targets, no spaces/quotes added to ID lists, no NaN strings, and exact headers:

```text
matching_results.tsv: source1_entity_id<TAB>matched_entity_ids
candidate_pairs.tsv:  source1_entity_id<TAB>candidate_entity_ids
```

Candidate IDs must equal the actual pre-matcher population. Every final match is a subset. Keep detailed scores in sidecars, not extra columns in official outputs. Persist the final matching-file hash so the packaged artifact equals the one uploaded.

### 18.2 Validation sequence

1. Internal streaming check: exact required S1 set, one row each, valid headers/UTF-8, target existence/source prefixes, no intra-list duplicates, strict match subset and exact candidate/scored-pair equality.
2. Check per-country counts against test S1: US663,106; India809,986; France259,452. Do not use those values to exclude any unexpected label in a future input.
3. Inspect output cardinality/empty/candidate distributions for catastrophic drift, without forcing them to resemble train.
4. Run the supplied validator against the actual test files. From the repository root:

```powershell
python student_resource/student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/student_resource/dataset/test --check-ids
```

The official validator stores candidate sets in memory and may be expensive at hundreds of millions of pairs. Budget a larger validation machine if necessary. If memory prevents a full candidate pass, retain the official matching-file check and add an independently verified streaming/external-sort candidate audit covering every invariant; document exactly which checks ran. Do not call a skipped check a PASS.

5. Reproduce from a clean extracted package with pinned dependencies and the supplied data. Verify output hashes or explicitly documented deterministic equivalence.

### 18.3 Required archive structure

```text
<team_name>_submission.zip
  output/
    matching_results.tsv
    candidate_pairs.tsv
  code/
    business_entity_resolution/
      src/
      README.md
      requirements.txt
      configs/ and final model artifacts as needed
  Documentation_template.md
```

The README must give exact preprocessing, training/retrieval, inference, validation and packaging commands, expected resources, seeds, resume behavior, paths and required model artifacts. Keep it self-contained for regeneration, including any legally redistributable weights or documented pinned acquisition/retraining procedure needed for the provided-data workflow. Do not rely on private Drive state or an unnamed notebook kernel.

Fill the supplied template with the final method actually used: measured data facts, candidate strategies and counts, retained recall/oracle, normalization and features, model architecture/license, threshold selection, honest validation split/score, error taxonomy, resource usage and reproducible entry points. Distinguish labeled US/India results from unvalidated French expectations. Do not package stale sample outputs or mark the0.8886 development score as a private leaderboard result.

## 19. Final acceptance checklist

- [ ] Exact metric and singleton tests pass; every evaluation includes all required S1.
- [ ] Fresh grouped evaluation and exposure history are recorded; all learned artifacts respect splits.
- [ ] Data manifests confirm inputs and target membership; full pools include distractors.
- [ ] Unicode/combining marks, empty fields, country/field abbreviations and number parsing are checked.
- [ ] Candidate oracle provides sufficient margin; all trimming/floor/ANN losses are measured.
- [ ] Candidate expansion, scoring and features match between development and test.
- [ ] OOF predictions and thresholds are reproducible; reported score provenance is explicit.
- [ ] Singleton, one-match, India, source, missing-address and common-name slices are reviewed.
- [ ] France has a tested generic route and explicit uncertainty; no fabricated French score.
- [ ] Optional neural/consistency stages produce independently measured gains and fit the runtime budget.
- [ ] Chunk manifests, atomic resume, multi-device compatibility and exact merge coverage pass.
- [ ] Final files correspond to one model/config run; candidate set equals scored population.
- [ ] All test S1, including France, appear once; all target IDs exist; all matches are candidates.
- [ ] Required validator/internal checks pass on final files; public upload and packaged matching hashes agree.
- [ ] Licenses/parameter counts and no-external-data provenance are documented.
- [ ] Clean package reproduces both outputs and contains the filled methodology template.

## 20. Research references and claim discipline

External research in this audit was limited to algorithm/API/model-license verification. No external business data was used.

1. [LightGBM classifier API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html): current fit interface, subsampling and importance semantics.
2. [sparse_dot_topn project](https://github.com/ing-bank/sparse_dot_topn): sparse top-N multiplication and input-format behavior.
3. [multilingual-e5-small publisher model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md): eligible proposed multilingual encoder and serialization/pooling conventions.
4. [all-MiniLM-L6-v2 publisher model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/README.md): Apache-2.0 correction and English model scope.
5. [Sentence Transformers losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.losses.MultipleNegativesRankingLoss): contrastive-learning and duplicate-batch considerations.
6. [Faiss index selection](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index): index/memory/accuracy trade-offs.

None of these references establishes a challenge score. All proposed thresholds, budgets, model choices and schedules must earn their place through the experiments above. The immediate bottleneck is now quantified: current retrieval caps the old sample below0.98, while the matcher remains about0.091 below that ceiling. Solving both, while completing an honest and reproducible end-to-end submission, is the highest-value next phase.
