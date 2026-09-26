# Current progress and plan to maximize verified macro F0.5

**26 September 2026 — revised objective requested by the user:** maximize reliable end-to-end F0.5 with the available three devices. There is no fixed 0.98 acceptance threshold. Numerical oracle floors and loss budgets in the earlier plan are historical aspirations, not blockers for useful improvements.

## 1. What is established now

The latest repository reports contain a reproducible B0 screening baseline and a full-pool candidate-depth frontier. They do **not** contain completed new Mac or RTX model results, nor an end-to-end matcher trained/evaluated on the untrimmed union. Device handoff readiness is not a measured quality improvement.

| Measurement on screen_2k | Overall | India | US |
|---|---:|---:|---:|
| B0 actual macro F0.5 | 0.904586 | 0.875299 | 0.933873 |
| B0 K100 oracle | 0.985159 | 0.974536 | 0.995783 |
| K250 oracle | 0.988796 | 0.980223 | 0.997368 |
| Untrimmed union oracle | 0.990452 | 0.983216 | 0.997688 |

The depth sweep uses the same 2,000 screening IDs, with 1,000 per country. K100 to untrimmed raises overall oracle by **0.0052930**, India's by **0.0086805**, and US's by **0.0019055**. Mean candidate count rises from 100 to 339.244, approximately 3.39 times as many scored pairs. This is a useful retrieval tradeoff to test, not an automatic final-score gain. Oracle is the best possible score restricted to a candidate set; it is not matcher accuracy.

For current B0, remaining loss is 0.0954142: retrieval contributes 0.0148407 and matching/decisions 0.0805736. About **84.4%** of present loss is downstream of candidate generation. This supports substantial matcher effort alongside focused India retrieval, rather than waiting for an arbitrary oracle threshold before improving the matcher.

## 2. New experiment executed during this review

Used the existing saved model probabilities, complete ground truth from the evaluation bundle, and disjoint calibration_5k/screen_2k populations. No training, new retrieval, holdout evaluation or production-policy overwrite occurred.

All policies were selected **only on calibration_5k**, then evaluated on screen_2k. Candidate probabilities and model weights remained fixed. Search used thresholds 0.05 through 0.995 in increments of 0.005 and `t_singleton >= t_match`. Country-specific policies use the calibrated global policy as a fallback for countries without labels.

| Policy | Calibration F0.5 | Screen F0.5 | Screen change vs B0 | Paired 95% interval for change |
|---|---:|---:|---:|---|
| B0: 0.70 / 0.70 | 0.912060 | 0.904586 | — | — |
| Fine global threshold: 0.665 | 0.912367 | 0.904294 | -0.000292 | [-0.002629, +0.002011] |
| Fine global dual: 0.715 / 0.685 | 0.912741 | 0.904599 | +0.000013 | [-0.000993, +0.000909] |
| Country-specific dual thresholds | 0.914114 | **0.906774** | **+0.002189** | **[-0.000155, +0.004699]** |

Country-specific policy: India `0.72 / 0.70`, US `0.585 / 0.585`, global fallback `0.715 / 0.685`. On screening, India is unchanged at **0.875299**; US improves to **0.938250**. Overall pair precision changes from 0.964653 to 0.960685 and pair recall from 0.836403 to 0.847338. Singleton false merges rise from 12 to 13; false-empty predictions on non-singletons fall from 48 to 43.

**Decision:** keep country thresholds as a promising challenger. Do not yet replace B0. Its gain is about **0.219 percentage points**, but the 95% interval includes zero. The interval uses 2,000 country-stratified paired query bootstrap resamples with seed 42; it does not account for repeated experiment selection or all training uncertainty. Confirm the frozen policy on comparison_15k, separately reporting the 13k queries outside screen_2k. Do not retune thresholds using that comparison result. France performance remains unknown.

Artifacts:

- [Machine-readable results and input hashes](B0_decision_audit.json).
- [Calibration-selected policies](../../runs/parallel-v1/d1/decision_audit_v1/calibration_selected_policies.json).
- Per-query scores: `runs/parallel-v1/d1/decision_audit_v1/per_query_scores.parquet`.
- Reproduction: `code/business_entity_resolution/src/er/analyze_b0_decisions.py`; from repository root, set `PYTHONPATH=code/business_entity_resolution/src` and run `python -m er.analyze_b0_decisions` using the project interpreter.

The vectorized evaluator was checked against the shared exact scorer on synthetic cases covering empty truth, no candidates, unretrieved true targets, false positives, partial recall, country overrides and query gates. Targeted tests passed: **4/4**.

## 3. What the errors suggest

Saved predictions reproduce B0's exact score. Counting full truth, including unretrieved matches:

| Query outcome | Queries | Contribution to overall macro loss |
|---|---:|---:|
| Perfect prediction set | 1,095 | 0 |
| False positives only | 111 | 0.0182034 |
| False negatives only | 717 | 0.0621992 |
| Both false positives and false negatives | 77 | 0.0150116 |

These categories are descriptive, not proposed prediction rules. They show that simply raising all thresholds to favor precision is poorly justified. The US calibration result also suggests the common threshold is too restrictive for at least some US cases.

Answer-key diagnostics give further direction: removing every predicted false positive would add approximately **0.0266761**; adding every retrieved-but-rejected true match while retaining existing false positives would add **0.0542381**. These are not deployable methods, not expected gains, and not additive. They suggest that hard-positive discrimination and conditional decisions deserve attention alongside false-positive prevention. Retrieval-missing positives still require better candidates.

## 4. Remaining engineering gaps — bounded repairs, not a restart

1. **Oracle call-site bug:** the current B0 training source still passed `(target_id, score)` tuples to a scorer expecting target IDs. This review fixed the overall and per-country calls to use IDs. Existing saved B0 oracle values were independently confirmed; no retraining was needed for that correction.
2. **Handoff paths are stale:** the original `b0_portable_bundle.joblib` is absent. Current files are `b0_train_features.joblib` and `b0_eval_features.joblib`. The original manifest, report and provenance builder still reference the old combined file. Use the actual split files for D2; update/version producer and manifests before regeneration. Preserve historical provenance instead of pretending the old hash identifies the new files.
3. **D3 training package is incomplete:** the training bundle has X/y/query groups/folds/feature names but no training target IDs. The text exporter collects targets from calibration/evaluation candidates and truth, not a complete keyed training-pair population. Having 19,000 query texts and 589,223 target texts does not provide supervised training pair mappings. Export training `(query_id,target_id,label)` keys and referenced texts, separately from evaluation labels, before neural fine-tuning. Evaluation true targets included for diagnostics must not be added to natural candidates.
4. **Feature improvements remain available:** the 23-feature schema still merges channel scores into `tfidf_max`; it lacks independent channel score/rank/presence columns. Empty fuzzy similarities need actual feature-level guards. Fix these through versioned feature experiments with retraining, rather than silently changing inputs to the saved model.
5. **Reproducibility is better but incomplete:** the manifest records some hashes, but not every channel vectorizer/matrix, structured/duplicate cache, current split bundle and source patch. Freeze the files actually consumed by the next experiment. The progress log's standalone holdout filename does not exist here; locked assignments are in `splits/f05-v1/splits.json`.

Do not rerun multimillion-document indexing just to make documentation look complete. Repair the specific interfaces, retain B0, and start experiments using verified artifacts.

## 5. Revised next experiments, in order of decision value

### Immediate: freeze and confirm the inexpensive challenger

Keep B0 0.70/0.70 as the reference. Evaluate the already selected country policy on the larger comparison population when its natural B0 features are available. Generate comparison features once, cache them with pair keys, and reuse them across D2 candidates. Report full-15k and non-screen-13k results, per-country changes and singleton tradeoffs. A reliable small gain is valuable; no minimum absolute F0.5 target is required.

### Device 1: candidate depth with actual matcher scores

Before a large channel grid, compare these natural candidate policies with consistent training/calibration:

1. US K100 / India K100 reference.
2. US K100 / India K250.
3. US K100 / India untrimmed existing union.
4. Both countries untrimmed control if cost allows.

First, scoring the larger candidate sets with frozen B0 plus separately recalibrated thresholds is a cheap distribution-shift diagnostic. Then retrain the best candidate policy on matching training candidates before promotion. Record all added false positives, recovered positives and per-query changes. Candidate count, not oracle alone, determines cost.

After that, test India address K150→300, word/token-weighted address retrieval and separate Unicode/Latin views, one conceptual change at a time. Advance a route if its final macro gain justifies its cost, even if its oracle remains below former numeric goals. Remove a route only after leave-one-channel-out comparison on the full union; pair count alone is not evidence of usefulness.

### Device 2: discriminating missed positives from hard negatives

Load the actual split feature files, reproduce B0, and use the country policy as a frozen challenger. The old 100-tree defect is already repaired (B0 has 743 trees); prioritize feature and training-distribution changes over repeating a large tree-count grid.

First bounded model screen: existing 63 leaves/min-child 80 versus a smaller 31-leaf regularized arm and a 127-leaf/less restrictive-depth arm, with training-inner early stopping and identical query groups. Then add independent channel features, missingness guards and discriminative token evidence. Use naturally retrieved hard positives and same-name/different-location negatives. Increase distinct training identities 12k→25k only after a clean comparable setup and adequate memory.

Keep standard binary loss as the control. Focal/custom asymmetric objectives are later experiments requiring separate calibration; a precision-weighted metric does not imply a particular class weight will improve macro F0.5. Avoid changing loss, sampling, features and thresholds together without ablations.

### Device 3: complementary neural evidence

While the keyed training export is repaired, benchmark a frozen multilingual encoder on the existing natural calibration/screen pairs. Encoding is label-blind; adaptation must use training labels only. Pass keyed scores to D2 for a controlled feature comparison.

Once training pairs/texts exist, compare a small task-trained pair classifier or bi-encoder against its frozen control. Use the established hard-positive/hard-negative taxonomy. Full-pool dense encoding comes after evidence that representation misses justify its cost; do not encode every target merely because a GPU is available. The RTX 3050/16 GB resource limits and small-batch fallbacks in the device plan still apply.

## 6. New selection rule

The objective is **the best reproducible end-to-end macro F0.5 under feasible runtime/memory**, with uncertainty and country behavior visible. An experiment is valuable if it produces a stable positive paired delta or explains an important failure mode cheaply. Negative experiments should be recorded and stopped, not dressed up as milestones.

Freeze thresholds on calibration. Screen cheaply on 2k, confirm finalists on larger development, and use locked holdout only after selection. Small uncertain gains remain candidates; stable gains need not reach any arbitrary 0.98, 0.995 oracle, or 0.0005 delta floor. Prefer the cheaper/simpler solution when quality differences remain unresolved. Preserve legitimate multiple matches and complete query coverage. Never choose a model because its report has the largest number on a different population.

The current best **established reference** remains **0.904586**. The current best **new screening candidate in this review** is **0.906774**, pending confirmation. No claim of private-test performance is made.
