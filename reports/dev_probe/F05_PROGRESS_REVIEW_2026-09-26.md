# Progress review: is >0.98 supported by the current results?

**Review date:** 26 September 2026. **Decision: no, the current evidence does not establish a trajectory to >0.98.** The target is still a research objective; it is neither proven impossible nor close to demonstrated. The immediate priority is to repair evaluation, then improve retrieval and matching in parallel on three devices.

This review supersedes promotion claims in E06–E09 and the completion labels in `progress.md`. Original reports remain unchanged as historical records. No model was trained, no holdout was evaluated, and no submission was generated during this review.

## 1. Evidence reviewed

All existing files under `reports/` at review time: the two smoke JSONs, `GATE_A_E00_E02_smoke.md`, the US/India E02/E03 MD/JSON pairs, E06 MD/JSON, and E07/E08/E09 MD/JSON. These were checked against the current retrieval, feature, matcher, calibration and training source; three saved feature caches; the existing implementation plan; and the progress log. Graphify was used to navigate the report/code relationships; direct source and artifact inspection determined the findings.

| Evidence | What it actually establishes | Decision |
|---|---|---|
| E00/Gate A | Historical frozen oracle 0.9793097093 on an exposed 20k sample; older model config reports 0.8886 | Historical reference, not the current pipeline's score |
| US/India smoke | 200 queries each against a capped 200k pool; pipeline executes | Engineering smoke only |
| E02/E03 US | 300 queries, full 6,186,873-target pool; union oracle **0.9984713**, pair recall **99.5279%** | Encouraging small-probe retrieval result |
| E02/E03 India | 300 queries, full 4,133,346-target pool; union oracle **0.9858318**, pair recall **96.5193%** | Clear improvement within this probe; inadequate safety margin |
| E06 India | K100 oracle **0.9811691**, losing 17 positives versus untrimmed | Do not promote K100 as the default |
| E07 scaled | Reported macro **0.9050761** on 3,000 validation queries | Label-aware candidate population invalidates end-to-end interpretation |
| E08 | 600-query calibration: **0.9207743 → 0.9224410** | Earlier, smaller, contaminated setup; not progress beyond scaled E07 |
| E09 | Reported macro **0.9037012**, below E07 | No promotion; same evaluation defect, plus a regression |

The small retrieval probes are not the entire fixed 20k probe. They use source-file order after membership filtering, not a newly randomized representative sample. The historical 20k, new 300-query retrieval slices, 600-query calibration slice and 3,000-query matcher slice must not be connected into a score-versus-time curve. All latest experiment timestamps are from the same day. There is no defensible estimate of days remaining until 0.98.

## 2. Critical finding: ground truth enters validation candidates

In `code/business_entity_resolution/src/er/training/train_matcher.py`, lines 219–248, the code takes natural top-100 retrieval candidates and appends missing true targets with RRF score `0.001`. This happens for both training and validation queries. `train_dense_matcher.py`, lines 294–333, does the same.

This changes the evaluation input using the answer key. GroupKFold does not correct it: an out-of-fold classifier still sees label-conditioned candidate lists and synthetic retrieval features. The special RRF value, potentially absent channel evidence, and guaranteed positivity also create a shortcut the model can learn. Training-only positive augmentation can be a deliberate experiment; it must never enter validation, calibration, holdout or test candidate generation.

### Read-only cache verification performed in this review

Loaded the local five-element feature-cache tuples `(X, y, groups, validation_candidates, validation_truth)`. Feature index 22 is `rrf_best` in the inspected schema. Counted rows with `abs(rrf_best - 0.001) <= 1e-8`; every such row was positive. Recomputed natural-candidate oracle by removing those rows, intersecting remaining target IDs with truth, and averaging `5h / (g + 4h)`, with empty truth scoring 1 for the oracle.

| Cache under `cache/features/` | Groups / validation queries | Synthetic positives, all / validation | Validation queries affected | Natural top-100 oracle | Augmented oracle |
|---|---:|---:|---:|---:|---:|
| `train_pairs_1500_300.joblib` | 3,600 / 600 | 385 / 62 | 53 | 0.9895353752 | 1.0000000000 |
| `train_pairs_6000_1500.joblib` | 15,000 / 3,000 | 1,632 / 283 | 229 | 0.9909794621 | 1.0000000000 |
| `train_pairs_dense_6000_1500.joblib` | 15,000 / 3,000 | 1,632 / 283 | 229 | 0.9909794621 | 1.0000000000 |

The scaled lexical and dense caches have the same validation query IDs, candidate target sets and ground-truth sets; ordering differs. Their natural pair recall is **97.3191%**. The smaller cache's natural pair recall is **97.0207%**. These are **cache-reconstructed diagnostics**, not a fresh retrieval rerun or a corrected model score. They do not validate cache provenance, representativeness or generalization. Simply deleting synthetic pairs from existing predictions would also leave a model trained on the shortcut; clean retraining is required.

The 0.99098 combined oracle does not contradict India's 0.98117 K100 oracle: they concern different query samples. It does show that a substantial matcher/decision gap remains on the cached population even before generalization questions are addressed.

## 3. Other blockers found in the implementation

1. **Normalization improvements are bypassed by training adapters.** Both training scripts implement `extract_record_dict`: name core is only the first token, house tokens come from a broad digit regex, and unit tokens are always empty. The shared normalizer's richer components therefore do not reach those matcher features. Zero unit-feature importance is not evidence that units are useless.
2. **Mixed score semantics remain downstream.** `features.py` calls `max()` across channel scores for `tfidf_max`, including structured-channel scores. Upstream channel provenance is helpful, but the matcher still needs separate channel score/rank/presence columns. Several raw fuzzy functions receive empty strings without explicit guards; test their actual behavior and encode missingness consistently.
3. **Boosting is capped by omission.** `training/matcher.py` does not set `num_boost_round` in OOF training and uses early-stopping patience 200. Both reports stop at 100 iterations. Set an explicit limit and a meaningful stopping protocol; more trees are an experiment, not a promised fix.
4. **OOF schema and report reproducibility need repair.** OOF datasets are built from arrays without explicit feature names, while callers assert semantic feature names. The current E07 source's policy grid does not reproduce the report's full dual-threshold/resolved output. `run_calibration_sweep.py` is zero bytes. The saved reports cannot simply be assumed to come from the current tree.
5. **Cache keys are insufficient.** Pair caches use query counts; retrieval caches use country/mode; embedding caches use country/query count. They do not identify input hashes, exact IDs/order, source revision, feature schema and model revision. Preserve existing files but create a new versioned namespace and rebuild.
6. **Development is not a globally excluded validation set.** Training and nominal development identities enter the same grouped OOF population, and final refitting uses all pairs. Grouped OOF can support development, but these results are not an independent untouched holdout. Explicitly separate fit, calibration, comparison and locked evaluation roles.
7. **E09 is not the planned multilingual retrieval or cross-encoder experiment.** The default checkpoint is local `all-MiniLM-L6-v2_openvino`; the experiment adds cosine similarity to existing candidate pairs. It does not add unseen target candidates, train an ER encoder or implement the original plan's routed cross-encoder. Hardware execution is an engineering milestone, not a quality gate. Its source also changes `min_child_samples` versus E07, so the comparison is not a pure dense-feature ablation.
8. **Runtime metadata is incomplete.** Reported 10–12-second training runs do not establish cold-cache end-to-end runtime for all preprocessing, encoding, indexing, fitting and inference. Device/GPU labels are partly hard-coded. Measure actual device selection, cache hits, elapsed stages and peak memory.

## 4. Correct interpretation of the gains and remaining margin

### Retrieval

On the same India 300-query probe, joint retrieval to all lexical channels improves oracle from **0.9351197 to 0.9851366**. Adding structured retrieval contributes another **0.0006952**, recovering three additional positives. Duplicate expansion adds 14 candidates and **zero additional positives** on this probe. In US, duplicate expansion adds 43 candidates and zero additional positives. Candidate additions are not recovered matches.

Name-only added to joint has no oracle gain on these small probes; full address provides the major measured benefit. Do not remove name retrieval from the final system on this evidence alone: measure leave-one-channel-out losses from the complete union on a larger shared sample.

The E06 sentence claiming K100 oracle loss `<0.003` is wrong. Actual loss is **0.9858318029 - 0.9811690804 = 0.0046627225**. K100 leaves only **0.0011691** matcher-loss allowance to reach 0.98 on that India probe. Untrimmed leaves **0.0058318**. K200 reaches 0.9852418 and loses three positives; it is a useful experiment, not an already approved production choice. The original 40–60 candidate ambition is contradicted by this India frontier.

### Matching and calibration

E09 minus E07 is **-0.0013748791**, or about **-0.1375 percentage points**. E09's feature importance ranking cannot overrule that outcome. Even the contaminated reported E07 score is **0.0749239** below 0.98; closing this would require eliminating about **78.9% of its remaining loss**, if this were a valid comparable starting point. It is an illustration of scale, not a forecast.

E08's improvement is exactly **1/600 = 0.0016667**, with singleton false merges reduced from four to three and unchanged reported pair precision/recall. It predates the scaled E07 run. For scaled E07, all 27 singleton false merges contribute 27/3000 = **0.009** macro loss; even fixing all of them without side effects would only bring the reported score to about **0.9141**. Threshold tuning alone cannot plausibly account for the whole gap in these artifacts.

E07's target-conflict step adds only **0.0000625**, and E09's adds **0.00001664**. Keep this as a later ablation. Preserve multiple legitimate targets per S1; any ownership constraint must be justified and measured, not confused with forcing one match per query.

### France and uncertainty

France has no labeled training evaluation in these reports. An overall private-test claim cannot be inferred from US/India alone. Report country results separately and use cross-country transfer as a stress test, not a France score. The previously audited test query proportions (US about 38.27%, India 46.75%, France 14.98%) help prioritize effort, but cannot supply the missing France accuracy or hidden leaderboard composition.

## 5. Revised gates and experiment order

| Gate | Required evidence | What happens if it fails |
|---|---|---|
| R0: evaluation integrity | Label-blind natural candidates; independent query roles; schema/cache identity; exact metric and all-query accounting checks | No score promoted; repair first |
| R1: clean baseline | Same frozen candidate and query manifests; per-query predictions; threshold chosen on separate calibration IDs; natural oracle and actual score together | Diagnose reproducibility/feature defects before larger training |
| R2: retrieval headroom | Full-pool common probe; aim oracle ≥0.995 and pair recall ≥99% per labeled country; minimum research floor >0.99 | Prioritize misses, especially India; report frontier honestly |
| R3: matcher progress | Paired macro gains on fixed comparison queries, confidence intervals, slice/cost stability; clean 0.95/0.97 checkpoints before 0.98 claims | Error taxonomy and training-size/feature curves; avoid blind architecture expansion |
| R4: combined pipeline | Best retrieval and matcher recombined and retrained; natural candidates, calibration and routing frozen | Do not combine individually best scores arithmetically |
| R5: target evidence | Independent locked evaluation >0.98, uncertainty reported; preferably lower 95% bound >0.98 on labeled distribution | State target unproven; preserve best verified model |
| R6: deployment | France path, all test IDs, candidate-to-score parity, runtime and supplied validator verified | No claim of submission readiness |

The ≥0.995 oracle target allocates roughly 0.005 retrieval loss and up to 0.015 downstream loss at a final 0.98. On identical query sets, report the exact decomposition `1 - final = (1 - oracle) + (oracle - final)`. These are planning budgets, not guaranteed attainable metrics. Per-country targets are robustness objectives, not additional challenge scoring rules.

## 6. Three-device execution

- [Shared comparison and artifact protocol](../experiments/THREE_DEVICE_PROTOCOL.md)
- [Device 1: this device, integrity, baseline and India retrieval](../experiments/DEVICE_1_LOCAL_BASELINE.md)
- [Device 2: M4 Air, controlled matcher and calibration experiments](../experiments/DEVICE_2_MAC_MATCHER.md)
- [Device 3: i5 HX / RTX 3050, multilingual and neural challengers](../experiments/DEVICE_3_CHALLENGERS.md)

All three can implement and prepare their independent work immediately. Expensive scored runs depend on the common repaired evaluation contract. Neither adding two devices nor running more experiments makes contaminated scores meaningful; the shared protocol is what makes parallelism useful.
