# Local-first plan to maximize verified macro F0.5

**Updated 26 September 2026, following the user's hardware change.** All required work belongs on the current Windows/Intel Arc device. Google Colab A100 is an optional parallel execution environment for expensive neural experiments. The Mac and RTX 3050 assignments are retired. The objective remains the highest reliable end-to-end F0.5 achievable at practical cost, with no fixed 0.98 gate.

This is the authoritative execution plan. The older `F05_098_IMPLEMENTATION_PLAN.md` retains the challenge contract, original audit and technical references. The separate [Colab A100 runbook](reports/experiments/COLAB_A100_RUNBOOK.md) defines the optional GPU workload and exchange format. No cloud job, upload or training was started by this plan revision.

## 1. Progress verified at this revision

Inspected current reports, source, Git history and local artifacts. Latest recorded commit is `10132b1`, which adds keyed training pairs and partitioned text records. An existing working-tree edit removes `load_record_text_provenance` from `io.py`; it is preserved. No new quality report after the threshold audit was found.

| Item | Current evidence | Status |
|---|---|---|
| Clean B0 model | 12k training identities, 1.2M natural pairs, 743 trees; 0.9045857751 on screen_2k | Established screening reference |
| Country-threshold challenger | 0.9067743787; paired delta +0.0021886; 95% interval [-0.0001553, +0.0046990] | Promising, not confirmed on larger development |
| Candidate depth | Same 2k query IDs: K100 oracle 0.9851593; untrimmed 0.9904524 | Retrieval-only evidence; actual untrimmed matcher score unknown |
| India | Actual B0 0.875299; oracle K100 0.974536, untrimmed 0.983216 | Main country-specific retrieval/matching opportunity |
| US | Actual B0 0.933873; oracle K100 0.995783, untrimmed 0.997688 | Matcher/decision improvements have priority |
| Keyed training export | 1,200,000 rows, zero duplicate query/target pairs; row order, labels, folds and RRF agree with feature arrays | Previous neural-training mapping blocker resolved |
| Text coverage | 931,658 training targets; no missing query/target text for keyed training pairs | Ready for a compact neural data export |
| Artifact provenance | All seven files in `manifest.json`'s bundle hash map match their SHA-256 values | Existing bundle verified; source/cache provenance still needs completion |
| Larger comparison / holdout / production | No new result establishing these stages | Pending |

The partitioned text files each contain 19,000 query records. Select supervised training queries using `train_12k.json`, not every query present in a text file. `eval_text_records.joblib` contains **588,371 natural evaluation candidate targets** and **852 diagnostic unretrieved targets** in separate maps. The diagnostic map must not be used to add evaluation candidates.

The best established reference remains B0. Infrastructure completion is valuable, but does not itself increase measured F0.5. On B0, about 84.4% of remaining macro loss is downstream of retrieval; 717/2,000 screening queries have false negatives without false positives. These findings justify matcher experiments and selective India candidate expansion together.

## 2. Execution ownership

| Work | Current device | Optional Colab A100 |
|---|---|---|
| Split/exposure manifests, exact scorer, report generation | Own and run | Consume pinned versions |
| Lexical/structured retrieval, candidate-depth experiments | Run locally; cache full-pool results | No duplicate sparse indexing by default |
| Feature extraction, CPU LightGBM, calibration, ensembles | Run locally | Return keyed neural scores/features |
| Frozen multilingual embedding inference | Small Arc/CPU smoke if useful | Bulk encoding when local cost warrants it |
| ER encoder fine-tuning / pair reranker training | Prepare data and evaluate returned results | Primary optional GPU workloads |
| Dense full-pool encoding/search | Own IDs, manifests and integration | Conditional accelerator after complementary recovery is demonstrated |
| Final selection, holdout, export, validator/package | Own and run | Optional inference service for a frozen retained model, if needed |

Local work must remain useful if Colab is unavailable. Do not make the tree baseline, larger-set validation or final package depend on a successful neural experiment.

## 3. Local scheduling and resource policy

Keep the existing **12-thread local ceiling**, counting nested BLAS/OpenMP/LightGBM workers. A thread count is not proof of CPU affinity; do not claim exclusion of particular CPU cores without measuring affinity. Run one substantial memory-heavy job at a time. In particular, avoid simultaneous full-pool sparse indexing, candidate-dictionary construction and a large LightGBM fit.

Measure available physical RAM, peak process RSS and free disk before expanding candidates. Do not infer local RAM from the Arc report's “16GB” string. Stream candidate/features by country and query shard, use compact IDs and float32 arrays, and free one country's indexes before loading the next if memory requires it. A chunk of all Python record dictionaries can cost much more than the numeric feature matrix.

Colab may run a GPU job concurrently while this device runs a retrieval or matcher job. Reports, checksum verification and small threshold analyses can overlap locally when actual memory headroom permits. There is no need for distributed training or a live remote-control service.

Use the existing B0 bundle read-only. New runs go under `runs/local-v2/<experiment>/<run-id>/`; incoming GPU artifacts go under `runs/colab-v1/<experiment>/<run-id>/`. Those are proposed output locations. Keep model, threshold and data manifests together; never overwrite B0 because a challenger finishes first.

## 4. Fixed evaluation protocol

Retain existing `splits/f05-v1/parallel-v1/` roles: train_12k, calibration_5k, screen_2k and comparison_15k, plus inner training folds. The screen is a subset of the 15k comparison set. Report results on both the full 15k and the **13k outside screen_2k** when checking a screened challenger. Those 13k are development and become exposed once evaluated; they are not the locked holdout.

Choose stopping iterations and supervised model parameters through training-inner folds. Select thresholds/calibration using calibration_5k. Rank candidates using paired comparison scores and uncertainty. All evaluation candidates must be generated without labels; retain complete truth for false-negative accounting, including unretrieved true targets and queries with no candidates.

The locked holdout lives in `splits/f05-v1/splits.json`. Do not evaluate it repeatedly during local or cloud searches. Evaluate the selected frozen pipeline once, report uncertainty and limitations, then use the selected deployment procedure. France has no labeled training evaluation: retain a global fallback and treat cross-country tests as stress tests, not France accuracy estimates.

Use exact macro F0.5 and paired query-level bootstrap intervals. Record actual score, oracle, their gap, pair precision/recall, singleton false merges, false-empty predictions, match-count/country/missingness slices and runtime. A small, replicated gain can be retained; there is no absolute score or minimum-delta floor. Uncertain differences remain provisional; favor lower cost if quality remains unresolved.

## 5. Ordered local experiment queue

### L00 — stabilize current artifact readers and experiment outputs

This is a bounded compatibility pass, not a baseline rebuild.

1. Use `er.io.load_b0_bundle` for the split train/eval feature files. Current B0 data hashes already match; do not regenerate them merely because old filenames appear in a report.
2. Make text consumers accept `train_text_records.joblib` and `eval_text_records.joblib`. `analyze_b0_decisions.py` still opens the absent monolithic `record_text_provenance.joblib`; current provenance-generation source still writes a monolithic output while the saved manifest describes split outputs. Align producer/consumer schemas and explicitly distinguish diagnostic targets.
3. Preserve the in-progress `io.py` change and reconcile its intended loader interface during implementation. Do not restore removed code blindly.
4. Add configurable candidate policy, run directory and evaluation role to the reusable runner. `train_clean_b0.py` currently hard-codes K100 and writes shared model/report paths; it is not yet a safe sweep runner.
5. Persist pair keys, per-query metrics, ordered candidate hashes, feature schema, source revision/patch hash, environment and all retrieval-component hashes. Save a small parity fixture so local/cloud evaluation uses identical semantics.

**Done when:** existing B0 scores reproduce through current split-file interfaces, run outputs are isolated, and a challenger can run without modifying the baseline artifacts. No full-pool retraining is required just to meet this gate.

### L01 — inexpensive local matcher screen on cached B0 features

Absorb the former Mac workload here. Reuse the 1.2M-pair feature matrix and fixed folds; avoid retrieval while testing model capacity.

- Control: the exact B0 parameters and calibrated 0.70/0.70 policy.
- Compact arm: 31 leaves, depth 7, minimum leaf data 100.
- Capacity arm: 127 leaves with depth 9, minimum leaf data 80.
- Regularization arm only if useful: the better capacity with minimum leaf data 200.

Use explicit maximum rounds (initially 2,000), patience 100, fixed seeds and training-inner validation. B0 already has 743 trees; the old 100-tree omission is not the current bottleneck. Calibrate every model independently on calibration_5k, and include the fixed country-policy comparison. Keep standard binary loss as the initial control; do not start with a large custom-loss grid.

**Done when:** a small table identifies whether capacity changes help and which model(s) deserve larger evaluation. Save negative results. Stop expanding the grid if effects are negligible or unstable.

### L02 — build comparison features once and confirm candidates

Generate natural K100 candidates and features for the 13k comparison IDs not yet in the screening cache. Reuse the existing 2k screen artifacts only if normalization/model/candidate hashes match. If reconstructing all 15k is simpler, compare the overlapping 2k hashes as a consistency check.

Score B0, the fixed country-threshold challenger and the best L01 model(s). Never choose new thresholds from these labels. The country policy currently uses India 0.72/0.70, US 0.585/0.585 and global fallback 0.715/0.685. Its current screening gain is entirely from US; confirm that behavior and any singleton tradeoff on the larger population.

**Done when:** retain or reject each frozen contender using full-15k and non-screen-13k paired metrics. The larger comparison cache becomes reusable infrastructure for later local and neural challengers.

### L03 — candidate depth with actual scores

Retrieve the complete natural union once per needed role/country and retain channel provenance; derive nested K policies from the same ordered candidates rather than repeating all channel searches per K.

| Arm | US policy | India policy | Purpose |
|---|---|---|---|
| C0 | K100 | K100 | B0 retrieval reference |
| C1 | K100 | K250 | Moderate India recovery/cost |
| C2 | K100 | Existing untrimmed union | India recall control |
| C3, conditional | Untrimmed | Untrimmed | Test whether US tail recovery warrants its cost |

First score candidate arms with a frozen model and separately calibrated policies as a cheap distribution-shift diagnostic. Then train on matching natural training candidates for the best candidate arm(s) before promotion. Report actual macro and new false positives, not just oracle. Untrimmed means the union of the specified per-channel retrieval lists; it is not exhaustive matching against every target.

**Done when:** choose a final-score/cost operating point. More candidates are retained only when measured quality or useful complementary coverage warrants them.

### L04 — richer features, hard examples and identity scaling

Use the selected candidate policy and add one conceptual group at a time:

1. Separate score/rank/presence for joint, name, address and structured channels; replace the misleading mixed `tfidf_max` behavior with versioned columns.
2. Verify guarded empty-field similarities and parser-to-feature unit/house/postal evidence; add missingness/conflict flags and rare-token overlap.
3. Add candidate competition/margin context without answer-key information. Diagnose same-name/different-location and same-address/different-business errors.
4. Mine training-only hard negatives and difficult positives; exclude all known aliases from negatives. Keep natural evaluation candidate lists unsampled.
5. Create nested train_25k and, only if useful, train_50k manifests from the supervised training partition. Increase unique identity diversity, not just pairs per old identity. Recalibrate after any training-negative sampling.

**Done when:** a feature/identity learning curve supports the added complexity. No meaningful improvement after two bounded expansions means return to the error taxonomy rather than automatically adding another model family.

### L05 — targeted retrieval representations

Proceed using the current India misses and per-channel recovery evidence. Test address K150→300, word/token-weighted address retrieval, and an additional Unicode/Latin representation separately. Reuse compatible indexes. Measure leave-one-channel-out effects on the full union before removing routes.

Use global score consistency when sharding a full target pool. Smaller pools are engineering smokes only. Changes to dictionaries/normalization derived from labels use training data, never comparison/holdout answers. Retain a channel because it improves the final model or provides independently useful recovery at acceptable cost, not because it reaches an old numeric oracle target.

### L06 — integrate optional GPU results

Accept a returned bundle only after ID/schema/source/hash checks. Frozen encoder scores can join the local pair matrix directly. For a supervised neural model used as a learned feature/stacking input, obtain training-only OOF neural predictions or use a separate training partition for the combiner; do not stack in-sample neural predictions against clean evaluation outputs.

Calibrate locally. Compare the tree winner, neural-only decisions and a simple calibrated fusion, then rerun the selected combination on the larger development set. A neural score trained on old candidates must be rescored/retrained as needed for a changed union. Do not add standalone gains arithmetically.

### L07 — locked finalist and complete production path

Freeze the model(s), feature and candidate schemas, thresholds, routing and country fallbacks. Evaluate the locked holdout once. Test France execution, stream all required S1 queries, export exact scored candidates and matching subsets, and run the supplied validation/package procedure in the original audit plan.

If a retained neural component cannot run locally within the final runtime budget, either obtain its frozen bulk predictions in Colab with complete manifests and reproducible local score ingestion, or select the strongest feasible local-only finalist. Keep enough local capability to reproduce the non-neural baseline and validate all outputs even if a Colab session ends.

## 6. Parallel GPU queue

The [Colab runbook](reports/experiments/COLAB_A100_RUNBOOK.md) provides G00–G04 details. Run one bounded cloud experiment at a time:

1. G00: verify/import the current training/evaluation bundle and benchmark the actual GPU.
2. G01: frozen compact multilingual encoder features on existing natural pairs; return keyed scores for local training.
3. G02: one task-trained pair model or bi-encoder chosen from the error analysis; use training-only supervision and inner validation.
4. G03: full-pool dense retrieval only if expected unique recovery justifies encoding millions of records.
5. G04: routed reranking/OOF feature export only if the simpler neural route improves actual macro.

Local L01/L02 work can overlap G00/G01. Local L03/L04 can overlap G02. Freeze artifact versions for each GPU job; an ongoing local schema experiment must not silently change the cloud job's inputs. If no A100 is available, continue L00–L05 and postpone GPU work.

## 7. Runtime planning — measured facts and limits

Measured local B0: **48.23 minutes total**, including **41.04 minutes retrieval**, approximately **1.01 minutes feature extraction**, **37.26 seconds grouped OOF training**, and **18.71 seconds calibration/evaluation**. These are one observed workload using cached retrieval artifacts and 19k total queries, not cold-cache timings for every future pipeline. Refitting, loading/export and other overhead are included in the total but not fully split in the old timing table.

The previous chat's Mac/RTX and A100-like speed estimates were unbenchmarked planning guesses; they are not promises and do not apply to this revised workflow. In particular, 300-query retrieval runs included setup/fitting, so do not scale their total runtime linearly by query count. No A100 throughput has been measured for this project.

For planning, allocate one local B0-sized pass at roughly its observed 48-minute scale, then replace that estimate using the first new run's stage timings. Cache reuse makes L01 much cheaper than repeating that entire pass for each tree configuration. A proposed six-fit screen would spend about 3.7 minutes in OOF fitting **if each exactly matched B0's 37.26 seconds**; deeper models, more pairs, refit, calibration and I/O add cost. This is a calculation under stated assumptions, not a benchmark.

Before a long GPU run, measure 10k representative encodings and 200 training steps after warmup. Estimate `encoding_time = record_count / measured_records_per_second` and `training_time = total_optimizer_steps * measured_seconds_per_step`, with validation/checkpoint/I/O overhead added separately. For pair reranking use routed pair count and measured pairs/second. Check host memory and disk as well as GPU memory.

Checkpoint at every completed stage and preserve intermediate candidate/features. The total experiment queue is adaptive: stop weak arms and scale promising ones. There is no defensible fixed end date for every optional arm before these benchmarks and decisions.

## 8. Immediate next session

1. Complete L00 split-reader/run-output compatibility, using the already verified bundle.
2. Run L01's bounded cached-feature screen locally.
3. Build L02's remaining comparison features once; confirm the country policy and best local model.
4. In parallel, if an A100 session is available, run G00/G01 on the frozen B0 bundle.
5. Proceed to L03 candidate-depth training and L04 features based on those results; report the best verified model after each stage.

This sequence moves directly into quality experiments without repeating the original audits or making local progress wait for cloud availability.
