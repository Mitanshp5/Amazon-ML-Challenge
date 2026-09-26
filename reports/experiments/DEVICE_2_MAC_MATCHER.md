# Device 2 — MacBook Air M4, 16 GB unified memory

> **RETIRED device assignment:** this workload now runs on the current Windows device under L01/L02/L04 of the [active local-first plan](../../LOCAL_COLAB_IMPLEMENTATION_PLAN.md). The experiment ideas below are historical references. No Mac execution or transfer is required. The keyed training-pair export is now complete.

> **Current priority update:** maximize verified F0.5. B0 has 743 trees, so prioritize feature/distribution changes after a bounded capacity screen, rather than treating the old 100-tree cap as unresolved. The actual handoff files are `b0_train_features.joblib` and `b0_eval_features.joblib` under `runs/parallel-v1/d1/b0_baseline/`; the old combined file is absent. The training bundle supports existing-feature tree fits but lacks target IDs for training-pair reconstruction. See the [latest review](../dev_probe/F05_MAXIMIZATION_REVIEW.md) for the new country-threshold challenger and larger-set confirmation protocol.

**User-confirmed hardware:** 10-core CPU, 10-core GPU, 16 GB unified memory. **Role:** controlled matcher, feature and calibration experiments. Follow the [shared protocol](THREE_DEVICE_PROTOCOL.md). This is a plan, not evidence that these experiments have run.

## Why this workload belongs here

Device 1 can build candidates once and export compact numeric features; this Mac can explore model/decision quality without rebuilding million-record retrieval indexes. CPU LightGBM is the primary path. The M4 GPU is not assumed to accelerate LightGBM or to run the Windows OpenVINO export. Neural GPU work primarily belongs on Device 3.

Use an isolated native Apple Silicon environment and pin tested versions. Start with 4–6 CPU threads for a single fit, measure sustained throughput and memory, then adjust within the 10-core CPU budget. CPU cores are heterogeneous, and 10 threads are not guaranteed to be the best sustained setting. Avoid simultaneous model fits and memory-heavy feature construction. Leave room for the OS; initially keep process RSS around 8–10 GB or below, watching swap/memory pressure. Free disk must be checked locally.

## Required inputs from Device 1

Receive contract-v1, `train_12k`/calibration/comparison manifests, B0 natural candidate target IDs, source-normalized field tables, channel provenance, feature schema, float32 feature shards, training labels, baseline model parameters and per-query baseline scores. Verify hashes and row-to-pair mappings before fitting.

The input must exclude synthetic `rrf=0.001` positives from all evaluation populations. Do not use the old 1.5M-pair joblib cache as the clean bundle. Training labels are separate from evaluator truth files. Do not rebuild splits from file order. If a variant needs a new feature, materialize it as a keyed sidecar and send its implementation/schema back to D1.

## D2-00 — portability and memory baseline

Fit the exact B0 configuration on the common train_12k set. Use the same fixed seeds, folds and threshold protocol. Compare pair IDs, candidate counts, feature values on a deterministic sample and final metrics; minor floating-point differences may occur across platforms, but investigate material prediction changes before selecting models.

Measure read time, matrix conversion, LightGBM dataset construction, fit peak memory and scoring time. With `P` pairs and `F` float32 features, the raw matrix alone uses `4PF` bytes. For 1.5M × 40 features this is about 240 MB, but IDs, Python objects, folds, histograms and temporary copies can multiply memory use. Stream IDs and avoid lists of millions of dictionaries. Keep only needed folds/arrays resident.

**Gate:** reproduce the clean baseline within a predeclared numerical tolerance and finish without sustained swapping. If memory fails, reduce **training identities** or train-shard construction size; do not shrink the comparison truth/candidate universe silently.

## D2-01 — meaningful boosting-depth/regularization curve

The old OOF helper implicitly stops at 100 rounds. Test this hypothesis on the repaired features and fixed candidates before attributing low performance to insufficient model architecture.

| Stage | Arms | Controls |
|---|---|---|
| Tree-count diagnostic | Fixed 100 / 500 / 1,000 rounds | Same leaves, learning rate 0.05, features, identity set and seed |
| Capacity screen | Leaves 31 / 63 / 127 | Explicit max 2,000 rounds, patience 100 on training-inner validation |
| Regularization screen | `min_data_in_leaf` 20 / 100 / 300 for the best capacity | Fix other parameters; identify rare-pattern overfitting |
| Refined finalist | Learning rate 0.03 versus 0.05 | Increase maximum rounds only if needed; record actual learning curves |

Run these as stages, **not the full Cartesian product**. Binary log loss is a fitting diagnostic; select the deployed policy by exact macro on the permitted development roles. Comparison labels may not choose early-stopping iterations. Pass actual feature names to every fold model.

**Deliverables:** score-versus-iteration and memory/time tables, model artifacts, calibration policy and paired comparison deltas. Stop expanding this search if two staged changes produce no credible improvement; proceed to error-driven features.

## D2-02 — repaired and richer pair features

Use a fixed training configuration and common candidates. Compare the following conceptual groups individually before combining winners:

1. **Field correctness:** full suffix-reduced name core, parsed house/unit/postal evidence, explicit missingness and guarded empty-string similarities. The training adapter must actually supply the fields. This repair may already be in B0; if so treat it as mandatory, not a new experimental gain.
2. **Channel provenance:** separate joint/name/address/structured scores, ranks and presence, plus RRF and channel count. Do not mix incomparable score scales into one maximum.
3. **Discriminative token evidence:** training-corpus-fitted rare-name and rare-address overlaps, unmatched discriminative tokens, numeric agreement/conflict by field. Keep generic names and same-address/different-business negatives in evaluation slices.
4. **Query-level context:** best-versus-second score margin, candidate score distribution, competing targets, missing-field flags and source-specific support. Compute these using natural candidate lists, with no label lookups. Avoid fixed candidate-rank acceptance rules that hide missed aliases.
5. **Representation parity:** primary Unicode versus an additional accent-folded view; measure short-name and non-Latin slice behavior separately.

Do not repeatedly handcraft rules from comparison mistakes. Use training error analysis to design variants, freeze them, then compare. A feature's gain importance is a diagnostic; retain it for measured prediction improvement or necessary correctness, not high importance alone.

## D2-03 — training identity scaling and hard negatives

Train the best simple configuration on nested **12k → 25k → 50k total identities**, increasing only while memory and gain justify it. Millions of pairs from a few repeated identities are not equivalent to more supervised identity diversity.

Compare full natural negatives against a training-only negative sampling policy: preserve all natural positives, keep highest-scoring hard negatives, and sample additional negatives with recorded inclusion rates. Keep all evaluation candidates unsampled. Sampling changes score calibration, so recalibrate on the full natural calibration population; log weights/probabilities and compare against a feasible full-negative control.

Mine difficult same-name/different-address, same-address/different-name and numeric-conflict pairs using training-only out-of-fold models. Exclude other known aliases of the query from the negative pool. Never use comparison/holdout positives for mining. Do not create positives by changing business numbers or inventing unsupported identity labels.

**Decision:** scale only if the learning curve shows gain beyond uncertainty. If more training does not help, identify the dominant error class rather than requesting a bigger run automatically. Memory fallback is the largest successful earlier identity set, clearly labeled.

## D2-04 — calibration and decisions

Reuse saved probabilities; no refit is required for each threshold point. Compare:

- One global threshold, selected on calibration IDs.
- The existing query-max gate plus pair threshold, with both thresholds selected on calibration IDs. This is an operational empty-set decision rule, not knowledge that a query is a true singleton.
- A fitted probability calibration method (e.g. logistic calibration), using grouped or reserved calibration data; compare against uncalibrated output. Avoid flexible calibration when slice counts are too small.
- A learned no-match gate trained from training identities using candidate-summary features; calibrate separately and evaluate both singleton false merges and false-empty predictions for genuine matches.
- Country/source-specific thresholds only if counts and repeated comparisons support them; retain a global fallback for France. Do not tune a France threshold without France labels.

Report threshold surfaces, not only the single maximum. A one-query improvement on 600 examples was enough to explain old E08's entire gain; require larger-set confirmation. Do not sacrifice non-singleton recall to improve singleton statistics alone.

## D2-05 — ensembles and ownership (lower priority)

After distinct clean models exist, test calibrated score averages or a small stacking model trained on training-only OOF predictions. Align `(query_id,target_id)` keys exactly; models scoring different candidate sets require explicit treatment of missing scores and a shared union.

Test target-ownership resolution only as an ablation against independent decisions. It must not force one target per S1. The old measured gains were tiny, so this is lower priority than D2-01–04. Inspect false reassignment and recall loss, not only removed duplicate claims.

## What can run concurrently and what to return

While D1 builds retrieval and D3 encodes/trains neural models, run one substantial Mac fit at a time. Threshold analysis on a completed score file may overlap if measured RAM/CPU headroom allows. Do not start an entire parameter grid as parallel processes on 16 GB unified memory.

Return the exact winning configuration, feature code/schema, model, calibration, keyed scores, per-query differences, bootstrap interval, country/singleton diagnostics, runtime and peak memory. Also return negative results so other devices avoid repeating them. After D1 changes retrieval, rerun the winner on those candidates before integration; a matcher winner on old candidates is not automatically the best final pipeline.

**Recommended order:** D2-00 → D2-01 → D2-02 → D2-03 → D2-04 → optional D2-05. Do not spend the whole budget on tree grids while missingness/feature or candidate defects remain.
