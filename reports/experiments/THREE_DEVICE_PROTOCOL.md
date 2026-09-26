# Three-device experiments: shared protocol and selection rules

> **Objective update:** maximize verified end-to-end F0.5; no fixed 0.98/0.995 score gate or mandatory 0.0005 improvement floor. Follow the [latest review](../dev_probe/F05_MAXIMIZATION_REVIEW.md) for current results, experiment priority and handoff gaps. Retain paired evaluation, calibration separation, uncertainty and resource controls below. Earlier numerical targets are historical guidance only.

**26 September 2026. Status: execution plan, not completed experiments.** Read the [progress review](../dev_probe/F05_PROGRESS_REVIEW_2026-09-26.md) first. Existing E07–E09 scores use ground-truth-augmented validation candidates. Do not use them as promotion baselines.

## 1. Device allocation

| Device | Known resources | Primary responsibility | Independent work now |
|---|---|---|---|
| [Device 1: current Windows machine](DEVICE_1_LOCAL_BASELINE.md) | Existing log specifies 12 CPU-thread cap and Intel Arc 140T/OpenVINO; physical RAM/free disk not verified in this review | Repair common evaluation; clean reference matcher; full-pool retrieval and integration | Evaluation/cache repairs, retrieval channel preparation |
| [Device 2: MacBook Air M4](DEVICE_2_MAC_MATCHER.md) | User-confirmed 10 CPU cores, 10 GPU cores, **16 GB unified memory**; disk unknown | Controlled CPU matcher/feature/calibration experiments on portable feature arrays | Mac environment and memory smoke, feature adapters, experiment configs |
| [Device 3: i5 HX / RTX 3050](DEVICE_3_CHALLENGERS.md) | User-confirmed 13th-gen i5 HX, RTX 3050, 16 GB system RAM; exact CPU SKU, VRAM and disk unknown | Multilingual embeddings, domain fine-tuning and a bounded pair reranker; CPU fallback | GPU/memory smoke, configurations, data checks, branch and artifact preparation |

The M4's GPU and CPU share 16 GB; this is not 16 GB RAM plus separate VRAM. The Arc report's “16GB” label is not independently verified dedicated memory. Hardware discovery through Windows CIM was denied in this review; no hardware sizing claim is based on that query. Do not start several full-pool builds on either device without measuring memory.

## 2. Dependency and handoff order

1. **D1 publishes contract v1:** exact scorer, label-blind candidate API, normalized-record adapter, cache manifests, query-role manifests and tiny synthetic correctness fixtures.
2. **In parallel:** D2 prepares the Mac environment and bounded feature/model variants; D3 prepares one capability-appropriate challenger; D1 creates clean retrieval/feature artifacts. No device needs to wait to implement its independent experiments.
3. **D1 publishes baseline bundle B0:** natural candidate IDs and provenance, named feature arrays, training-only labels, calibration/comparison labels in evaluator-owned files, baseline predictions, hashes and environment. D2 can then train without building multimillion-document indexes.
4. **Run independent experiments:** D1 retrieval; D2 matcher/calibration; D3 complementary representations or model. Publish immutable run directories, not overwritten `E07`/`E09` filenames.
5. **Integrate finalists on D1:** align pair IDs, regenerate features for changed candidates, retrain/recalibrate the combined pipeline, and compare against B0 on the same development IDs.
6. **Lock one configuration:** evaluate the reserved holdout once, report uncertainty and failures, then consider full-test execution under the original packaging plan. Holdout is not a three-device leaderboard.

## 3. Shared query roles

Retain `splits/f05-v1/splits.json`, its exposure ledger and locked holdout. Verify disjointness with all historical and new supervised training IDs. Treat all previously inspected probe/cached-validation IDs as development-exposed; never relabel them “unseen.”

Create and persist **proposed** manifests under `splits/f05-v1/parallel-v1/`:

- `train_12k.json`, `train_25k.json`, `train_50k.json`: nested identity-group samples from the train partition only, balanced/stratified and selected deterministically, not source-file prefixes. Record actual country counts; the numeric suffix means total identities, not per-country identities.
- `calibration_5k.json` and `comparison_15k.json`: disjoint deterministic partitions of the existing 20k development probe, preserving country and match-count representation where feasible. If fewer eligible IDs remain after exposure/split verification, freeze the actual counts and document the reason rather than silently backfilling.
- `screen_2k.json`: a fixed stratified subset of comparison IDs for cheap screening, marked repeatedly exposed. Finalists are compared on all comparison IDs.
- `inner_train_folds.json`: supervised training identity folds for fitting/early stopping/hard-negative mining. No calibration/comparison/holdout group may enter these folds.

Thresholds and probability calibration are fit on calibration IDs; experiment ranking uses comparison IDs. Label-free transforms and full-pool indexing follow the same declared policy for every device. Learned encoders, mined supervised negatives and normalization dictionaries follow training-group boundaries. Cross-country stress tests need their own role manifests, with calibration from the training country rather than tuning on the supposedly unseen country.

## 4. Mandatory integrity checks before any promoted score

1. Run candidate generation with no ground-truth argument or evaluator import. Removing/permuting labels must leave the ordered candidate checksum unchanged. No `0.001` synthetic positive path may execute in evaluation.
2. Score every expected S1 exactly once, including no-candidate queries and singletons. Use the exact per-entity F0.5, not a pairwise/micro proxy.
3. Preserve all true targets in the evaluator when measuring false negatives; do not restrict truth to retrieved targets. Verify predictions are subsets of actual scored candidates.
4. Assert training/calibration/comparison/holdout group disjointness and log exposure. Do not train on comparison groups and later call predictions on them validation.
5. Pin feature names/order/types and normalization version across devices. Guard empty-field similarities and verify unit/house/postal components through the actual training adapter.
6. Cache manifests must identify raw input SHA-256, ordered query/target IDs, source revision plus dirty-tree patch hash, normalization/features, retriever parameters, model/tokenizer revisions, numeric precision and environment. Evaluation candidate cache identity must not depend on label values. Training label hashes belong in a separate supervised-data manifest.
7. Separate cold-cache preprocessing, warm-cache inference, fitting and report-writing times. Measure peak process memory and actual GPU/backend selection; do not copy a hard-coded device label into evidence.

## 5. Portable artifact bundle

Use `runs/parallel-v1/<device>/<experiment>/<run-id>/` and matching reports. Never share a live mutable `cache/` between devices. Preserve old checkpoints and create new namespaces.

Each run must contain:

- `manifest.json`: experiment ID, parent baseline ID, source/environment/hardware fingerprints, query/pool/feature/candidate hashes, seeds, parameters, start/end, peak RAM/VRAM, full timing stages and cache-hit status.
- `metrics.json` and `report.md`: exact macro, natural oracle, oracle-minus-actual gap, pair precision/recall, singleton false merges with numerator/denominator, candidate size p50/p95/p99/max, completeness/zero-hit, per-country/source/match-count/missingness/script slices.
- `query_scores.parquet`: query ID, per-query F0.5, TP/FP/FN, truth/prediction/candidate counts. Compare uncertainty by query, not by candidate pair.
- `pair_scores.parquet`: query ID, target ID, score and model version; include scored candidates even when rejected. Keep natural candidate provenance and rank/score sidecars.
- Model, exact feature schema, fitted calibration and decision policy; training log and environment lock/export.

Use Parquet/Arrow or sharded NumPy arrays plus JSON metadata for cross-platform exchange. Prefer compact integer ID mappings and float32 numeric features. Do not send the Windows OpenVINO model or opaque joblib objects as the only portable experiment input. Copy challenge data only to authorized local/team devices under the challenge rules; no connector upload or external service is required by this plan.

## 6. Fair comparison and choosing a winner

For every A/B pair, fix query IDs, training IDs, natural candidate universe (unless retrieval is the changed component), seed/folds, feature versions and threshold-selection protocol. E09's different `min_child_samples` must be held constant when isolating dense-feature value.

Use a paired bootstrap over query identities, stratified by country, e.g. 2,000 resamples with a saved seed; report mean macro difference and 95% interval. This interval measures sampling variability conditional on the trained models, not all training uncertainty. Rerun finalist training with two additional seeds when compute permits. Multiple experiments create selection bias: these intervals screen development results, and do not replace the final locked evaluation.

Default **proposed** promotion rule: clean paired macro improvement at least 0.0005, interval lower bound above zero, no unexplained material country/singleton regression, and feasible measured cost. A country drop exceeding 0.001 triggers review rather than silent averaging. Smaller gains can still be retained if replicated on the larger development set, stable across seeds, cheap and complementary; uncertain results remain provisional.

Retrieval has a separate gate: measure newly recovered positives, lost positives, exact oracle change and added pair cost. Aim at oracle ≥0.995 and recall ≥99% in both labeled countries; >0.99 is the minimum research headroom target. Then retrain the matcher on that new candidate distribution. Higher oracle alone does not prove higher final F0.5.

Never select the numerically highest value from different sample sizes, candidate caps or threshold-tuning populations. Never add two standalone gains to predict an ensemble score. Evaluate the actual combined pipeline and recalibrate it. If no challenger wins, keep B0.

## 7. Resource control and parallel scheduling

“Parallel” primarily means **one substantial experiment per device at the same time**. Within a device, reuse expensive candidate/features/embeddings and run a bounded sequence. Small analysis/report tasks can overlap if memory allows. Avoid three full CPU thread pools inside one process or a CPU fit competing with a GPU job for the same unified memory.

For each new workload run a 200-query smoke, then a 2k-query benchmark with the real pool. Estimate fixed setup cost separately from per-query/per-pair cost. Log peak RSS, swap and disk before deciding batch sizes. Capped target pools are permissible engineering smokes, never quality evidence. Full-pool sharding must preserve globally comparable retrieval scores and merge global top-K correctly.

Reserve substantial RAM for the OS and temporary arrays; on the 16 GB Mac target a process RSS around 8–10 GB maximum initially and lower it if memory pressure or sustained swapping appears. These are starting limits, not measured capacity. Check free storage for inputs + caches + two concurrent artifact generations before launching.

Dense storage lower bound is `number_of_targets × dimensions × bytes_per_value`, before IDs, ANN structures and working memory. At 384 dimensions and float32 the known training target pools require about 6.35 GB (India) or 9.50 GB (US), separately. Streaming/sharding is mandatory on a 16 GB machine for large dense retrieval; a matrix fitting in memory does not mean an index build fits.

## 8. Execution boundaries

These Markdown files specify work to implement. New CLI flags/manifests described here are **proposed**, not already available commands. Do not launch the current matcher scripts unchanged: they contain the evaluation contamination identified in the review. This documentation update does not start training, install software, transfer files or consume another device's resources.

Before implementation on each device, read local `AGENTS.md` and use graphify where available. After actual source edits, run the required graph update. Preserve shared core changes centrally, exchange pinned patches/commits, and keep device-specific outputs separate. Do not merge competing edits to the evaluator independently on all three devices.
