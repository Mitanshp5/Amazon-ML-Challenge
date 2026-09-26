# Device 1 — current Windows/Intel Arc machine

**Role:** own evaluation correctness, establish the clean reference, improve retrieval, and integrate the best verified components. Follow the [shared protocol](THREE_DEVICE_PROTOCOL.md). Status: proposed work; no runs launched by this plan update.

## Resources and boundaries

Keep the existing project policy of at most 12 CPU threads total for the main workload. Use `venv/Scripts/python.exe` for this repository. The logs report Intel Arc 140T/OpenVINO; confirm actual backend/device/memory before neural work. System RAM and free disk were not verified in this review. Run one full-pool fitting/index job at a time until measured otherwise. Do not treat the report's GPU “16GB” string as a dedicated-memory guarantee.

Own shared changes in `metrics.py`, `normalization.py`, `features.py`, `training/matcher.py`, the training adapters, split/cache manifests and evaluator. D2/D3 consume a pinned contract instead of independently redefining it. Their challengers should use new modules/configs and independent run directories.

## D1-00 — repair and publish the evaluation contract (mandatory)

**Hypothesis:** present numbers are confounded by answer-key candidate augmentation and inconsistent feature/provenance paths; fixing these is required before measuring gains.

Implementation:

1. Remove ground-truth augmentation from every evaluation path in both training scripts. Start the new baseline without training augmentation too; reintroduce training-only augmentation later as a separately labeled experiment with realistic recomputed features and no sentinel shortcut.
2. Make candidate generation label-blind by API design. Generate and hash natural candidates before loading labels in the evaluator. Persist zero-candidate queries explicitly.
3. Replace duplicate record adapters with the shared normalizer. Pass real name core, house, unit and postal components. Keep raw fields and language/script views available for later experiments.
4. Define named channel features instead of mixed `tfidf_max`; test presence and missingness. Keep a compatibility variant if needed to isolate the gain from the feature fix.
5. Explicitly pass feature names to OOF datasets, set boosting rounds/patience, and separate training inner folds, calibration and comparison roles. Implement the currently empty calibration entry point with reproducible policy artifacts.
6. Create `parallel-v1` manifests and versioned caches. Mark old pair/embedding/model caches as historical and ineligible for clean scoring; preserve them rather than deleting them.
7. Publish synthetic fixtures and checks for label-invariant candidates, exact singleton metric, all-query coverage, group separation, schema order, empty inputs and parser-to-feature integration.

**Deliverable:** contract-v1 source revision/patch hash, passing meaningful correctness checks, role manifests and schema. **Gate:** D2/D3 score promotion waits for this, although independent implementation/preparation can proceed.

## D1-01 — clean reference B0 and true loss decomposition

Use train_12k identities and the shared calibration/comparison manifests. Retrieve against complete country target pools. Preserve the full lexical+structured union as the primary reference. Keep top100 as a diagnostic arm only; its India headroom is too small to assume it is acceptable.

Train a CPU LightGBM reference on natural candidates with the repaired schema. Use training-only inner validation for early stopping. Initial explicit limit: 2,000 rounds, patience 100, learning rate 0.05; record actual iteration curve rather than assuming these values are optimal. Fixed parameters and sample sizes are shared with D2. Final threshold selection uses calibration IDs; report comparison results separately.

Persist all pair probabilities and query metrics. Compute `(1-oracle)` and `(oracle-final)` on the identical query population. Break errors into retrieval misses, wrong-branch/name false positives, house/unit/postal conflicts, missing-address cases, multiple-target omissions and singleton false merges. Inspect the largest **macro loss** contributors, not only pair counts.

**Deliverable:** B0 portable feature/candidate bundle for D2, scores and error taxonomy for D3. **Stop rule:** if provenance/schema/label-invariance checks fail, fix them before a training-size sweep. If clean score is worse than the old reported 0.9051, report it honestly; it is not a regression claim against an invalid evaluation.

## D1-02 — India lexical expansion (first quality priority)

**Hypothesis:** full-address and representation diversity recover misses more cheaply than adding a cosine feature to already retrieved pairs.

Run sequential staged arms using the same complete India pool and frozen screen/comparison IDs:

| Arm | Change from B0 | What to learn |
|---|---|---|
| L0 | Existing joint100/name100/address150/structured reference | Reproduce natural union and timing |
| L1 | Address K150 → 300; then 500 only if marginal recovery continues | Cutoff versus representation misses |
| L2 | Joint K100 → 200, with L0 address budget | Independent joint-tail contribution |
| L3 | Word-token address retrieval with token-frequency weighting, unioned with char retrieval | Recovery of reordered/partial addresses |
| L4 | Unicode-primary plus separate Latin-fold/transliteration view; preserve original scripts | Script mismatch recovery and false-positive cost |
| L5 | Larger vocabulary or lower minimum document frequency, one change at a time | Rare names/address token losses |

Do not fit all channel variants simultaneously. Reuse immutable matrix/vectorizer caches only when their full manifest matches. If RAM is insufficient, use correctly normalized/sharded sparse search and global score merging, not a smaller target pool masquerading as the main result.

**Metrics:** union oracle, pair recall, completeness, zero-hit rate, unique recovered/lost positives, candidate counts and build/query cost. Check the full US comparison slice for transfer/regression on finalists. Aim to reduce India's retrieval-loss budget toward 0.005; no fixed gain is promised.

## D1-03 — structured and duplicate channels

Test name+postal, name+house+locality, rare-name+address-token, and explicit unit-aware routes. Do not use bare house numbers as globally discriminative keys. Preserve candidate provenance and log overflow; loosen blocking only in an arm whose extra candidate cost is measured.

Compare duplicate expansion off/on using **recovered positives**, not added candidates. Existing India/US probes added 14/43 candidates but no positive links. Run leave-one-channel-out ablations of the complete union; absence of marginal gain in one insertion order is not sufficient to remove a channel.

Prioritize routes using D1-01 miss taxonomy, while avoiding ad hoc rules copied from individual comparison answers. Learn thresholds/rules from training/calibration data and compare fixed rules on comparison IDs.

## D1-04 — compression only after quality is measured

With the best raw union, compare K100,150,200,250 and untrimmed; include a proposed adaptive-budget arm that expands low-margin/missing-field queries and preserves distinct-channel evidence. Fit adaptive decisions without comparison labels.

For every arm record oracle loss **and actual end-to-end score after appropriate retraining/calibration**, not only retrieval recall. Proposed maximum compression oracle loss for a finalist: 0.0005 versus its raw union, subject to actual-score and runtime gains. K100 already loses 0.0046627 on the existing India probe and fails this budget. K200 is not automatically accepted either.

## D1-05 — merge winners from the other devices

Receive D2 model/features/policy and D3 channel/model outputs with manifests. Recompute features for newly added candidates; recalibrate scores under the new candidate distribution. Evaluate baseline, retrieval-only winner, matcher-only winner, and combined winner on the same comparison IDs. An ensemble may need separate calibration and must be tested explicitly.

Keep the best verified baseline as rollback. Run independent locked evaluation only after selection, then France inference smoke, end-to-end streaming, candidate/output parity and supplied-validator checks from the main implementation plan. This stage does not force one target per S1.

## Scheduling and handoffs

- **First handoff:** D1-00 contract and tiny fixture; D2/D3 stop diverging on evaluator semantics.
- **Second handoff:** D1-01 B0 feature bundle; D2 runs matcher experiments while D1-02/03 run locally and D3 prepares challengers.
- **Third handoff:** best retrieval candidates/provenance and updated feature shards; D2 confirms its winner on the new distribution.
- **Final handoff:** integrated finalist and locked policy; no device tunes on locked-holdout outcomes.

Within this device, run memory-heavy retrieval fits sequentially. Reuse fitted channels across query batches. Run report/paired-statistics work alongside only if observed memory headroom permits. Record benchmark-derived wall time before scheduling overnight work; 300-query report runtimes and cached 12-second fits are insufficient to promise a completion time.
