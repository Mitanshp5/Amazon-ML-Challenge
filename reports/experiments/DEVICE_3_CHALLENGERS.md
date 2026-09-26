# Device 3 — 13th-gen i5 HX, RTX 3050, 16 GB RAM

**User-confirmed hardware:** 13th-generation i5 HX processor, RTX 3050, 16 GB system RAM. Exact CPU SKU/core count, GPU VRAM and available disk remain unverified. **Role:** complementary multilingual retrieval and task-trained neural matching. Follow the [shared protocol](THREE_DEVICE_PROTOCOL.md). These are proposed experiments, not completed model improvements.

## 1. Resource setup and stop conditions

Run `nvidia-smi` locally and record GPU name, dedicated memory, driver and current allocation. Confirm the Python training environment can actually use CUDA. Do not assume that system RAM is GPU VRAM or that every RTX 3050 variant has the same memory. Start with one GPU job and a small number of data-loader workers; do not launch training and encoding simultaneously.

Use a separate pinned environment appropriate to this device; do not copy the Windows Arc/OpenVINO environment as the CUDA training environment. Keep model weights, tokenizer and revision in the run manifest. D1's existing OpenVINO files are an inference artifact, not automatically a trainable CUDA checkpoint.

Conservative starting points, to be measured rather than promised:

| Workload | Initial microbatch | Initial length | Adjustment |
|---|---:|---:|---|
| Frozen encoder inference | 16 texts | 128 tokens | Increase only after peak-memory/throughput benchmark |
| Bi-encoder fine-tuning | 2–4 identity examples | 128 tokens per record | Accumulate gradients; record actual contrastive negative population |
| Pair classifier/reranker | 2–4 record pairs | 192 tokens total | Reduce batch before dropping important fields; compare truncation |

Use mixed precision only after a finite-loss and score-parity smoke; benchmark memory in the actual framework. If training does not fit, try freezing lower layers or training a small head before abandoning the experiment. Keep inference/scoring as a useful fallback. Gradient accumulation increases optimizer effective batch but **does not automatically increase in-batch contrastive negatives**.

With 16 GB system RAM, stream records and use memory-mapped embeddings. Keep process RSS initially around 8–10 GB or lower if OS pressure appears. Do not load all US embeddings, Python ID dictionaries and a large index together. Measure disk space and write completed-shard manifests so interrupted jobs resume safely.

## 2. Checkpoint choices and eligibility

Use at most two initial multilingual encoder challengers, not an unrestricted model sweep:

- **First:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Its publisher documents multilingual support, 384-dimensional embeddings, mean pooling, a 128-token default and Apache-2.0 licensing. Follow its serialization/pooling configuration and pin the revision. [Publisher model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
- **Second, only after the first is measured:** `intfloat/multilingual-e5-small`, a multilingual embedding checkpoint under MIT licensing. Follow its model-card prefix, pooling and normalization conventions; treat symmetric entity-record matching versus asymmetric retrieval serialization as an explicit controlled choice. [Publisher model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md).

These are research candidates, not endorsements of expected ER accuracy. Verify the exact downloaded checkpoint's parameter count/license against the supplied challenge rules, record provenance, and use only challenge-provided records/labels for task adaptation. The main plan records the permitted model license and 8B-parameter limit; model suitability does not waive other competition constraints. Do not download external business identity data or query external matching/geocoding services.

The existing English `all-MiniLM-L6-v2` cosine feature is a control, not a completed multilingual experiment. Its E09 regression is a reason for controlled tests, not proof that every neural representation will fail.

## D3-00 — serialization and backend correctness

Receive the shared normalized-record adapter and natural candidate bundle. Use deterministic field-labeled text, for example `name: ... address: ...`; preserve Unicode and numeric components. Compare joint text against separate name/address vectors only after the first joint baseline works. Record truncation rates by field and country; do not let a long name erase the house/postal-bearing address.

Check padding invariance, correct attention-mask pooling, finite embeddings, L2 normalization when using cosine, deterministic ID-to-vector mapping and batch-size parity. Test 200 record pairs on CPU/GPU within a declared numeric tolerance. Cached embeddings must depend on model/tokenizer revision, serialization, max length, precision and ordered record IDs.

**Deliverable:** verified encoder wrapper plus throughput/peak-memory report. This can proceed while D1 repairs the evaluation contract because it needs no validation labels.

## D3-01 — frozen multilingual pair features (cheap first test)

Encode only the unique records needed by B0's natural training/calibration/comparison candidates, in shards. This is a matcher-feature experiment, **not a full-pool retrieval experiment**. Start with the first encoder and joint-text cosine.

Send keyed cosine sidecars to D2; compare B0 versus B0+cosine with identical training IDs, features otherwise, LightGBM parameters, seeds and calibration roles. Follow with separate name/address cosine only if joint cosine shows useful complementary errors or a plausible field-loss failure mode. Compare the second encoder only on the same sample/protocol.

**Promotion:** actual paired macro gain and stable country/singleton behavior, not cosine separation plots or feature importance. If neither frozen encoder helps, retain the negative result and prioritize task supervision rather than more frozen model downloads.

## D3-02 — genuine dense retrieval with a complete target pool

**Hypothesis:** multilingual embeddings recover true targets absent from the lexical/structured union, especially India representation misses. Begin with India; full US comes after useful marginal recovery and cost are established.

1. Encode the **entire country target pool** using the selected frozen encoder, deterministic shards and stable target IDs. Use provided record fields only. Encode the common query manifests separately.
2. For a small fixed query sample, stream exact cosine top-K across every target shard. Maintain global top-K per query by merging shard results; do not report recall against only a convenient subset of targets.
3. Screen dense K50/100/200 unioned with B0. Report dense-only and union oracle, uniquely recovered positives and added candidate cost. Evaluation truth is loaded only after candidate generation.
4. If exhaustive search is too slow, build a bounded-memory approximate index, then compare its candidate results and true-positive retrieval against full-pool exact search on the fixed small query sample. Tune index accuracy on development, not holdout. Store index parameters and training-sample seed.
5. Send recovered candidates with score/rank/presence to D1. Rebuild affected matcher features and recalibrate on the new natural distribution; do not infer final gain from oracle gain alone.

A 384-dimensional float32 India target matrix alone is about **6.35 GB**, and US about **9.50 GB**, excluding IDs/index/working memory. Float16 storage can reduce disk size, but requires score/ranking parity checks; cast only active blocks if the search implementation needs float32. Shard sizes must follow measured memory, not these raw-matrix estimates.

**Stop rule:** if dense retrieval adds no meaningful unique recovery at feasible cost, do not encode every remaining country automatically. Keep useful pair features separately. Conversely, a weak cosine matcher feature does not alone disprove dense retrieval's complementary coverage.

## D3-03 — task-trained bi-encoder with hard negatives

Run after D3-00 and the shared split contract. Start with the same frozen multilingual checkpoint as the control. Build supervision only from train_12k; scale to 25k/50k if justified.

- Positives: all available true S1-to-target aliases in selected **training identities**, preserving multiple-positive semantics.
- Negatives: high-ranked wrong targets from lexical/structured retrieval and training-only dense retrieval. Include same-name/different-location, similar address/different-name and conflicting-number pairs. Exclude every known positive alias of the query.
- Loss: begin with a pairwise/contrastive objective that supports explicit hard negatives and multiple positives. Do not treat another alias of the same identity as an in-batch negative. Keep query groups out of calibration/comparison throughout mining and training.
- Small staged arms: frozen encoder control; upper-layers/head adaptation; full fine-tuning only if VRAM fits and partial adaptation shows a plausible benefit. Initial training screen: one epoch at learning rates 1e-5 and 2e-5, then up to three epochs for the better arm with training-inner stopping.
- Preserve a frozen pretrained-model comparison to detect loss of cross-country generalization. Add US-only→India and India-only→US stress tests when affordable; these are proxies, not France validation.

Measure both pair-scoring quality and retrieval unique recovery; a model optimized for one may not improve the other. New weights require new embedding/index caches. Do not reuse old vectors under a renamed checkpoint. Stop training if transfer collapses or improvements vanish on the common comparison set.

## D3-04 — small task-trained pair reranker (later, conditional)

Only proceed after a clean tree baseline and error taxonomy show substantial confusion **among retrieved pairs**. Start from an eligible compact multilingual encoder with a binary classification head over serialized record pairs. This is task training, not assuming a generic semantic similarity score is an entity-match probability.

Train on natural training positives and realistic hard negatives; include singleton-query negatives and multiple-target positives. Use small microbatches, gradient accumulation and group-safe early stopping. Compare a standalone pair score and its addition to D2's tree model. Fit any fusion on training-only OOF scores and calibrate on the calibration manifest.

For deployment, score only an **ambiguity route** if full inference cost is prohibitive: low score margin, missing address or a bounded uncertain probability band learned on calibration data. The route is fixed before comparison evaluation. Compare against scoring all candidates on a manageable diagnostic query subset to measure routing loss. Never use an unexplained universal top10 cutoff; retain the tree decision for un-reranked candidates.

Benchmark actual pairs/second and routed fraction. Project `test_queries × mean_candidates × routed_fraction / measured_pair_throughput`, adding retrieval, feature and I/O time separately. The model must justify both paired macro improvement and projected full-test runtime.

## D3-05 — complementary CPU fallback

If CUDA is unavailable, VRAM is too small, or full-pool dense work is impractical, use this machine for an independent lexical representation arm: token-weighted address retrieval or a sorted-neighborhood name/address route not already assigned to D1. Agree one disjoint experiment ID first to avoid duplicate work. Use full-pool sharding, the same manifests, and exact union/oracle reporting. Alternatively score cached neural embeddings with a small CPU classifier and provide keyed sidecars to D2.

This fallback should not copy D1's entire sweep. The purpose is complementary coverage or model evidence, not three machines repeating the same setup.

## Scheduling and return package

**Recommended order:** D3-00 → D3-01 → small D3-02 full-pool query benchmark → choose D3-02 expansion or D3-03 based on evidence → optional D3-04. Do not run full fine-tuning and full-pool encoding at once on this GPU/16 GB system.

Return checkpoints/tokenizer/revisions, scripts/configs, complete encoding manifests, keyed pair features/candidate scores, quality reports, peak CPU/GPU memory, throughput and failure cases. Separate “encoder runs,” “adds useful candidates,” and “improves final F0.5” in the report. Only the last, under the shared clean protocol, makes it a final matcher winner.
