# Optional Colab A100 experiments alongside the local workflow

**Authority:** [local-first implementation plan](../../LOCAL_COLAB_IMPLEMENTATION_PLAN.md). This replaces the old RTX 3050 assignment. This device remains the source of truth for splits, evaluation, selection and packaging. Colab is used only when a bounded GPU workload warrants it; no Mac or second PC is required.

This document specifies work to implement and execute later. There is no new runnable Colab notebook or completed A100 benchmark in this revision. Do not assume the old `entity_resolution_a100.ipynb` implements the clean B0 split/candidate contract; port the shared package and parity fixture instead of adopting old notebook results.

## 1. Session setup and persistence

Verify the actual allocated GPU with `nvidia-smi`, plus framework CUDA visibility, VRAM, host RAM and free disk. Do not assume that selecting a GPU guarantees an A100 or a particular session duration. Colab states that hardware availability and usage limits vary; its VMs are temporary. Copy an input archive to runtime-local storage for active reads and keep completed artifacts/checkpoints in persistent storage or download them. [Official Colab FAQ](https://research.google.com/colaboratory/faq.html).

Pin a tested CUDA/framework/tokenizer environment and checkpoint revision. Keep local CPU/OpenVINO dependencies separate from the cloud training environment. Record precision, pooling, normalization, max length, seed and package versions. No production score should depend on an unrecorded notebook cell state.

Run one GPU experiment at a time. Start conservatively, then increase the batch size until measured throughput stops improving or memory becomes limiting. For a compact encoder, initial inference batch 128 and training microbatch 16 at 128 tokens are **proposed benchmark starting points**, not guaranteed capacity. For paired records, start at 192 total tokens and log truncation by field. Use supported mixed precision only after a finite-loss/parity smoke. Gradient accumulation does not automatically enlarge contrastive in-batch negative sets.

Save resumable state after each epoch and during longer epochs: model/tokenizer, optimizer/scheduler, random state, epoch/step, sampler progress, input/config hashes and validation history. A completion marker is written only after output hashes are recorded. Partial shards are not valid final outputs. Session loss should cost at most the work since the most recent completed checkpoint.

## 2. Existing input package and required filtering

Current verified root: `runs/parallel-v1/d1/b0_baseline/`.

| File | Purpose | Observed size, decimal MB |
|---|---|---:|
| `train_pairs.parquet` | 1.2M keyed natural training pairs; labels/folds/RRF | 15.16 |
| `train_text_records.joblib` | Query texts and 931,658 training target texts | 43.88 |
| `eval_text_records.joblib` | Natural evaluation target texts plus separate diagnostic targets | 27.81 |
| `calibration_predictions.parquet` | 500k keyed calibration pairs and B0 scores | 10.67 |
| `screen_predictions.parquet` | 200k keyed screening pairs and B0 scores | 4.57 |
| `b0_train_features.joblib` | Optional local tree feature reference | 32.67 |
| `b0_eval_features.joblib` | Optional evaluator data/feature reference | 27.47 |

The first five files total approximately **102.1 MB**, excluding model weights, code and manifests. All seven total about **162.2 MB**. Compressed file size is not resident memory consumption.

A cloud export should contain the pinned shared source, feature/text schema, `train_12k.json`, `inner_train_folds.json`, query-role manifests, verified input hashes and deterministic pair ordering. Prefer explicit Parquet text tables and JSON metadata when preparing a portable archive; existing local joblib can be read in a pinned compatible environment as a transitional format.

For training, select query IDs from **train_12k only**. Both text files contain 19k query records spanning roles. Keyed pair labels/folds must remain aligned; this was checked row-for-row against the existing local feature bundle during the latest review.

For scoring, derive candidate target IDs exclusively from the saved natural pair lists. Exclude `diagnostic_unretrieved_targets` from candidate generation. Export evaluation pair keys/texts without `is_match` or truth maps into the GPU scoring interface; keep label-based selection and final metrics local. Training-only truth aliases may be used to avoid false negatives in hard-negative mining, but comparison/holdout labels may not be used for that purpose.

The compact package supports pair-scoring experiments. It is **not a full-country retrieval corpus**: a result searching only these targets must not be called full-pool dense retrieval.

## 3. G00 — import validation and hardware benchmark

Verify archive/file hashes, query-role disjointness, training-pair uniqueness, schema and text coverage before training. Run the shared exact-metric fixture and a small encoder CPU/GPU parity check. Check padding/attention-mask pooling, deterministic target IDs, finite vectors and normalization.

Benchmark 10k representative records for encoding after warmup; log records/sec, sequence-length distribution, peak GPU/host memory and tokenization/I/O cost. Benchmark 200 training steps with the chosen objective before extrapolating an epoch. Persist the benchmark report locally with the other run metadata.

**Exit:** usable environment and a measured compute estimate, or a clear fallback decision. If no suitable GPU is allocated, postpone the GPU experiment and continue the local queue.

## 4. G01 — frozen multilingual features

Begin with one previously planned compact encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or `intfloat/multilingual-e5-small`. Preserve their documented pooling/prefix conventions, pin the revision and record license/parameter count under the challenge rules. The prior [neural experiment reference](DEVICE_3_CHALLENGERS.md) contains publisher links; only its modeling ideas remain relevant, not the retired hardware allocation.

Encode unique query/target records for natural training/calibration/screen pairs in shards. Serialize name and address deterministically with field boundaries; keep numeric and Unicode evidence. First return joint-text cosine, then separate name/address similarity only if the first experiment or error analysis warrants the cost.

Return `(query_id, target_id, neural_score, checkpoint_revision, serialization_version)` and corresponding train/calibration/screen role identifiers. The local device joins by pair IDs and fits an identical tree control versus control-plus-neural-feature, with separately calibrated decisions. Feature importance or embedding separation alone is not a promotion criterion.

**Exit:** paired local macro result and cost. No credible complement after one or two justified encoders means stop expanding frozen model choice and reassess supervision/serialization.

## 5. G02 — one task-trained neural challenger

Choose the objective based on current errors: a pair classifier for confusion among retrieved pairs, or a bi-encoder if representation misses/semantic candidate recovery are the main opportunity. Do not run every architecture simply because an A100 is available.

Use training positives and hard negatives from natural retrieval. Keep multiple-positive identity semantics; known aliases are not negatives. Use the fixed inner training folds for model selection/early stopping. Start with one epoch, then at most three for a promising arm. A small learning-rate screen such as 1e-5 versus 2e-5 is sufficient initially; record all settings and actual step timings.

Return frozen model weights/tokenizer, serialization, checkpoint selection history and keyed calibration/comparison scores. Fine-tuning may run alongside local candidate/feature experiments, but the GPU run must remain tied to its initial manifest. A newly selected local candidate universe requires an explicit new scoring job.

If neural scores feed a local learned stacker, produce training-only OOF scores or reserve disjoint training groups for combiner fitting. Full-training in-sample scores must not be treated as OOF features. A simple separately calibrated score fusion is a cheaper diagnostic before expensive multi-fold neural retraining.

**Exit:** actual calibrated development macro gain, per-country/singleton effects and projected deployment cost. Stop when gains vanish outside the screen or transfer stress tests deteriorate substantially.

## 6. G03 — conditional full-pool dense retrieval

Only launch after the earlier experiments/error analysis justify its cost. This stage needs every target record for the chosen country, stable IDs and source hashes; the compact pair bundle is insufficient. Start with India and the fixed screening queries.

Encode all targets in resumable shards. Compare exact streamed top-K against an approximate index on a fixed query subset; preserve global top-K when merging shards. Union dense K50/100/200 with the local lexical/structured candidates, returning score/rank/provenance. All candidate generation remains label-blind.

For 384-dimensional float32 vectors, raw India training-target embeddings occupy about 6.35 GB and US about 9.50 GB before IDs, indexes and work buffers. A large GPU does not guarantee host-RAM/index-build capacity. Memory-map or shard as needed; validate any lower-precision ranking change.

Return candidate deltas and index/encoder manifests. The local device measures unique positive recovery and then retrains/calibrates the matcher under the new candidate distribution. Stop if added candidate cost does not translate into useful quality.

## 7. G04 — routed pair reranker, if justified

A neural pair model must improve errors remaining after the selected local tree model. Compare score fusion on existing candidates first. If full-test pair scoring is too costly, learn an ambiguity route on training/calibration data and freeze it before comparison: e.g. a small score margin or uncertain probability band.

Measure routing loss against reranking all candidates on a manageable diagnostic sample. Retain the local model decision for un-reranked pairs; do not impose a universal top10 or one-target-per-query cap. Estimate final runtime from routed pair count and measured pairs/second, with retrieval and I/O added separately.

## 8. Return contract and local acceptance

Each GPU result must include code revision/patch hash, environment, hardware, input and role hashes, model/tokenizer revision, training history, checkpoint, keyed outputs, coverage/duplicate checks, throughput, peak memory and completed-shard hashes. Export enough to rerun the exact model and serialization.

The local device verifies all expected query/pair IDs, rejects unknown/duplicate pairs, handles no-candidate queries explicitly and joins by keys, never by assumed row order. Scores for a new candidate set cannot silently substitute for old model inputs. Calibration and promotion happen locally on the same manifests as CPU contenders.

The cloud job is useful only if it yields reproducible complementary evidence at acceptable end-to-end cost. Local L00–L05 and final packaging can proceed without any neural promotion.
