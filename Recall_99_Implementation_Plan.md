Amazon ML Challenge — implementation plan for 99% blocking recall

Prepared for Pranil Agrawal, 26 September 2026. Based on the supplied metric screenshot and `ML-Devs-main.zip`. This is a proposed implementation plan; the notebooks have not been modified or retrained.

The recommended architecture is a hybrid candidate retriever followed by an entity-matching classifier: retain complementary lexical, structural, and multilingual signals, measure the recall lost at every filtering step, and optimize final decisions for the challenge's macro-F0.5 score.

The ZIP contains code and documentation, but no dataset TSVs, candidate checkpoints, trained weights, or executed notebook outputs. Findings below are grounded in source inspection and small normalization checks. No improvement percentage has been measured. In particular, 99% is an acceptance target, not a guarantee.

1. Define the target correctly

The screenshot measures pair-weighted blocking recall: the fraction of true S1-to-S2/S3 links present in the candidate set. It does not measure final model recall or the leaderboard score.

| Slice | Current hits / true links | Blocking recall | Missed links | Minimum hits for 99% | Additional hits needed |
|---|---:|---:|---:|---:|---:|
| India | 25,239 / 27,605 | 91.43% | 2,366 | 27,329 | 2,090 |
| US | 38,564 / 41,391 | 93.17% | 2,827 | 40,978 | 2,414 |
| Overall | 63,803 / 68,996 | 92.47% | 5,193 | 68,307 | 4,504 |

On this same sample, at least 86.73% of the currently missed links must be recovered. At most 689 true links may remain missing. These counts change when the evaluation set changes.

For the same true-link population, final pair recall equals blocking recall multiplied by the matcher's recall conditional on retrieval. Thus 99% blocking recall with a 98% conditional matcher gives 97.02% final recall. If the desired target is 99% final recall, a useful development target is at least 99.5% blocking recall and 99.5% conditional matcher recall, whose product is 99.0025%. Whether that operating point has acceptable precision must be measured.

Use 99% overall and in each labeled country as the primary blocking gate, with 99.5% as a stretch target. Select final predictions using macro-F0.5 over all S1 records, including singletons, as specified in the supplied challenge README. Also publish the final precision–recall trade-off. France is test-only: its true recall cannot be certified from the provided training labels.

2. Fix the observed implementation problems first

Notebook cell numbers below are zero-based JSON cell indices. Stable cell IDs are included where helpful.

| Location in supplied code | Observed behavior | Planned change |
|---|---|---|
| Local notebook, cell 10: `normalize_ascii` | ASCII encoding with `ignore` removes Hindi/Kannada names completely. | Preserve the original script in the main representation; add a separate Latin/transliterated view. |
| Local notebook, cell 10: `normalize_text` | The supposedly Unicode-preserving `[^\w\s]` cleanup removes Indic combining marks. A check turns `श्री गणेश ट्रेडर्स` into broken syllables. | Preserve Unicode letter, mark, and number categories, and test complete Indic words. |
| Local notebook, cell 10: `ABBR` | Merging US/India and French dictionaries changes US `ste` to `societe`. A check gives `123 main street societe 5`. | Dispatch abbreviation rules by field and country; keep unknown-country fallback. |
| Local notebook, cell 19: US second pass | The first pass asks for 80 candidates; the second asks for 60 at the same 0.02 threshold, then trims the merged result to 60. | Make expansion a true set union with larger retrieval depth. Assert that expansion preserves all earlier candidates. |
| Local notebook, cell 19: fallback scores | Fuzzy string scores and TF-IDF cosine scores are mixed as if they were comparable. | Store each score under its own channel; use rank fusion for ordering. |
| Local notebook, cell 22 (`p2c-ibf`) | The challenger adds candidates but then forces `K_TOP=50`; it can lose existing true matches. | Evaluate the untrimmed union first. Apply caps only after measuring their recall cost. |
| Local notebook, cells 19/22 | Oversized key buckets are removed; some fallback postings are taken in input order. | Refine frequent buckets using additional keys and query-dependent ranking. Log overflow losses. |
| Local notebook, cells 19/30 | Validation has US fallback retrieval; test blocking runs the primary TF-IDF route only. | Use the same retrieval implementation and configuration for development, holdout, and test. |
| Local notebook, cells 19/30/32 | Candidate caches use manual version/country/query-count names; test parts resume by row count. | Fingerprint data, exact IDs, preprocessing, model, index, and configuration; reject incompatible parts. |
| Local notebook, cells 24–26; A100, cell 16 | The set named `val_matches` supplies model training examples. Local grouped OOF is useful, but there is no independent final holdout after repeated blocker/model selection. | Establish explicit train/development/locked-holdout roles and rebuild learned artifacts accordingly. |
| A100 notebook, cell 16 | Encoder positives come only from the PIN-blocked skeleton. Hard positives missed by that blocker never enter training. | Construct positives from all training GT links, joining their raw records directly. |
| Local notebook, cell 24 | Empty names can count as exact matches, and `_char3('')` yields `{''}`, making two empty names appear identical. | Add field-availability flags and make empty-empty lexical agreement non-positive evidence. |

The screenshot reports mean K=50 for India and 40 for US. The ZIP instead has K=90/80 and comments describing a later 94.15% baseline. Those comments are not saved experimental results. Reproduce and name the exact current baseline before comparing changes; do not attribute the screenshot to the newer configuration automatically.

The code also fits its models using unpinned dependencies. Pin a verified environment rather than guessing API compatibility. For example, current LightGBM 4.7 documents `eval_X`/`eval_y`, while older environments use `eval_set`; the existing arguments are not inherently a bug [4].

3. Establish trustworthy evaluation and diagnose the missing links

Create persistent train/development/holdout ID manifests, initially around 80/10/10, stratified by country, match-count bucket, and script where feasible. Group by S1 entity. If any true S2/S3 identity is linked to multiple S1 records, place the entire connected labeled group in one split. Preserve a record of prior exposure; select an unexamined holdout and retrain from base weights if previously used entities would contaminate it.

All labeled positive/negative training examples must respect these identity boundaries. Grouped OOF inside the training split can support stacking and calibration. Development selects retrieval settings and thresholds. The locked holdout receives the selected pipeline once; changing the pipeline in response to it turns it into development data.

Keep the full applicable S2+S3 retrieval universe during evaluation. Do not reduce the denominator to true matches that happen to be in a sampled pool. Indexing unlabeled corpus text is separate from using holdout labels for training. Document the corpus-fitting policy and apply it consistently to test. Never insert GT matches into development/holdout candidate lists.

Use a deterministic stratified sample of about 20,000 development queries for iteration, followed by the full development split. Replace `sorted(val_matches)[:SAMPLE_S1]`, which samples an ID prefix, with a seeded/hash-selected sample. Run country-specific smoke queries explicitly; `head(2000)` does not ensure every country is covered.

Export a development miss table by joining every missing GT ID to its actual record. Include both raw names and addresses, script/field availability, country/source, each channel's score/rank, and the stage that dropped the pair. Review a stratified set of at least 200 misses, then quantify the following categories across the full table:

- Name removed or damaged by normalization; same-script versus cross-script mismatch.
- Missing address, postal code, or business name; zero-vector lexical representation.
- True match below the score floor or beyond top-K.
- Franchise/common-name collision, same-building collision, or source crowding.
- Oversized structured bucket discarded or truncated.
- Found in a raw channel but lost during union compression or matcher filtering.
- Incorrect/missing country or missing GT record; flag integrity issues without silently changing denominators.

For low-ranked GT links, compute their pairwise similarity directly even when they were not retrieved. Obtain exact ranks on a selected query subset against the full shard. This separates ranking-depth failures from representation failures without retaining every possible pair.

Every run should report micro blocking recall, mean per-S1 recall among non-singletons, the proportion of matched S1 records for which all true links were retrieved, and retrieval's oracle macro-F0.5 ceiling obtained from `prediction = candidates ∩ truth`. Also report mean/median/p95/p99/max K, empty candidate rate, latency, throughput, and peak RAM. Break down recall by country, source, script, missing fields, and match-count bucket. Compute uncertainty by bootstrapping S1 groups, not independently resampling correlated links.

4. Build multiple safe text representations

Preserve raw names and addresses. Produce separate fields for Unicode-normalized text, Latin accent-folded text, legal-suffix-reduced names, optional offline transliteration, and lightly normalized full addresses. Field labels and missing-value markers should be identical in training and inference.

Use NFKC plus case folding for a Unicode view while preserving combining marks. Keep Latin accent folding separate from Indic cleanup. Transliteration is an additional noisy retrieval channel, not a replacement for the original script or proof that two names denote the same business. Use local deterministic conversion and evaluate actual English/romanized spellings from the supplied training pairs.

Split name and address dictionaries. In US addresses, `ste` can mean suite; French name rules differ. Resolve ambiguous words such as `st` with field/context rather than replacing them everywhere. Keep name-with-suffix and name-without-suffix variants; do not treat every address abbreviation as a legal suffix.

Extract country-aware postal-code candidates and house/unit/range tokens from raw addresses before destructive cleanup. Do not call the first arbitrary 5–6 digit number a confirmed postal code. Preserve letters in values such as `12A` and distinctions between building, suite, and postal numbers. Treat conflicting or missing components as features, not unconditional rejection rules.

Acceptance checks: Indic letters/marks survive; missing fields remain explicitly missing; US/French abbreviation collisions disappear; raw IDs and records remain unchanged; all existing countries and unseen country labels work.

5. Replace the single retrieval view with a complementary union

Retain the baseline route while evaluating the following additions. Values below are starting experiment settings, not claimed optima.

| Channel | Input and method | Initial returned depth | Main purpose |
|---|---|---:|---|
| Unicode name | Name-only char TF-IDF, initially `char_wb`, 3–5 grams | 100 | Typos, word-order/suffix variation, same-script names |
| Latin name variants | Accent-folded/compact/transliterated names; char retrieval | 50 | Accents, concatenated names, cross-script spelling overlap |
| Full address | Address-only char TF-IDF, preserving number tokens | 100 | Address bridges when names disagree or are missing |
| Name + full address | Existing joint route with corrected normalization and explicit field weighting | 100 | Preserve useful combined evidence |
| Structured keys | Rare name tokens, nonempty exact variants, postal + token, house + street token; ranked postings | 50 initial additions | Exact or rare evidence that dense/char scores dilute |
| Multilingual dense | Fine-tuned or frozen multilingual record embeddings | 100 | Cross-script and broader name/address variation |

Initially form a deduplicated union with the baseline and no global cap. Record channel membership, channel-specific scores/ranks, and candidate source. Measure each channel's marginal recovered GT links; channel gains overlap and must not be added as independent percentages.

For the lexical channels, sweep K=50/100/200/400 and score floor=0.02/0.005/0 on selected development queries. A zero score floor does not recover a pair whose representations share no features; empty vectors require other channels. Tune vocabulary limits and document-frequency filters after checking memory and missing rare features. Process one country/view at a time and keep matrices sparse.

For frequent structured keys, use compound intersections and a secondary lexical rank. Avoid taking the first N records or dropping the whole bucket. Exact name equality alone must not force a final match: franchise branches and unrelated businesses can share names. If one source crowds out the other, compare combined retrieval with explicit S2/S3 quotas.

Start rank fusion with `sum(weight[channel] / (60 + rank[channel]))`, using ranks starting at one and equal channel weights as the baseline. Keep the raw scores as independent matcher features. Rank fusion orders candidates; it does not guarantee recall after truncation.

After the union reaches the recall target, test final budgets of 200, 400, and 800. Try protecting the top 20 from each useful channel family and the retained baseline before filling the remainder by fused rank. Measure recall before and after this compression. If mandatory reservations exceed a budget, expand the budget or explicitly measure a revised policy; never truncate silently. Keep the smallest feasible policy that still passes the gate.

Adaptive expansion should consider saturated lists, common-name frequency, missing fields, script mismatch, channel disagreement, and the density of the score tail. A large top-1/top-2 gap is insufficient: each S1 can have several true matches, including difficult lower-ranked variants. Audit expansion coverage using development labels. Expanded candidate sets must be supersets of the preceding sets before any separately reported compression.

Country shards remain the efficient default. Recheck the supplied documentation's claim of zero cross-country GT links. Provide a generic country route and a bounded country-agnostic fallback for missing/unreliable country values when audit evidence warrants it. Do not assume France needs a smaller budget just because it has no labels.

6. Add a multilingual bi-encoder trained for business identity

Start with `intfloat/multilingual-e5-small` as the dense-channel challenger. Its publisher lists an MIT license, multilingual support, and 384-dimensional embeddings [1]. Those properties make it a reasonable compact candidate for this task; its performance on these records remains to be established.

Serialize a record consistently, for example `query: name: <name> address: <address> country: <country>`. Treat record similarity as symmetric and initially use the `query:` prefix on both sides, following the model card's symmetric-task guidance. Do not encode entity IDs. Use attention-mask-aware pooling and L2 normalization; use the same tokenizer, pooling, truncation, and normalization after export [1].

Training proposal:

- Build positive pairs from every available GT link in the training partition, including positives absent from all current candidate routes. Rotate positives so one easy match does not dominate an entity.
- Mine hard negatives from the hybrid retriever: same/common name at another address, similar address with another business, near spellings, and candidates that fooled the previous matcher. Mix in some random negatives and include singleton queries in classifier training.
- Exclude every known same-entity record from negatives. Use one identity group per contrastive batch or explicit multi-positive masking. String-level duplicate removal alone is insufficient when different strings are aliases of one business.
- Use a group-aware MultipleNegativesRankingLoss baseline or a multi-positive contrastive loss. Sentence Transformers documents in-batch alternatives as negatives and recommends avoiding duplicates; extend that protection to identity groups here [2].
- Start with 1–3 epochs, learning rate around 2e-5, maximum length 256, and a batch size that fits the measured GPU budget. Check truncation rates and compare 128/256/512 if needed. These are search starting points, not fixed requirements.
- Use only supplied-data noise patterns for augmentation: punctuation, abbreviation, token order, plausible typos, and selective field dropout. Preserve identity-critical information; indiscriminate number swaps can create false positives.
- Compare the pretrained and fine-tuned encoders on normal and out-of-domain development slices. A fine-tuned model can lose multilingual generalization, so keep the pretrained checkpoint as a baseline.

Build normalized-vector indexes per country. On a selected query subset, benchmark against exact `IndexFlatIP` search over the full applicable corpus. For scale, evaluate HNSW or IVF; tune `efSearch` or `nprobe` using both exact-neighbor overlap and actual GT recall [3]. Measure approximation loss separately from encoder representation loss. A proposed approximation-loss budget is at most 0.1 percentage point of blocking recall, with the final hybrid still above the target.

The A100 notebook currently uses GPU encoding with a CPU `IndexFlatIP`; the displayed label does not make the search GPU-based. Choose and document the actual index device. Its full-scale section is a stub and must be implemented, including resumable embedding generation and explicit ID-to-vector mapping.

7. Strengthen matching while preserving retrieval gains

Use a pooled LightGBM pair classifier as the first complete matcher. It should score every candidate in the retained union. Add country specialists only if independent evaluation demonstrates a gain and preserve a pooled fallback for France/unseen countries.

Extend the current 18-feature set with separate Unicode/Latin name and full-address similarities, IDF-weighted token overlap, dense similarity, each retrieval channel's rank/score/presence, script agreement, field availability, legal-suffix variation, house/range/unit agreement or conflict, postal evidence, name frequency, and candidate-list context. Do not interpret equal empty fields as evidence. Do not merge heterogeneous scores into one feature called `tfidf`.

Train with all retrieved positives plus hard negatives from the expanded retrieval distribution. The current 1:6 sampling ratio can be a starting point; replace purely random candidate negatives with a mix of hard and random examples. Include singletons and train on entities whose true matches were missed, so the model sees realistic no-positive candidate sets. Compare per-entity weighting against ordinary pair weighting to prevent large candidate sets from dominating the macro objective.

It is acceptable to add missing training GT pairs for encoder learning or explicitly tracked matcher augmentation. It is never acceptable to inject them into evaluation candidates. If augmenting matcher training, mark the altered sampling distribution and calibrate on natural, unaugmented candidate sets.

Retrieval models, mining, feature generation, and matcher OOF predictions must respect fold boundaries when producing honest OOF estimates. Generate full, unsampled development candidate predictions for threshold tuning; negative subsampling changes the probability distribution.

If LightGBM's remaining errors justify more GPU cost, add a compact multilingual cross-encoder trained for binary entity matching. A concrete option is the multilingual encoder backbone above with a newly trained pair-classification head on serialized record pairs. Compare it with the LightGBM baseline; the existing off-the-shelf MS MARCO reranker demo is not a trained entity matcher.

Route ambiguous pairs to the cross-encoder and leave other pairs' decisions with the calibrated LightGBM model. Avoid an unconditional top-10 cutoff: it creates another recall ceiling. Record the recall of the routing decision and every final rejection stage. Evaluate any score fusion on out-of-fold predictions rather than training predictions. Start without graph transitive closure, automatic exact-name acceptance, or one-to-one assignment; none is justified by this multi-match task.

8. Calibrate decisions and test robustness

Tune the existing empty-list gate and pair-match threshold jointly on development macro-F0.5, using all S1 queries, including zero-candidate queries and true singletons. Report micro precision/recall, conditional matcher recall, macro-F0.5, complete-match-set accuracy, and singleton false-positive rate. If probabilities are calibrated, use grouped held-out/OOF predictions from the unsampled candidate distribution.

Use one global fallback threshold. Country-specific thresholds are optional and require enough validation evidence. France cannot receive a defensibly tuned threshold from nonexistent French labels. Compare a pooled model/global threshold with country specialists using leave-one-country-out US/India experiments as generalization diagnostics; these are supplementary tests, not substitutes for the main stratified evaluation or proof of French performance.

Run controlled stress slices based on supplied-data patterns: accents, Indic script, abbreviation/suffix changes, reordered addresses, missing fields, common names, similar businesses at one address, and unseen country labels. Measure both clean and perturbed performance. Retain original labels only for transformations that plausibly preserve identity.

For a genuine 99% final-recall requirement, produce a threshold sweep showing the precision and macro-F0.5 achievable subject to recall ≥99%. If no point satisfies the desired precision/cost constraints, report that result; lowering thresholds until the metric reads 99% does not establish a robust model.

9. Make full-scale inference feasible and reproducible

The existing plan reports 1,732,544 test queries and about 10 million S2/S3 records; these counts could not be rechecked without the dataset. At mean K=200, that scale creates approximately 346.5 million candidate pairs, about 4.55 times the screenshot's mean-K=44 workload. A dense matrix of 40 float32 features for all those pairs alone would be about 55.4 GB.

Compute and score features in chunks, writing outputs immediately. Use integer record IDs internally, columnar/sharded candidate storage, and sparse matrices. Avoid Python object lists and an all-test-pairs feature matrix. Materialize raw/normalized data once instead of rereading and renormalizing the entire corpus for each experiment.

One set of 10 million 384-dimensional float32 embeddings needs about 15.36 GB before index, ID, graph, or temporary-build overhead. Additional embedding views multiply storage. Keep one country/index active at a time on a 32 GB machine, and benchmark peak build memory; a CPU flat/HNSW index does not automatically become half-size because embeddings were cached as float16. Try compression only after quantifying its recall loss.

Benchmark representative 5,000-query slices per country with the full corpus to estimate retrieval, feature-generation, and optional cross-encoder throughput. Use measured rates to choose K and routing; do not promise full-test runtimes from the sample-only demos.

Create one shared library used by local, A100, and sample notebooks. Save a manifest containing data fingerprints, exact query/pool IDs, split seed, normalization version, full retrieval configuration, encoder revision/checkpoint, index parameters, feature schema/order, matcher version, thresholds, and dependency versions. Write chunk outputs atomically and resume only when manifests match.

`candidate_pairs.tsv` must contain the candidates actually scored by the matching stage, including the LightGBM-scored pairs rejected later. Persist score/provenance sidecars so test features reproduce development features. Generate exactly one output row per required S1, including France and empty predictions. Check exact ID-set equality, not just total row count. Assert no duplicate candidates and `final_matches ⊆ scored_candidates`.

Run the supplied validator with `--check-ids`. Its subset check is a warning, so add a strict internal assertion. At hundreds of millions of candidates its in-memory dictionaries/sets may be too large: retain the official validator, validate the final matching file, and implement a streaming/external-sort candidate audit for IDs, coverage, duplicates, and subset checks. Do not silently omit those checks to fit memory.

10. Implement in this order, with measurable gates

| Stage | Work | Completion evidence |
|---|---|---|
| A | Baseline manifest, honest split, miss joins, metric report, stale-cache protection | Reproducible baseline and complete accounting of its missed links; no claim based only on comments |
| B | Unicode/field-aware normalization; remove shrinking fallback/caps; unify train/test retrieval | Normalization checks pass; expansion is monotonic; same-record retrieval parity passes |
| C | Independent lexical/structured channels; untrimmed union; K/floor sweep | Channel marginal gains, union recall, and recall lost to compression are measured |
| D | Multilingual encoder with all training positives and group-aware negatives; ANN benchmark | Dense channel adds justified recall; approximation loss is measured; selected hybrid targets ≥99% |
| E | Retrain LightGBM on the final retrieval distribution; calibrate thresholds | Development macro-F0.5 improves without an unexplained singleton or important-slice regression |
| F | Optional cross-encoder only if remaining classifier errors justify it | End-to-end metric gain exceeds measured routing/latency cost; no hidden top-10 ceiling |
| G | Frozen holdout, country/noise diagnostics, streamed test pipeline, format audit | Actual holdout metrics and uncertainty reported; exact output coverage and candidate consistency pass |

Do not claim that stages B/C/D will yield specific intermediate recall percentages. If raw union recall remains below 99%, use the miss taxonomy to choose the next representation or data-quality intervention. If the union reaches 99% but the retained candidate set does not, fix compression. If blocking passes and final recall does not, focus on the classifier and decision thresholds.

Suggested source layout and notebook migration:

| Proposed file | Responsibility | Existing code to migrate/replace |
|---|---|---|
| `src/er/normalization.py` | Safe country/field-aware text views and address components | Local cell 10; A100 normalization cell |
| `src/er/splits.py` | Persistent identity-group train/dev/holdout manifests | Hash split cells and ID-prefix sampling |
| `src/er/retrieval/lexical.py` | Sparse name/address/joint retrieval | Local cell 19 and duplicated test cells |
| `src/er/retrieval/structured.py` | Frequency-aware key postings and overflow handling | Local cell 22 and US backfill |
| `src/er/retrieval/dense.py` | Embeddings, ID maps, ANN build/search and exact benchmark | A100 cells 14/22; local OpenVINO path |
| `src/er/retrieval/hybrid.py` | Provenance, union, fusion, measured budgets and expansion | Local cells 19/22/28/30 |
| `src/er/training/encoder.py` | All-GT positives, grouped contrastive batches, hard-negative mining | A100 cell 16 |
| `src/er/features.py` | Shared feature schema and chunked extraction | Local cell 24 (`p3a-feats`) |
| `src/er/training/matcher.py` | Grouped matcher fitting and optional cross-encoder | Local cell 25; A100 reranker demo |
| `src/er/calibration.py` | Full-candidate thresholds and optional score calibration | Local cell 26 (`p3c-thresh`) |
| `src/er/evaluation.py` | Recall curves, oracle ceiling, macro-F0.5, miss audit, slice/CI reports | Local cell 20 and scorer cells |
| `src/er/pipeline.py` | Shared inference, manifests, atomic resume, streaming output checks | Local cells 28/30/32 |
| `configs/hybrid_recall.yaml` | Versioned experiment settings and selected final configuration | Scattered notebook constants |
| `tests/` | Unicode, empty-field, group-boundary, union, parity, resume, and output-invariant checks | Add focused checks for the identified failure modes |

Keep notebooks as thin launch/report interfaces. Update the existing plans and runbooks once measured configurations are selected so contradictory K values and obsolete recall gates are not carried forward. Preserve the old baseline artifacts for comparisons.

Suggested experiment ledger: baseline; correctness fixes; separate lexical views; structured union; frozen multilingual encoder; fine-tuned encoder; candidate compression/expansion; retrained matcher; calibrated decisions; optional cross-encoder. Store configuration hash, per-slice recalls, recovered/lost GT links, K distribution, macro-F0.5, singleton errors, runtime, and RAM for each. Promote one change at a time unless a dependency requires a combined experiment.

The first implementation session should complete stages A and the correctness part of B. The next major experiment should compare the current joint TF-IDF route against independent name-only and full-address retrieval at larger diagnostic K. This establishes how much of the gap is candidate truncation versus lost representation before investing in neural training.

Technical references and evidence

Repository evidence: `notebooks/entity_resolution_local.ipynb`, `notebooks/entity_resolution_a100.ipynb`, `student_resource/student_resource/README.md`, `student_resource/student_resource/utils/validate_submission.py`, and the existing implementation plans/runbooks in the supplied ZIP. Dataset counts and historical improvements recorded only in those plans are not independently verified measurements in this review.

1. [Publisher model card: intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md) — license, multilingual embeddings, prefix conventions, and pooling. The model choice is a proposed experiment; published benchmark scores do not predict this challenge's recall.
2. [Sentence Transformers: MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.losses.MultipleNegativesRankingLoss) — contrastive pairs/triplets, in-batch negatives, and duplicate-aware batches. Identity-aware masking is an additional task-specific requirement proposed here.
3. [Faiss: guidelines for choosing an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) — memory and search-accuracy trade-offs. Exact-versus-ANN GT recall benchmarking is required for this proposed pipeline.
4. [LightGBM classifier API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) — version-sensitive fit/validation interfaces; pin and verify the chosen environment.

All proposed training augmentation uses only the supplied data. Do not use business lookup, geocoding, external identity databases, or internet record enrichment; the supplied challenge rules prohibit them. Retain the license and revision of every included model and respect the stated MIT/Apache-2.0 and parameter-limit requirements.
