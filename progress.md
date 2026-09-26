# Amazon ML Challenge: Business Entity Resolution Progress Log

**Active plan:** [local-first execution, L00–L07](LOCAL_COLAB_IMPLEMENTATION_PLAN.md), with optional parallel [Colab A100 experiments, G00–G04](reports/experiments/COLAB_A100_RUNBOOK.md). All Mac/RTX device assignments are retired. No fixed 0.98 target is required for promotion.

**Latest verified progress (after commit `10132b1`):** all seven bundle SHA-256 values match; `train_pairs.parquet` has 1.2M unique pairs whose query order, labels, folds and RRF match the existing training matrix. All keyed training pairs have query/target text coverage. The neural-training mapping blocker is resolved. No later model-score improvement is reported. Existing in-progress `io.py` changes are preserved.

> **26 September 2026 review correction:** the milestone table below is historical and does not establish quality-gate completion. Cache inspection confirmed 283 ground-truth-injected validation pairs across 229/3,000 queries in scaled E07/E09; their reported scores are not clean end-to-end estimates. E09 finished at 0.9037012, below E07's 0.9050761. E06 K100 loses 0.0046627 oracle on the India probe and is not a promoted default. See the [evidence review](reports/dev_probe/F05_PROGRESS_REVIEW_2026-09-26.md), [updated implementation plan](F05_098_IMPLEMENTATION_PLAN.md), and [three-device protocol](reports/experiments/THREE_DEVICE_PROTOCOL.md) for current status and next actions. This correction does not alter the original experiment records below.

**Current objective:** maximize verified entity-macro F0.5 within practical runtime and memory; no fixed 0.98 gate.

**Latest review:** [F0.5 maximization review](reports/dev_probe/F05_MAXIMIZATION_REVIEW.md). B0 remains the established reference at **0.904586**. A calibration-selected country-threshold challenger reaches **0.906774** on screen_2k, with delta 95% CI **[-0.000155, +0.004699]**; provisional, not deployed. Untrimmed oracle is 0.990452; its actual matcher score is still unmeasured.

**Current artifacts:** use `b0_train_features.joblib`, `b0_eval_features.joblib`, `train_pairs.parquet`, `train_text_records.joblib` and `eval_text_records.joblib`. The old monolithic feature/text filenames are absent; update remaining consumers before rerunning them. Keep the 852 diagnostic unretrieved targets separate from natural evaluation candidates. Locked holdout assignments are in `splits/f05-v1/splits.json`.
**Hardware Policy Enforced:**
- **CPU:** Strictly capped to **12 threads / cores** (8 P-cores + 4/6 E-cores; zero LP-E ultra-efficiency core contention).
- **GPU:** **Intel(R) Arc(TM) 140T (16GB)** utilized via OpenVINO (`optimum[openvino]`, FP16, `PERFORMANCE_HINT: THROUGHPUT`, `NUM_STREAMS: AUTO`, batch size 512, throughput ~6,700 texts/sec).
- **Python Environment:** Pinned strictly to `d:\Amazon_ML_Challange\venv\Scripts\python.exe`.

---

## 1. Milestone Tracker (all required work local)

| Milestone | Device | Description | Status | Verified Metrics | Key Artifacts |
|---|---|---|---|---|---|
| **D1-00** | D1 (Win) | **Evaluation Contract Repair** | **COMPLETED** | 19/19 unit tests passing. Zero GT injection. | `splits/f05-v1/parallel-v1/`<br>`candidate_generation.py`<br>`test_evaluation_contract.py` |
| **D1-01** | Local | **Clean Baseline B0 & Untrimmed Control** | **ESTABLISHED SCREENING BASELINE** | Actual `0.904586`; K100 oracle `0.985159`; untrimmed oracle `0.990452` | Split feature/text bundles, keyed pairs, prediction parquets and `manifest.json` under `runs/parallel-v1/d1/b0_baseline/` |
| **Decision audit** | D1 (Win) | **Calibration-only country thresholds** | **PROVISIONAL CHALLENGER** | Screen F0.5 `0.906774`, delta `+0.002189`; CI includes zero | `reports/dev_probe/B0_decision_audit.json` |
| **L00** | Local | **Split-reader and isolated-run compatibility** | **NEXT** | Preserve verified artifacts; align remaining old filename consumers | Active plan L00 |
| **L01/L02** | Local | **Matcher screen and larger comparison** | **READY AFTER L00** | Bounded cached-feature fits; confirm on full 15k and non-screen 13k | Existing matrices/folds; reusable comparison features |
| **L03/L05** | Local | **Candidate depth and India retrieval** | **QUEUED** | Actual macro/cost across K100, India K250/untrimmed, then new channels | Current depth frontier is oracle-only |
| **L04** | Local | **Features, hard examples and identity scaling** | **QUEUED** | Improve paired macro with stable slice behavior | Versioned schemas; train-only mining |
| **G00/G01** | Optional Colab A100 | **GPU benchmark and frozen multilingual features** | **DATA VERIFIED; RUNNER TO PREPARE** | No A100 result yet | Keyed pair/text bundle; pinned cloud export |
| **G02–G04 / L06** | Optional A100 + local integration | **Task-trained neural challenger and fusion** | **CONDITIONAL** | Retain only measured complementary gains | OOF or separate combiner training; local calibration |
| **Final Holdout** | Locked | **Unbiased Final Verification** | **LOCKED** | Strict blind evaluation after winner selection | Holdout assignments in `splits/f05-v1/splits.json` |

---

## 2. Local and optional cloud execution

- **Local:** owns all source, splits, retrieval, CPU training, calibration, selection, holdout and final output validation. Start L00, then the cached-feature screen and reusable comparison cache.
- **Optional A100:** can encode/train in parallel on a pinned export after a device/memory benchmark. Return keyed scores and model/provenance artifacts for local evaluation. Local work continues if Colab is unavailable.
- **Training export:** verified row alignment and complete text coverage. Filter the 19k query-text map by train_12k for supervision; diagnostic evaluation targets never become candidate additions.
- **Runtime:** measured B0 total is 48.23 minutes, dominated by retrieval. Earlier Mac/RTX estimates do not apply to this plan. No project A100 benchmark exists; use G00 to estimate encoding/training time before long runs.
