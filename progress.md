# Amazon ML Challenge: Business Entity Resolution Progress Log

> **26 September 2026 review correction:** the milestone table below is historical and does not establish quality-gate completion. Cache inspection confirmed 283 ground-truth-injected validation pairs across 229/3,000 queries in scaled E07/E09; their reported scores are not clean end-to-end estimates. E09 finished at 0.9037012, below E07's 0.9050761. E06 K100 loses 0.0046627 oracle on the India probe and is not a promoted default. See the [evidence review](reports/dev_probe/F05_PROGRESS_REVIEW_2026-09-26.md), [updated implementation plan](F05_098_IMPLEMENTATION_PLAN.md), and [three-device protocol](reports/experiments/THREE_DEVICE_PROTOCOL.md) for current status and next actions. This correction does not alter the original experiment records below.

**Current objective:** maximize verified entity-macro F0.5 within practical runtime and memory; no fixed 0.98 gate.

**Latest review:** [F0.5 maximization review](reports/dev_probe/F05_MAXIMIZATION_REVIEW.md). B0 remains the established reference at **0.904586**. A calibration-selected country-threshold challenger reaches **0.906774** on screen_2k, with delta 95% CI **[-0.000155, +0.004699]**; provisional, not deployed. Untrimmed oracle is 0.990452; its actual matcher score is still unmeasured.

**Handoff correction:** the combined `b0_portable_bundle.joblib` referenced below is absent; use the existing `b0_train_features.joblib` and `b0_eval_features.joblib`. Training target IDs/pair mappings still need export before D3 fine-tuning. The locked holdout is assigned in `splits/f05-v1/splits.json`, not the standalone filename shown in the older table.
**Hardware Policy Enforced:**
- **CPU:** Strictly capped to **12 threads / cores** (8 P-cores + 4/6 E-cores; zero LP-E ultra-efficiency core contention).
- **GPU:** **Intel(R) Arc(TM) 140T (16GB)** utilized via OpenVINO (`optimum[openvino]`, FP16, `PERFORMANCE_HINT: THROUGHPUT`, `NUM_STREAMS: AUTO`, batch size 512, throughput ~6,700 texts/sec).
- **Python Environment:** Pinned strictly to `d:\Amazon_ML_Challange\venv\Scripts\python.exe`.

---

## 1. Audited Multi-Device Milestone Tracker

| Milestone | Device | Description | Status | Verified Metrics | Key Artifacts |
|---|---|---|---|---|---|
| **D1-00** | D1 (Win) | **Evaluation Contract Repair** | **COMPLETED** | 19/19 unit tests passing. Zero GT injection. | `splits/f05-v1/parallel-v1/`<br>`candidate_generation.py`<br>`test_evaluation_contract.py` |
| **D1-01** | D1 (Win) | **Clean Baseline B0 & Untrimmed Control** | **ESTABLISHED SCREENING BASELINE** | **Clean Macro $F_{0.5}$:** `0.904586`<br>**K100 Oracle:** `0.985159`<br>**Untrimmed Oracle:** `0.990452`<br>**Retrieval Loss:** `0.014841`<br>**Matcher Loss:** `0.080574` | `runs/parallel-v1/d1/b0_baseline/b0_portable_bundle.joblib`<br>`calibration_predictions.parquet`<br>`screen_predictions.parquet`<br>`record_text_provenance.joblib`<br>`manifest.json` |
| **Decision audit** | D1 (Win) | **Calibration-only country thresholds** | **PROVISIONAL CHALLENGER** | Screen F0.5 `0.906774`, delta `+0.002189`; CI includes zero | `reports/dev_probe/B0_decision_audit.json` |
| **D1-02** | D1 (Win) | **Candidate Depth & India Retrieval** | **NEXT** | Improve actual macro F0.5 at measured cost | Compare India K250/untrimmed with US K100, then address/token views |
| **D2-01** | D2 (Mac) | **Matcher & Decision Optimization** | **EXISTING-FEATURE FITS READY** | Improve paired macro; confirm threshold challenger on larger development | Split train/eval bundles; channel features, hard positives/negatives |
| **D3-01** | D3 (RTX) | **Multilingual Dense Challenger** | **FROZEN SCORING READY; TRAINING EXPORT INCOMPLETE** | Measure complementary final-score gain | Export training target keys/texts before fine-tuning |
| **Final Holdout** | Locked | **Unbiased Final Verification** | **LOCKED** | Strict blind evaluation after winner selection | Holdout assignments in `splits/f05-v1/splits.json` |

---

## 2. Multi-Device Handoff Readiness
- **Device 2 (MacBook Air M4):** Use `b0_train_features.joblib` and `b0_eval_features.joblib` under `runs/parallel-v1/d1/b0_baseline/`, plus the schema/role manifests and baseline policy. Existing-feature LightGBM experiments do not require rebuilding retrieval indexes.
- **Device 3 (RTX 3050):** Text provenance and keyed calibration/screen predictions support frozen encoder scoring. They are not a complete fine-tuning dataset; export training target IDs, labels and referenced texts first.
- **Device 1 (Windows / Arc 140T):** Compare actual end-to-end scores across candidate depths, confirm the country-policy challenger, then pursue India channel improvements. Preserve B0 as rollback.
