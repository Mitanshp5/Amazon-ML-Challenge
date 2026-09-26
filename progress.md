# Amazon ML Challenge: Business Entity Resolution Progress Log

> **26 September 2026 review correction:** the milestone table below is historical and does not establish quality-gate completion. Cache inspection confirmed 283 ground-truth-injected validation pairs across 229/3,000 queries in scaled E07/E09; their reported scores are not clean end-to-end estimates. E09 finished at 0.9037012, below E07's 0.9050761. E06 K100 loses 0.0046627 oracle on the India probe and is not a promoted default. See the [evidence review](reports/dev_probe/F05_PROGRESS_REVIEW_2026-09-26.md), [updated implementation plan](F05_098_IMPLEMENTATION_PLAN.md), and [three-device protocol](reports/experiments/THREE_DEVICE_PROTOCOL.md) for current status and next actions. This correction does not alter the original experiment records below.

**Target Objective:** Entity-Macro $F_{0.5} > 0.98$ on held-out test data  
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
| **D1-02** | D1 (Win) | **India Lexical Expansion** | **NEXT** | Target: India candidate oracle $\ge 0.99$ | Address $K=150 \to 300$, token frequency weighting, Unicode views |
| **D2-01** | D2 (Mac) | **Matcher & Decision Optimization** | **READY FOR DISTRIBUTION** | Target: Close matcher loss (`0.08057` $\to$ `<0.015`) | Boosting sweeps, focal/asymmetric loss, hard negative mining |
| **D3-01** | D3 (RTX) | **Multilingual Dense Challenger** | **READY FOR DISTRIBUTION** | Target: Semantic complement to lexical retrieval | `multilingual-e5-small` fine-tuning on `record_text_provenance.joblib` |
| **Final Holdout** | Locked | **Unbiased Final Verification** | **LOCKED** | Strict blind evaluation of winner | `splits/f05-v1/holdout_unexposed_198351.json` |

---

## 2. Multi-Device Handoff Readiness
- **Device 2 (MacBook Air M4):** Needs only `runs/parallel-v1/d1/b0_baseline/b0_portable_bundle.joblib` (60.1 MB) shared via Drive/Local network. Can immediately train LightGBM matchers without large TSVs or index builds.
- **Device 3 (RTX 3050):** Needs `runs/parallel-v1/d1/b0_baseline/record_text_provenance.joblib` (27.8 MB) and prediction parquets to fine-tune neural encoders.
- **Device 1 (Windows / Arc 140T):** Ready to execute D1-02 India Lexical Expansion.
