# Amazon ML Challenge: Business Entity Resolution Progress Log

> **26 September 2026 review correction:** the milestone table below is historical and does not establish quality-gate completion. Cache inspection confirmed 283 ground-truth-injected validation pairs across 229/3,000 queries in scaled E07/E09; their reported scores are not clean end-to-end estimates. E09 finished at 0.9037012, below E07's 0.9050761. E06 K100 loses 0.0046627 oracle on the India probe and is not a promoted default. See the [evidence review](reports/dev_probe/F05_PROGRESS_REVIEW_2026-09-26.md), [updated implementation plan](F05_098_IMPLEMENTATION_PLAN.md), and [three-device protocol](reports/experiments/THREE_DEVICE_PROTOCOL.md) for current status and next actions. This correction does not alter the original experiment records below.

**Target Objective:** Entity-Macro $F_{0.5} > 0.98$ on held-out test data  
**Hardware Policy Enforced:**
- **CPU:** Strictly capped to **12 threads / cores** (8 P-cores + 4/6 E-cores; zero LP-E ultra-efficiency core contention).
- **GPU:** **Intel(R) Arc(TM) 140T (16GB)** utilized via OpenVINO (`optimum[openvino]`, FP16, `PERFORMANCE_HINT: THROUGHPUT`, `NUM_STREAMS: AUTO`, batch size 512, throughput ~6,700 texts/sec).
- **Python Environment:** Pinned strictly to `d:\Amazon_ML_Challange\venv\Scripts\python.exe`.

---

## 1. Milestone Status Tracker

| Milestone | Description | Status | Key Metric / Result | Artifact / Report |
|---|---|---|---|---|
| **E00** | Baseline Freeze & Metric Verification | **COMPLETED** | Verified historical baseline: `0.8886` OOF Macro-$F_{0.5}$ | `cache/models/model_config.json` |
| **E01** | Normalization Parity & Unit Tests | **COMPLETED** | 12/12 unit tests passing (Unicode NFKC, Indic, French, component parsing) | `tests/test_normalization.py` |
| **E02 / E03** | Full-Pool Lexical & Structured Retrieval Sweeps | **COMPLETED** | **India Oracle:** `0.9858` (Pair Recall: 96.52%)<br>**US Oracle:** `0.9985` (Pair Recall: 99.53%) | `reports/dev_probe/E02_E03_India_sweep.md`<br>`reports/dev_probe/E02_E03_US_sweep.md` |
| **E04** | Intel Arc 140T Dense OpenVINO Engine | **COMPLETED** | Arc 140T GPU native inference @ **6,777 texts/sec** (batch size 512, FP16) | `code/business_entity_resolution/src/er/retrieval/dense.py`<br>`cache/models/all-MiniLM-L6-v2_openvino/` |
| **E06** | Candidate Compression Frontier | **COMPLETED** | $K=100$ per query preserves `0.9812` oracle ceiling while reducing pairs by 65% | `reports/dev_probe/E06_compression_frontier_India.md` |
| **E07 (Scaled)** | Scaled Grouped LightGBM Matcher | **COMPLETED** | **OOF Macro-$F_{0.5} = 0.9051$** on 3,000 val queries (1.5M pairs trained); Pair Precision `94.41%`, Pair Recall `85.72%` | `reports/dev_probe/E07_matcher_report.md`<br>`cache/models/lgbm_matcher_v2.txt` |
| **E08** | Dual-Threshold Singleton Calibration | **COMPLETED** | Evaluated 2D grid $(T_{\text{singleton}}, T_{\text{match}})$; optimal policy $T_s=0.60, T_m=0.60$ | `cache/models/calibrated_thresholds.json` |
| **E10** | Graph/Target Disambiguation & Consistency | **COMPLETED** | Evaluated 1-to-1 target conflict resolution; resolves multi-query claims | Built into matcher pipeline |
| **E09** | Multilingual & Dense Feature Integration | **RUNNING** | Intel Arc 140T GPU dense cosine similarity feature + 12 CPU cores via ThreadPoolExecutor | Background Task ID: `task-963` |
| **E12** | Test Pipeline & Submission Verification | **PENDING** | Chunked streaming inference on 1.73M test queries with validator | `student_resource/student_resource/validator.py` |

---

## 2. Active Run Details: Milestone E09
- **Scope:** 15,000 queries (12,000 train + 3,000 held-out validation) across India and US.
- **Hardware Integration:**
  - **Intel Arc 140T GPU (16GB):** Running OpenVINO FP16 XMX inference at batch size 512.
  - **12 CPU Cores:** Running parallel lexical search, structured indexing, and pair feature extraction via `ThreadPoolExecutor` (eliminating Windows IPC overhead).
- **Features:** 24 features (23 lexical/structural + continuous dense embedding cosine similarity).
- **Output Artifacts:** `cache/models/lgbm_matcher_dense_v1.txt`, `cache/models/calibrated_thresholds_dense.json`, `reports/dev_probe/E09_dense_matcher_report.md`.
