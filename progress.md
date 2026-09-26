# Amazon ML Challenge: Implementation Progress Log

**Last Updated:** 2026-09-26 14:27 IST  
**Repository Branch:** `main`  
**Execution Environment:** `d:\Amazon_ML_Challange\venv`  
**Hardware Detected & Active:**
- **CPU:** 16 Cores @ 3.99 GHz
- **GPU:** Intel(R) Arc(TM) 140T GPU (16GB shared VRAM)
- **NPU:** Intel(R) AI Boost
- **Runtime:** Intel OpenVINO 2026.4.0 + HuggingFace Optimum Intel + PyTorch 2.14.0 + LightGBM 4.7.0

---

## 1. Experiment Status Overview

| Experiment | Phase | Description | Status | Key Results / Metrics |
|---|---|---|---|---|
| **E00** | Phase A | Historical baseline freeze & fresh grouped splits | **DONE** | 22 baseline artifacts locked in `runs/historical_2026-09-26_baseline/`; fresh seed `f05-v1` splits (dev, holdout, train, 20k probe). |
| **E01** | Phase B | Unicode, Indic & French normalization parity | **DONE** | Unicode primary views, Latin folding extra view, field-dispatched `ste`, legal-vs-street separation. 10/10 pytest passing. |
| **E02** | Phase C | Independent lexical channels (joint, name, address) | **DONE** | Evaluated on full target pools (India 4.13M, US 6.18M). Measured significant complementary recall over joint alone. |
| **E03** | Phase C | Structured compound keys & duplicate observation expansion | **DONE** | **India:** Oracle F0.5 = **0.9858**, Recall = **96.52%**.<br>**US:** Oracle F0.5 = **0.9985**, Recall = **99.53%**, Zero-hit = **0.00%**. |
| **E04** | Phase D | OpenVINO Intel Arc 140T GPU Dense Retriever | **DONE** | OpenVINO GPU integration verified @ **928.2 texts/sec**. Precompiled IR saved to `cache/models/all-MiniLM-L6-v2_openvino`. Attention-mask pooling + Faiss IndexFlatIP implemented. |
| **E06** | Phase C | Candidate compression & RRF budget sweep | **DONE** | Fused multi-channel candidates via RRF; swept $K=40..250$. $K=100$ retains **0.9812** oracle on India while slashing candidate volume by 65%. |
| **E07** | Phase E | Supervised LightGBM Matcher (23 schema-pinned features) | **DONE** | **Exact OOF Macro-F0.5 = 0.9009** (up from baseline 0.8886). Pair Precision = **94.43%**, Recall = **84.72%**. Optimal threshold = **0.60**. |
| **E08** | Phase F | Dual threshold calibration & singleton modeling | *Ready* | Calibrate singleton gate ($T_{\text{singleton}}$) vs match gate ($T_{\text{match}}$) to suppress the 11.76% false merge rate. |
| **E12** | Phase G | Test inference streaming, validator verification & packaging | *Queued* | Final test set streaming prediction and submission package generator. |

---

## 2. Detailed Technical Findings

### 2.1 E02 & E03 Full-Pool Retrieval Sweeps
- **Parsing Bug Fix:** Caught and resolved unparsed comma-separated target IDs in ground truth loading (`load_gt_map`), which previously masked true recall numbers.
- **India Pool (4,133,346 records):**
  - Baseline `joint_only`: Oracle F0.5 = 0.9351, Pair Recall = 83.63%.
  - Full union + duplicate expansion: Oracle F0.5 = **0.9858** ($+0.0507$), Pair Recall = **96.52%** ($+12.89\%$).
  - All-true-matches-retrieved rate surged from 65.96% to **89.47%**.
- **US Pool (6,186,873 records):**
  - Baseline `joint_only`: Oracle F0.5 = 0.9927, Pair Recall = 97.26%.
  - Full union + duplicate expansion: Oracle F0.5 = **0.9985** ($+0.0058$), Pair Recall = **99.53%** ($+2.27\%$).
  - Zero-hit rate: **0.00%** (100% of tested entities had at least one true match retrieved).

### 2.2 E04 Intel Arc 140T GPU Acceleration
- Verified hardware offload with `OpenVINO 2026.4` on `Intel(R) Arc(TM) 140T GPU (16GB)`.
- Batch benchmark: 1,000 dense entity embeddings processed in **1.077 seconds** (**928 texts/sec**).
- Attention-mask aware mean pooling implemented to eliminate padding bias.

### 2.3 E06 Compression Frontier (India)
- $K=40$: Oracle F0.5 = 0.9673, Recall = 91.16%
- $K=60$: Oracle F0.5 = 0.9778, Recall = 93.89%
- $K=80$: Oracle F0.5 = 0.9786, Recall = 94.26%
- $K=100$: Oracle F0.5 = **0.9812**, Recall = **94.92%** (Recommended balance)
- $K=150$: Oracle F0.5 = **0.9835**, Recall = **95.67%**
- $K=200$: Oracle F0.5 = **0.9852**, Recall = **96.24%**
- $K=\text{untrimmed}$ ($K=342.7$): Oracle F0.5 = **0.9858**, Recall = **96.52%**

### 2.4 E07 Supervised Matcher (LightGBM)
- **Training Data:** 360,385 candidate pairs generated via multi-channel candidate retrieval across India and US.
- **Cross-Validation:** 3-fold GroupKFold by S1 entity group.
- **Exact Out-of-Fold Macro-F0.5:** **`0.9009`** at optimal threshold `0.60` (surpassing the previous `0.8886` baseline ceiling).
- **Pair Precision:** `94.43%` | **Pair Recall:** `84.72%`.
- **Top Features by Gain:**
  1. `rrf_best` (480,610 gain) — Multi-channel fusion rank
  2. `addr_word_jac` (79,558 gain) — Component address overlap
  3. `house_equal` (39,116 gain) — Parsed house number match
  4. `name_partial` (30,585 gain) — Partial business name match
  5. `addr_set` (27,661 gain) — Address token set ratio
  6. `house_conflict` (26,448 gain) — Conflicting house number penalty
- **Artifact:** Model saved and schema-asserted at `cache/models/lgbm_matcher_v2.txt`.
