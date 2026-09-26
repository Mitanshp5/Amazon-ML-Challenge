# Scaled Supervised Matcher Training & Exact Macro F0.5 Report

**Execution Date:** 2026-09-26 16:49:37
**Total Runtime:** 12.5s | **CPU Cores Used:** 12 (P/E cores only) | **Total Pairs Trained:** 1,501,632 | **Features:** 23

## 1. Out-of-Fold Validation Performance

- **Optimal Policy:** $T_{\text{singleton}} = 0.60$, $T_{\text{match}} = 0.60$
- **Exact Out-of-Fold Macro-F0.5:** **`0.9051`**
- **Pair Precision:** `94.41%`
- **Pair Recall:** `85.72%`
- **Singleton False-Merge Rate:** `17.53%` (`27` out of `154` singletons)
- **1-to-1 Target Conflict Disambiguation Macro-F0.5:** **`0.9051`** (Precision: `94.42%`)
- **Model Checkpoint:** [`cache/models/lgbm_matcher_v2.txt`](file:///d:/Amazon_ML_Challange/cache/models/lgbm_matcher_v2.txt)
- **Calibrated Policy Checkpoint:** [`cache/models/calibrated_thresholds.json`](file:///d:/Amazon_ML_Challange/cache/models/calibrated_thresholds.json)

## 2. Top-10 Dual-Threshold Operating Points

| Rank | $T_{\text{singleton}}$ | $T_{\text{match}}$ | Exact Macro-F0.5 |
|---|---|---|---|
| 1 | **0.60** | **0.60** | **`0.9051`** |
| 2 | **0.65** | **0.65** | **`0.9045`** |
| 3 | **0.65** | **0.60** | **`0.9043`** |
| 4 | **0.70** | **0.65** | **`0.9038`** |
| 5 | **0.60** | **0.55** | **`0.9036`** |
| 6 | **0.70** | **0.60** | **`0.9035`** |
| 7 | **0.70** | **0.70** | **`0.9032`** |
| 8 | **0.65** | **0.55** | **`0.9027`** |
| 9 | **0.55** | **0.55** | **`0.9024`** |
| 10 | **0.70** | **0.55** | **`0.9018`** |

## 3. Top-10 Feature Importances (Gain)

| Rank | Feature | Total Gain | Description |
|---|---|---|---|
| 1 | `rrf_best` | 1,998,474.9 | Schema-pinned feature |
| 2 | `addr_word_jac` | 316,557.4 | Schema-pinned feature |
| 3 | `house_equal` | 161,329.6 | Schema-pinned feature |
| 4 | `addr_set` | 120,784.8 | Schema-pinned feature |
| 5 | `name_partial` | 115,978.2 | Schema-pinned feature |
| 6 | `house_conflict` | 112,147.8 | Schema-pinned feature |
| 7 | `name_wratio` | 95,215.4 | Schema-pinned feature |
| 8 | `name_sort` | 73,857.3 | Schema-pinned feature |
| 9 | `tfidf_max` | 67,787.7 | Schema-pinned feature |
| 10 | `name_len_ratio` | 61,783.2 | Schema-pinned feature |

## 4. Key Takeaways & Gate Promotion

- **Previous Baseline Model OOF Score:** `0.8886` (reported in `model_config.json`, 18 features).
- **Scaled Matcher OOF Score:** **`0.9051`** (conflict resolved: **`0.9051`**).
- **Schema Assertion:** Model artifact asserts exact 23 feature names and ordering at inference.
- **CPU Affinity:** Capped strictly to 12 cores (8P + 4E) avoiding ultra-efficiency cores.
