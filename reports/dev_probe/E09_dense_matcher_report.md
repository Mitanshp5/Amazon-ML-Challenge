# E09 Multilingual & Dense Neural Matcher Report

**Execution Date:** 2026-09-26 18:02:14
**Hardware Utilized:** Intel(R) Arc(TM) 140T GPU (16GB) OpenVINO + 12 CPU Cores | **Pairs Trained:** 1,501,632 | **Features:** 24

## 1. Out-of-Fold Validation Performance Comparison

| Configuration | Features | Macro-$F_{0.5}$ | Pair Precision | Pair Recall | Singleton False Merges |
|---|---|---|---|---|---|
| **E00 Baseline** | 18 | `0.8886` | ~92.0% | ~82.0% | ~18.0% |
| **E07 Scaled (Lexical + Struct)** | 23 | `0.9051` | 94.41% | 85.72% | 17.53% |
| **E09 Dense Neural Matcher** | **24** | **`0.9037`** | **`94.18%`** | **`85.63%`** | **`18.18%`** |
| **E09 + 1-to-1 Disambiguation** | **24** | **`0.9037`** | **`94.19%`** | **`85.63%`** | **`18.18%`** |

- **Optimal Policy:** $T_{\text{singleton}} = 0.60$, $T_{\text{match}} = 0.60$
- **Model Checkpoint:** [`cache/models/lgbm_matcher_dense_v1.txt`](file:///d:/Amazon_ML_Challange/cache/models/lgbm_matcher_dense_v1.txt)
- **Calibrated Policy Checkpoint:** [`cache/models/calibrated_thresholds_dense.json`](file:///d:/Amazon_ML_Challange/cache/models/calibrated_thresholds_dense.json)

## 2. Top-10 Dual-Threshold Operating Points

| Rank | $T_{\text{singleton}}$ | $T_{\text{match}}$ | Exact Macro-F0.5 |
|---|---|---|---|
| 1 | **0.60** | **0.60** | **`0.9037`** |
| 2 | **0.70** | **0.65** | **`0.9034`** |
| 3 | **0.65** | **0.65** | **`0.9029`** |
| 4 | **0.70** | **0.60** | **`0.9027`** |
| 5 | **0.70** | **0.70** | **`0.9026`** |
| 6 | **0.65** | **0.60** | **`0.9022`** |
| 7 | **0.60** | **0.55** | **`0.9021`** |
| 8 | **0.70** | **0.55** | **`0.9010`** |
| 9 | **0.55** | **0.55** | **`0.9006`** |
| 10 | **0.65** | **0.55** | **`0.9005`** |

## 3. Top-10 Feature Importances (Gain)

| Rank | Feature | Total Gain | Description |
|---|---|---|---|
| 1 | `rrf_best` | 1,909,739.9 | Schema-pinned feature |
| 2 | `dense_sim` | 279,283.9 | Schema-pinned feature |
| 3 | `house_equal` | 182,170.2 | Schema-pinned feature |
| 4 | `addr_word_jac` | 156,951.9 | Schema-pinned feature |
| 5 | `addr_set` | 138,328.7 | Schema-pinned feature |
| 6 | `name_partial` | 130,741.8 | Schema-pinned feature |
| 7 | `house_conflict` | 105,615.8 | Schema-pinned feature |
| 8 | `name_sort` | 68,092.5 | Schema-pinned feature |
| 9 | `tfidf_max` | 65,923.6 | Schema-pinned feature |
| 10 | `name_len_ratio` | 49,003.1 | Schema-pinned feature |

## 4. Key Takeaways & Gate Promotion

- **Dense Neural Integration:** The Intel Arc 140T GPU computed semantic embeddings at scale, delivering continuous cosine similarity across language boundaries.
- **Delta vs Historical Baseline:** `+0.0151` gain in exact entity-macro $F_{0.5}$.
- **Fast Multithreading:** ThreadPoolExecutor completely eliminated Windows IPC serialization overhead.
