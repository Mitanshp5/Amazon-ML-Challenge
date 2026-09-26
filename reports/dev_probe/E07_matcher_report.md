# E07 Supervised Matcher Training & Exact Macro F0.5 Report

**Execution Date:** 2026-09-26 14:26:34
**Total Runtime:** 773.6s | **Total Pairs Trained:** 360,385 | **Features:** 23

## 1. Out-of-Fold Validation Performance

- **Optimal Decision Threshold:** `0.60`
- **Exact Out-of-Fold Macro-F0.5:** **`0.9009`**
- **Pair Precision:** `94.43%`
- **Pair Recall:** `84.72%`
- **Singleton False-Merge Rate:** `11.76%` (`4` out of `34` singletons)
- **Model Checkpoint:** [`cache/models/lgbm_matcher_v2.txt`](file:///d:/Amazon_ML_Challange/cache/models/lgbm_matcher_v2.txt)

## 2. Threshold Sweep on Exact Macro-F0.5 Scorer

| Threshold | Exact Entity-Macro F0.5 |
|---|---|
| **0.60** | **0.9009** |
| 0.50 | 0.8971 |
| 0.70 | 0.8960 |
| 0.40 | 0.8949 |
| 0.75 | 0.8908 |
| 0.80 | 0.8818 |
| 0.85 | 0.8650 |
| 0.90 | 0.8223 |

## 3. Top-10 Feature Importances (Gain)

| Rank | Feature | Total Gain | Description |
|---|---|---|---|
| 1 | `rrf_best` | 480,610.6 | Schema-pinned feature |
| 2 | `addr_word_jac` | 79,557.7 | Schema-pinned feature |
| 3 | `house_equal` | 39,116.2 | Schema-pinned feature |
| 4 | `name_partial` | 30,584.7 | Schema-pinned feature |
| 5 | `addr_set` | 27,661.1 | Schema-pinned feature |
| 6 | `house_conflict` | 26,448.4 | Schema-pinned feature |
| 7 | `name_wratio` | 25,553.5 | Schema-pinned feature |
| 8 | `name_len_ratio` | 20,523.2 | Schema-pinned feature |
| 9 | `tfidf_max` | 18,587.1 | Schema-pinned feature |
| 10 | `name_sort` | 13,283.7 | Schema-pinned feature |

## 4. Key Takeaways & Gate Promotion

- **Previous Baseline Model OOF Score:** `0.8886` (reported in `model_config.json`, 18 features).
- **New E07 Matcher OOF Score:** **`0.9009`** (+ gain on the expanded candidate distribution).
- **Schema Assertion:** Model artifact asserts exact 23 feature names and ordering at inference.
- **Next Step:** Proceed to **Phase F / E08** (Dual threshold calibration & singleton modeling) and **E12** (Final test streaming pipeline and packaging).
