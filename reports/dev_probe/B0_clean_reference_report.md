# Device 1: Clean Reference Baseline B0 Report (D1-01)

**Date:** 2026-09-26 19:30:10
**Status:** Clean, Uncontaminated Baseline (Zero GT Injection)
**Evaluation Manifest:** `screen_2k` (2,000 queries: India=1,000, US=1,000)

## 1. Verified Metrics & True Loss Decomposition

$$\text{Total Loss} = 1 - F_{0.5}^{\text{final}} = (1 - F_{0.5}^{\text{oracle}}) + (F_{0.5}^{\text{oracle}} - F_{0.5}^{\text{final}})$$

| Metric | Overall | India | US | Target Headroom |
|---|---|---|---|---|
| **Clean Macro $F_{0.5}$** | **0.904586** | **0.875299** | **0.933873** | Gap to >0.98: 0.075414 |
| **Natural Oracle Macro $F_{0.5}$** | **0.985159** | **0.974536** | **0.995783** | Ceiling |
| **Retrieval Loss $(1 - \text{Oracle})$** | **0.014841** | **0.025464** | **0.004217** | Budget: $\le 0.005$ |
| **Matcher Loss $(\text{Oracle} - \text{Final})$** | **0.080574** | **0.099237** | **0.061910** | Budget: $< 0.015$ |

## 2. Threshold Calibration on `calibration_5k`
- **Optimal $T_{\text{singleton}}$:** `0.70`
- **Optimal $T_{\text{match}}$:** `0.70`
- **Calibration Macro $F_{0.5}$:** `0.912060`
- **Calibration Singleton Accuracy:** `227 / 257` correct (`30` false merges, `88.33%` accuracy)
- **Screening (`screen_2k`) Singleton Accuracy:** `85 / 97` correct (`12` false merges, `87.63%` accuracy)

## 3. Top Feature Gains
| Rank | Feature | Importance Gain |
|---|---|---|
| 1 | `rrf_best` | 934,004.42 |
| 2 | `addr_word_jac` | 222,767.26 |
| 3 | `house_equal` | 115,888.08 |
| 4 | `name_partial` | 84,134.15 |
| 5 | `addr_set` | 77,490.92 |
| 6 | `n_channels` | 59,549.44 |
| 7 | `name_sort` | 50,102.32 |
| 8 | `name_len_ratio` | 47,900.62 |
| 9 | `name_wratio` | 44,270.82 |
| 10 | `addr_sort` | 38,068.41 |

## 4. Multi-Device Handoff Status
- **B0 Portable Feature Bundle:** Published to `runs/parallel-v1/d1/b0_baseline/b0_portable_bundle.joblib` (60.1 MB).
- **Clean Model Artifact:** Saved to `cache/models/b0_clean_matcher.txt`.
- **Calibrated Policy:** Saved to `cache/models/b0_calibrated_thresholds.json`.
- **Next Parallel Steps:**
  - **Device 2 (MacBook Air M4):** Can immediately load `b0_portable_bundle.joblib` and run CPU matcher experiments (D2-01 boosting sweeps, hard negative mining, feature ablations) without building multi-million document indices.
  - **Device 3 (RTX 3050):** Can consume candidate sets and benchmark multilingual dense challengers (D3-01).
  - **Device 1 (Windows / Arc 140T):** Can proceed to D1-02 India Lexical Expansion.
