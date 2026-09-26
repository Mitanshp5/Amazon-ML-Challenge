# E08 Dual-Threshold Singleton Calibration Report

**Execution Date:** 2026-09-26 14:31:06
**Validation Queries:** 600 | **Total Runtime:** 0.45s

## 1. Baseline vs Dual-Threshold Policy Comparison

| Policy | $T_{\text{singleton}}$ | $T_{\text{match}}$ | Exact Macro-F0.5 | Pair Precision | Pair Recall | Singleton False-Merges | FM Rate |
|---|---|---|---|---|---|---|---|
| **Single Threshold (Baseline)** | 0.60 | 0.60 | `0.9208` | 96.13% | 87.17% | 4/34 | 11.76% |
| **Dual Threshold (Optimal)** | **0.65** | **0.60** | **`0.9224`** | **96.13%** | **87.17%** | **3/34** | **8.82%** |
| **Delta** | — | — | **`+0.0017`** | — | — | **-1** | — |

## 2. Top-10 Dual-Threshold Operating Points

| Rank | $T_{\text{singleton}}$ | $T_{\text{match}}$ | Exact Macro-F0.5 | Pair Precision | Pair Recall | Singleton FM Rate |
|---|---|---|---|---|---|---|
| 1 | **0.65** | **0.60** | **0.9224** | 96.13% | 87.17% | 8.82% |
| 2 | **0.65** | **0.55** | **0.9215** | 95.27% | 88.95% | 8.82% |
| 3 | **0.65** | **0.65** | **0.9212** | 96.70% | 85.87% | 8.82% |
| 4 | **0.60** | **0.60** | **0.9208** | 96.13% | 87.17% | 11.76% |
| 5 | **0.55** | **0.55** | **0.9198** | 95.27% | 88.95% | 11.76% |
| 6 | **0.60** | **0.55** | **0.9198** | 95.27% | 88.95% | 11.76% |
| 7 | **0.50** | **0.50** | **0.9185** | 94.41% | 90.10% | 11.76% |
| 8 | **0.65** | **0.50** | **0.9185** | 94.41% | 90.00% | 8.82% |
| 9 | **0.50** | **0.45** | **0.9183** | 93.84% | 91.45% | 11.76% |
| 10 | **0.65** | **0.45** | **0.9183** | 93.83% | 91.35% | 8.82% |

## 3. Key Takeaways & Gate Promotion

- **Singleton Protection:** By setting $T_{\text{singleton}} = 0.65$, false merges among unassigned singletons are suppressed, protecting the macro average.
- **Recall Retention:** Setting $T_{\text{match}} = 0.60$ ensures non-singletons retain high recall for confident secondary pairs.
- **Calibrated Policy Checkpoint:** [`cache/models/calibrated_thresholds.json`](file:///d:/Amazon_ML_Challange/cache/models/calibrated_thresholds.json).
