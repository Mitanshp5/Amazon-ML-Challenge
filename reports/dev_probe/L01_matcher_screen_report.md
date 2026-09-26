# L01: Local Matcher Screen Report

**Date:** 2026-09-26 21:59:49
**Hardware:** Local Windows / 12 CPU threads
**Baseline Reference:** B0 screening macro $F_{0.5} = 0.904586$ (global 0.70/0.70)
**Candidate Set:** Frozen natural candidates on `screen_2k` (Zero retrieval rerun)

## 1. Capacity Screen Results

| Arm | Leaves | Depth | MinChild | Trees | Calib F0.5 | Screen F0.5 (Base 0.70) | Screen F0.5 (Country Dual) | India F0.5 | US F0.5 | Delta vs B0 Ref | Train Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **control_b0** | 63 | 7 | 80 | 743 | 0.914114 | 0.904586 | **0.906774** | 0.875299 | 0.938250 | **+0.002189** | 45.4s |
| **compact_31** | 31 | 7 | 100 | 1166 | 0.912179 | 0.905429 | **0.906533** | 0.875653 | 0.937413 | **+0.001947** | 61.3s |
| **capacity_127** | 127 | 9 | 80 | 431 | 0.912470 | 0.906131 | **0.907269** | 0.877717 | 0.936822 | **+0.002684** | 39.7s |
| **regularized_127** | 127 | 9 | 200 | 483 | 0.914510 | 0.905637 | **0.906958** | 0.878660 | 0.935256 | **+0.002372** | 44.4s |

## 2. Key Observations
- All models were fitted using 3-fold inner cross-validation on `train_12k` (1.2M pairs).
- Early stopping patience was set to 100 with a 2,000-tree ceiling.
- Thresholds were calibrated strictly on `calibration_5k`, completely disjoint from `screen_2k`.

Total execution time: 230.49s.
