# L02: Comparison 15k Confirmation Report

**Date:** 2026-09-26 23:25:44
**Hardware:** Windows / 12 CPU threads
**Candidate Set:** Natural K100 candidates (100% label-blind)
**Total Queries Evaluated:** 15,000 (`comparison_15k`)
**Partitions:**
- `screen_2k`: 2,000 queries (screening subset)
- `unexposed_13k`: 13,000 queries (strictly out-of-screen development)

---

## 1. Full Comparison (15,000 Queries) Results

| Model | Policy | Full 15k F0.5 | India F0.5 | US F0.5 | Delta vs B0 Ref | 95% Bootstrap CI |
|---|---|---|---|---|---|---|
| **control_b0** | baseline (0.70/0.70) | 0.909473 | 0.885822 | 0.933124 | Ref | - |
| **control_b0** | country_dual | **0.910485** | 0.885539 | 0.935431 | **+0.001012** | [+0.000073, +0.002048] |
| **capacity_127** | baseline (0.70/0.70) | 0.908700 | 0.883249 | 0.934152 | -0.000773 | [-0.002052, +0.000488] |
| **capacity_127** | country_dual | **0.909067** | 0.882780 | 0.935354 | **-0.000406** | [-0.001745, +0.000980] |
| **regularized_127** | baseline (0.70/0.70) | 0.909870 | 0.885348 | 0.934392 | +0.000397 | [-0.000882, +0.001685] |
| **regularized_127** | country_dual | **0.910161** | 0.885562 | 0.934761 | **+0.000688** | [-0.000731, +0.002087] |

---

## 2. Unexposed Development (13,000 Queries) Results

| Model | Policy | Unexp 13k F0.5 | India F0.5 | US F0.5 | Delta vs B0 Ref | 95% Bootstrap CI |
|---|---|---|---|---|---|---|
| **control_b0** | baseline (0.70/0.70) | 0.910225 | 0.887441 | 0.933009 | Ref | - |
| **control_b0** | country_dual | **0.911056** | 0.887114 | 0.934998 | **+0.000831** | [-0.000244, +0.001926] |
| **capacity_127** | baseline (0.70/0.70) | 0.909096 | 0.884076 | 0.934116 | -0.001129 | [-0.002537, +0.000308] |
| **capacity_127** | country_dual | **0.909354** | 0.883663 | 0.935045 | **-0.000872** | [-0.002366, +0.000603] |
| **regularized_127** | baseline (0.70/0.70) | 0.910521 | 0.886378 | 0.934664 | +0.000296 | [-0.001024, +0.001777] |
| **regularized_127** | country_dual | **0.910689** | 0.886625 | 0.934753 | **+0.000464** | [-0.001022, +0.001914] |

---

## 3. Screening Parity Check (2,000 Queries)
- **control_b0 baseline:** `0.904586` (Expected: 0.904586)
- **control_b0 country_dual:** `0.906774` (Expected: 0.906774)

Total execution time: 2363.37s.
