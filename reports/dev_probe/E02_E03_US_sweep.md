# E02 + E03 Retrieval & Structured Expansion Report: US

**Execution Date:** 2026-09-26 13:23:28
**Country:** US | **Probe Queries:** 300 | **Full Pool Size:** 6,186,873 | **Total Runtime:** 1588.7s

## 1. Individual Retrieval Channels

| Channel | Oracle Macro-F0.5 | Pair Recall | Mean K | All Retrieved Rate | Zero Hit Rate | Runtime (s) |
|---|---|---|---|---|---|---|
| `joint` | **0.9927** | 97.26% | 100.0 | 90.28% | 0.00% | 624.4s |
| `name_only` | **0.8728** | 74.50% | 100.0 | 48.26% | 6.60% | 250.0s |
| `address_only` | **0.9727** | 93.01% | 150.0 | 78.47% | 1.04% | 275.0s |
| `structured` | **0.7643** | 61.38% | 55.7 | 35.76% | 15.28% | 0.0s |

## 2. Cumulative Union & Marginal Gains (E02 + E03)

| Stage | Oracle Macro-F0.5 | Pair Recall | Mean K | All Retrieved Rate | Zero Hit Rate |
|---|---|---|---|---|---|
| `joint_only` | **0.9927** | 97.26% | 100.0 | 90.28% | 0.00% |
| `union_joint_name` | **0.9927** | 97.26% | 150.7 | 90.28% | 0.00% |
| `union_all_lexical` | **0.9985** | 99.53% | 295.6 | 98.26% | 0.00% |
| `union_lexical_structured` | **0.9985** | 99.53% | 331.8 | 98.26% | 0.00% |
| `union_all_incl_duplicate_expansion` | **0.9985** | 99.53% | 332.0 | 98.26% | 0.00% |

## 3. Key Findings & Gate Verdict

- **Baseline Analogue (`joint_only`):** Oracle F0.5 = `0.9927`, Pair Recall = `97.26%`, Mean K = `100.0`.
- **Full Union + Duplicate Expansion:** Oracle F0.5 = `0.9985`, Pair Recall = `99.53%`, Mean K = `332.0`.
- **Oracle Delta:** `+0.0057` macro F0.5 gain.
- **Recall Delta:** `+2.27%` pair recall gain.
- **Duplicate Observations:** Recovered `43` candidate expansions from `36768` identical target signatures.
- **Gate Advancement:** Meaningful unique recovery confirmed beyond joint baseline. Advances E02 and E03 to promotion gate.
