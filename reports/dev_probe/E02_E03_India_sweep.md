# E02 + E03 Retrieval & Structured Expansion Report: India

**Execution Date:** 2026-09-26 12:51:38
**Country:** India | **Probe Queries:** 300 | **Full Pool Size:** 4,133,346 | **Total Runtime:** 1201.9s

## 1. Individual Retrieval Channels

| Channel | Oracle Macro-F0.5 | Pair Recall | Mean K | All Retrieved Rate | Zero Hit Rate | Runtime (s) |
|---|---|---|---|---|---|---|
| `joint` | **0.9351** | 83.63% | 100.0 | 65.96% | 2.11% | 442.0s |
| `name_only` | **0.7966** | 62.65% | 100.0 | 36.84% | 11.23% | 158.7s |
| `address_only` | **0.9405** | 85.42% | 150.0 | 61.75% | 2.11% | 301.4s |
| `structured` | **0.8127** | 69.71% | 56.9 | 44.56% | 13.33% | 0.0s |

## 2. Cumulative Union & Marginal Gains (E02 + E03)

| Stage | Oracle Macro-F0.5 | Pair Recall | Mean K | All Retrieved Rate | Zero Hit Rate |
|---|---|---|---|---|---|
| `joint_only` | **0.9351** | 83.63% | 100.0 | 65.96% | 2.11% |
| `union_joint_name` | **0.9351** | 83.63% | 173.0 | 65.96% | 2.11% |
| `union_all_lexical` | **0.9851** | 96.24% | 310.0 | 88.77% | 0.70% |
| `union_lexical_structured` | **0.9858** | 96.52% | 342.6 | 89.47% | 0.70% |
| `union_all_incl_duplicate_expansion` | **0.9858** | 96.52% | 342.7 | 89.47% | 0.70% |

## 3. Key Findings & Gate Verdict

- **Baseline Analogue (`joint_only`):** Oracle F0.5 = `0.9351`, Pair Recall = `83.63%`, Mean K = `100.0`.
- **Full Union + Duplicate Expansion:** Oracle F0.5 = `0.9858`, Pair Recall = `96.52%`, Mean K = `342.7`.
- **Oracle Delta:** `+0.0507` macro F0.5 gain.
- **Recall Delta:** `+12.89%` pair recall gain.
- **Duplicate Observations:** Recovered `14` candidate expansions from `50598` identical target signatures.
- **Gate Advancement:** Meaningful unique recovery confirmed beyond joint baseline. Advances E02 and E03 to promotion gate.
