# E06 Candidate Compression Frontier Report: India

**Execution Date:** 2026-09-26 13:46:23
**Country:** India | **Probe Queries:** 300 | **Total Runtime:** 147.0s

## Compression / Oracle Trade-off Frontier

| Budget $K$ | Mean Candidates | Oracle Macro-F0.5 | Oracle Loss vs Untrimmed | Pair Recall | All Retrieved Rate | Zero Hit Rate | Lost GT Links |
|---|---|---|---|---|---|---|---|
| **40** | 40.0 | **0.9673** | `-0.0186` | 91.16% | 76.49% | 1.05% | 57 |
| **50** | 50.0 | **0.9740** | `-0.0119` | 92.47% | 78.95% | 0.70% | 43 |
| **60** | 60.0 | **0.9778** | `-0.0080` | 93.89% | 81.75% | 0.70% | 28 |
| **80** | 80.0 | **0.9786** | `-0.0072` | 94.26% | 82.81% | 0.70% | 24 |
| **100** | 100.0 | **0.9812** | `-0.0047` | 94.92% | 84.91% | 0.70% | 17 |
| **120** | 120.0 | **0.9820** | `-0.0038` | 95.20% | 85.61% | 0.70% | 14 |
| **150** | 150.0 | **0.9835** | `-0.0023` | 95.67% | 86.67% | 0.70% | 9 |
| **200** | 200.0 | **0.9852** | `-0.0006` | 96.24% | 88.42% | 0.70% | 3 |
| **250** | 250.0 | **0.9854** | `-0.0004` | 96.33% | 88.77% | 0.70% | 2 |
| **untrimmed** | 342.7 | **0.9858** | `+0.0000` | 96.52% | 89.47% | 0.70% | 0 |

## Key Insights & Recommended Operating Point

- **Untrimmed Union:** Oracle F0.5 = `0.9858`, Recall = `96.52%`, Mean K = `342.7`.
- **Budget K=100:** Oracle F0.5 = `0.9812`, Recall = `94.92%`, Mean K = `100.0`, Lost GT = `17`.
- **Budget K=120:** Oracle F0.5 = `0.9820`, Recall = `95.20%`, Mean K = `120.0`, Lost GT = `14`.
- **Efficiency Reduction:** K=100 reduces candidate volume by ~65-70% while preserving nearly all oracle ceiling (oracle loss < 0.003).
