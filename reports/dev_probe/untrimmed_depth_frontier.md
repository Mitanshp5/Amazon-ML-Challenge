# Untrimmed Candidate Control & K-Depth Frontier Report

**Date:** 2026-09-26 19:47:41
**Scope:** `screen_2k` (2,000 queries: India=1,000, US=1,000)
**Candidate Generation:** Natural Multi-Channel Union (Joint, Name, Address, Structured, Dupes) with Zero GT Injection.

## 1. Candidate Depth Frontier vs Oracle Ceilings

| Candidate Policy | Overall Oracle $F_{0.5}$ | Pair Recall % | Mean $K$ / query | India Oracle $F_{0.5}$ | India Recall % | US Oracle $F_{0.5}$ | US Recall % |
|---|---|---|---|---|---|---|---|
| **K=50** | **0.977958** | 94.46% | 50.0 | **0.964063** | 91.55% | **0.991853** | 97.32% |
| **K=100 (B0)** | **0.985159** | 96.16% | 100.0 | **0.974536** | 93.84% | **0.995783** | 98.43% |
| **K=150** | **0.986252** | 96.52% | 150.0 | **0.975830** | 94.34% | **0.996675** | 98.66% |
| **K=200** | **0.986857** | 96.76% | 200.0 | **0.976653** | 94.66% | **0.997061** | 98.83% |
| **K=250** | **0.988796** | 97.11% | 250.0 | **0.980223** | 95.24% | **0.997368** | 98.94% |
| **Untrimmed (Natural Union)** | **0.990452** | 97.38% | 339.2 | **0.983216** | 95.67% | **0.997688** | 99.06% |

## 2. Key Findings: Untrimmed Control vs Truncated B0
1. **India K100 Truncation Loss:**
   - Truncating India candidates to K=100 loses **0.008681** in oracle macro $F_{0.5}$ compared to the untrimmed natural union.
   - At untrimmed depth (mean $K \approx 345.8$), India oracle reaches **0.983216**.
2. **US Headroom:**
   - US is virtually saturated even at K=100 (0.995783), reaching 0.997688 untrimmed.
3. **Implications for Device 1 (D1-02 Lexical Expansion):**
   - Simply expanding $K$ beyond 100 on existing channels recovers valuable positives for India, but still leaves an oracle gap to $>0.99$.
   - Therefore, **representation diversity** (address $K=150 \to 300$, word-token address with token frequency weighting, and preserved Unicode primary views) is mandatory to push India oracle above 0.99.

## 3. Coverage Quality Breakdown
| Policy | India Zero-Hit % | India 100% Coverage % | US Zero-Hit % | US 100% Coverage % |
|---|---|---|---|---|
| **K=50** | 1.48% | 78.46% | 0.10% | 91.53% |
| **K=100 (B0)** | 1.06% | 83.32% | 0.00% | 94.77% |
| **K=150** | 1.06% | 84.79% | 0.00% | 95.29% |
| **K=200** | 1.06% | 85.43% | 0.00% | 95.82% |
| **K=250** | 0.74% | 86.69% | 0.00% | 96.23% |
| **Untrimmed (Natural Union)** | 0.53% | 87.96% | 0.00% | 96.55% |
