# Full Validation Benchmark Report — Phase 3 (Amazon ML Challenge 2026)

**Evaluation Scope**: Complete validation set of **220,683 Source 1 entities** evaluated against all **10,320,219 Source 2 & Source 3 records**.
**Execution Profile**: Completed in **1492.8s** with peak RSS of **5462.3 MB**.

## 1. Full Blocking Comparison (Phase 3A)

| Strategy | Recall (%) | Total Cand Pairs | Avg / S1 | Median | P95 | P99 | Max | Red Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Country Only** | **100.00%** | 1,183,902,323,766 | 5364719.18 | 6186873.0 | 6186873.0 | 6186873.0 | 6186873 | 48.02% |
| **2. Exact Normalized Name** | **21.63%** | 2,189,774 | 9.92 | 1.0 | 60.0 | 179.0 | 458 | 99.999904% |
| **3. Token-Based (Canonical)** | **58.55%** | 18,334,089 | 83.08 | 13.0 | 503.0 | 857.0 | 991 | 99.999195% |
| **4. Character Prefix** | **20.30%** | 97,658,299 | 442.53 | 500.0 | 500.0 | 500.0 | 500 | 99.995712% |
| **5. Address-Assisted** | **55.98%** | 4,104,740 | 18.6 | 4.0 | 92.0 | 266.0 | 1856 | 99.99982% |
| **6. Combined Multi-Strategy Union** | **71.93%** | 29,750,113 | 134.81 | 100.0 | 508.0 | 876.0 | 1824 | 99.998694% |

### Combined Union Ablation & Marginal Recall Contribution
- **Exact Normalized Name**: Found 165,135 true links (**21.63%**)
- **Canonical Token (Marginal)**: Added +285,176 true links (**+37.35%**)
- **Address-Assisted (Marginal)**: Added +118,072 true links (**+15.46%**)
- **Character Prefix Fallback (Marginal)**: Added +17,617 true links (**+2.31%**)

## 2. Full Deterministic Baseline Comparison (Phase 3B)

**Best Configuration on Validation Split**: `conservative_precision` at threshold **$	heta = 0.7$**
- **Macro $F_0.5$**: **0.6165**
- **Macro Precision**: **0.6807**
- **Macro Recall**: **0.5395**

### Threshold Grid Evaluation
| Configuration | Threshold ($	heta$) | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Aggregate TP | Aggregate FP | Aggregate FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `conservative_precision` | 0.70 | **0.6165** | 0.6807 | 0.5395 | 413,907 | 337,192 | 349,685 |
| `balanced` | 0.70 | **0.6104** | 0.6758 | 0.5327 | 408,484 | 342,505 | 355,108 |
| `conservative_precision` | 0.75 | **0.5944** | 0.6668 | 0.5061 | 386,359 | 335,263 | 377,233 |
| `balanced` | 0.75 | **0.5899** | 0.6634 | 0.5003 | 381,646 | 336,293 | 381,946 |
| `name_heavy` | 0.70 | **0.5793** | 0.6480 | 0.5026 | 384,932 | 400,497 | 378,660 |
| `conservative_precision` | 0.80 | **0.5770** | 0.6544 | 0.4824 | 366,984 | 334,315 | 396,608 |
| `balanced` | 0.80 | **0.5733** | 0.6517 | 0.4777 | 363,187 | 334,238 | 400,405 |
| `name_heavy` | 0.75 | **0.5722** | 0.6473 | 0.4835 | 368,554 | 358,989 | 395,038 |
| `name_heavy` | 0.80 | **0.5677** | 0.6463 | 0.4726 | 359,253 | 338,382 | 404,339 |
| `conservative_precision` | 0.65 | **0.4855** | 0.5193 | 0.5690 | 444,750 | 7,585,876 | 318,842 |
| `balanced` | 0.65 | **0.4818** | 0.5162 | 0.5628 | 439,811 | 7,593,767 | 323,781 |
| `name_heavy` | 0.65 | **0.4497** | 0.4876 | 0.5237 | 409,760 | 7,750,745 | 353,832 |

### Score Distributions (True Matches vs. False Candidates)
- **True Matches**: Mean = 0.8398, Median = 0.9444, P75 = 0.9714, P90 = 0.9938
- **False Candidates**: Mean = 0.3049, Median = 0.2248, P90 = 0.65, P95 = 0.65, P99 = 0.85

## 3. Source-Specific Diagnostics (Phase 3C: S2 vs S3)

| Source | True Matches | Blocking Recall (%) | Total Candidates | Precision | Recall | $F_{0.5}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S2** | 369,516 | 73.07% | 20,531,337 | 0.5385 | 0.5785 | 0.5461 |
| **S3** | 394,076 | 70.86% | 9,218,776 | 0.5651 | 0.5079 | 0.5527 |

## 4. Singleton Analysis (Phase 3D)

- **Total True Singletons**: 12,325
- **Correctly Predicted as Empty**: 6,095 (**49.45%** accuracy)
- **False Merges on Singletons**: 6,230

## 5. Measured Memory & Runtime Safety (Phase 3F)

| Country Partition | Records Indexed | Runtime (s) | Peak RSS (MB) | Post-Cleanup RSS (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **India** | 4,133,346 | 582.34s | 3540.0 MB | 3540.0 MB |
| **US** | 6,186,873 | 893.58s | 5462.3 MB | 5462.3 MB |

## 6. Key Error Patterns (Phase 3E)

1. **Same-Name Different-Location (Franchises/Chains)**: High name agreement, state match, but street/city diverge.
2. **Devanagari / Latin Transliteration**: Devanagari characters fail ASCII string overlap unless transliterated.
3. **Missing Address Matches on Generic Corporate Names**: Unanchored entities causing false merges.
4. **Landmark-Heavy Descriptions**: Landmark text obscuring municipal house/PIN codes.