# Phase 3 Report: Candidate Generation / Blocking & Deterministic Matching Baseline

## Executive Summary
Phase 3 evaluated five individual candidate-blocking strategies plus a combined multi-strategy union on the validation split, followed by a deterministic matching baseline over the generated candidate pairs.

The resulting deterministic baseline achieves a **Macro $F_{0.5}$ score of 0.7570** (Precision: **0.8639**, Recall: **0.5814**, Singleton Accuracy: **94.92%**) using zero machine learning classifiers.

---

## Part 1: Phase 3A — Blocking & Candidate Generation Experiments

### Comparison Table across Blocking Strategies

| Strategy | Blocking Recall (%) | Avg Candidates / S1 | Median Candidates | P95 Candidates | Extrapolated Full Val Pairs | Reduction Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Country Only** | **100.00%** | 5,364,570.60 | 6,186,873 | 6,186,873 | 1,183,902,323,766 | 48.0300% |
| **2. Exact Normalized Name** | 20.49% | 1.12 | 1 | 4 | 248,268 | 99.9999% |
| **3. Token-Based (Canonical)** | 40.01% | 2.95 | 2 | 9 | 651,014 | 99.9999% |
| **4. Character Prefix (4-char)** | 41.03% | 311.66 | 433 | 500 | 68,778,284 | 99.9969% |
| **5. Address-Assisted (Num + Init)** | 53.88% | 2.58 | 2 | 7 | 570,244 | 99.9999% |
| **6. Combined Multi-Strategy Union** | **75.87%** | **80.57** | **100** | **100** | **17,780,870** | **99.9992%** |

### Key Blocking Insights:
1. **Country Partitioning (Strategy 1)**: Lossless hard block (100% recall), but alone reduces the search space by only 48.03% ($\sim 1.18$ trillion pairs), making downstream comparison intractable.
2. **Exact Normalized Name (Strategy 2)**: Extremely high selectivity (1.12 cands/S1), but misses $\sim 79.5\%$ of true matches due to legal suffixes, abbreviations, word reordering, and typos.
3. **Token-Based Canonical (Strategy 3)**: Recovers legal suffix variations and word-order transpositions (`Crystal Staffing Solutions LLC` $\leftrightarrow$ `Llc Crystal Staffing Solutions`), doubling recall to 40.01% with only 2.95 cands/S1.
4. **Address-Assisted Blocking (Strategy 5)**: Strong signal (53.88% recall, 2.58 cands/S1) by coupling address numbers/PIN codes with business name prefixes.
5. **Combined Union (Strategy 6)**: Reaches **75.87% blocking recall** while achieving a **99.9992% reduction ratio** ($\sim 80.57$ candidates per S1, $\sim 17.8\text{M}$ pairs across full validation).

---

## Part 2: Phase 3B — Deterministic Matching Baseline

A deterministic feature computation pipeline was constructed using interpretable signals without any ML model:
- **Name Signals**: Exact normalized match, canonical token match, token Jaccard, character 3-gram Jaccard, length ratio.
- **Address Signals**: Exact normalized match, token Jaccard, numeric token overlap, missing address indicator.
- **Country Signal**: Strict country equality.

### Score Distributions (True Matches vs. False Candidates)
- **True Ground-Truth Matches**:
  - Mean score: **0.8237** | Median: **0.9385** | P75: **0.9714** | P90: **0.9933**
- **False Candidates (Distractors)**:
  - Mean score: **0.1413** | Median: **0.1130** | P90: **0.2739** | P95: **0.3583** | P99: **0.6500**

There is clear multimodal separation. 99% of negative candidate pairs score below 0.65.

### Decision Threshold & Configuration Optimization (Validation Split)

| Configuration | Threshold ($\theta$) | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy | False Merges |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`conservative_precision`** | **0.70** | **0.7570** | **0.8639** | **0.5814** | **94.92%** | **72** |
| `balanced` | 0.70 | 0.7481 | 0.8567 | 0.5700 | 94.92% | 72 |
| `conservative_precision` | 0.75 | 0.7270 | 0.8445 | 0.5390 | 94.92% | 72 |
| `name_heavy` | 0.70 | 0.7237 | 0.8395 | 0.5379 | 94.92% | 96 |
| `balanced` | 0.75 | 0.7230 | 0.8415 | 0.5327 | 94.92% | 71 |
| `conservative_precision` | 0.80 | 0.7059 | 0.8279 | 0.5131 | 94.92% | 71 |
| `name_heavy` | 0.75 | 0.7062 | 0.8295 | 0.5105 | 94.92% | 76 |

---

## Part 3: Qualitative Error Analysis

### 1. Major False-Positive Patterns (False Merges)
- **Same Business Name at Different Physical Locations (Chains/Franchises)**:
  - S1: `Pediatric Dental Partners`, `909 Beach Drive, Oak Island, NC`
  - False Cand: `Pediatric Dental Partners LLC`, `1646 MORNINGSIDE DR, BURLINGTON, NC`
  - *Root Cause*: High name similarity ($1.0$) combined with same state (`NC`), but street address and city differ.
  - *Remedy for ML*: City/locality extraction and street number mismatch penalty.
- **Blank Address Matches on Generic Business Names**:
  - S1: `Creative Consulting Private Limited`, `Kerala, Malappuram`
  - False Cand: `CREATIVE CONSULTING PRIVATE LIMITED`, address: `""` (missing)
  - *Root Cause*: Neutral imputation for missing addresses permits matching common names.

### 2. Major False-Negative Patterns (Missed Matches)
- **Devanagari $\leftrightarrow$ Latin Script Transliterations**:
  - S1: `Ss Food Private Limited`
  - S2: `एसएस फूड प्राइवेट लिमिटेड`
  - *Root Cause*: Pure string Jaccard between Latin and Devanagari is 0.0.
  - *Remedy for Phase 4/5*: Rule-based phonetic transliteration or multilingual subword embeddings.
- **Heavy Landmark vs Municipal Address Discrepancies**:
  - S1: `Af-684, Nandgram Near Mother India Public School, Ghaziabad`
  - S3: `Af-684, Ghaziabad, UP`
  - *Root Cause*: Token Jaccard is diluted by lengthy landmark tokens (`Mother India Public School`).
  - *Remedy for ML*: Numeric token matching (`684`) and city-anchor extraction.
