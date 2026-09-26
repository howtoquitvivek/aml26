# Phase 7 P3: Consonant Signature Blocker Experiment

## Objective
Target the ~19.7% of ground truth links completely missed by the baseline blocker. These missed links generally suffer from token-level variations (e.g., typos, cross-script transliteration like *Laxmi* vs *Laxmee*) in the name, combined with a lack of shared numbers in the address. The existing keys (Exact, Canonical, AddrOld, KeyB, KeyE) require strictly exact matching on whole tokens or prefixes, causing them to fail entirely on these variants.

## New Blocking Logic: Consonant Signatures
The proposed logic bypasses token boundaries and vowel-transliteration mismatches by generating a "Consonant Signature" of the longest, rarest tokens in the entity.
1. Extract the top 2 longest alphabetic tokens (≥ 5 characters) from both the name and address.
2. Strip vowels (a, e, i, o, u, y, h) to form a consonant signature (e.g., `venkateshwara` -> `vnkswr`).
3. Generate two types of composite keys:
   - `P3#{NameSignature}#{AddressSignature}`
   - `P3N#{NameSignature}#{AddressNumber}` (using address numbers ≥ 3 digits).

## 5,000-S1 Validation Results

### 1. Baseline Blocker (Pre-K Unlimited)
- **Total GT Links:** 17,123
- **Recovered GT Links:** 13,620
- **Recall:** 79.54%
- **Missed Links:** 3,503
- **Total Candidates Generated:** 790,247
- **Avg Candidates / S1:** 158

### 2. P3 Consonant Signature Blocker (Pre-K Unlimited)
- **Recovered GT Links:** 15,100
- **Recall:** 88.19%
- **Missed Links:** 2,023
- **Total Candidates Generated:** 25,489,060
- **Avg Candidates / S1:** 5,097

## Extrapolation & Analysis
- **Newly Recovered Links (5k sample):** +1,480
- **Extrapolated Full-Set Recovery:** ~65,322 links (out of 150,612 previously missed).
- **Extrapolated Full-Set Candidates:** **~1.09 Billion candidates**

## Decision Gate
**Status: REJECTED for production.**
While the Consonant Signature keys successfully recover over 43% of the highly-difficult missed links, they cause an absolutely massive candidate explosion. The candidate volume increases by ~32x. Scoring 1.09 billion candidates through the XGBoost downstream pipeline is computationally prohibitive and violates the strict requirement to avoid broad candidate explosions (like the previous TwoTok issue). The issue arises because 3-letter consonant signatures paired with common 3-digit numbers (like `100`) collide with tens of thousands of irrelevant records.

The P3 code is retained for reference and diagnostic use in the EC2 environment.
