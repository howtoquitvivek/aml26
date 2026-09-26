# Phase 3 Historical Archive

This directory contains historical iterations of the Phase 3 blocking and matching pipeline. These versions were superseded as the system evolved.

**⚠️ DISCLAIMER:** The experiments and metrics in this archive are historically important for understanding the blocker development, but their results are **NOT authoritative** and have been completely superseded by the final Phase 6 pipeline.

## v1: Capped Blocking
**Files:** `v1_capped_blocking/`

**Known Results:**
- Macro F0.5: 0.6165

**Known Problems:**
- Relied on positional truncation (`[:500]`) and asymmetric candidate capping.
- The matcher achieved artificial precision because dangerous False Positives were arbitrarily discarded before reaching the matcher.

## v2: True / Unconstrained Blocking
**Files:** `v2_unconstrained_blocking/`

**Known Results / Problems:**
- Removed artificial budget capping to expose true blocking capabilities.
- Revealed severe candidate explosions (up to 18,000+ candidates for generic terms).
- Made full evaluation completely impractical locally without running out of memory.

## Later Composite Blocking Experiments
- Following the unconstrained blocking failure, we introduced composite keys (CFG1, CFG2, CFG3) to constrain the candidate space without relying on positional cutoffs.
- CFG1 became the baseline blocker (combining Exact, Canon, TwoTok, AddrOld, and Key B).
- Further experiments (available in `experiments/phase3/`) refined this by identifying that `idx_two_tok` was the primary cause of candidate explosions, leading to frequency-capping strategies.
