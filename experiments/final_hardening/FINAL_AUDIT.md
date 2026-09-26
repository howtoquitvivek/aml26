# Final Production Hardening Audit

## Baseline Pipeline
Our authoritative baseline uses a **P3.2 Blocker + XGBoost Matcher**.
* **Blocker**: Uses strict exact match keys (normalized names, canonical names, address numeric overlaps, canonical tokens + address numerics, salient address consonants) combined with P3.2 generic-filtered consonant signatures. P3.2 keys are clipped at a posting list frequency threshold of 100.
* **Candidate Truncation**: Truncates retrieved candidates per query S1 at $K=300$.
* **Matcher**: We extract 15 deterministic features (including Jaccard similarities and length difference ratios) and use a frozen XGBoost model.
* **Final Threshold**: 0.90
* **Authoritative Performance**:
    * S1 count: 220,683
    * GT Links: 763,592
    * Macro F0.5: 0.8393
    * TP: 589,876
    * FP: 55,808

## Memory-Safe Architecture Redesign
During hardening on our 16GB EC2 instance, dynamic index generation across the entire dataset caused massive out-of-memory cascading failures. To ensure production robustness:

1. **Deterministic Versioned Indexes**: Index generation was decoupled from the pipeline and partitioned by country. 
   - Script: `src/index_builder.py`
   - Output: `experiments/final_hardening/index/idx_India.pkl`, `idx_US.pkl`, etc.
   - This keeps memory overhead strictly under ~4GB during peak unpickling and <1GB at runtime.
2. **Decoupled Candidate Generation**: Re-architected as `src/candidate_generation.py` which dynamically swaps index shards from disk per country to guarantee safe inference limits.
3. **Robust Feature Extraction**: All feature logic was strictly localized to `src/feature_extraction.py` enforcing version 2 schemas.
4. **Reproducible Training**: The `src/train_matcher.py` explicitly caps candidates, correctly samples GT matches, and deterministically scales negative weights avoiding memory explosion.

## Missing Link Gap Analysis
A crucial bottleneck in pushing beyond F0.5 = 0.8393 is the 93,601 Completely Missed GT Links by the P3.2 blocker.
By utilizing our safely partitioned `CandidateGenerator`, we execute a strict subset evaluation on the failed GT links, generating `missed_gt.json` to systematically extract patterns for the new **Miss-Recovery Blocker**.

## Final Steps Remaining
1. Await `missed_gt.json` and perform the manual Gap Analysis.
2. Implement a safe Union Miss-Recovery Blocker that focuses strictly on filling the 93k hole.
3. Retrain the XGBoost matcher using the larger 100k S1 sample generated from `train_matcher.py`.
4. Run `evaluate_pipeline.py` for the final definitive result.
