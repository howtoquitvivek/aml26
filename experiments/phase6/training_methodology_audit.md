# Phase 6B: Final Training Methodology Audit

An audit of the proposed chunked incremental XGBoost training strategy was conducted to evaluate its methodological validity for the final 1.98M-S1 test inference preparation.

## 1. Mathematical Validity of Chunked XGBoost Training
**Finding: NOT VALID.**
Unlike Neural Networks that utilize Stochastic Gradient Descent (SGD) and can be safely trained iteratively in mini-batches, Gradient Boosting Decision Trees (GBDTs) like XGBoost do not natively support continuous "updating" of tree structures.
When passing a previous `xgb_model` to XGBoost on a new data chunk, the algorithm does not adjust existing trees. It either:
1. **Adds new trees** (causing tree explosion—e.g., 100 chunks × 100 trees = 10,000 trees, completely violating our frozen `n_estimators=100` architecture).
2. **Updates leaf weights only** (using `process_type='update'`), which fixes the tree splits to whatever was learned *only in the first chunk*, entirely defeating the purpose of reading the rest of the dataset.

Sequential incremental training is mathematically **not equivalent** to full-dataset training.

## 2. Inconsistency of `scale_pos_weight`
**Finding: SEVERE RISK.**
If `scale_pos_weight` is recalculated independently per chunk, the model will experience contradictory base loss calculations. Chunk A might have a positive/negative ratio of 1:100, while Chunk B has 1:80. The trees trained on Chunk B would pull the global probabilities off-center relative to Chunk A, destroying global calibration and severely damaging the validity of our frozen `0.90` threshold.

## 3. Vulnerability to Chunk Ordering
**Finding: SEVERE RISK (Catastrophic Forgetting).**
Because incremental boosting adds trees to correct the residuals of the previous state, the chunk order will dictate the model's final behavior. The final 10 trees in our 100-tree model would exclusively optimize for the exact data distribution of the *final chunk*, completely ignoring the 1.96M S1s processed earlier. 

## 4 & 5. Safer Alternative: Bounded Simultaneous Training
Given that the model utilizes only **15 highly-stable heuristic features**, we already proved empirical convergence: a model trained on just 1,500 S1s (114,000 pairs) perfectly generalized to 220,683 unseen S1s (10M pairs) with zero degradation.

There is zero mathematical or empirical justification to train on 200 million pairs. The feature-space variance is fully captured in a much smaller sample.

**Recommendation: The Bounded Deterministic Sample**
Rather than risking incremental training artifacts, we should execute standard, simultaneous training on a sufficiently large, bounded subsample:
* **Dataset:** Randomly sample exactly **50,000 S1 records** (seed=42) from `train_s1_ids.txt`.
* **Method:** Generate K=100 candidates for these 50k S1s simultaneously.
* **Volume:** ~5 million candidate pairs.
* **Integrity:** This guarantees a single, mathematically sound gradient-boosting run, a globally accurate `scale_pos_weight`, and strict adherence to the frozen Phase 6A configuration (`n_estimators=100`, `threshold=0.90`).

## 6. Resource Estimation (50,000 S1 Bounded Run)
* **Candidate Volume:** ~5,000,000 pairs (approx. 50x larger than the dev set).
* **RAM (Peak RSS):** ~4.3 GB (Well within the 10GB hard limit).
* **Feature Extraction Runtime:** ~5.5 minutes.
* **XGBoost Training Runtime:** ~10-15 seconds.
* **Total Training Time:** ~6 minutes.

## Verdict
Do not use chunked incremental training. Train the final Phase 6 model simultaneously on a fixed 50,000-S1 subsample drawn from the 1.98M training partition.
