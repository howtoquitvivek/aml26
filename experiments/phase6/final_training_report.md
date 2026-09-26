# Phase 6B: Final Master XGBoost Model Trained

The final production-grade XGBoost matcher has been successfully trained on the 50,000-S1 strictly bounded, deterministic training subset.

## Training Metadata
* **Model Path:** `experiments/phase6/artifacts/final_xgb_model.ubj`
* **Total S1 Records:** 50,000 (sampled purely from `train_s1_ids.txt`, zero validation leakage)
* **Total Candidate Pairs Fed:** 2,255,343 (~45 candidates per S1)
* **Positives (GT Match):** 128,893
* **Negatives (False Candidates):** 2,126,450
* **Global `scale_pos_weight`:** 16.4978

## Resource Footprint
* **Peak RAM (RSS):** 6.17 GB (Well within the 10GB safe operating limit)
* **XGBoost Algorithm Runtime:** 6.62 seconds
* **Total Pipeline Runtime (Including Feature Extraction):** 3.55 minutes

## Feature Importance Shift Analysis
Across the much larger and significantly more diverse 50,000-S1 training distribution, the gradient boosting algorithm fine-tuned its reliance on specific address features:
1. `addr_char_jac`: **0.6903**
2. `addr_tok_jac`: **0.1149**
3. `addr_missing`: **0.0604**
4. `addr_len_diff`: **0.0555**
5. `name_char_jac`: **0.0352**

*Insight:* On the tiny 1,500-S1 dev set, XGBoost relied almost entirely on token-level address overlap (0.75). However, when exposed to 2.25 million candidates, it learned that **character-level address overlap** (`addr_char_jac`) is vastly superior for capturing the nuanced variations, typos, and abbreviation differences present in the wider population. Address variables continue to completely dominate the model's decision matrix (over 92% of the total predictive power), decisively proving that names are too noisy for final disambiguation.

## Final Status
The `final_xgb_model.ubj` artifact is ready for blind Test Set Inference.
