# Phase 6B: Final Test Inference Preparation Plan

The Phase 6A XGBoost configuration is completely frozen, audited, and ready for test-set inference. Before executing the final run, we must prepare the final training procedure that safely scales the model without hitting memory limits (as the VM is constrained to 10GB of total RAM).

## 1. Exact Labeled S1 Records for Final Training
The original `experiments/train_s1_ids.txt` partition contains 542,909 S1 records. Generating K=100 candidates for all 542k records would yield ~54 million pairs, which requires roughly 45GB of RAM to hold in a single matrix and would instantly trigger an Out-Of-Memory (OOM) crash.

Because our model contains only 15 heuristic features, XGBoost completely converged and generalized perfectly (0.8031 F0.5) using just the 1,500-S1 dev set (110k pairs). We do not need 54 million pairs.

**Decision:** We will train the final model on a fixed, deterministic subsample of **50,000 S1 records** randomly drawn from `train_s1_ids.txt` (seed=42). 

## 2. Candidate Generation & Negatives
Candidates will be extracted via the exact frozen Phase 6 methodology: `CFG1_NoTwoTok_KeyE` at `K=100`. The negative labels will naturally constitute all candidate pairs retrieved by the blocker that do not exist in the Ground Truth mappings.

## 3. Inclusion of the 1,500 Dev Records
**Decision:** We **will** include the 1,500 `dev_s1_ids.txt` records in the final training dataset (bringing the total training size to 51,500 S1s). 
*Rationale:* The model architecture, hyperparameters, and the 0.90 threshold are now permanently frozen. Because no further hyperparameter tuning or threshold selection will occur, adding the dev set into the final training corpus adds safe, known-good diversity without causing any methodological leakage.

## 4. Frozen Threshold
The threshold will remain strictly hardcoded to **0.90**. No validation labels or final-training predictions will be used to alter it.

## 5. Leakage Prevention
We will strictly assert that no IDs from `val_s1_ids.txt` or `test.csv` enter the 51,500-S1 training pool. The final model is evaluated strictly on blind test data.

## 6. Resource Requirements
* **Total Training Candidate Pairs:** ~5.15 million
* **Estimated Feature Extraction Runtime:** ~5 to 6 minutes
* **Estimated XGBoost Training Time:** ~10 seconds
* **Estimated Peak RAM (RSS):** ~4.3 GB (safely beneath the 10GB hard limit).

## 7. Proposed Final Pipeline
1. **Train Model:** `python3 experiments/phase6/train_final_xgb.py`
   * Extracts features for the 51.5k S1s.
   * Trains `XGBClassifier` with the frozen Phase 6 configuration.
   * Saves the model to `experiments/phase6/artifacts/final_xgb_model.ubj`.
2. **Test Inference:** `python3 experiments/phase6/run_test_inference.py`
   * Loads the saved model.
   * Processes the blind test S1 records via `CFG1_NoTwoTok_KeyE` + K=100.
   * Scores with the saved XGBoost model at threshold `0.90`.
   * Outputs the final `submission.csv`.
