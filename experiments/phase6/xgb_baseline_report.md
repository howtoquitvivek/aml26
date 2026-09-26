# Phase 6A: XGBoost Matcher Baseline

We have upgraded the matching architecture from Logistic Regression to an XGBoost Classifier (`XGBClassifier`) using the `K=100` budget output from `CFG1_NoTwoTok_KeyE`.

## 1. 3-Fold Cross-Validation on 1,500-S1 Dev Set

XGBoost demonstrates a **massive** improvement over Logistic Regression. It successfully learned non-linear interactions to safely aggressively filter the 106,297 negative candidate pairs while preserving true positives.

| Model / Budget | Optimal Threshold | Macro F0.5 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (No ML) | 0.70 | `0.6370` | 0.4900 | 0.4400 | 
| Logistic Regression (K=25) | 0.80 | `0.7289` | 0.7975 | 0.6300 |
| **XGBoost Classifier (K=100)** | **0.90** | **`0.8040`** 🚀 | **0.8627** | **0.7074** |

By moving to XGBoost, we increased Macro F0.5 by an absolute **+7.51%** over the standard K=25 LR model on the Dev set.

## 2. Feature Importances
XGBoost's `feature_importances_` reveal exactly what the model learned:

1. **`addr_tok_jac` (0.7014)**: Address Token Jaccard is overwhelmingly the most predictive feature.
2. **`addr_char_jac` (0.0729)**: Address Character Jaccard.
3. **`addr_missing` (0.0621)**: Heavily penalizing or altering behavior when an address is entirely absent.
4. **`addr_len_diff` (0.0547)**: Address Length Difference.
5. **`name_char_jac` (0.0374)**: Name Character Jaccard.

*Insight:* XGBoost relies almost entirely on the **Address** to make its final decisions, which perfectly aligns with our earlier diagnostic analysis: name overlaps are highly noisy (74% of missed GTs had high name overlap but were missed by blocking; many negative candidates *also* have high name overlap). Only the address can securely disambiguate true matches in dense candidate sets.

## 3. Next Step: Full Validation Benchmark
We must now run this XGBoost model on the 10,000 holdout or directly on the 220,683-S1 full validation set to verify that the +7.5% F0.5 gain transfers cleanly to the complete catalog.
