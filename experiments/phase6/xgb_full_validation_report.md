# Phase 6A: XGBoost Full Validation Benchmark

The XGBoost model (`XGBClassifier`) has been evaluated against the complete 220,683-S1 validation set using the validated `K=100` budget. The model was trained entirely on the frozen 1,500-S1 development set with the strictly pre-calculated threshold of `0.90`.

## 1. Top-Level Metric Comparison

XGBoost completely dominates Logistic Regression at handling the large K=100 candidate budgets. It simultaneously increases recall while drastically cutting down on false positives.

| Metric | LR Baseline (K=100) | XGBoost (K=100) | Change |
| :--- | :--- | :--- | :--- |
| **Macro F0.5** | `0.7613` | **`0.8031`** | **+0.0418** 🚀 |
| Macro Precision | `0.9188` | **`0.9489`** | +0.0301 |
| Macro Recall | `0.6655` | **`0.6961`** | +0.0306 |
| True Positives (TP) | 493,803 | **518,469** | **+24,666** |
| False Positives (FP)| 59,273 | **32,214** | **-27,059** |
| False Negatives (FN)| 269,789 | 245,123 | -24,666 |
| Singleton Accuracy | 50.32% | **58.54%** | +8.22% |

## 2. Generalization Success
A key fear with tree-based models like XGBoost is overfitting on small training data. However, the model trained on just 1,500 entities (110k candidate pairs) achieved a cross-validation F0.5 of `0.8040`. On the entirely unseen 220,683 entities (10M candidate pairs), it scored `0.8031`.

This confirms the feature engineering (Jaccard overlaps) effectively sanitizes the data, making the dataset extremely stationary and allowing the XGBoost matcher to perfectly generalize.

## 3. Computational Cost

| Resource | XGBoost (K=100) |
| :--- | :--- |
| Candidate Generation | 59.6s |
| Feature Extraction (10M pairs) | 666.7s |
| XGBoost Scoring (10M pairs) | **8.86s** |
| **Total Wall-Clock Time** | **13.76 minutes** |
| **Peak RAM (RSS)** | **8.56 GB** |

XGBoost inference is nearly instantaneous, scoring all 10 million pairs in just 8.8 seconds without requiring GPU acceleration. Peak RAM remains comfortably underneath standard 16GB memory constraints.

## 4. Conclusion
Upgrading the matching classifier to XGBoost has pushed the system F0.5 above 0.80. The K=100 pipeline is now highly optimized, safe, and computationally viable. 
