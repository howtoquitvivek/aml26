# Phase 7: Priority 2 - Downstream K Evaluation (Corrected)

## 1. Experimental Methodology
We performed a highly controlled isolation of the XGBoost inference step to evaluate the downstream impact of Top-K candidate thresholds. We reused the Phase 4 blocker index, retrieved up to `K=300` candidates per S1 on the validation set, and extracted matching features for all 18.6 million candidate pairs. The frozen `final_xgb_model.ubj` (threshold = 0.90) scored the candidates. 

We then evaluated downstream performance by slicing these identical predictions at `K=100`, `150`, `200`, and `300`. This isolates the `K` parameter perfectly without retraining or changing features.

**Critical Evaluation Fix:** The initial report accidentally used Micro F0.5. This version correctly uses `src.evaluation.evaluate_predictions` to compute the official **Macro F0.5**, ensuring singletons and edge cases are perfectly aligned with the authoritative metric.

- **Total S1 Validation Entities:** 220,683
- **Feature Extraction & Scoring Runtime:** 1,742 seconds (~29 minutes)

## 2. Downstream Results (Threshold = 0.90)

| Metric | K=100 (Baseline) | K=150 | K=200 | K=300 |
|---|---|---|---|---|
| **Total Candidates** | 10.01M | 12.87M | 15.14M | 18.65M |
| **Avg Cands/S1** | 45.38 | 58.33 | 68.62 | 84.52 |
| **True Positives (TP)** | 531,828 | 542,459 | 548,933 | 554,499 |
| **False Positives (FP)** | 39,014 | 42,046 | 43,999 | 46,164 |
| **False Negatives (FN)**| 231,764 | 221,133 | 214,659 | 209,093 |
| **Macro Precision** | 0.8726 | 0.8763 | 0.8785 | 0.8800 |
| **Macro Recall** | 0.7009 | 0.7133 | 0.7209 | 0.7276 |
| **Macro F0.5** | **0.8066** | **0.8138** | **0.8182** | **0.8216** |
| **Singleton Accuracy** | 79.00% | 77.91% | 77.44% | 77.00% |

### Deltas relative to K=100:

**K=150:**
- Δ Macro Precision: +0.0036
- Δ Macro Recall: +0.0123
- Δ Macro F0.5: +0.0071
- Additional TP: +10,631
- Additional FP: +3,032

**K=200:**
- Δ Macro Precision: +0.0058
- Δ Macro Recall: +0.0200
- Δ Macro F0.5: +0.0115
- Additional TP: +17,105
- Additional FP: +4,985

**K=300:**
- Δ Macro Precision: +0.0073
- Δ Macro Recall: +0.0267
- Δ Macro F0.5: +0.0149
- Additional TP: +22,671
- Additional FP: +7,150

## 3. Critical Analysis of Candidate Quality
An extremely counterintuitive (but highly beneficial) phenomenon occurs when increasing K: **Both Macro Recall AND Macro Precision increase simultaneously.** 

How does precision increase if we add more global False Positives (+7,150 FP at K=300)? 
Because the official metric averages precision *per entity*. The XGBoost model is acting as an incredibly effective filter. When we feed it candidates from the deep ranks (101-300), it rejects the vast majority of the garbage. The ~3.1-to-1 ratio of true-to-false matches it accepts from this deep tail actually *repairs* the denominators of entities that previously had terrible precision. 

For example, if an S1 entity at K=100 had 1 FP and 0 TP, its precision was 0.0. By expanding to K=300, we find 1 TP and add 0 FPs for that entity. Its precision leaps from 0.0 to 0.5. The model's discrimination power is so strong that feeding it more candidates universally improves the per-entity macro scores.

However, we do see the impact of candidate bloat on singletons:
- As K increases, the model is exposed to more look-alikes for entities that have no true matches. Singleton accuracy drops from 79.0% (K=100) to 77.0% (K=300).

## 4. Decision Gate

**RECOMMENDATION:**
K = 200 (or K=300 depending on final inference budget constraints).

**Additional F0.5 vs K=100:**
+0.0115 (K=200) or +0.0149 (K=300)

**Reason:**
Unlike the original micro-averaged assumption, increasing candidate volume strongly benefits the actual Macro F0.5 metric, yielding a massive **+0.0115 absolute bump** simply by changing the blocker configuration from `K=100` to `K=200`. 

`K=200` hits the undeniable sweet spot of the curve. It captures the vast majority of the F0.5 gains while keeping the candidate increase to a modest +5.1 million (+51%). 

Going to `K=300` provides another +0.0034 F0.5 bump, but it costs an additional 3.5 million candidates. If our final inference time constraints allow generating and scoring 18.6 million candidates, K=300 is technically superior. However, K=200 is highly recommended as the most efficient balance of metric gain and computational budget.
