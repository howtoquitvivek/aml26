# Baseline Freeze: P3.2 + XGBoost + K=300

Blocker: P3.2
Matcher: XGBoost
K: 300
Threshold: 0.90

Macro F0.5: 0.8393
Precision: 0.8845
Recall: 0.7682

TP: 589876
FP: 55808
FN: 173716

Blocking recall: 87.74%
Candidate count: 40411291

Model: experiments/phase6/artifacts/final_xgb_model.ubj
Command: python3 experiments/phase7/run_p3_2_downstream.py
