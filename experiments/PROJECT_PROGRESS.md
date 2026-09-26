# Amazon ML Challenge 2026: Project Progress

This document tracks the chronological progress of the entity resolution project, detailing findings, methodologies, validations, and final modeling decisions.

## Phase 1: Data Exploration and Initial Findings
*   **Artifacts:** `experiments/phase1/`
*   **Focus:** Understanding the structure, scale, and noise patterns of the dataset.
*   **Findings:** The ground truth (`train_ground_truth.tsv`) contained 1.98M Source 1 records mapped to 0 or more Source 2/3 records. Exact matching string structures revealed noise categories such as legal suffix differences, abbreviation inconsistencies, address ordering mismatches, transliteration (English/Hindi), and missing components.

## Phase 2: Evaluation & Validation Methodology
*   **Artifacts:** `experiments/phase2/`
*   **Focus:** Establishing a leak-free validation strategy and implementing the official metric.
*   **Methodology:**
    *   Implemented `Macro F0.5` scoring in `evaluation.py`.
    *   Executed a 90/10 deterministic stratified split (`train_s1_ids.txt` vs `val_s1_ids.txt`) on the training dataset based on target match counts (handling singletons proportionally).
    *   Established strict boundary constraints to prevent leakage between training and validation data.

## Phase 3: Deterministic Blocking & Feature Engineering
*   **Artifacts:** `experiments/phase3/`
*   **Focus:** Building a high-recall, memory-efficient candidate generator (blocker).
*   **Diagnostic Experiments:**
    *   Attempted simple blocking methods (exact name, canonical keys).
    *   Encountered low recall due to strict token overlap requirements.
*   **Phase 3.1 Unconstrained Blocking:** Investigated broader string distance and prefix matching techniques to capture typographic errors and cross-script variations.
*   **Outcome:** Unified multi-strategy blocking schema (Country-partitioned inverted indices) implemented in `src/blocking.py`.

## Phase 4: Logistic Regression Matcher Pipeline
*   **Artifacts:** `experiments/phase4/`
*   **Focus:** Baseline machine learning model implementation.
*   **Methodology:**
    *   Developed a pipeline to extract address, name, and alphanumeric feature vectors.
    *   Trained a Logistic Regression model as a baseline.
    *   Conducted a strict 10,000-S1 holdout audit (`holdout_10k`) which identified the need for stronger blocker performance.

## Phase 5: Blocker Ceiling & Candidate Arithmetic
*   **Artifacts:** `experiments/phase5/`
*   **Focus:** Analyzing the absolute limits of the blocking methodology.
*   **Blocker Diagnostic (Full Validation Set):**
    *   Total validation GT links: 763,592
    *   Generated before Top-K constraint: 612,980
    *   Blocking recall: 80.2759%
    *   Blocking loss: 19.7241%
    *   Additional GT links lost due to budget constraints (K=100): 45,689
*   **Outcome:** Finalized the blocker configuration (`CFG1_NoTwoTok_KeyE`) at a fixed budget of `K=100`.

## Phase 6: XGBoost Matcher (Current State)
*   **Artifacts:** `experiments/phase6/`
*   **Focus:** Training and auditing the final non-linear classification model.

### Phase 6A: XGBoost Development & Leakage Correction
*   Discovered that the initial 1,500-S1 dev set leaked into the 220,683-S1 validation set.
*   Generated a clean, completely disjoint `clean_dev_1500` set to correctly tune XGBoost hyperparameters and select thresholds.
*   **Clean Full Validation Result (Baseline):**
    *   Macro F0.5 = 0.8037
    *   Precision = 0.9495
    *   Recall = 0.6957
    *   TP = 518,223 | FP = 32,513 | FN = 245,369
    *   Singleton accuracy = 58.83%

### Phase 6B: Final Model Training
*   **Constraint:** Scaling from 1,500 to the full 1.98M S1 set for final training exceeded the strict 10GB memory limit.
*   **Methodology:** Trained a definitive model over a strictly bounded, deterministic 50,000-S1 population.
*   **Final Training Metrics:**
    *   Records: 50,000 S1
    *   Candidates Fed: 2,255,343
    *   Positives: 128,893 | Negatives: 2,126,450
    *   `scale_pos_weight`: 16.4978
    *   Peak RSS: 6.17 GB
    *   Training Time: 6.62 sec (Total Pipeline: 3.55 min)

## Current Frozen Configuration (Final Test Inference Status)
The modeling configuration is now completely frozen and validated against a leak-free 10,000-S1 test inference dry-run.

*   **Blocker:** `CFG1_NoTwoTok_KeyE`
*   **Candidate Budget:** `K=100`
*   **Features:** 15 base features
*   **Algorithm:** `XGBClassifier`
*   **Hyperparameters:** `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`
*   **Threshold:** `0.90`
*   **Final Model Path:** `experiments/phase6/artifacts/final_xgb_model.ubj`

The output formatting logic is certified correct. The repository is staged for the final blind test execution.
