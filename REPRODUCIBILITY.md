# Entity Resolution - Final Reproducible Pipeline

This repository contains the finalized, hardened machine learning pipeline for the Amazon ML Challenge 2026. This pipeline deterministically scores and pairs entities across massive multi-source datasets (S1, S2, and S3).

## Project Purpose
To provide a highly scalable, fully deterministic, CPU-bound pipeline capable of ingesting 1.7M queries and resolving them against a 9.9M-record reference database, achieving optimal matching under strict formatting and resource constraints.

## Pipeline Architecture & Frozen Configuration
The pipeline executes sequentially in a single process (`n_jobs=1`) to guarantee strict deterministic execution across hardware. It has been frozen at the highest-scoring validation checkpoint (Macro F0.5 = 0.8043).

**Exact Frozen Configuration:**
- **Candidate Generator 1 (Primary):** `P3.2` Blocker
- **Candidate Generator 2 (Secondary):** `Miss-Recovery` (MR) Blocker
- **Maximum Candidates Evaluated (K):** `300` (sampled if candidate sets exceed this limit)
- **Feature Extraction:** Canonical 15-feature schema
- **XGBoost Model:** `experiments/final_hardening/artifacts/xgb_15000.ubj`
- **XGBoost Inference Threshold:** `0.98`

## Repository Structure
```text
.
├── dataset/                # Excluded from git (place train/test datasets here)
├── experiments/            # Iterative testing, notebook history, and final artifacts
│   └── final_hardening/
│       ├── artifacts/      # Contains the required xgb_15000.ubj model
│       └── index/          # Generated indices (excluded from git)
├── output/                 # Excluded from git (generated candidate_pairs and matching_results go here)
├── src/                    # The core pipeline source code (features, blockers, inference logic)
├── utils/                  # Tools like `validate_submission.py`
├── REPRODUCIBILITY.md      # This document
└── requirements.txt        # Python dependencies
```

## Required Environment
- **OS**: Linux (tested on Ubuntu 24.04)
- **Python**: Python 3.12+
- **Resources**: CPU-bound pipeline. Recommend an instance with at least 16 GB of RAM. The pipeline peak memory usage (RSS) on the massive 1.7M test dataset is exactly **10.4 GB**.

---

## 🚀 FROM FRESH CLONE: How to Run the Pipeline

### 1. Clone the Repository and Setup the Environment
```bash
git clone <repository-url>
cd aml26
git checkout improve/entity-resolution-v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Layout
Place your datasets into the `dataset/test/` directory. You must ensure the directory structure matches the following exactly (these files are intentionally excluded from Git):
```text
dataset/
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```
*Note: The script also works on the training set if you provide the path to `dataset/train/train_source*.tsv`.*

### 3. Run Final Inference
The inference script combines index building and inference. 
You do **not** need to manually build indexes beforehand. The inference script incorporates robust validation: it will verify whether a complete set of index components exists for every required country inside `--index_dir`. If they are absent or partially built, it will automatically build them using `os.makedirs(..., exist_ok=True)`.

Execute the following command to begin:
```bash
export PYTHONPATH=.
python3 src/run_inference.py \
  --s1 dataset/test/test_source1.tsv \
  --s2 dataset/test/test_source2.tsv \
  --s3 dataset/test/test_source3.tsv \
  --index_dir experiments/final_hardening/test_index \
  --out output/matching_results.tsv \
  --cand output/candidate_pairs.tsv \
  --model experiments/final_hardening/artifacts/xgb_15000.ubj \
  --threshold 0.98
```
**Expected Runtime:** ~3 hours and 20 minutes for the full 1.73M dataset.

### 4. Validate Outputs
The output directories (`output/`) are automatically created. Once inference concludes, use the provided validator utility to structurally audit the output TSV artifacts:
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```
You should see: `PASS — no blocking issues found. Safe to submit.`

---

## Advanced Documentation & Notes

### Determinism and Reproducibility
The pipeline is locked at `n_jobs=1` inside XGBoost and utilizes purely sequential Python iterations to ensure outputs do not diverge due to OS-level race conditions or stochastic set orderings across nodes.

### Generated Directories and `.gitignore`
A fresh clone will intentionally lack generated directories such as `output/` and `experiments/final_hardening/test_index/`. This is by design to keep the git repository lightweight. The pipeline code dynamically checks and generates these structures safely.

### The 15-Feature Canonical Schema
The feature extraction outputs a strict sequence:
1. `name_exact` 2. `name_canonical_exact` 3. `name_tok_jac` 4. `name_char_jac` 5. `name_len_diff` 6. `addr_exact` 7. `addr_missing` 8. `addr_tok_jac` 9. `addr_char_jac` 10. `addr_num_jac` 11. `addr_len_diff` 12. `postal_overlap` 13. `salient_addr_jac` 14. `is_s2` 15. `country_match`

### Known Limitations
Because of the strict sequential nature of the script, performance cannot scale by simply adding more CPU cores. If faster iteration is required, the chunking and random.sample iteration ordering must be carefully redesigned to preserve deterministic execution in a multiprocessing (`fork`) environment.
