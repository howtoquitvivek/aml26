# Repository Audit

A. Current authoritative pipeline:
- Blocker: P3.2 blocker (baseline + consonant signatures + generic token filtering, freq_threshold=100)
- Matcher: XGBoost (K=300, frozen, threshold=0.90)

B. Current authoritative model:
- `experiments/phase6/artifacts/final_xgb_model.ubj`

C. Current authoritative blocker:
- P3.2 blocker implementation via `get_p3_2_keys` from `experiments/phase7/run_p3_2_downstream.py`

D. Current validation split:
- `dataset/validation/validation_s1.tsv`
- `dataset/validation/validation_s1_ground_truth.tsv`
- Uses all 220,683 rows and evaluates on S2+S3.

E. Current metrics:
- Macro F0.5: 0.8393
- Precision: 0.8845
- Recall: 0.7682
- TP: 589876
- FP: 55808
- FN: 173716
- Blocker Recall: 87.74%
- Candidates: 40,411,291

F. Known bugs previously discovered:
- Positional [:500] posting list truncation (fixed).
- Asymmetric candidate limit < 100 on merged sets (fixed).
- Candidate selection from unordered sets caused non-determinism (fixed).

G. Known obsolete approaches:
- Original P3 N-gram blocker (P3N/4+ digit numbers).
- Original dev benchmark testing on true targets.
- Original K=300 using Micro F0.5.

H. Known resource hazards:
- Full corpus TF-IDF similarity.
- Unrestricted character fuzzy matching across 4M+ records.
- Storing millions of `s2_texts` simultaneously in memory without generators.

I. Files retained:
- `src/*` (until final upgrade)
- `dataset/*`
- Authoritative baseline models

J. Files to be removed:
- Temporary and failed scripts in `experiments/phase7/` (e.g., `run_union_blocker.py`, `tune_miss_recovery.py`, `test_hf_speed.py`, `dump_misses.py`)
- Unused cache files and indexes (`indexes/*`, `cache/*`)

K. Files to be regenerated:
- New deterministic indexes under a versioned manifest.

L. Historical-only files:
- Reports from `phase1` to `phase6`.

M. Files that must NEVER be reused:
- Old broken index generation scripts (e.g., `src/indexer.py` with `[:500]`).

N. Exact reproducibility commands:
- Baseline: `python3 experiments/phase7/run_p3_2_downstream.py`
