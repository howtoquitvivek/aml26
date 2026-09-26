# Project-Wide Correctness & Reproducibility Audit

**Audit Date**: September 25, 2026  
**Auditor**: AntiGravity Agent  
**Scope**: All source code (`src/`), experiment scripts & results (`experiments/`), notebooks (`notebooks/`), unit tests (`tests/`), and specifications (`.ps/ps.md`, `README.md`, `Documentation_template.md`, `requirements.txt`).  
**Mode**: Read-Only Audit (No production source code was modified).

---

## 1. Executive Summary

This comprehensive audit was initiated following the discovery of two significant methodological issues in `experiments/run_full_validation_phase3.py`:
1. **Posting-list truncation** (`idx[k] = idx[k][:500]`): Inverted lists for high-frequency keys were truncated to the first 500 records encountered during file scan order.
2. **Arbitrary candidate set capping** (`len(combined) < 100`): The combined candidate generation logic stopped adding address-assisted candidates whenever name-based candidates reached 100.

### Key Audit Conclusions:
1. **Evaluation Metric (`src/evaluation.py`) is Mathematically Sound**: The official macro $F_{0.5}$ metric is implemented exactly according to `.ps/ps.md`. Singleton handling ($1.0$ for true empty/pred empty; $0.0$ for false merge or missed matches) and competition examples match official specs with $100\%$ precision. However, input validation (detecting missing/extra S1 entities or intra-list duplicate IDs) is missing compared to `utils/validate_submission.py`.
2. **Validation Split (`src/split_data.py`) is Strictly Leak-Free**: The 10% stratified validation split (220,683 S1 entities) is strictly partitioned at the S1 level, reproducible (`seed=42`), disjoint, exhaustive, and exhibits zero target leakage.
3. **Earlier Sample Report (`experiments/phase3_report.md`) is Seriously Compromised**: The reported score of $F_{0.5} = 0.7570$ in `experiments/phase3_report.md` was obtained from a toy sample of 1,000 entities against an artificial candidate pool containing only 500,000 background records alongside ground-truth targets (`sample_true_targets`). Over 95% of background distractors were omitted, artificially inflating precision to 0.8639.
4. **Full Validation Benchmark (`experiments/phase3_full_report.md`) is Genuine but Artificially Constrained**: The full validation run evaluated all 220,683 validation S1 entities against all 10,320,219 S2/S3 records (1,492.8s, 5.46 GB RAM). Its headline score ($F_{0.5} = 0.6165$, Recall = 0.5395, Precision = 0.6807, Blocking Recall = 71.93%) is authentic and realistic, but represents a *constrained* pipeline due to the 500-posting cap and 100-candidate cap.
5. **No Test Set Contamination**: The test dataset (`dataset/test/`) has not been used for any training, tuning, feature fitting, or threshold selection.

---

## 2. Critical Findings

### [CRITICAL-01] Target Injection and Subsampled Background Pool in `run_phase3_benchmark.py`
- **File**: [`experiments/run_phase3_benchmark.py`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/run_phase3_benchmark.py#L97-L103); reported in [`experiments/phase3_report.md`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/phase3_report.md).
- **Exact Behavior**:
  ```python
  is_target = eid in sample_true_targets
  if is_target or count < 250000:
      idx.add_record(eid, name, addr, ctry)
  ```
  Only 1,000 validation S1 entities were evaluated against a pool constructed by explicitly taking all true ground-truth targets (`sample_true_targets`) and only 250,000 background records from each of S2 and S3 (total 500k records vs 10.32M actual records).
- **Why It Matters**: This omitted ~95% of negative distractors in the S2/S3 universe. Candidate pool false positives were artificially suppressed, inflating reported precision to **0.8639** and Macro $F_{0.5}$ to **0.7570**. When evaluated against the full 10.32M records, true precision dropped to **0.6807** and $F_{0.5}$ dropped to **0.6165**.
- **Affects Already-Reported Results**: Yes. The headline results in `experiments/phase3_report.md` ($F_{0.5} = 0.7570$) are invalid for real-world full-data inference.
- **Affects Future Experiments**: No, provided all future work relies on the full validation benchmark.
- **Recommended Fix**: Mark `experiments/phase3_report.md` and `experiments/phase3_results.json` as deprecated preliminary toy samples. Never cite 0.7570 as the project baseline.
- **Rerun Required**: Already superseded by `experiments/run_full_validation_phase3.py`.

---

### [CRITICAL-02] Silent Positional Posting-List Truncation (`[:max_block_size]` / `[:500]`)
- **File**: [`src/blocking.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/blocking.py#L85-L98), [`experiments/run_full_validation_phase3.py`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/run_full_validation_phase3.py#L170-L174), [`experiments/run_phase3.py`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/run_phase3.py#L145).
- **Exact Behavior**:
  ```python
  for idx in [self.idx_exact, self.idx_canon, self.idx_two_tokens, self.idx_char_prefix, self.idx_addr_num]:
      for k in list(idx.keys()):
          if len(idx[k]) > self.max_block_size:
              idx[k] = idx[k][: self.max_block_size]
  ```
  Inverted index posting lists exceeding `max_block_size = 500` were truncated using a Python list slice to the first 500 elements.
- **Why It Matters**:
  1. **Source-Order Dependence**: Retained records depend purely on the physical scan order of records in `train_source2.tsv` and `train_source3.tsv`. If the input TSV is sorted or shuffled, completely different candidate pairs are selected.
  2. **Systematic Loss of True Matches**: For entities sharing frequent name tokens (e.g. major retail chains, banks, common prefixes like "royal restaurant" or "star enterprise"), true matching records appearing after row index 500 in the source file are permanently discarded from candidate generation.
  3. **Skewed Prefix Indexing**: For character prefixes (`canon_name[:4]`), thousands of entities share prefixes like `amer`, `indi`, `shri`, or `smit`. Slicing to 500 records renders the index arbitrary and non-selective.
- **Affects Already-Reported Results**: Yes. It artificially constrained blocking recall to **71.93%** in `experiments/phase3_full_report.md`. Uncapped or statistically pruned blocking may yield higher recall.
- **Affects Future Experiments**: Yes. If not fixed, candidate generation in Phase 4 and training pair extraction in Phase 5 will silently drop true matches.
- **Recommended Fix**: Eliminate positional list slicing. Replace with principled frequency management:
  - If a key has extreme block size (exceeds an upper selectivity threshold, e.g. >1,000), treat it as a stop-word/non-selective key and discard the key entirely from low-selectivity strategies (like character prefix), OR intersect it with a secondary selective attribute (e.g., country + city or postal code), rather than taking the first 500 rows.
  - For exact name matching (`idx_exact`), never truncate; exact name equality is inherently high-precision.
- **Rerun Required**: Yes. A clean, non-truncated full validation blocking benchmark must be rerun to establish true uncapped recall.

---

### [CRITICAL-03] Arbitrary Combined Candidate Set Cap (`len(combined) < 100`) Causing Asymmetric Dropping of Address Candidates
- **File**: [`experiments/run_full_validation_phase3.py`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/run_full_validation_phase3.py#L203-L215), [`src/blocking.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/blocking.py#L154-L167).
- **Exact Behavior**:
  ```python
  combined = set()
  combined.update(c_exact)
  combined.update(c_canon)
  if len(combined) < 100:
      combined.update(c_addr)
  if len(combined) < 15:
      for idx_val in c_char:
          combined.add(idx_val)
          if len(combined) >= 100:
              break
  ```
- **Why It Matters**:
  1. **Asymmetric Dropping of High-Precision Address Candidates**: If an S1 entity matches 100 records via name-based blocking (very common for generic names or franchise brands), `c_addr` is completely bypassed. Yet, address-assisted candidates (matching exact building number or PIN code) are the *exact* candidates needed to identify the true location of a franchise.
  2. **Non-Deterministic Selection in Fallback**: Iterating over `c_char` (a Python set) and breaking at 100 adds candidates in arbitrary hash-seed order.
- **Affects Already-Reported Results**: Yes. It directly depressed the reported blocking recall in `experiments/phase3_full_report.md`.
- **Affects Future Experiments**: Yes. Downstream ML models will never receive candidate pairs for entities where name candidates saturate the quota.
- **Recommended Fix**: Do not sequentially choke off subsequent strategies based on raw candidate count. Form the full union of high-quality strategies (exact, canonical, address-assisted), and only apply top-$k$ filtering if accompanied by a ranking score (e.g., token Jaccard or BM25/TF-IDF), never by arbitrary arrival order.
- **Rerun Required**: Yes.

---

## 3. High Findings

### [HIGH-01] Missing Global One-to-One Match Assignment Constraint (Duplicate S2/S3 Merges Across S1s)
- **File**: [`experiments/run_full_validation_phase3.py`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/run_full_validation_phase3.py#L361), [`src/baseline_matcher.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/baseline_matcher.py).
- **Exact Behavior**:
  `pred_set = {cid for cid, (sc, _, _, _) in cand_scores[cfg].items() if sc >= th}`
  Predictions are evaluated for each S1 entity independently. If candidate `S2-00123` scores 0.72 with `S1-A` and 0.71 with `S1-B`, both entities predict `S2-00123`.
- **Why It Matters**: Phase 1 EDA confirmed that in the ground truth, **zero** S2 or S3 records link to multiple S1 entities (`s2_entities_matching_multiple_s1 = 0`, `s3_entities_matching_multiple_s1 = 0`). Predicting the same S2/S3 entity for multiple S1s guarantees at least one False Positive. Because $F_{0.5}$ penalizes False Positives twice as heavily as False Negatives, this omission directly caused hundreds of thousands of avoidable False Positives in the baseline (337,192 total FPs).
- **Affects Already-Reported Results**: Yes. It depressed the Phase 3 baseline score ($F_{0.5} = 0.6165$).
- **Affects Future Experiments**: Critical for Phase 5.
- **Recommended Fix**: Implement a global bipartite post-processing step (e.g., max-weight matching or greedy highest-score reservation) ensuring each S2/S3 record is assigned to at most one S1 entity.
- **Rerun Required**: No for Phase 3 baseline (it was a simple deterministic baseline), but mandatory for Phase 5 ML.

---

### [HIGH-02] Missing Format Validation & Denominator Checks in `src/evaluation.py`
- **File**: [`src/evaluation.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/evaluation.py#L120-L178, #L181-L198).
- **Exact Behavior**:
  1. `evaluate_predictions(ground_truth, predictions)` loops over `ground_truth.keys()` and calls `predictions.get(s1_id, set())`. If a user passes an incomplete prediction dictionary missing 50,000 S1 entities, the missing entities are treated as predicting `set()`. For true singletons, this awards an automatic score of $1.0$.
  2. Extra keys in `predictions` that do not exist in `ground_truth` are silently ignored without penalty.
  3. `load_tsv_mapping` silently overwrites duplicate S1 rows and silently deduplicates intra-list duplicate IDs with `{x.strip() for x in ...}`.
- **Why It Matters**: This conflicts with [`utils/validate_submission.py`](file:///home/howtoquitvivek/amazon-ml-2026/utils/validate_submission.py#L141-L203), which treats missing S1 rows, duplicate S1 rows, and intra-list duplicate IDs as fatal submission rejections. Code evaluated using `src/evaluation.py` could appear to score well locally but fail official submission validation.
- **Affects Already-Reported Results**: No (in full validation, all 220,683 IDs were verified to match exactly).
- **Affects Future Experiments**: Yes, poses a silent verification risk for candidate and prediction generators.
- **Recommended Fix**: Add assertion checks in `src/evaluation.py` verifying `set(predictions.keys()) == set(ground_truth.keys())` and logging warnings/errors if malformed rows or intra-list duplicate IDs are encountered.
- **Rerun Required**: No.

---

### [HIGH-03] Non-Deterministic Character Prefix Candidate Selection
- **File**: [`src/blocking.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/blocking.py#L161-L167), [`experiments/run_full_validation_phase3.py`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/run_full_validation_phase3.py#L209-L215).
- **Exact Behavior**:
  ```python
  for idx_val in c_char:  # c_char is a set
      combined.add(idx_val)
      if len(combined) >= 100:
          break
  ```
- **Why It Matters**: Iterating over a Python `set` is order-dependent on Python's hash randomization seed (`PYTHONHASHSEED`). Two identical runs with different process hash seeds will select different candidate sets whenever `len(c_char)` pushes `combined` over 100.
- **Affects Already-Reported Results**: Marginal (minor fluctuation in character prefix candidate selection).
- **Affects Future Experiments**: Yes, violates strict reproducibility.
- **Recommended Fix**: If capping is ever applied, sort candidates by a deterministic key (e.g. record ID or similarity score) before slicing.
- **Rerun Required**: Addressed when rerunning the uncapped benchmark.

---

## 4. Medium Findings

### [MEDIUM-01] Global Legal Suffix Stripping Collapsing Legitimate Indian & US Name Tokens
- **File**: [`src/normalization.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/normalization.py#L24-L38, #L147-L148).
- **Exact Behavior**:
  `LEGAL_SUFFIX_SET` combines suffixes across US, India, and France into a single static set (`"sa"`, `"co"`, `"pc"`, `"gie"`, `"snc"`, `"pvt"`, `"llc"`, etc.). In `normalize_and_canonicalize`, any token present in this set is stripped globally.
- **Why It Matters**:
  - The token `"sa"` is a French legal suffix (*Société Anonyme*), but in India and the US, "Sa" can be a valid name word (e.g., "Sa Re Ga Ma", "Sa Food", "Sa Enterprises"). Under current normalization, "Sa Food" has "sa" stripped, leaving only "food".
  - The token `"pc"` is a US legal suffix (*Professional Corporation*), but in technology businesses it represents "Personal Computer" (e.g. "PC Care", "Apex PC Solutions").
  - The token `"co"` is stripped anywhere, affecting names like "Co Op" or "T & Co".
- **Affects Already-Reported Results**: Causes minor false name collapses for short business names.
- **Affects Future Experiments**: Yes, in Phase 4 normalization.
- **Recommended Fix**: Condition legal suffix stripping on country context (apply French suffixes only to French records, Indian only to Indian records), or enforce position constraints (e.g., strip legal suffixes only at the end of a name string rather than arbitrary token removal).
- **Rerun Required**: No immediate rerun; incorporate into Phase 4 normalization refinement.

---

### [MEDIUM-02] Omission of Accented French Legal Suffixes
- **File**: [`src/normalization.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/normalization.py#L31).
- **Exact Behavior**: Suffix list includes `"sarl"`, `"sas"`, `"sasu"`, `"sci"`, `"sa"`, `"eurl"`, `"snc"`, `"gie"`, but omits accented variants like `"sàrl"`.
- **Why It Matters**: In French business data (which appears in the test set), "SARL" is frequently spelled "Sàrl". Since Unicode normalization preserves accented characters (`à`), "sàrl" is not stripped, while "sarl" is stripped, preventing exact canonical match between spelling variants.
- **Affects Already-Reported Results**: No (training and validation datasets contain only US and India).
- **Affects Future Experiments**: Yes, affects test set evaluation on France.
- **Recommended Fix**: Add `"sàrl"` to the legal suffix dictionary.
- **Rerun Required**: No.

---

### [MEDIUM-03] Report Terminology Ambiguity in `experiments/phase3_report.md`
- **File**: [`experiments/phase3_report.md`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/phase3_report.md#L1-L7).
- **Exact Behavior**: The document is titled "Phase 3 Report" and leads with:
  *"The resulting deterministic baseline achieves a Macro $F_{0.5}$ score of 0.7570 (Precision: 0.8639, Recall: 0.5814, Singleton Accuracy: 94.92%)."*
  Nowhere in the executive summary or title does it state that this was measured on a 1,000-entity sample with artificial negative subsampling.
- **Why It Matters**: Creates ambiguity between the 1k toy sample ($F_{0.5} = 0.7570$) and the genuine full-validation baseline ($F_{0.5} = 0.6165$).
- **Affects Already-Reported Results**: Documentation and transparency issue.
- **Affects Future Experiments**: No.
- **Recommended Fix**: Add a clear disclaimer banner at the top of `experiments/phase3_report.md` pointing to `experiments/phase3_full_report.md`.
- **Rerun Required**: No.

---

### [MEDIUM-04] Hardcoded Country Lists in Benchmark Sampling Scripts
- **File**: [`experiments/run_phase3.py`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/run_phase3.py#L93-L94), [`experiments/run_phase3_benchmark.py`](file:///home/howtoquitvivek/amazon-ml-2026/experiments/run_phase3_benchmark.py#L62-L63).
- **Exact Behavior**:
  ```python
  us_ids = [k for k, v in val_records.items() if v["country"] == "US"][:600]
  in_ids = [k for k, v in val_records.items() if v["country"] == "India"][:400]
  ```
- **Why It Matters**: Violates the open-set country rule from `.ps/ps.md` ("Treat country as an open set of string labels: do not hard-code, filter, or one-hot your pipeline to only {US, India}"). If these scripts were ever run on test data containing France, French records would be ignored.
- **Affects Already-Reported Results**: No (validation set has only US and India).
- **Affects Future Experiments**: Yes, if sampling utilities are reused.
- **Recommended Fix**: Dynamically discover countries from `country_partitions.keys()` (as done properly in `run_full_validation_phase3.py`).
- **Rerun Required**: No.

---

## 5. Low Findings

### [LOW-01] Domain Regex Does Not Support Multi-Part TLDs or Subdomains
- **File**: [`src/normalization.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/normalization.py#L18-L21).
- **Exact Behavior**: `DOMAIN_PATTERN` uses `([a-z0-9\-]+)\.(?:com|org|net|co|io|in|co\.in|fr|biz|info|us)`. Subdomains (e.g. `shop.brand.com`) or unlisted TLDs (e.g. `.co.uk`, `.store`, `.agency`) are not extracted.
- **Why It Matters**: Minor loss in domain normalization for businesses named after websites.
- **Affects Already-Reported Results**: Negligible.
- **Affects Future Experiments**: Low impact.
- **Recommended Fix**: Broaden domain matching regex in Phase 4.
- **Rerun Required**: No.

---

### [LOW-02] Silent TSV Overwriting on Duplicate S1 Keys in `load_tsv_mapping`
- **File**: [`src/evaluation.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/evaluation.py#L196).
- **Exact Behavior**: In `load_tsv_mapping`, lines are parsed into a dictionary `mapping[s1_id] = ids`. If an input file contains duplicate rows for the same `s1_id`, earlier rows are silently overwritten without warning.
- **Why It Matters**: `utils/validate_submission.py` specifically rejects duplicate S1 rows.
- **Affects Already-Reported Results**: No (no duplicates exist in source files).
- **Affects Future Experiments**: Low diagnostic risk.
- **Recommended Fix**: Add duplicate row detection to `load_tsv_mapping`.
- **Rerun Required**: No.

---

### [LOW-03] Heuristic Neutral Address Imputation in Baseline Matcher
- **File**: [`src/baseline_matcher.py`](file:///home/howtoquitvivek/amazon-ml-2026/src/baseline_matcher.py#L124-L125, #L139-L141).
- **Exact Behavior**: When candidate address is missing, `addr_subscore` is set to 0.50 (or 0.85 for exact names).
- **Why It Matters**: This heuristic rewards common names that happen to lack addresses in S2/S3, contributing to false merges.
- **Affects Already-Reported Results**: Known baseline heuristic limitation.
- **Affects Future Experiments**: Handled by ML model in Phase 5 via learned missingness indicators.
- **Recommended Fix**: In Phase 5 ML feature extraction, provide `addr_missing` as an explicit indicator feature and allow tree-based models to learn the exact penalty.
- **Rerun Required**: No.

---

## 6. File-by-File Audit

| File | Status | Audit Findings & Verification |
| :--- | :---: | :--- |
| `src/evaluation.py` | **PASS (with warnings)** | Exact mathematical implementation of macro $F_{0.5}$ and singleton scoring. Matches all `.ps/ps.md` examples. Lacks strict key-equality assertions ([HIGH-02]) and duplicate detection ([LOW-02]). |
| `src/split_data.py` | **PASS** | 100% leak-free, S1-level, reproducible (`seed=42`), stratified split. Train and validation IDs are strictly disjoint and exhaustive. Zero test data involvement. |
| `src/normalization.py` | **PASS (with warnings)** | Unicode NFKC, Devanagari and Latin accents preserved. Identified cross-lingual legal suffix collision ([MEDIUM-01]), missing French accented suffix ([MEDIUM-02]), and restricted domain regex ([LOW-01]). |
| `src/blocking.py` | **FAIL (methodological)** | Contains hard block pruning to 500 records ([CRITICAL-02]) and arbitrary combined candidate capping at 100 ([CRITICAL-03]). Must be revised before Phase 4. |
| `src/baseline_matcher.py` | **PASS (as baseline)** | Pure deterministic pairwise feature calculator. Country equality gate strictly enforced. Lacks global 1-to-1 assignment constraint ([HIGH-01]) and uses heuristic missing imputation ([LOW-03]). |
| `experiments/run_eda.py` | **PASS** | Memory-safe chunked streaming. Accurately computed all schema and missingness distributions. No data leakage. |
| `experiments/sample_noise_patterns.py` | **PASS** | Accurate qualitative pattern extraction across 30,000 true pairs. |
| `experiments/run_phase3_benchmark.py` | **FAIL (methodological)** | Injected ground-truth match targets into a reduced 500k candidate pool ([CRITICAL-01]). Biased metrics. |
| `experiments/run_phase3.py` | **OBSOLETE** | Unused draft script that previously failed due to memory exhaustion. |
| `experiments/run_full_validation_phase3.py` | **QUALIFIED PASS** | Genuine full-dataset execution (220,683 S1s vs 10.32M S2/S3s; 1,492.8s; peak RAM 5.46 GB). Validated 100% match existence. Suffered from 500-posting cap and 100-candidate cap inherited from `blocking.py`. |
| `experiments/phase3_results.json` | **DEPRECATED** | Contains biased results from `run_phase3_benchmark.py`. |
| `experiments/phase3_report.md` | **MISLEADING** | Reports $F_{0.5} = 0.7570$ based on biased 1k sample ([MEDIUM-03]). |
| `experiments/phase3_full_results.json` | **TRUSTWORTHY (CONSTRAINED)** | Real full validation run across 220,683 entities ($F_{0.5} = 0.6165$), but candidate set was capped at 100. |
| `experiments/phase3_full_report.md` | **TRUSTWORTHY (CONSTRAINED)** | Authoritative full-scale benchmark report, accurately representing the capped baseline. |
| `notebooks/01_data_exploration.ipynb` | **PASS** | Clean exploratory notebook displaying dataset statistics and noise patterns. |
| `tests/test_evaluation.py` | **PASS** | 7 unit tests verifying singleton logic, competition examples, and macro averaging. All pass. |
| `tests/test_normalization.py` | **PASS** | Verifies Unicode preservation, Devanagari, legal canonicalization, and addresses. |
| `tests/test_baseline_matcher.py` | **PASS** | Verifies Jaccard logic, feature computation, and cross-country gating. |
| `README.md` & `.ps/ps.md` | **ALIGNED** | Full alignment with problem statement, constraints, evaluation metric, and file schemas. |
| `Documentation_template.md` | **ALIGNED** | Submission methodology template ready for Phase 8 documentation. |
| `requirements.txt` | **PASS** | Pinned core dependencies (`pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `tqdm`, `joblib`). |

---

## 7. Which Existing Results Are Trustworthy?

### Trustworthy Results:
1. **Phase 1 EDA Statistics (`experiments/eda_statistics.json`)**:
   - Total row counts (26,435,994 rows).
   - Missingness rates (0% missing names/countries; S2/S3 addresses ~2.7–3.4% blank; S1 addresses 100% complete).
   - Cardinality findings: 0 cross-country matches; 0 S2/S3 entities matching multiple S1 entities in ground truth.
2. **Phase 2 Validation Split (`experiments/validation_split_summary.json`, `val_s1_ids.txt`, `train_s1_ids.txt`)**:
   - 220,683 validation S1 entities strictly stratified by country and match count.
   - Completely disjoint and reproducible (`seed=42`).
3. **Data Integrity Ground-Truth Verification (`run_full_validation_phase3.py`)**:
   - 100% of all 763,592 validation ground-truth matches exist in `train_source2.tsv` and `train_source3.tsv` (369,516 S2, 394,076 S3).
4. **Country Partitioning as a Hard Block**:
   - Verified 100% recall with zero cross-country true matches.
5. **Phase 3 Full Validation Deterministic Metrics (`experiments/phase3_full_report.md`)**:
   - The headline metrics ($F_{0.5} = 0.6165$, Precision = 0.6807, Recall = 0.5395, Peak RSS = 5,462.3 MB) are authentic, leak-free measurements over the entire 220,683-entity validation set under the stated capping conditions.

### Untrustworthy / Deprecated Results:
1. **Sample Benchmark Results (`experiments/phase3_results.json`, `experiments/phase3_report.md`)**:
   - The reported Macro $F_{0.5} = 0.7570$ and Precision = 0.8639 are **completely untrustworthy** for real evaluation due to target injection and the omission of 95% of background distractors.
2. **Uncapped Blocking Recall Ceiling**:
   - The reported blocking recall of **71.93%** is an underestimate of true multi-strategy potential because high-frequency blocks were truncated to 500 and candidate sets were capped at 100.

---

## 8. Which Results Must Be Rerun?

1. **Uncapped / Statistically Pruned Full-Validation Blocking Benchmark**:
   - **Why**: To establish the true recall ceiling of candidate generation without arbitrary positional list slicing (`[:500]`) or asymmetric address skipping (`len(combined) < 100`).
   - **Scope**: Evaluate on all 220,683 validation S1 entities against all 10.32M records using memory-safe inverted indices with frequency-based stop-wording instead of positional slicing.
2. **Deterministic Baseline With 1-to-1 Bipartite Resolution (Optional Baseline Update)**:
   - Applying a greedy 1-to-1 assignment constraint on the full validation predictions will establish how much of the 337,192 False Positives can be eliminated prior to ML modeling.

---

## 9. Exact Remediation Order

To maintain strict scientific integrity, the following remediation sequence must be followed:

1. **Step 1: Deprecate Biased Sample Reports**
   - Update `experiments/phase3_report.md` with an explicit notice that its numbers reflect a 1k toy sample with synthetic background downsampling, pointing readers to `experiments/phase3_full_report.md`.
2. **Step 2: Hardening `src/evaluation.py`**
   - Add assertion checks verifying that `set(predictions.keys()) == set(ground_truth.keys())`.
   - Add duplicate row and duplicate intra-list candidate ID checks to mirror `utils/validate_submission.py`.
3. **Step 3: Refactor `src/blocking.py` Candidate Generation**
   - Remove the arbitrary `idx[k][:500]` slice.
   - For high-frequency non-selective keys (e.g. block size > 1,000 in prefix index), treat as a non-selective stop-key or combine with city/postal-code anchors rather than taking an arbitrary 500 records.
   - Remove the `if len(combined) < 100` condition that suppresses address-assisted candidates.
   - Ensure deterministic candidate ranking before any size throttling.
4. **Step 4: Refine `src/normalization.py` for Multi-Country Safety**
   - Condition legal suffixes on country context (US, India, France).
   - Add accented French suffixes (`"sàrl"`).
5. **Step 5: Run Clean Full Validation Benchmark**
   - Provide the clean benchmark script for user execution in the terminal to measure true uncapped recall and baseline $F_{0.5}$.

---

## 10. Recommended Gate Before Phase 4

Before any Phase 4 (Advanced Candidate Generation) or Phase 5 (Pairwise ML) code is developed:

> [!IMPORTANT]
> **PHASE 4 ENTRY GATE CRITERIA**:
> 1. `src/blocking.py` has zero positional slicing (`[:500]`) and zero asymmetric candidate dropping (`if len(combined) < 100`).
> 2. `src/evaluation.py` enforces exact S1 key matching and flags intra-list duplicate IDs.
> 3. The true uncapped blocking recall on the full 220,683 validation entities is measured and recorded as the official blocking recall ceiling.
> 4. The baseline to beat for all future ML models is formally documented as **Macro $F_{0.5} = 0.6165$** (or the updated clean baseline), **never** $0.7570$.

---

## 11. Findings Summary Table

| Finding ID | Description | Severity | Affects Existing Results? | Rerun Required? |
| :--- | :--- | :---: | :---: | :---: |
| **CRITICAL-01** | Target injection & 500k background pool in sample benchmark (`run_phase3_benchmark.py`) | **CRITICAL** | Yes (invalidates `phase3_report.md`) | Already superseded by full run |
| **CRITICAL-02** | Silent posting list truncation (`[:500]`) in `blocking.py` and `run_full_validation_phase3.py` | **CRITICAL** | Yes (suppressed full blocking recall to 71.93%) | **Yes** (rerun clean blocking) |
| **CRITICAL-03** | Combined candidate cap (`len(combined) < 100`) asymmetrically dropping address candidates | **CRITICAL** | Yes (suppressed full blocking recall) | **Yes** (rerun clean blocking) |
| **HIGH-01** | Missing 1-to-1 bipartite assignment constraint generating duplicate S2/S3 matches across S1s | **HIGH** | Yes (inflated baseline False Positives to 337k) | No for baseline; mandatory for Phase 5 |
| **HIGH-02** | Missing S1 entity set validation & intra-list duplicate detection in `evaluation.py` | **HIGH** | No (full run had exact IDs) | No |
| **HIGH-03** | Non-deterministic set iteration order in fallback character candidate selection | **HIGH** | Minor | Yes (resolved in clean blocking rerun) |
| **MEDIUM-01** | Global legal suffix set stripping valid Indian/US tokens ("sa", "co", "pc") | **MEDIUM** | Minor (rare name collisions) | No (resolve in Phase 4) |
| **MEDIUM-02** | Missing accented French legal suffix (`sàrl`) in `normalization.py` | **MEDIUM** | No (train/val are US/India) | No (resolve in Phase 4) |
| **MEDIUM-03** | Ambiguous executive summary in `phase3_report.md` claiming 0.7570 without sample qualification | **MEDIUM** | Documentation issue | No (update markdown header) |
| **MEDIUM-04** | Hardcoded country list (`["US", "India"]`) in preliminary sampling scripts | **MEDIUM** | No (only run on train data) | No |
| **LOW-01** | Incomplete domain TLD and subdomain regex coverage in `normalization.py` | **LOW** | Negligible | No |
| **LOW-02** | Silent TSV overwrite on duplicate S1 rows in `load_tsv_mapping` | **LOW** | No (no duplicate rows in data) | No |
| **LOW-03** | Heuristic missing address imputation bias in deterministic scoring | **LOW** | Contributed to baseline FPs | No (handled by ML in Phase 5) |
