# Phase 6C: Final Test Inference Preflight Audit

Prior to writing and executing the final test inference script (`experiments/phase6/run_test_inference_xgb.py`), its planned architecture and logic have been strictly audited against the 12 requirements. 

## Audit Results

1. **Reads ONLY dataset/test/*:** **PASS.** The script will strictly hardcode the paths `dataset/test/test_source1.tsv`, `dataset/test/test_source2.tsv`, and `dataset/test/test_source3.tsv`. It will explicitly forbid any `dataset/train/` paths.
2. **No train/validation ground truth:** **PASS.** `train_ground_truth.tsv` and `train_s1_ids.txt` will not be imported or referenced. The final `.ubj` model contains no external dataset linkages.
3. **Candidate generation methodology:** **PASS.** The test S2/S3 records will be ingested to build the exact same inverted index schema used in Phase 4. Candidate generation will strictly call `query_custom_s1(..., budget=100, cfg="CFG1_NoTwoTok_KeyE")`.
4. **`candidate_pairs.tsv` strictly reflects XGBoost input:** **PASS.** The script will generate the candidate pairs file during the feature extraction loop. The candidates written for each S1 will be exactly the IDs present in the `budget_eids` list that proceeds to feature engineering. Format: `S1_ID \t Cand1,Cand2,...`
5. **Matches are a subset of candidates:** **PASS.** Because the model scores only the candidates from `budget_eids`, any ID that passes the `0.90` threshold is guaranteed mathematically to exist in `candidate_pairs.tsv`.
6. **`matching_results.tsv` has exactly one row per test S1:** **PASS.** The outer loop will process every single row from `test_source1.tsv`. It will emit exactly one row to `matching_results.tsv` for every S1, regardless of how many candidates passed.
7. **Singletons produce an empty list:** **PASS.** If no candidates pass the `0.90` threshold (or if the blocker returned 0 candidates), the row will be written as `S1_ID \t ` (an empty list after the tab), exactly fulfilling the official requirement.
8. **No duplicate matched IDs:** **PASS.** The passing IDs will be collapsed into a `set` before joining into the comma-separated string, guaranteeing uniqueness.
9. **Every matched ID exists in test S2/S3:** **PASS.** Because candidate generation operates strictly on the test S2/S3 index built in step 1, every matched ID is guaranteed to exist.
10. **Source IDs are preserved exactly:** **PASS.** No stripping, lowering, or formatting will be applied to the raw ID strings read from the first column of the TSV files.
11. **Output format matches specification:** **PASS.** Both output files will be strictly tab-separated (`\t`) with comma-separated ID lists (`cand1,cand2`), adhering to `utils/validate_submission.py`. The header will be `source1_entity_id \t matched_entity_ids` (and `candidate_entity_ids` for the candidate file).
12. **Run official validator:** **PLANNED.** I will write a dry-run flag (`--dry-run`) to execute the script on the first 10,000 test S1s. I will then run `python3 utils/validate_submission.py --check-ids` on the output to guarantee zero format errors before scaling to the full 1.7M test set.

## Conclusion
The proposed inference script architecture completely satisfies the Phase 6C integrity requirements. The frozen model (`n_estimators=100`, `threshold=0.90`) remains perfectly untouched.
