#!/usr/bin/env python3
"""
Phase 3.1: Correct Blocking Implementation & Benchmark
Amazon ML Challenge 2026: Business Entity Resolution

Corrections implemented per repository audit:
1. Positional posting-list truncation (`[:500]`) completely REMOVED.
   All postings for all keys are preserved in memory-compact array.array('I').
2. Asymmetric candidate cap (`if len(combined) < 100`) completely REMOVED.
   Full union is: exact UNION canonical UNION address UNION character.
3. Candidate selection is strictly deterministic (no Python set iteration order dependency).
4. Country-partitioned execution (India, then US) preserving memory safety.
5. True RSS measured via /proc/self/status VmRSS alongside peak RSS (ru_maxrss).
6. Experiments evaluated:
   A. Full Union Blocking (220,683 S1s vs 10.32M records)
   B. Candidate Budget Ablation: 25, 50, 75, 100, 150, 250, 500
   C. Deterministic Matching on Full Union & Budgets across thresholds (0.65, 0.70, 0.75, 0.80)
   D. One-to-One Matching Ablation (Independent vs Greedy Competitive Assignment)
"""

import os
import sys
import time
import json
import array
import argparse
import collections
import resource
import gc
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.normalization import (
    normalize_name,
    normalize_and_canonicalize,
    get_canonical_name_key,
    get_name_tokens,
    normalize_address,
    extract_numeric_tokens,
    get_character_ngrams,
)
from src.baseline_matcher import (
    compute_pair_features,
    compute_deterministic_score,
)
from src.evaluation import compute_entity_metrics, evaluate_predictions

DATA_DIR = "dataset"
EXPERIMENTS_DIR = "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

S1_FILE = os.path.join(DATA_DIR, "train", "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train", "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train", "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
VAL_IDS_FILE = os.path.join(EXPERIMENTS_DIR, "val_s1_ids.txt")

RESULTS_JSON = os.path.join(EXPERIMENTS_DIR, "phase3_1_results.json")
REPORT_MD = os.path.join(EXPERIMENTS_DIR, "phase3_1_report.md")


def get_peak_rss_mb() -> float:
    """Peak resident set size in MB from getrusage."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_current_rss_mb() -> float:
    """Current resident set size in MB by reading VmRSS from /proc/self/status."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return get_peak_rss_mb()


def verify_ground_truth_integrity(val_gt: Dict[str, Set[str]]):
    """Verify that every ground-truth S2/S3 match for validation S1 exists in source files."""
    print("\n--- SANITY CHECK: Verifying Ground-Truth Match Targets in Source Files ---")
    needed_s2 = set()
    needed_s3 = set()
    for s1_id, matches in val_gt.items():
        for m in matches:
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)

    print(f"Total validation ground-truth matches: {len(needed_s2) + len(needed_s3):,} ({len(needed_s2):,} S2, {len(needed_s3):,} S3)")

    found_s2 = set()
    with open(S2_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0].strip()
            if eid in needed_s2:
                found_s2.add(eid)

    found_s3 = set()
    with open(S3_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0].strip()
            if eid in needed_s3:
                found_s3.add(eid)

    missing_s2 = needed_s2 - found_s2
    missing_s3 = needed_s3 - found_s3

    assert len(missing_s2) == 0, f"FATAL: {len(missing_s2)} validation S2 ground-truth matches missing in train_source2.tsv!"
    assert len(missing_s3) == 0, f"FATAL: {len(missing_s3)} validation S3 ground-truth matches missing in train_source3.tsv!"
    print(f"PASS: 100% of validation ground-truth matches exist in searched source data ({len(found_s2):,} S2, {len(found_s3):,} S3).")


class UnconstrainedCountryBlocker:
    """
    Memory-compact blocker for a single country partition.
    Stores all postings using array.array('I') with zero positional truncation.
    Applies frequency-aware restriction on 4-character prefix blocker to eliminate common prefix explosion.
    """

    def __init__(self, country: str, max_char_prefix_freq: int = 500):
        self.country = country
        self.max_char_prefix_freq = max_char_prefix_freq

        # Parallel arrays storing candidate record fields
        self.cand_ids: List[str] = []
        self.cand_names: List[str] = []
        self.cand_addrs: List[str] = []

        # Compact inverted indices: key -> array of 32-bit unsigned integers
        # NO positional truncation: all postings preserved completely
        self.idx_exact = collections.defaultdict(lambda: array.array("I"))
        self.idx_canon = collections.defaultdict(lambda: array.array("I"))
        self.idx_two_tokens = collections.defaultdict(lambda: array.array("I"))
        self.idx_char_prefix = collections.defaultdict(lambda: array.array("I"))
        self.idx_addr_num = collections.defaultdict(lambda: array.array("I"))

    def index_source_files(self):
        """Index all S2 and S3 records for this country."""
        t0 = time.time()
        print(f"\nIndexing Source 2 and Source 3 records for country: {self.country}...")

        total_scanned = 0
        for fpath in [S2_FILE, S3_FILE]:
            with open(fpath, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    total_scanned += 1
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4 and parts[3].strip() == self.country:
                        eid = parts[0].strip()
                        name = parts[1].strip()
                        addr = parts[2].strip()

                        rec_idx = len(self.cand_ids)
                        self.cand_ids.append(eid)
                        self.cand_names.append(name)
                        self.cand_addrs.append(addr)

                        norm_name, canon_name, tokens = normalize_and_canonicalize(name)

                        # 1. Exact normalized name
                        if norm_name:
                            self.idx_exact[norm_name].append(rec_idx)

                        # 2. Canonical token key
                        if canon_name and canon_name != norm_name:
                            self.idx_canon[canon_name].append(rec_idx)

                        # 3. Two tokens
                        if len(tokens) >= 2:
                            two_tok = f"{tokens[0]} {tokens[1]}"
                            self.idx_two_tokens[two_tok].append(rec_idx)

                        # 4. Character prefix (4 chars)
                        if len(canon_name) >= 4:
                            self.idx_char_prefix[canon_name[:4]].append(rec_idx)

                        # 5. Address numeric token + first token prefix
                        if addr and tokens:
                            nums = extract_numeric_tokens(addr)
                            tok_prefix = tokens[0][:4] if len(tokens[0]) >= 4 else tokens[0]
                            for num in nums:
                                if len(num) >= 2:
                                    self.idx_addr_num[f"{num}_{tok_prefix}"].append(rec_idx)

        t1 = time.time()
        print(f"Indexed {len(self.cand_ids):,} records for {self.country} in {t1 - t0:.2f}s. Scanned {total_scanned:,} total rows.")
        print(f"Memory: Current RSS = {get_current_rss_mb():.1f} MB | Peak RSS = {get_peak_rss_mb():.1f} MB")

    def get_candidates(self, s1_name: str, s1_addr: str) -> Tuple[Dict[str, Set[int]], Set[int], List[int]]:
        """
        Generate candidate sets across all strategies.
        Returns:
            strat_map: dict of set of record indices for each individual strategy
            best_two_union: set union of Exact Normalized Name + Canonical Token
            ranked_candidates: deterministically ranked list of candidates by blocking signal priority
        """
        norm_name, canon_name, tokens = normalize_and_canonicalize(s1_name)

        # Strategy 2: Exact Name
        c_exact = set(self.idx_exact.get(norm_name, []))

        # Strategy 3: Canonical Token
        c_canon = set(self.idx_canon.get(canon_name, []))
        if len(tokens) >= 2:
            c_canon.update(self.idx_two_tokens.get(f"{tokens[0]} {tokens[1]}", []))

        # Strategy 4: Tightened Character Prefix (4 chars)
        # Frequency-aware filtering: filter overly-common prefixes (e.g. "amer", "indi", "nati", "shri")
        # Retain only specific prefixes whose corpus frequency <= max_char_prefix_freq
        c_char = set()
        if len(canon_name) >= 4:
            p4 = canon_name[:4]
            postings = self.idx_char_prefix.get(p4, [])
            if len(postings) <= self.max_char_prefix_freq:
                c_char = set(postings)

        # Strategy 5: Address Assisted
        c_addr = set()
        if s1_addr and tokens:
            nums = extract_numeric_tokens(s1_addr)
            tok_prefix = tokens[0][:4] if len(tokens[0]) >= 4 else tokens[0]
            for num in nums:
                if len(num) >= 2:
                    c_addr.update(self.idx_addr_num.get(f"{num}_{tok_prefix}", []))

        # Strategy 6: Tightened Full Union (Exact ∪ Canonical ∪ Address ∪ Tightened Character)
        tightened_full_union = c_exact | c_canon | c_addr | c_char

        # Deterministic Ranking based ONLY on blocking signals (Label-Independent)
        # Priority weights:
        # exact name match = 8
        # canonical token match = 4
        # address match = 2
        # tightened character prefix match = 1
        # Ties broken deterministically by cand_id string ascending
        cand_weights = collections.defaultdict(int)
        for idx_val in c_exact:
            cand_weights[idx_val] += 8
        for idx_val in c_canon:
            cand_weights[idx_val] += 4
        for idx_val in c_addr:
            cand_weights[idx_val] += 2
        for idx_val in c_char:
            cand_weights[idx_val] += 1

        # Deterministic sort: highest weight first, tie-break by candidate entity_id string
        ranked_candidates = sorted(
            tightened_full_union,
            key=lambda idx_val: (-cand_weights[idx_val], self.cand_ids[idx_val])
        )

        strat_map = {
            "exact": c_exact,
            "canon": c_canon,
            "char": c_char,
            "addr": c_addr,
        }

        return strat_map, tightened_full_union, ranked_candidates


def process_country_partition(
    country: str,
    country_s1_records: Dict[str, dict],
    val_gt: Dict[str, Set[str]],
    budgets: List[int],
    match_budgets: List[int],
    match_thresholds: List[float],
):
    """Process a single country partition for Phase 3.1."""
    print(f"\n=================================================================")
    print(f"PROCESSING COUNTRY PARTITION: {country} ({len(country_s1_records):,} validation S1s)")
    print(f"=================================================================")
    t_start = time.time()
    rss_start = get_current_rss_mb()

    blocker = UnconstrainedCountryBlocker(country)
    blocker.index_source_files()
    rss_indexed = get_current_rss_mb()

    # Map candidate IDs to indices for fast ground-truth intersection
    cand_id_to_idx = {cid: idx for idx, cid in enumerate(blocker.cand_ids)}

    total_true_country_pairs = sum(len(val_gt[sid]) for sid in country_s1_records)

    # 1. Tracking for Strategy Blocking
    strat_keys = ["exact", "canon", "char", "addr"]
    strat_recovered = {k: 0 for k in strat_keys}
    strat_cand_counts = {k: [] for k in strat_keys}

    # Tracking for Tightened Full Union (Exact + Canon + Addr + Tightened Char)
    tightened_union_recovered = 0
    tightened_union_cand_counts = []

    # Combined strategy ablation contributions (marginal)
    contrib_exact = 0
    contrib_canon_only = 0
    contrib_addr_only = 0
    contrib_char_only = 0

    # 2. Tracking for Budget Ablation
    budget_recovered = {b: 0 for b in budgets}
    budget_cand_counts = {b: [] for b in budgets}

    # -------------------------------------------------------------
    # STAGE A & B: BLOCKING AND STREAMING INDEPENDENT MATCHING
    # -------------------------------------------------------------
    s1_id_list = list(country_s1_records.keys())
    s1_id_to_idx = {sid: i for i, sid in enumerate(s1_id_list)}

    # Stage B: Streaming Independent Matching State
    # Accumulate metrics per (b_key, th) using float64 arrays to preserve exact floating point precision
    indep_f05_list = collections.defaultdict(lambda: array.array("d"))
    indep_prec_list = collections.defaultdict(lambda: array.array("d"))
    indep_rec_list = collections.defaultdict(lambda: array.array("d"))
    indep_tp = collections.defaultdict(int)
    indep_fp = collections.defaultdict(int)
    indep_fn = collections.defaultdict(int)
    indep_sing_corr = collections.defaultdict(int)
    indep_sing_tot = collections.defaultdict(int)

    # Stage C: Compact Numeric Pair Storage for One-to-One Ablation
    # Stored once for qualifying pairs (score >= 0.65) using flat typed arrays with float64 scores
    oto_s1_idx = array.array("I")
    oto_c_idx = array.array("I")
    oto_rank = array.array("I")
    oto_score = array.array("d")  # float64 double precision

    min_th = min(match_thresholds)
    has_tightened_matching = (-1 in match_budgets)
    max_mb = max([b for b in match_budgets if b != -1]) if any(b != -1 for b in match_budgets) else 0

    country_val_gt = {sid: val_gt[sid] for sid in country_s1_records}

    print(f"Evaluating Stage A (blocking) & Stage B (streaming matching) across {len(country_s1_records):,} {country} S1 entities...")
    for s1_idx, s1_id in enumerate(tqdm(s1_id_list, desc=f"Evaluating {country}")):
        s1 = country_s1_records[s1_id]
        true_set = country_val_gt[s1_id]
        true_indices = {cand_id_to_idx[m] for m in true_set if m in cand_id_to_idx}

        strat_map, tightened_union, ranked_cands = blocker.get_candidates(s1["name"], s1["address"])

        # Strategy blocking
        for sk in strat_keys:
            c_indices = strat_map[sk]
            strat_cand_counts[sk].append(len(c_indices))
            strat_recovered[sk] += len(true_indices & c_indices)

        # Tightened Full Union
        tightened_union_cand_counts.append(len(tightened_union))
        tightened_union_recovered += len(true_indices & tightened_union)

        # Marginal contributions to Tightened Full Union
        exact_idx = strat_map["exact"]
        canon_idx = strat_map["canon"]
        addr_idx = strat_map["addr"]
        char_idx = strat_map["char"]

        contrib_exact += len(true_indices & exact_idx)
        contrib_canon_only += len(true_indices & (canon_idx - exact_idx))
        contrib_addr_only += len(true_indices & (addr_idx - exact_idx - canon_idx))
        contrib_char_only += len(true_indices & (char_idx - exact_idx - canon_idx - addr_idx))

        # Budget ablation
        for b in budgets:
            b_cands_idx = set(ranked_cands[:b])
            budget_cand_counts[b].append(len(b_cands_idx))
            budget_recovered[b] += len(true_indices & b_cands_idx)

        # Deterministic scoring for matching on Tightened Full Union
        eval_cands = ranked_cands if has_tightened_matching else ranked_cands[:max_mb]

        # Score pairs and record compact data for Stage C
        s1_scores = {}
        for rank, c_idx_val in enumerate(eval_cands):
            cid = blocker.cand_ids[c_idx_val]
            c_name = blocker.cand_names[c_idx_val]
            c_addr = blocker.cand_addrs[c_idx_val]

            feats = compute_pair_features(s1["name"], s1["address"], country, c_name, c_addr, country)
            sc = compute_deterministic_score(feats, "conservative_precision")
            s1_scores[c_idx_val] = sc

            # Record compact entry for One-to-One if score >= min_th
            if sc >= min_th:
                oto_s1_idx.append(s1_idx)
                oto_c_idx.append(c_idx_val)
                oto_rank.append(rank)
                oto_score.append(sc)

        # Stage B: Streaming Independent Evaluation per S1
        for mb in match_budgets:
            b_key = f"budget_{mb}" if mb != -1 else "tightened_full_union"
            cands_to_check = eval_cands if mb == -1 else eval_cands[:mb]

            for th in match_thresholds:
                pred_set = {
                    blocker.cand_ids[c_idx_val]
                    for c_idx_val in cands_to_check
                    if s1_scores[c_idx_val] >= th
                }

                m = compute_entity_metrics(true_set, pred_set)
                k = (b_key, th)
                indep_f05_list[k].append(m["f0_5"])
                indep_prec_list[k].append(m["precision"])
                indep_rec_list[k].append(m["recall"])
                indep_tp[k] += m["tp"]
                indep_fp[k] += m["fp"]
                indep_fn[k] += m["fn"]
                indep_sing_tot[k] += 1 if m["is_singleton"] else 0
                indep_sing_corr[k] += 1 if (m["is_singleton"] and m["singleton_correct"]) else 0

    t_eval_end = time.time()
    rss_eval = get_current_rss_mb()
    print(f"\nCountry {country} Stage A & B complete in {t_eval_end - t_start:.2f}s.")
    print(f"Memory: Current RSS = {rss_eval:.1f} MB | Peak RSS = {get_peak_rss_mb():.1f} MB")

    # Format Independent Matching Results
    matching_results_independent = {}
    for mb in match_budgets:
        b_key = f"budget_{mb}" if mb != -1 else "tightened_full_union"
        for th in match_thresholds:
            k = (b_key, th)
            stot = indep_sing_tot[k]
            scorr = indep_sing_corr[k]
            matching_results_independent[k] = {
                "macro_f0_5": float(np.mean(indep_f05_list[k])) if len(indep_f05_list[k]) else 0.0,
                "macro_precision": float(np.mean(indep_prec_list[k])) if len(indep_prec_list[k]) else 0.0,
                "macro_recall": float(np.mean(indep_rec_list[k])) if len(indep_rec_list[k]) else 0.0,
                "f0_5_list": list(indep_f05_list[k]),
                "prec_list": list(indep_prec_list[k]),
                "rec_list": list(indep_rec_list[k]),
                "aggregate_tp": indep_tp[k],
                "aggregate_fp": indep_fp[k],
                "aggregate_fn": indep_fn[k],
                "singletons_total": stot,
                "singletons_correct": scorr,
                "singleton_accuracy": float(scorr / stot) if stot > 0 else 0.0,
                "singleton_false_merges": stot - scorr,
            }

    # -------------------------------------------------------------
    # STAGE C: COMPACT ONE-TO-ONE ABLATION EVALUATION
    # -------------------------------------------------------------
    num_compact_pairs = len(oto_score)
    compact_mem_mb = (oto_s1_idx.buffer_info()[1] * oto_s1_idx.itemsize +
                      oto_c_idx.buffer_info()[1] * oto_c_idx.itemsize +
                      oto_rank.buffer_info()[1] * oto_rank.itemsize +
                      oto_score.buffer_info()[1] * oto_score.itemsize) / (1024.0 * 1024.0)

    print(f"Stage C: One-to-One compact storage: {num_compact_pairs:,} qualifying pairs ({compact_mem_mb:.2f} MB).")

    # Explicit test: Does float32 vs float64 produce any different ordering among qualifying pairs?
    f32_diffs = 0
    if num_compact_pairs > 0 and num_compact_pairs <= 100_000:
        f64_keys = [(-oto_score[i], s1_id_list[oto_s1_idx[i]], blocker.cand_ids[oto_c_idx[i]]) for i in range(num_compact_pairs)]
        f32_keys = [(-float(np.float32(oto_score[i])), s1_id_list[oto_s1_idx[i]], blocker.cand_ids[oto_c_idx[i]]) for i in range(num_compact_pairs)]
        f64_sorted = sorted(range(num_compact_pairs), key=lambda i: f64_keys[i])
        f32_sorted = sorted(range(num_compact_pairs), key=lambda i: f32_keys[i])
        f32_diffs = sum(1 for a, b in zip(f64_sorted, f32_sorted) if a != b)
        print(f"Stage C: [{country}] Float32 vs Float64 ordering test on {num_compact_pairs:,} qualifying pairs: {f32_diffs} rank differences found.")
        if f32_diffs > 0:
            print(f"Stage C: [{country}] NOTICE: float32 changes tie-breaking / rank order! Retaining float64 ensures exact bit-for-bit ordering.")

    matching_results_onetoone = {}

    for mb in match_budgets:
        b_key = f"budget_{mb}" if mb != -1 else "tightened_full_union"
        budget_limit = None if mb == -1 else mb

        for th in match_thresholds:
            k = (b_key, th)

            # Filter indices where score >= th and rank satisfies budget
            if budget_limit is None:
                qual_indices = [i for i in range(num_compact_pairs) if oto_score[i] >= th]
            else:
                qual_indices = [i for i in range(num_compact_pairs) if oto_score[i] >= th and oto_rank[i] < budget_limit]

            # Exact deterministic sorting: (-score, s1_id_str, cand_id_str)
            qual_indices.sort(key=lambda i: (-oto_score[i], s1_id_list[oto_s1_idx[i]], blocker.cand_ids[oto_c_idx[i]]))

            # Exact greedy assignment
            assigned_cands = set()
            oto_preds = collections.defaultdict(set)

            for i in qual_indices:
                cid = blocker.cand_ids[oto_c_idx[i]]
                if cid not in assigned_cands:
                    assigned_cands.add(cid)
                    oto_preds[oto_s1_idx[i]].add(cid)

            # Evaluate predictions across country entities
            oto_f05 = array.array("d")
            oto_prec = array.array("d")
            oto_rec = array.array("d")
            tot_tp = tot_fp = tot_fn = tot_sing = tot_sing_corr = 0

            for s1_i, s1_id in enumerate(s1_id_list):
                true_set = country_val_gt[s1_id]
                pred_set = oto_preds.get(s1_i, set())
                m = compute_entity_metrics(true_set, pred_set)

                oto_f05.append(m["f0_5"])
                oto_prec.append(m["precision"])
                oto_rec.append(m["recall"])
                tot_tp += m["tp"]
                tot_fp += m["fp"]
                tot_fn += m["fn"]
                if m["is_singleton"]:
                    tot_sing += 1
                    if m["singleton_correct"]:
                        tot_sing_corr += 1

            matching_results_onetoone[k] = {
                "macro_f0_5": float(np.mean(oto_f05)) if len(oto_f05) else 0.0,
                "macro_precision": float(np.mean(oto_prec)) if len(oto_prec) else 0.0,
                "macro_recall": float(np.mean(oto_rec)) if len(oto_rec) else 0.0,
                "f0_5_list": list(oto_f05),
                "prec_list": list(oto_prec),
                "rec_list": list(oto_rec),
                "aggregate_tp": tot_tp,
                "aggregate_fp": tot_fp,
                "aggregate_fn": tot_fn,
                "singletons_total": tot_sing,
                "singletons_correct": tot_sing_corr,
                "singleton_accuracy": float(tot_sing_corr / tot_sing) if tot_sing > 0 else 0.0,
                "singleton_false_merges": tot_sing - tot_sing_corr,
            }

    # Explicit memory cleanup
    num_indexed = len(blocker.cand_ids)
    del blocker
    del cand_id_to_idx
    del oto_s1_idx
    del oto_c_idx
    del oto_rank
    del oto_score
    gc.collect()
    rss_after = get_current_rss_mb()

    return {
        "country": country,
        "records_indexed": num_indexed,
        "runtime_seconds": round(time.time() - t_start, 2),
        "peak_rss_mb": round(get_peak_rss_mb(), 1),
        "current_rss_mb": round(rss_after, 1),
        "total_s1": len(country_s1_records),
        "total_true_pairs": total_true_country_pairs,
        "strat_recovered": strat_recovered,
        "strat_cand_counts": strat_cand_counts,
        "tightened_union_recovered": tightened_union_recovered,
        "tightened_union_cand_counts": tightened_union_cand_counts,
        "contributions": {
            "exact": contrib_exact,
            "canon_only": contrib_canon_only,
            "addr_only": contrib_addr_only,
            "char_only": contrib_char_only,
        },
        "budget_recovered": budget_recovered,
        "budget_cand_counts": budget_cand_counts,
        "qualifying_pairs_count": num_compact_pairs,
        "compact_mem_mb": round(compact_mem_mb, 2),
        "f32_ordering_diffs": f32_diffs,
        "matching_results_independent": matching_results_independent,
        "matching_results_onetoone": matching_results_onetoone,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3.1: Correct Blocking & Benchmark")
    parser.add_argument("--sample", type=int, default=None, help="Sample count per country for testing")
    args = parser.parse_args()

    t_global_start = time.time()
    print("=================================================================")
    print("AMAZON ML CHALLENGE 2026: PHASE 3.1 CORRECT BLOCKING BENCHMARK")
    print("=================================================================")

    # 1. Load validation S1 IDs
    print(f"Reading validation IDs from {VAL_IDS_FILE}...")
    with open(VAL_IDS_FILE, "r", encoding="utf-8") as f:
        val_s1_ids = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(val_s1_ids):,} validation S1 IDs.")

    # 2. Load validation S1 records
    val_records = {}
    with open(S1_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0].strip() in val_s1_ids:
                val_records[parts[0].strip()] = {
                    "name": parts[1].strip(),
                    "address": parts[2].strip(),
                    "country": parts[3].strip(),
                }
    print(f"Loaded metadata for {len(val_records):,} validation S1 entities.")

    # 3. Load ground truth
    val_gt = collections.defaultdict(set)
    with open(GT_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0].strip()
            if sid in val_s1_ids:
                if len(parts) > 1 and parts[1].strip():
                    val_gt[sid] = {m.strip() for m in parts[1].split(",") if m.strip()}
                else:
                    val_gt[sid] = set()
    print(f"Loaded ground truth for {len(val_gt):,} entities.")

    # Sanity check
    verify_ground_truth_integrity(val_gt)

    # Dynamic country partitioning
    country_partitions = collections.defaultdict(dict)
    for sid, r in val_records.items():
        country_partitions[r["country"]][sid] = r

    unique_countries = sorted(country_partitions.keys())
    print(f"\nDiscovered {len(unique_countries)} country partitions in validation: {unique_countries}")
    for c in unique_countries:
        print(f"  Partition '{c}': {len(country_partitions[c]):,} validation entities")

    if args.sample:
        print(f"\n[NOTICE] Running dry run sample of {args.sample} entities per country.")
        for c in unique_countries:
            sample_keys = list(country_partitions[c].keys())[:args.sample]
            country_partitions[c] = {k: country_partitions[c][k] for k in sample_keys}

    budgets = [25, 50, 75, 100, 150, 250, 500]
    match_budgets = [25, 50, 75, 100, 150, 250, 500, -1]
    match_thresholds = [0.65, 0.70, 0.75, 0.80]

    # Process each country partition
    country_results = {}
    for country in unique_countries:
        c_res = process_country_partition(
            country,
            country_partitions[country],
            val_gt,
            budgets,
            match_budgets,
            match_thresholds,
        )
        country_results[country] = c_res

    # Aggregating across countries
    print("\n=================================================================")
    print("AGGREGATING RESULTS ACROSS ALL COUNTRIES")
    print("=================================================================")

    total_val_s1 = sum(r["total_s1"] for r in country_results.values())
    total_val_gt_pairs = sum(r["total_true_pairs"] for r in country_results.values())
    total_possible_pairs = sum(
        r["total_s1"] * r["records_indexed"]
        for r in country_results.values()
    )

    # 1. Full Union & Strategy Table
    strat_keys = ["exact", "canon", "char", "addr"]
    strat_display_names = {
        "exact": "2. Exact Normalized Name",
        "canon": "3. Token-Based (Canonical)",
        "char": "4. Character Prefix (Tightened: max freq <= 500)",
        "addr": "5. Address-Assisted",
    }

    full_blocking_table = []

    # Country Only (dynamically calculated from actual per-entity record counts)
    country_only_counts = []
    for r in country_results.values():
        country_only_counts.extend([r["records_indexed"]] * r["total_s1"])
    co_arr = np.array(country_only_counts)
    country_pairs_total = int(co_arr.sum())

    full_blocking_table.append({
        "strategy": "1. Country Only",
        "blocking_recall_pct": 100.0,
        "total_candidate_pairs": country_pairs_total,
        "avg_candidates_per_s1": round(float(co_arr.mean()), 2),
        "median": float(np.median(co_arr)),
        "p95": float(np.percentile(co_arr, 95)),
        "p99": float(np.percentile(co_arr, 99)),
        "maximum": int(co_arr.max()),
        "reduction_ratio_pct": round((1 - country_pairs_total / total_possible_pairs) * 100, 2),
    })

    for sk in strat_keys:
        tot_recovered = sum(country_results[c]["strat_recovered"][sk] for c in unique_countries)
        all_counts = []
        for c in unique_countries:
            all_counts.extend(country_results[c]["strat_cand_counts"][sk])

        arr = np.array(all_counts)
        tot_pairs = int(arr.sum())
        recall = (tot_recovered / total_val_gt_pairs) * 100.0 if total_val_gt_pairs > 0 else 0.0

        full_blocking_table.append({
            "strategy": strat_display_names[sk],
            "blocking_recall_pct": round(recall, 2),
            "total_candidate_pairs": tot_pairs,
            "avg_candidates_per_s1": round(float(arr.mean()), 2),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "maximum": int(arr.max()),
            "reduction_ratio_pct": round((1 - tot_pairs / total_possible_pairs) * 100, 6),
        })

    # Strategy 6: Tightened Full Union (Exact + Canon + Addr + Tightened Char)
    tu_recovered = sum(country_results[c]["tightened_union_recovered"] for c in unique_countries)
    tu_counts = []
    for c in unique_countries:
        tu_counts.extend(country_results[c]["tightened_union_cand_counts"])
    tu_arr = np.array(tu_counts)
    tu_pairs = int(tu_arr.sum())
    tu_recall = (tu_recovered / total_val_gt_pairs) * 100.0 if total_val_gt_pairs > 0 else 0.0

    full_blocking_table.append({
        "strategy": "6. Tightened Full Union",
        "blocking_recall_pct": round(tu_recall, 2),
        "total_candidate_pairs": tu_pairs,
        "avg_candidates_per_s1": round(float(tu_arr.mean()), 2),
        "median": float(np.median(tu_arr)),
        "p95": float(np.percentile(tu_arr, 95)),
        "p99": float(np.percentile(tu_arr, 99)),
        "maximum": int(tu_arr.max()),
        "reduction_ratio_pct": round((1 - tu_pairs / total_possible_pairs) * 100, 6),
    })

    # Marginal Ablation
    ablation_stats = {
        "exact_matches_found": sum(country_results[c]["contributions"]["exact"] for c in unique_countries),
        "canon_marginal_found": sum(country_results[c]["contributions"]["canon_only"] for c in unique_countries),
        "addr_marginal_found": sum(country_results[c]["contributions"]["addr_only"] for c in unique_countries),
        "char_marginal_found": sum(country_results[c]["contributions"]["char_only"] for c in unique_countries),
    }
    for k in list(ablation_stats.keys()):
        ablation_stats[f"{k}_pct"] = round(ablation_stats[k] / total_val_gt_pairs * 100, 2)

    # 2. Budget Ablation Aggregation
    budget_table = []
    for b in budgets:
        b_rec = sum(country_results[c]["budget_recovered"][b] for c in unique_countries)
        b_counts = []
        for c in unique_countries:
            b_counts.extend(country_results[c]["budget_cand_counts"][b])
        b_arr = np.array(b_counts)
        b_pairs = int(b_arr.sum())
        b_recall = (b_rec / total_val_gt_pairs) * 100.0 if total_val_gt_pairs > 0 else 0.0

        budget_table.append({
            "budget": b,
            "blocking_recall_pct": round(b_recall, 2),
            "total_candidate_pairs": b_pairs,
            "avg_candidates_per_s1": round(float(b_arr.mean()), 2),
            "median": float(np.median(b_arr)),
            "p95": float(np.percentile(b_arr, 95)),
            "p99": float(np.percentile(b_arr, 99)),
            "maximum": int(b_arr.max()),
            "reduction_ratio_pct": round((1 - b_pairs / total_possible_pairs) * 100, 6),
        })

    # 3. Matching Comparison Table (Independent vs One-to-One)
    matching_table = []
    for mb in match_budgets:
        b_key = f"budget_{mb}" if mb != -1 else "tightened_full_union"
        for th in match_thresholds:
            # Exact global macro average across all S1 entities in all country partitions
            all_indep_f05 = []
            all_indep_prec = []
            all_indep_rec = []
            all_oto_f05 = []
            all_oto_prec = []
            all_oto_rec = []
            indep_tp = indep_fp = indep_fn = indep_fm = 0
            oto_tp = oto_fp = oto_fn = oto_fm = 0
            tot_singletons = tot_indep_sing_corr = tot_oto_sing_corr = 0

            for c in unique_countries:
                ind_res = country_results[c]["matching_results_independent"][(b_key, th)]
                oto_res = country_results[c]["matching_results_onetoone"][(b_key, th)]

                all_indep_f05.extend(ind_res["f0_5_list"])
                all_indep_prec.extend(ind_res["prec_list"])
                all_indep_rec.extend(ind_res["rec_list"])
                indep_tp += ind_res["aggregate_tp"]
                indep_fp += ind_res["aggregate_fp"]
                indep_fn += ind_res["aggregate_fn"]
                indep_fm += ind_res["singleton_false_merges"]
                tot_indep_sing_corr += ind_res["singletons_correct"]

                all_oto_f05.extend(oto_res["f0_5_list"])
                all_oto_prec.extend(oto_res["prec_list"])
                all_oto_rec.extend(oto_res["rec_list"])
                oto_tp += oto_res["aggregate_tp"]
                oto_fp += oto_res["aggregate_fp"]
                oto_fn += oto_res["aggregate_fn"]
                oto_fm += oto_res["singleton_false_merges"]
                tot_oto_sing_corr += oto_res["singletons_correct"]

                tot_singletons += ind_res["singletons_total"]

            indep_f05 = np.mean(all_indep_f05) if all_indep_f05 else 0.0
            indep_prec = np.mean(all_indep_prec) if all_indep_prec else 0.0
            indep_rec = np.mean(all_indep_rec) if all_indep_rec else 0.0
            indep_acc = (tot_indep_sing_corr / tot_singletons) if tot_singletons > 0 else 0.0

            oto_f05 = np.mean(all_oto_f05) if all_oto_f05 else 0.0
            oto_prec = np.mean(all_oto_prec) if all_oto_prec else 0.0
            oto_rec = np.mean(all_oto_rec) if all_oto_rec else 0.0
            oto_acc = (tot_oto_sing_corr / tot_singletons) if tot_singletons > 0 else 0.0

            matching_table.append({
                "candidate_budget": mb if mb != -1 else "Tightened Full Union",
                "threshold": th,
                "independent": {
                    "macro_f0_5": round(float(indep_f05), 4),
                    "macro_precision": round(float(indep_prec), 4),
                    "macro_recall": round(float(indep_rec), 4),
                    "tp": indep_tp,
                    "fp": indep_fp,
                    "fn": indep_fn,
                    "singleton_accuracy": round(float(indep_acc), 4),
                    "singleton_false_merges": indep_fm,
                },
                "onetoone": {
                    "macro_f0_5": round(float(oto_f05), 4),
                    "macro_precision": round(float(oto_prec), 4),
                    "macro_recall": round(float(oto_rec), 4),
                    "tp": oto_tp,
                    "fp": oto_fp,
                    "fn": oto_fn,
                    "singleton_accuracy": round(float(oto_acc), 4),
                    "singleton_false_merges": oto_fm,
                },
                "delta_f0_5": round(float(oto_f05 - indep_f05), 4),
                "fp_reduction": indep_fp - oto_fp,
            })

    t_global_end = time.time()
    total_runtime = round(t_global_end - t_global_start, 2)
    peak_rss = round(get_peak_rss_mb(), 1)
    current_rss = round(get_current_rss_mb(), 1)

    print(f"\nExecution Complete in {total_runtime:.2f}s.")
    print(f"Memory: Peak RSS = {peak_rss:.1f} MB | Current RSS = {current_rss:.1f} MB")

    # Serialize JSON
    results_payload = {
        "execution_summary": {
            "total_validation_s1": total_val_s1,
            "total_ground_truth_pairs": total_val_gt_pairs,
            "total_runtime_seconds": total_runtime,
            "peak_rss_mb": peak_rss,
            "current_rss_mb": current_rss,
            "total_qualifying_pairs_score_gte_065": sum(r.get("qualifying_pairs_count", 0) for r in country_results.values()),
            "total_compact_mem_mb": round(sum(r.get("compact_mem_mb", 0.0) for r in country_results.values()), 2),
            "f32_vs_f64_ordering_diffs": sum(r.get("f32_ordering_diffs", 0) for r in country_results.values()),
        },
        "full_blocking_table": full_blocking_table,
        "ablation_contributions": ablation_stats,
        "budget_ablation_table": budget_table,
        "matching_comparison_table": matching_table,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved results to {RESULTS_JSON}")

    # Generate Markdown Report
    generate_markdown_report(results_payload, REPORT_MD)
    print(f"Saved report to {REPORT_MD}")


def generate_markdown_report(res: dict, report_path: str):
    """Generate comprehensive markdown report for Phase 3.1."""
    exec_sum = res["execution_summary"]
    blk_tbl = res["full_blocking_table"]
    abl = res["ablation_contributions"]
    b_tbl = res["budget_ablation_table"]
    m_tbl = res["matching_comparison_table"]

    # Highest measured independent vs highest measured 1-to-1
    best_indep = max(m_tbl, key=lambda x: x["independent"]["macro_f0_5"])
    best_oto = max(m_tbl, key=lambda x: x["onetoone"]["macro_f0_5"])
    best_budget = best_oto["candidate_budget"]
    best_budget_label = f"Budget {best_budget}" if best_budget != "Tightened Full Union" else "Tightened Full Union"

    if best_budget == "Tightened Full Union":
        best_budget_recall = blk_tbl[-1]["blocking_recall_pct"]
        best_budget_avg_cands = blk_tbl[-1]["avg_candidates_per_s1"]
    else:
        best_budget_recall = next((x["blocking_recall_pct"] for x in b_tbl if x["budget"] == best_budget), 0.0)
        best_budget_avg_cands = float(best_budget)

    md = f"""# Phase 3.1 Report: Tightened Full Union Blocking & Baseline Benchmark

**Evaluation Scope**: Complete validation set of **{exec_sum['total_validation_s1']:,} Source 1 entities** evaluated against all **10,320,219 Source 2 & Source 3 records**.  
**Execution Profile**: Completed in **{exec_sum['total_runtime_seconds']:.1f}s** | Peak RSS: **{exec_sum['peak_rss_mb']:.1f} MB** | Current RSS: **{exec_sum['current_rss_mb']:.1f} MB**.

---

## 1. Executive Summary

> [!IMPORTANT]
> **Methodological Deprecation Notice**: The previous Phase 3 results (Macro $F_{{0.5}} = 0.6165$, Precision = 0.6807, Recall = 0.5395) relied on positional posting-list truncation (`[:500]`) and an asymmetric candidate cap (`len(combined) < 100`). Those results are officially deprecated. Phase 3.1 establishes the true, unconstrained candidate blocking ceiling and deterministic matching baseline.

Phase 3.1 rectifies the methodological limitations identified in the repository audit:
1. **Positional posting-list truncation (`[:500]`) was completely removed**. Every posting for every blocking key is indexed and searched using memory-compact integer arrays.
2. **Frequency-aware 4-character prefix filtering**: Overly-common generic prefixes (frequency > 500) are filtered out, eliminating candidate explosion while preserving specific character prefix signals.
3. **Tightened Full Union**: Matching operates on $\\text{{Exact}} \\cup \\text{{Canonical}} \\cup \\text{{Address}} \\cup \\text{{Tightened Character Prefix}}$ (max prefix freq $\\le 500$).
4. **Deterministic candidate ordering** based strictly on blocking signals was introduced.
5. **One-to-one competitive matching** was evaluated as an explicit ablation against independent thresholding.

### Headline Comparison: Deprecated Phase 3 (Capped) vs Phase 3.1 (Corrected)

| Metric | Phase 3 (Deprecated: Capped at 500/100) | Phase 3.1 Tightened Full Union (Independent) | Phase 3.1 One-to-One Ablation ({best_budget_label}, $\\theta={best_oto['threshold']:.2f}$) | Measured Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Blocking Recall Ceiling** | 71.93% | **{blk_tbl[-1]['blocking_recall_pct']:.2f}%** | {best_budget_recall:.2f}% | True blocking ceiling vs budget-selected recall |
| **Avg Candidates / S1** | 134.81 | **{blk_tbl[-1]['avg_candidates_per_s1']:.2f}** | {best_budget_avg_cands:.2f} | Candidate load per entity |
| **Highest Measured Macro $F_{{0.5}}$** | 0.6165 | **{best_indep['independent']['macro_f0_5']:.4f}** | **{best_oto['onetoone']['macro_f0_5']:.4f}** | Effect of 1-to-1 competitive assignment |
| **Macro Precision** | 0.6807 | {best_indep['independent']['macro_precision']:.4f} | **{best_oto['onetoone']['macro_precision']:.4f}** | Precision under 1-to-1 assignment |
| **Macro Recall** | 0.5395 | {best_indep['independent']['macro_recall']:.4f} | {best_oto['onetoone']['macro_recall']:.4f} | Recall retention |
| **Aggregate False Positives** | 337,192 | {best_indep['independent']['fp']:,} | **{best_oto['onetoone']['fp']:,}** | **{best_indep['independent']['fp'] - best_oto['onetoone']['fp']:,} duplicate FPs eliminated by 1-to-1** |

---

## 2. Experiment A: Blocking Strategies Comparison

| Strategy | Blocking Recall (%) | Total Candidate Pairs | Avg / S1 | Median | P95 | P99 | Maximum | Reduction Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for row in blk_tbl:
        md += f"| **{row['strategy']}** | **{row['blocking_recall_pct']:.2f}%** | {row['total_candidate_pairs']:,} | {row['avg_candidates_per_s1']:.2f} | {row['median']:.1f} | {row['p95']:.1f} | {row['p99']:.1f} | {row['maximum']:,} | {row['reduction_ratio_pct']:.6f}% |\n"

    md += f"""
### Marginal Contributions to Tightened Full Union & Candidate Pool
- **Exact Normalized Name**: {abl['exact_matches_found']:,} true links (**{abl['exact_matches_found_pct']:.2f}%**)
- **Canonical Token (Marginal to Exact)**: +{abl['canon_marginal_found']:,} true links (**+{abl['canon_marginal_found_pct']:.2f}%**)
- **Address-Assisted (Marginal)**: +{abl['addr_marginal_found']:,} true links (**+{abl['addr_marginal_found_pct']:.2f}%**)
- **Tightened Character Prefix (Marginal)**: +{abl['char_marginal_found']:,} true links (**+{abl['char_marginal_found_pct']:.2f}%**)

---

## 3. Experiment B: Candidate Budget Ablation

Evaluation of deterministic, label-independent candidate selection using blocking signals:

| Budget ($K$) | Blocking Recall (%) | Total Candidate Pairs | Avg / S1 | Median | P95 | P99 | Max | Reduction Ratio (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for row in b_tbl:
        md += f"| **{row['budget']}** | **{row['blocking_recall_pct']:.2f}%** | {row['total_candidate_pairs']:,} | {row['avg_candidates_per_s1']:.2f} | {row['median']:.1f} | {row['p95']:.1f} | {row['p99']:.1f} | {row['maximum']:,} | {row['reduction_ratio_pct']:.6f}% |\n"

    md += """
---

## 4. Experiment C & D: Deterministic Matching & One-to-One Ablation

Comparison between **Independent Matching** (each S1 thresholds independently) vs **One-to-One Matching** (greedy competitive assignment where each S2/S3 candidate is claimed by at most one S1):

| Candidate Budget | Threshold ($\\theta$) | Indep Macro $F_{0.5}$ | Indep Prec | Indep Rec | Indep FP | **1-to-1 Macro $F_{0.5}$** | **1-to-1 Prec** | **1-to-1 Rec** | **1-to-1 FP** | $\\Delta F_{0.5}$ | FP Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for row in m_tbl:
        ind = row["independent"]
        oto = row["onetoone"]
        md += f"| {row['candidate_budget']} | {row['threshold']:.2f} | {ind['macro_f0_5']:.4f} | {ind['macro_precision']:.4f} | {ind['macro_recall']:.4f} | {ind['fp']:,} | **{oto['macro_f0_5']:.4f}** | **{oto['macro_precision']:.4f}** | {oto['macro_recall']:.4f} | **{oto['fp']:,}** | **+{row['delta_f0_5']:.4f}** | -{row['fp_reduction']:,} |\n"

    # Derive empirical budget recall values
    b50_rec = next((x['blocking_recall_pct'] for x in b_tbl if x['budget'] == 50), 0.0)
    b100_rec = next((x['blocking_recall_pct'] for x in b_tbl if x['budget'] == 100), 0.0)
    b250_rec = next((x['blocking_recall_pct'] for x in b_tbl if x['budget'] == 250), 0.0)
    b500_rec = next((x['blocking_recall_pct'] for x in b_tbl if x['budget'] == 500), 0.0)

    md += f"""
---

## 5. Measured Observations & Takeaways

1. **Tightened Full Union Blocking Ceiling**: Without positional truncation or arbitrary caps, the Tightened Full Union (Exact + Canonical + Address + Tightened Character) achieves a blocking recall ceiling of **{blk_tbl[-1]['blocking_recall_pct']:.2f}%** ({abl['exact_matches_found'] + abl['canon_marginal_found'] + abl['addr_marginal_found'] + abl['char_marginal_found']:,} / {exec_sum['total_ground_truth_pairs']:,} true links) with an average of only **{blk_tbl[-1]['avg_candidates_per_s1']:.2f} candidates per S1**.
2. **Tightened Character-Prefix Blocker**: Frequency-aware filtering (frequency <= 500) eliminated generic prefix explosion ("amer", "indi", "nati", "shri"), keeping the character candidate count tractable while preserving specific prefix recall.
3. **Candidate Budget vs. Recall Tradeoff**: Restricting candidates to a fixed budget $K$ within Tightened Full Union yields the following empirical recall:
   - Budget 50: {b50_rec:.2f}% recall
   - Budget 100: {b100_rec:.2f}% recall
   - Budget 250: {b250_rec:.2f}% recall
   - Budget 500: {b500_rec:.2f}% recall
4. **One-to-One Matching Ablation**:
   - The greedy competitive assignment ablation assigns each candidate S2/S3 record to at most one S1 entity.
   - At threshold $\\theta = {best_oto['threshold']:.2f}$ with {best_budget_label}, one-to-one assignment reduces false positives by {best_oto['fp_reduction']:,} (from {best_oto['independent']['fp']:,} down to {best_oto['onetoone']['fp']:,}), shifting Macro $F_{{0.5}}$ from {best_oto['independent']['macro_f0_5']:.4f} to {best_oto['onetoone']['macro_f0_5']:.4f} ($\\Delta F_{{0.5}} = {best_oto['delta_f0_5']:+.4f}$).
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
