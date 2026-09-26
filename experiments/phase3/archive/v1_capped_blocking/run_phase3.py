#!/usr/bin/env python3
"""
Phase 3: Candidate Generation / Blocking & Deterministic Matching Baseline.
Executes Phase 3A (Blocking Experiments 1-5 + Combined) and Phase 3B (Deterministic Matcher).
Memory-safe streaming execution with country-by-country isolation.
"""

import os
import sys
import time
import json
import collections
import resource
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.normalization import (
    normalize_name,
    get_canonical_name_key,
    get_name_tokens,
    normalize_address,
    extract_numeric_tokens,
    get_character_ngrams,
)
from src.blocking import BlockingIndex
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

def get_peak_memory_mb():
    """Get peak resident set size in MB."""
    # On Linux ru_maxrss is in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0

def load_validation_ground_truth(val_s1_ids_set):
    """Load ground truth matching for validation entities."""
    print("Loading validation ground truth...")
    gt = {}
    with open(GT_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0].strip()
            if s1_id in val_s1_ids_set:
                if len(parts) > 1 and parts[1].strip():
                    gt[s1_id] = {m.strip() for m in parts[1].split(",") if m.strip()}
                else:
                    gt[s1_id] = set()
    return gt

def load_validation_s1_records(val_s1_ids_set):
    """Load S1 metadata (name, address, country) for validation entities."""
    print("Loading validation S1 records...")
    records = {}
    with open(S1_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                eid = parts[0].strip()
                if eid in val_s1_ids_set:
                    records[eid] = {
                        "name": parts[1].strip(),
                        "address": parts[2].strip(),
                        "country": parts[3].strip(),
                    }
    return records

def run_blocking_benchmarks(val_records, val_gt):
    """Run Blocking Experiments 1-5 and Combined Strategy."""
    print("\n=======================================================")
    print("PHASE 3A: CANDIDATE GENERATION & BLOCKING EXPERIMENTS")
    print("=======================================================")

    # Select representative evaluation sample for detailed comparative blocking benchmarks
    # We use a stratified sample of 15,000 validation entities (60% US, 40% India)
    rng = np.random.RandomState(42)
    us_val = [k for k, v in val_records.items() if v["country"] == "US"]
    in_val = [k for k, v in val_records.items() if v["country"] == "India"]

    sample_size = 15000
    us_sample_size = int(sample_size * 0.60)
    in_sample_size = sample_size - us_sample_size

    bench_s1_ids = set(rng.choice(us_val, us_sample_size, replace=False).tolist() +
                       rng.choice(in_val, in_sample_size, replace=False).tolist())

    print(f"Benchmark subset size: {len(bench_s1_ids):,} entities ({us_sample_size:,} US, {in_sample_size:,} India)")

    # 1. Strategy 1: Country Blocking (Analytical + Verification)
    # Total S2/S3 in US = 6,186,873; in India = 4,133,346
    # In validation: 132,364 US, 88,319 India
    # Total pairs on full validation = 132,364 * 6,186,873 + 88,319 * 4,133,346 = 1,183,902,323,766
    # Total possible without blocking = 220,683 * 10,320,219 = 2,277,496,890,577
    exp1_stats = {
        "strategy": "1. Country Only",
        "blocking_recall": 100.0,
        "avg_candidates_per_s1": float((132364 * 6186873 + 88319 * 4133346) / 220683),
        "median_candidates": 6186873.0,
        "p95_candidates": 6186873.0,
        "p99_candidates": 6186873.0,
        "total_candidate_pairs_full_val": 1183902323766,
        "reduction_ratio_pct": round((1 - 1183902323766 / 2277496890577) * 100, 2),
        "runtime_seconds": 0.0,
        "peak_memory_mb": 0.0,
        "notes": "Guarantees 100% recall (0 cross-country matches in GT), but leaves ~5.36M candidates/S1."
    }

    # Now build the BlockingIndex over all S2 + S3 records
    print("\nBuilding multi-strategy BlockingIndex over Source 2 & Source 3 (10.3M records)...")
    t0_idx = time.time()
    idx = BlockingIndex(max_block_size=500)

    # Stream S2
    with open(S2_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                idx.add_record(p[0].strip(), p[1].strip(), p[2].strip(), p[3].strip())

    # Stream S3
    with open(S3_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                idx.add_record(p[0].strip(), p[1].strip(), p[2].strip(), p[3].strip())

    idx.prune_large_blocks()
    t1_idx = time.time()
    print(f"Indexed {idx.total_indexed_records:,} records in {t1_idx - t0_idx:.2f}s. Peak RAM: {get_peak_memory_mb():.1f} MB")

    # Evaluate Strategies 2, 3, 4, 5, and Combined on benchmark subset
    strategies = [
        ("2. Exact Normalized Name", lambda r: idx.get_candidates_exact(r["name"], r["country"])),
        ("3. Token-Based (Canonical)", lambda r: idx.get_candidates_canonical(r["name"], r["country"])),
        ("4. Character Prefix", lambda r: idx.get_candidates_char_prefix(r["name"], r["country"])),
        ("5. Address-Assisted", lambda r: idx.get_candidates_address_assisted(r["name"], r["address"], r["country"])),
        ("6. Combined Union", lambda r: idx.get_candidates_combined(r["name"], r["address"], r["country"], max_candidates=100)),
    ]

    blocking_results = [exp1_stats]

    # Pre-calculate total true pairs in benchmark sample
    total_true_pairs_bench = sum(len(val_gt[sid]) for sid in bench_s1_ids)

    for strat_name, cand_fn in strategies:
        print(f"\nEvaluating {strat_name}...")
        t0 = time.time()
        cand_counts = []
        recovered_true_pairs = 0

        for sid in bench_s1_ids:
            r = val_records[sid]
            true_set = val_gt[sid]

            cands = cand_fn(r)
            cand_counts.append(len(cands))
            recovered_true_pairs += len(true_set & cands)

        t1 = time.time()
        arr = np.array(cand_counts)
        recall = (recovered_true_pairs / total_true_pairs_bench) * 100.0 if total_true_pairs_bench > 0 else 0.0
        avg_cands = float(arr.mean())
        med_cands = float(np.median(arr))
        p95_cands = float(np.percentile(arr, 95))
        p99_cands = float(np.percentile(arr, 99))
        tot_cands_sample = int(arr.sum())

        # Extrapolate total pairs to full validation set (220,683 S1 entities)
        extrapolated_full_val_pairs = int(avg_cands * len(val_records))
        full_possible_pairs = 2277496890577
        red_ratio = (1 - (extrapolated_full_val_pairs / full_possible_pairs)) * 100.0

        stat = {
            "strategy": strat_name,
            "blocking_recall": round(recall, 2),
            "avg_candidates_per_s1": round(avg_cands, 2),
            "median_candidates": med_cands,
            "p95_candidates": p95_cands,
            "p99_candidates": p99_cands,
            "sample_candidate_pairs": tot_cands_sample,
            "extrapolated_full_val_pairs": extrapolated_full_val_pairs,
            "reduction_ratio_pct": round(red_ratio, 6),
            "runtime_seconds": round(t1 - t0, 2),
            "peak_memory_mb": round(get_peak_memory_mb(), 1),
        }
        blocking_results.append(stat)
        print(f"  -> Recall: {recall:.2f}% | Avg cands: {avg_cands:.2f} | P95: {p95_cands} | Red ratio: {red_ratio:.4f}% | Time: {t1-t0:.2f}s")

    return idx, blocking_results, bench_s1_ids


def run_deterministic_matcher(idx, val_records, val_gt, bench_s1_ids):
    """Run Phase 3B: Deterministic matching baseline experiments."""
    print("\n=======================================================")
    print("PHASE 3B: DETERMINISTIC MATCHING BASELINE")
    print("=======================================================")

    # We need candidate S2/S3 record contents to compute pair features
    # Collect all candidate IDs needed for benchmark entities using Combined Blocker
    print("Generating candidate sets for benchmark entities...")
    bench_candidates = {}
    needed_cand_ids = set()

    for sid in bench_s1_ids:
        r = val_records[sid]
        cands = idx.get_candidates_combined(r["name"], r["address"], r["country"], max_candidates=100)
        bench_candidates[sid] = cands
        needed_cand_ids.update(cands)

    print(f"Total unique candidates to lookup: {len(needed_cand_ids):,}")

    # Stream S2 & S3 to fetch candidate record content
    cand_records = {}
    print("Fetching candidate records from S2 and S3...")
    for fpath in [S2_FILE, S3_FILE]:
        with open(fpath, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and p[0].strip() in needed_cand_ids:
                    cand_records[p[0].strip()] = {
                        "name": p[1].strip(),
                        "address": p[2].strip(),
                        "country": p[3].strip(),
                    }

    print(f"Retrieved {len(cand_records):,} candidate records.")

    # Compute features and score candidate pairs under multiple configurations
    configs = ["conservative_precision", "balanced", "name_heavy"]
    thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90]

    # Precompute features for all candidate pairs
    print("Computing candidate pair similarity features...")
    t0_feat = time.time()
    pair_features = collections.defaultdict(dict)
    true_pair_scores = []
    false_pair_scores = []

    for sid in bench_s1_ids:
        s1 = val_records[sid]
        true_set = val_gt[sid]
        for cid in bench_candidates[sid]:
            cand = cand_records.get(cid)
            if not cand:
                continue
            feats = compute_pair_features(
                s1["name"], s1["address"], s1["country"],
                cand["name"], cand["address"], cand["country"]
            )
            score_bal = compute_deterministic_score(feats, "balanced")
            pair_features[sid][cid] = feats

            if cid in true_set:
                true_pair_scores.append(score_bal)
            else:
                false_pair_scores.append(score_bal)

    t1_feat = time.time()
    print(f"Computed features for {len(true_pair_scores) + len(false_pair_scores):,} pairs in {t1_feat - t0_feat:.2f}s.")

    # Benchmark score distributions
    score_dist = {
        "true_matches": {
            "mean": float(round(np.mean(true_pair_scores), 4)) if true_pair_scores else 0.0,
            "median": float(round(np.median(true_pair_scores), 4)) if true_pair_scores else 0.0,
            "p10": float(round(np.percentile(true_pair_scores, 10), 4)) if true_pair_scores else 0.0,
            "p25": float(round(np.percentile(true_pair_scores, 25), 4)) if true_pair_scores else 0.0,
            "p75": float(round(np.percentile(true_pair_scores, 75), 4)) if true_pair_scores else 0.0,
            "p90": float(round(np.percentile(true_pair_scores, 90), 4)) if true_pair_scores else 0.0,
        },
        "false_candidates": {
            "mean": float(round(np.mean(false_pair_scores), 4)) if false_pair_scores else 0.0,
            "median": float(round(np.median(false_pair_scores), 4)) if false_pair_scores else 0.0,
            "p90": float(round(np.percentile(false_pair_scores, 90), 4)) if false_pair_scores else 0.0,
            "p95": float(round(np.percentile(false_pair_scores, 95), 4)) if false_pair_scores else 0.0,
            "p99": float(round(np.percentile(false_pair_scores, 99), 4)) if false_pair_scores else 0.0,
        }
    }

    # Evaluate grid of configurations and thresholds
    grid_results = []
    bench_gt = {sid: val_gt[sid] for sid in bench_s1_ids}

    for cfg in configs:
        for th in thresholds:
            preds = {}
            for sid in bench_s1_ids:
                s1_matches = set()
                for cid, feats in pair_features[sid].items():
                    sc = compute_deterministic_score(feats, cfg)
                    if sc >= th:
                        s1_matches.add(cid)
                preds[sid] = s1_matches

            eval_res = evaluate_predictions(bench_gt, preds)
            res_entry = {
                "config": cfg,
                "threshold": th,
                "macro_f0_5": round(eval_res["macro_f0_5"], 4),
                "macro_precision": round(eval_res["macro_precision"], 4),
                "macro_recall": round(eval_res["macro_recall"], 4),
                "singleton_accuracy": round(eval_res["singleton_accuracy"], 4),
                "singleton_false_merges": eval_res["singletons_total"] - eval_res["singletons_correct"],
                "aggregate_tp": eval_res["aggregate_tp"],
                "aggregate_fp": eval_res["aggregate_fp"],
                "aggregate_fn": eval_res["aggregate_fn"],
            }
            grid_results.append(res_entry)
            print(f"Config: {cfg:22s} | Th: {th:.2f} | F0.5: {res_entry['macro_f0_5']:.4f} | Prec: {res_entry['macro_precision']:.4f} | Rec: {res_entry['macro_recall']:.4f} | Singleton Acc: {res_entry['singleton_accuracy']:.4f}")

    # Sort to find best configuration
    grid_results.sort(key=lambda x: x["macro_f0_5"], reverse=True)
    best_config = grid_results[0]
    print(f"\nBest configuration: {best_config['config']} at threshold {best_config['threshold']} -> Macro F0.5: {best_config['macro_f0_5']:.4f}")

    # False Positive and False Negative Qualitative Inspection on Best Configuration
    best_preds = {}
    best_cfg_name = best_config["config"]
    best_th = best_config["threshold"]

    for sid in bench_s1_ids:
        s1_matches = set()
        for cid, feats in pair_features[sid].items():
            sc = compute_deterministic_score(feats, best_cfg_name)
            if sc >= best_th:
                s1_matches.add(cid)
        best_preds[sid] = s1_matches

    fp_examples = []
    fn_examples = []

    for sid in bench_s1_ids:
        s1 = val_records[sid]
        true_set = bench_gt[sid]
        pred_set = best_preds[sid]

        # False positives (predicted but not true)
        fps = pred_set - true_set
        for cid in fps:
            cand = cand_records.get(cid, {})
            if len(fp_examples) < 8:
                fp_examples.append({
                    "s1_id": sid,
                    "s1_name": s1["name"],
                    "s1_address": s1["address"],
                    "s1_country": s1["country"],
                    "false_match_id": cid,
                    "false_cand_name": cand.get("name"),
                    "false_cand_address": cand.get("address"),
                    "score": round(compute_deterministic_score(pair_features[sid][cid], best_cfg_name), 3),
                    "features": {k: round(v, 2) if isinstance(v, float) else v for k, v in pair_features[sid][cid].items()}
                })

        # False negatives (true but not predicted)
        fns = true_set - pred_set
        for cid in fns:
            cand = cand_records.get(cid, {})
            in_candidates = cid in bench_candidates[sid]
            sc = round(compute_deterministic_score(pair_features[sid][cid], best_cfg_name), 3) if in_candidates else None
            if len(fn_examples) < 8:
                fn_examples.append({
                    "s1_id": sid,
                    "s1_name": s1["name"],
                    "s1_address": s1["address"],
                    "s1_country": s1["country"],
                    "missed_match_id": cid,
                    "missed_cand_name": cand.get("name"),
                    "missed_cand_address": cand.get("address"),
                    "in_candidates": in_candidates,
                    "score": sc
                })

    # Predicted match count distribution under best config
    pred_counts = collections.Counter(len(p) for p in best_preds.values())
    pred_count_dist = {int(k): int(v) for k, v in sorted(pred_counts.items())}

    return {
        "score_distribution": score_dist,
        "grid_evaluations": grid_results,
        "best_configuration": best_config,
        "predicted_match_count_distribution": pred_count_dist,
        "false_positive_examples": fp_examples,
        "false_negative_examples": fn_examples,
    }


def main():
    t_start = time.time()
    print("Reading validation split IDs...")
    with open(VAL_IDS_FILE, "r", encoding="utf-8") as f:
        val_s1_ids = {line.strip() for line in f if line.strip()}

    print(f"Loaded {len(val_s1_ids):,} validation S1 IDs.")
    val_records = load_validation_s1_records(val_s1_ids)
    val_gt = load_validation_ground_truth(val_s1_ids)

    # Run Phase 3A: Blocking
    idx, blocking_results, bench_s1_ids = run_blocking_benchmarks(val_records, val_gt)

    # Run Phase 3B: Deterministic Matching Baseline
    matching_results = run_deterministic_matcher(idx, val_records, val_gt, bench_s1_ids)

    # Combine all results into final structure
    phase3_payload = {
        "benchmark_metadata": {
            "total_validation_s1_entities": len(val_s1_ids),
            "benchmark_sample_size": len(bench_s1_ids),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_execution_seconds": round(time.time() - t_start, 2),
            "peak_memory_mb": round(get_peak_memory_mb(), 1),
        },
        "phase_3a_blocking": {
            "comparison_table": blocking_results,
        },
        "phase_3b_matching": matching_results,
    }

    out_json = os.path.join(EXPERIMENTS_DIR, "phase3_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(phase3_payload, f, indent=2)

    print(f"\nSaved complete Phase 3 results to {out_json}")
    print(f"Total Phase 3 execution time: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
