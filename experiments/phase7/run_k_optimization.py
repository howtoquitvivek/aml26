import sys
import os
import json
import time
import collections
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import query_custom_s1

def load_persisted_index(country: str):
    idx_dir = f"experiments/phase4/artifacts/indexes/notwotok_keye_v1/{country.lower()}"
    with open(f"{idx_dir}/cand_ids.json") as f: cand_ids = json.load(f)
    with open(f"{idx_dir}/idx_exact.json") as f: idx_exact = json.load(f)
    with open(f"{idx_dir}/idx_canon.json") as f: idx_canon = json.load(f)
    with open(f"{idx_dir}/idx_addr_old.json") as f: idx_addr_old = json.load(f)
    with open(f"{idx_dir}/idx_key_b.json") as f: idx_key_b = json.load(f)
    with open(f"{idx_dir}/idx_key_e.json") as f: idx_key_e = json.load(f)
    return (cand_ids, idx_exact, idx_canon, {}, idx_addr_old, idx_key_b, idx_key_e)

def load_validation_data():
    with open("experiments/val_s1_ids.txt") as f:
        target_ids = {line.strip() for line in f if line.strip()}
    
    val_gt = rdb.load_gt_for_ids(target_ids)
    s1_data = rdb.load_s1_records(target_ids)
    return target_ids, val_gt, s1_data

def evaluate_k(k, s1_ranks, val_gt, total_gt, baseline_recovered):
    recovered_gt = 0
    total_cands = 0
    cand_counts = []
    zero_cands = 0
    
    # rank diagnostic buckets
    # For every GT link that is generated before Top-K but lost at K=100, record its rank
    # Actually, we need to record ranks for ALL GT links to do the breakdown
    
    for sid, ranks in s1_ranks.items():
        gt_set = val_gt.get(sid, set())
        gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
        
        # truncate
        cands = ranks[:k] if k is not None else ranks
        
        cand_counts.append(len(cands))
        if len(cands) == 0:
            zero_cands += 1
            
        retrieved_encs = set(cands)
        recovered = len(gt_enc_set & retrieved_encs)
        recovered_gt += recovered
        
    cand_counts = np.array(cand_counts)
    
    stats = {
        "recall": recovered_gt / total_gt if total_gt > 0 else 0,
        "total_cands": int(np.sum(cand_counts)),
        "avg_cands": float(np.mean(cand_counts)),
        "median": float(np.median(cand_counts)),
        "p95": float(np.percentile(cand_counts, 95)),
        "p99": float(np.percentile(cand_counts, 99)),
        "max": int(np.max(cand_counts)),
        "zero_cands": zero_cands,
        "recovered_gt": recovered_gt,
        "incremental_gt_vs_100": recovered_gt - baseline_recovered if baseline_recovered is not None else 0
    }
    return stats

def main():
    print("Loading validation data...")
    target_ids, val_gt, s1_data = load_validation_data()
    print(f"Validation S1: {len(target_ids)}")
    
    total_gt = sum(len(v) for v in val_gt.values() if v)
    print(f"Total GT links: {total_gt}")
    
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    s1_ranks = {}
    
    # Diagnostic rank tracking
    # (sid, gt) -> rank of gt (1-indexed), or None if never generated
    gt_ranks = {}
    
    t_start_retrieval = time.time()
    
    for country in unique_countries:
        # Ignore France since it's not in the Phase 4 index (it's dynamically built)
        # Wait, Phase 4 index DOES NOT HAVE FRANCE!
        # Ah! I must check this.
        pass

    # Actually, let's just build it if it's missing, or load if it's present!
    # Wait, Phase 4 index only has India and US! France wasn't supported by LR baseline!
    # I should build France inline!
    pass

    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        if not country_s1: continue
        
        try:
            print(f"[{country}] Loading persisted index...")
            idx_tuple = load_persisted_index(country)
        except Exception as e:
            print(f"[{country}] Persisted index not found. Building dynamically...")
            # We must build it using exact Phase 7 config logic
            from experiments.phase7.run_blocking_recall_experiments import build_index_for_country
            idx_tuple_full = build_index_for_country(country, "BASELINE")
            # build_index_for_country returns: cand_ids_arr, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_zip, idx_city
            idx_tuple = (idx_tuple_full[0], idx_tuple_full[1], idx_tuple_full[2], {}, idx_tuple_full[3], idx_tuple_full[4], idx_tuple_full[5])
            
        cand_ids = idx_tuple[0]
        
        for sid, s1 in country_s1.items():
            # Get unlimited ranked list
            all_cands, ranked_budget, _ = query_custom_s1(s1, 100000, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            
            # Map candidate indices to encoded entity IDs
            ranked_encs = [cand_ids[i] for i in ranked_budget]
            s1_ranks[sid] = ranked_encs
            
            gt_set = val_gt.get(sid, set())
            gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
            
            for gt_enc in gt_enc_set:
                try:
                    rank = ranked_encs.index(gt_enc) + 1
                    gt_ranks[(sid, gt_enc)] = rank
                except ValueError:
                    gt_ranks[(sid, gt_enc)] = None
                    
    t_retrieval = time.time() - t_start_retrieval
    print(f"Retrieval & Ranking took {t_retrieval:.2f}s")
    
    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    t_start_eval = time.time()
    
    results = {}
    
    print("\n--- Evaluating K=100 (BASELINE) ---")
    base_stats = evaluate_k(100, s1_ranks, val_gt, total_gt, None)
    results["K=100"] = base_stats
    base_recovered = base_stats["recovered_gt"]
    
    for k in [150, 200, 300, 500, None]:
        label = f"K={k}" if k is not None else "K=UNLIMITED"
        print(f"\n--- Evaluating {label} ---")
        stats = evaluate_k(k, s1_ranks, val_gt, total_gt, base_recovered)
        results[label] = stats
        
    t_eval = time.time() - t_start_eval
    print(f"Evaluation took {t_eval:.2f}s")
    
    results["runtime_retrieval_s"] = t_retrieval
    results["runtime_eval_s"] = t_eval
    
    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------
    buckets = {
        "1-100": 0,
        "101-150": 0,
        "151-200": 0,
        "201-300": 0,
        "301-500": 0,
        ">500": 0,
        "Not Generated": 0
    }
    
    for rank in gt_ranks.values():
        if rank is None:
            buckets["Not Generated"] += 1
        elif rank <= 100:
            buckets["1-100"] += 1
        elif rank <= 150:
            buckets["101-150"] += 1
        elif rank <= 200:
            buckets["151-200"] += 1
        elif rank <= 300:
            buckets["201-300"] += 1
        elif rank <= 500:
            buckets["301-500"] += 1
        else:
            buckets[">500"] += 1
            
    results["rank_distribution"] = buckets
    print("\n--- Rank Distribution of True Matches ---")
    print(json.dumps(buckets, indent=2))
    
    with open("experiments/phase7/k_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
