#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
import time
import collections

sys.path.insert(0, os.path.abspath("."))
from src.normalization import normalize_and_canonicalize, normalize_address, extract_numeric_tokens
import experiments.phase3.run_dev_benchmark as rdb

# Override build_country_index (No TwoTok, No Key E)
def build_country_index_notwotok(country: str):
    cand_ids = rdb.array.array("I")
    def mk(): return collections.defaultdict(lambda: rdb.array.array("I"))
    idx_exact = mk()
    idx_canon = mk()
    idx_addr_old = mk()
    idx_key_b = mk()
    
    for fpath in (rdb.S2_FILE, rdb.S3_FILE):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3].strip() == country:
                    eid = parts[0].strip()
                    name = parts[1].strip()
                    addr = parts[2].strip()
                    
                    rec_idx = len(cand_ids)
                    cand_ids.append(rdb.encode_id(eid))
                    
                    norm_name, canon_name, clean_toks = normalize_and_canonicalize(name)
                    nums, alphas = rdb.extract_address_features(addr)
                    
                    if norm_name: idx_exact[norm_name].append(rec_idx)
                    if canon_name and canon_name != norm_name: idx_canon[canon_name].append(rec_idx)
                    
                    if addr and clean_toks:
                        old_nums = extract_numeric_tokens(addr)
                        tok_p = clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0]
                        for n in old_nums:
                            if len(n) >= 2:
                                idx_addr_old[f"{n}_{tok_p}"].append(rec_idx)
                    
                    for k in rdb.get_key_b(canon_name, nums):
                        idx_key_b[k].append(rec_idx)
                        
    return cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b

# Override query_s1
def query_s1_notwotok(s1, budget, cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b):
    norm_name, canon_name, clean_toks = normalize_and_canonicalize(s1["name"])
    
    c_exact = set(idx_exact.get(norm_name, ()))
    c_canon = set(idx_canon.get(canon_name, ()))
    
    c_addr = set()
    if s1["address"] and clean_toks:
        old_nums = extract_numeric_tokens(s1["address"])
        tok_p = clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0]
        for n in old_nums:
            if len(n) >= 2:
                c_addr.update(idx_addr_old.get(f"{n}_{tok_p}", ()))
                
    c_key_b = set()
    nums, alphas = rdb.extract_address_features(s1["address"])
    for k in rdb.get_key_b(canon_name, nums):
        c_key_b.update(idx_key_b.get(k, ()))
        
    cfg1 = c_exact | c_canon | c_addr | c_key_b
    
    weights = {}
    for i in cfg1:
        w = 0
        if i in c_exact: w += rdb.W_EXACT
        if i in c_canon: w += rdb.W_CANON
        if i in c_addr: w += rdb.W_ADDR
        if i in c_key_b: w += rdb.W_KEY_B
        weights[i] = w
        
    ranked_all = sorted(cfg1, key=lambda i: (-weights[i], rdb.decode_id(cand_ids[i])))
    ranked_budget = ranked_all[:budget]
    
    return cfg1, ranked_budget, weights

# Copy run_evaluation logic
def run_evaluation_notwotok(sample_s1, val_gt, budget, threshold, matcher_config):
    t_start = time.time()
    unique_countries = sorted({r["country"] for r in sample_s1.values()})
    
    per_entity = {}
    cfg1_counts = []
    cfg1_total_gt = 0
    cfg1_recovered = 0
    
    for country in unique_countries:
        country_s1 = {sid: r for sid, r in sample_s1.items() if r["country"] == country}
        print(f"\n  [NoTwoTok] -- {country}: {len(country_s1)} S1 entities --")
        
        cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b = build_country_index_notwotok(country)
        print(f"  [NoTwoTok] Indexed {len(cand_ids):,} records | RSS={rdb.rss_mb():.0f} MB")
        
        pass1 = {}
        needed_encoded = set()
        
        for sid, s1 in country_s1.items():
            gt_set = val_gt[sid]
            cfg1, ranked_budget, _ = query_s1_notwotok(
                s1, budget, cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b
            )
            
            cfg1_eids = {rdb.decode_id(cand_ids[i]) for i in cfg1}
            cfg1_total_gt += len(gt_set)
            cfg1_recovered += len(gt_set & cfg1_eids)
            cfg1_counts.append(len(cfg1))
            
            budget_eids = []
            for i in ranked_budget:
                enc = cand_ids[i]
                needed_encoded.add(enc)
                budget_eids.append(rdb.decode_id(enc))
                
            pass1[sid] = {"gt_set": gt_set, "budget_eids": budget_eids}
            
        n_needed = len(needed_encoded)
        needed_eids_str = {rdb.decode_id(enc) for enc in needed_encoded}
        
        del cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, needed_encoded
        gc.collect()
        
        print(f"  [NoTwoTok] Pass 1 done. Fetching text for {n_needed:,} candidates...")
        
        cand_meta = {}
        for fpath in (rdb.S2_FILE, rdb.S3_FILE):
            with open(fpath, encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4:
                        eid = parts[0].strip()
                        if eid in needed_eids_str:
                            cand_meta[eid] = {"name": parts[1].strip(), "address": parts[2].strip()}
                    if len(cand_meta) >= len(needed_eids_str):
                        break
                        
        del needed_eids_str
        print(f"  [NoTwoTok] Pass 2 done. Evaluating matcher...")
        
        from src.baseline_matcher import compute_pair_features, compute_deterministic_score
        from src.evaluation import compute_entity_metrics
        
        for sid, res in pass1.items():
            s1 = country_s1[sid]
            gt_set = res["gt_set"]
            pred_set = set()
            
            for cid in res["budget_eids"]:
                cm = cand_meta.get(cid)
                if not cm: continue
                feats = compute_pair_features(
                    s1["name"], s1["address"], country,
                    cm["name"], cm["address"], country
                )
                score = compute_deterministic_score(feats, matcher_config)
                if score >= threshold:
                    pred_set.add(cid)
                    
            per_entity[sid] = compute_entity_metrics(gt_set, pred_set)
            
        del pass1, cand_meta
        gc.collect()
        
    t_total = time.time() - t_start
    
    f05_arr = np.array([m["f0_5"] for m in per_entity.values()])
    prec_arr = np.array([m["precision"] for m in per_entity.values()])
    rec_arr = np.array([m["recall"] for m in per_entity.values()])
    
    agg_tp = sum(m["tp"] for m in per_entity.values())
    agg_fp = sum(m["fp"] for m in per_entity.values())
    agg_fn = sum(m["fn"] for m in per_entity.values())
    
    singleton_total = sum(1 for m in per_entity.values() if m["is_singleton"])
    singleton_correct = sum(1 for m in per_entity.values() if m.get("singleton_correct", False))
    
    return {
        "macro_f0_5": round(float(f05_arr.mean()), 6),
        "macro_precision": round(float(prec_arr.mean()), 6),
        "macro_recall": round(float(rec_arr.mean()), 6),
        "aggregate_tp": agg_tp,
        "aggregate_fp": agg_fp,
        "aggregate_fn": agg_fn,
        "singleton_accuracy": round(singleton_correct / max(1, singleton_total), 6),
        "total_time": round(t_total, 1)
    }

def main():
    print("Loading dev sample...")
    with open(rdb.DEV_IDS_FILE) as f:
        dev_ids = {line.strip() for line in f if line.strip()}
        
    s1_data = rdb.load_s1_records(dev_ids)
    val_gt = rdb.load_gt_for_ids(dev_ids)
    
    print("Running NoTwoTok blocker + current matcher evaluation...")
    res = run_evaluation_notwotok(s1_data, val_gt, budget=25, threshold=0.70, matcher_config="conservative_precision")
    
    print("\n--- RESULTS ---")
    print(json.dumps(res, indent=2))
    
    with open("results/phase3/notwotok_matching_results.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
