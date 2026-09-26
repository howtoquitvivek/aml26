#!/usr/bin/env python3
import sys
import os
import gc
import json
import time
import collections
import random
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import get_key_e

def get_consonants(word):
    return "".join(c for c in word.lower() if c.isalpha() and c not in "aeiouyh")

def get_key_p3(name, addr):
    name_toks = [t for t in rdb.normalize_address(name).split() if t.isalpha()]
    addr_toks = [t for t in rdb.normalize_address(addr).split() if t.isalpha()]
    
    name_toks.sort(key=len, reverse=True)
    addr_toks.sort(key=len, reverse=True)
    
    nums = [t for t in rdb.normalize_address(addr).split() if t.isdigit() and len(t) >= 3]
    
    keys = set()
    for nt in name_toks[:2]:
        if len(nt) >= 5:
            nc = get_consonants(nt)
            if len(nc) >= 3:
                for at in addr_toks[:2]:
                    if len(at) >= 5:
                        ac = get_consonants(at)
                        if len(ac) >= 3:
                            keys.add(f"P3#{nc}#{ac}")
                for n in nums[:2]:
                    keys.add(f"P3N#{nc}#{n}")
    return keys

def build_index_for_country(country: str):
    def mk(): return collections.defaultdict(list)
    idx_exact = mk()
    idx_canon = mk()
    idx_addr_old = mk()
    idx_key_b = mk()
    idx_key_e = mk()
    idx_key_p3 = mk()
    
    cand_ids_arr = []
    
    for fpath in ("dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3].strip() == country:
                    cid = parts[0].strip()
                    name = parts[1].strip()
                    addr = parts[2].strip()
                    
                    rec_idx = len(cand_ids_arr)
                    cand_ids_arr.append(rdb.encode_id(cid))
                    
                    norm_name, canon_name, clean_toks = rdb.normalize_and_canonicalize(name)
                    
                    idx_exact[norm_name].append(rec_idx)
                    idx_canon[canon_name].append(rec_idx)
                    
                    nums, alphas = rdb.extract_address_features(addr)
                    if clean_toks:
                        tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
                        for n in nums:
                            if len(n) >= 2:
                                idx_addr_old[f"{n}_{tok_p}"].append(rec_idx)
                    
                    for k in rdb.get_key_b(canon_name, nums):
                        idx_key_b[k].append(rec_idx)
                        
                    for k in get_key_e(addr):
                        idx_key_e[k].append(rec_idx)
                        
                    for k in get_key_p3(name, addr):
                        idx_key_p3[k].append(rec_idx)
                        
    return cand_ids_arr, dict(idx_exact), dict(idx_canon), dict(idx_addr_old), dict(idx_key_b), dict(idx_key_e), dict(idx_key_p3)

def query_s1(s1, budget, cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_key_p3, use_p3):
    norm_name, canon_name, clean_toks = rdb.normalize_and_canonicalize(s1["name"])
    
    c_exact = set(idx_exact.get(norm_name, ()))
    c_canon = set(idx_canon.get(canon_name, ()))
    
    c_addr = set()
    if clean_toks:
        old_nums = rdb.extract_numeric_tokens(s1["address"])
        tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
        for n in old_nums:
            if len(n) >= 2:
                c_addr.update(idx_addr_old.get(f"{n}_{tok_p}", ()))
                
    nums, alphas = rdb.extract_address_features(s1["address"])
    c_key_b = set()
    for k in rdb.get_key_b(canon_name, nums):
        c_key_b.update(idx_key_b.get(k, ()))
        
    c_key_e = set()
    for k in get_key_e(s1["address"]):
        c_key_e.update(idx_key_e.get(k, ()))
        
    c_key_p3 = set()
    if use_p3:
        for k in get_key_p3(s1["name"], s1["address"]):
            c_key_p3.update(idx_key_p3.get(k, ()))
            
    all_cands = c_exact | c_canon | c_addr | c_key_b | c_key_e | c_key_p3
    
    W_EXACT = 40
    W_CANON = 30
    W_ADDR = 20
    W_KEY_P3 = 15
    W_KEY_B = 10
    W_KEY_E = 2 
    
    weights = {}
    for i in all_cands:
        w = 0
        if i in c_exact: w += W_EXACT
        if i in c_canon: w += W_CANON
        if i in c_addr:  w += W_ADDR
        if i in c_key_p3: w += W_KEY_P3
        if i in c_key_b: w += W_KEY_B
        if i in c_key_e: w += W_KEY_E
        weights[i] = w
        
    ranked_all = sorted(list(all_cands), key=lambda i: (-weights[i], rdb.decode_id(cand_ids[i])))
    ranked_budget = ranked_all[:budget]
    
    return all_cands, ranked_budget

def evaluate_config(country_s1, val_gt, cand_ids, indices, use_p3):
    idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_key_p3 = indices
    
    total_gt = 0
    gt_recovered_unlimited = 0
    gt_recovered_budget = 0
    cand_counts_unlimited = []
    
    budget = 300
    
    for sid, s1 in country_s1.items():
        gt_set = val_gt.get(sid, set())
        gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
        total_gt += len(gt_set)
        
        all_cands, ranked_budget = query_s1(s1, budget, cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_key_p3, use_p3)
        
        cand_counts_unlimited.append(len(all_cands))
        
        retrieved_encs_unlimited = {cand_ids[i] for i in all_cands}
        retrieved_encs_budget = {cand_ids[i] for i in ranked_budget}
        
        gt_recovered_unlimited += len(gt_enc_set & retrieved_encs_unlimited)
        gt_recovered_budget += len(gt_enc_set & retrieved_encs_budget)
        
    return {
        "total_gt": total_gt,
        "gt_recovered_unlimited": gt_recovered_unlimited,
        "gt_recovered_budget": gt_recovered_budget,
        "total_candidates": int(sum(cand_counts_unlimited)),
        "avg_cands": float(np.mean(cand_counts_unlimited)),
        "median_cands": float(np.median(cand_counts_unlimited)),
        "p95_cands": float(np.percentile(cand_counts_unlimited, 95)),
        "max_cands": int(np.max(cand_counts_unlimited))
    }

def main():
    print("Loading 5,000 S1 validation sample...")
    random.seed(42)
    with open("experiments/val_s1_ids.txt") as f:
        all_s1_ids = sorted([line.strip() for line in f if line.strip()])
    
    test_ids = set(random.sample(all_s1_ids, 5000))
    val_gt = rdb.load_gt_for_ids(test_ids)
    s1_data = rdb.load_s1_records(test_ids)
    
    countries = sorted({r["country"] for r in s1_data.values()})
    
    agg_base = collections.defaultdict(float)
    agg_p3 = collections.defaultdict(float)
    
    for country in countries:
        print(f"\n--- Processing {country} ---")
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        
        t0 = time.time()
        cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_key_p3 = build_index_for_country(country)
        indices = (idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_key_p3)
        print(f"Index built in {time.time()-t0:.1f}s")
        
        print("Evaluating BASELINE...")
        t0 = time.time()
        base_stats = evaluate_config(country_s1, val_gt, cand_ids, indices, use_p3=False)
        print(f"Baseline evaluated in {time.time()-t0:.1f}s")
        
        print("Evaluating P3...")
        t0 = time.time()
        p3_stats = evaluate_config(country_s1, val_gt, cand_ids, indices, use_p3=True)
        print(f"P3 evaluated in {time.time()-t0:.1f}s")
        
        for k, v in base_stats.items():
            if k == "total_gt":
                agg_base[k] += v
                agg_p3[k] += v
            elif "cands" in k and k != "total_candidates":
                # Average metrics are weighted later
                pass
            else:
                agg_base[k] += v
                agg_p3[k] += p3_stats[k]
                
        # To compute global aggregates for percentiles, we'd need to store all, but we'll approximate 
        # or just look at total candidates / total s1 for avg.
        
    print("\n=========================================")
    print("BASELINE (Pre-K Unlimited):")
    print(f"Total GT: {agg_base['total_gt']}")
    print(f"Recovered: {agg_base['gt_recovered_unlimited']}")
    print(f"Recall: {agg_base['gt_recovered_unlimited']/agg_base['total_gt']:.2%}")
    print(f"Missed: {agg_base['total_gt'] - agg_base['gt_recovered_unlimited']}")
    print(f"Total Cands: {agg_base['total_candidates']}")
    print(f"Avg Cands/S1: {agg_base['total_candidates'] / 5000:.1f}")
    
    print("\nBASELINE (K=300 Budget):")
    print(f"Recovered: {agg_base['gt_recovered_budget']}")
    print(f"Recall: {agg_base['gt_recovered_budget']/agg_base['total_gt']:.2%}")
    
    print("\n-----------------------------------------")
    print("P3 BLOCKER (Pre-K Unlimited):")
    print(f"Recovered: {agg_p3['gt_recovered_unlimited']}")
    print(f"Recall: {agg_p3['gt_recovered_unlimited']/agg_p3['total_gt']:.2%}")
    print(f"Missed: {agg_p3['total_gt'] - agg_p3['gt_recovered_unlimited']}")
    print(f"Total Cands: {agg_p3['total_candidates']}")
    print(f"Avg Cands/S1: {agg_p3['total_candidates'] / 5000:.1f}")
    
    print("\nP3 BLOCKER (K=300 Budget):")
    print(f"Recovered: {agg_p3['gt_recovered_budget']}")
    print(f"Recall: {agg_p3['gt_recovered_budget']/agg_p3['total_gt']:.2%}")
    
    print("\n-----------------------------------------")
    recovered_new = agg_p3['gt_recovered_unlimited'] - agg_base['gt_recovered_unlimited']
    print(f"Newly Recovered Links (Pre-K): {recovered_new}")
    
    extrapolated = recovered_new * (220683 / 5000)
    print(f"Extrapolated Full-Set Recovery: ~{int(extrapolated)} links")
    
    cand_increase = agg_p3['total_candidates'] - agg_base['total_candidates']
    print(f"Total Candidate Volume Increase: +{cand_increase} cands")
    
    extrapolated_cands = cand_increase * (220683 / 5000)
    print(f"Extrapolated Full-Set Candidates: ~{int(extrapolated_cands / 1000000)} Million")

if __name__ == "__main__":
    main()
