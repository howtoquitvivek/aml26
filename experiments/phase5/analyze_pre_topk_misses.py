#!/usr/bin/env python3
import sys
import os
import gc
import json
import time
import re
import numpy as np
import collections

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import extract_lr_features, query_custom_s1

def extract_numbers(s):
    return set(re.findall(r'\d+', s))

def count_non_ascii(s):
    return sum(1 for c in s if ord(c) > 127)

def main():
    print("1. Loading validation IDs...")
    with open("experiments/val_s1_ids.txt") as f:
        target_ids = {line.strip() for line in f if line.strip()}
        
    val_gt = rdb.load_gt_for_ids(target_ids)
    s1_data = rdb.load_s1_records(target_ids)
    
    pre_blocker_misses = []
    
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    print("2. Querying blocker to find pre-TopK misses...")
    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        
        idx_dir = f"experiments/phase4/artifacts/indexes/notwotok_keye_v1/{country.lower()}"
        with open(f"{idx_dir}/cand_ids.json") as f: cand_ids = json.load(f)
        with open(f"{idx_dir}/idx_exact.json") as f: idx_exact = json.load(f)
        with open(f"{idx_dir}/idx_canon.json") as f: idx_canon = json.load(f)
        with open(f"{idx_dir}/idx_addr_old.json") as f: idx_addr_old = json.load(f)
        with open(f"{idx_dir}/idx_key_b.json") as f: idx_key_b = json.load(f)
        with open(f"{idx_dir}/idx_key_e.json") as f: idx_key_e = json.load(f)
        idx_tuple = (cand_ids, idx_exact, idx_canon, {}, idx_addr_old, idx_key_b, idx_key_e)
        
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            if not gt_set: continue
            
            all_cands, _, _ = query_custom_s1(s1, 100, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            
            generated_eids = set()
            for i in all_cands:
                generated_eids.add(rdb.decode_id(cand_ids[i]))
                
            for gt in gt_set:
                if gt not in generated_eids:
                    pre_blocker_misses.append((sid, gt, country))
                    
        del idx_tuple, cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e
        gc.collect()

    print(f"Completely missed by blocker: {len(pre_blocker_misses)}")
    
    needed_eids = {gt for _, gt, _ in pre_blocker_misses}
    
    print("3. Parsing S2/S3 for missed GT bodies...")
    cand_meta = {}
    for fpath in (rdb.S2_FILE, rdb.S3_FILE):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    eid = parts[0].strip()
                    if eid in needed_eids:
                        cand_meta[eid] = {"name": parts[1].strip(), "address": parts[2].strip()}
                if len(cand_meta) >= len(needed_eids):
                    break
                    
    print("4. Analyzing blocker misses...")
    
    stats = {
        "total_misses": len(pre_blocker_misses),
        "cross_script": 0,
        "high_name_char_overlap": 0,
        "high_addr_char_overlap": 0,
        "shared_numbers": 0,
        "shared_pincode": 0,
        "shared_10digit_phone": 0,
        "exact_name_match": 0,
        "generic_name": 0,
        "india_misses": 0,
        "us_misses": 0,
        "has_useful_address_digits": 0
    }
    
    for sid, gt, country in pre_blocker_misses:
        if gt not in cand_meta: continue
        
        if country.lower() == "india": stats["india_misses"] += 1
        else: stats["us_misses"] += 1
        
        s1 = s1_data[sid]
        cm = cand_meta[gt]
        
        n1 = s1["name"].lower()
        n2 = cm["name"].lower()
        a1 = s1["address"].lower()
        a2 = cm["address"].lower()
        
        is_cross_script = False
        nonascii_1 = count_non_ascii(n1) + count_non_ascii(a1)
        nonascii_2 = count_non_ascii(n2) + count_non_ascii(a2)
        
        if (nonascii_1 > 5 and nonascii_2 == 0) or (nonascii_2 > 5 and nonascii_1 == 0):
            stats["cross_script"] += 1
            is_cross_script = True
            
        c1 = set(n1.replace(" ", ""))
        c2 = set(n2.replace(" ", ""))
        c_overlap = len(c1 & c2) / max(1, min(len(c1), len(c2)))
        if c_overlap > 0.7: stats["high_name_char_overlap"] += 1
        
        ca1 = set(a1.replace(" ", ""))
        ca2 = set(a2.replace(" ", ""))
        ca_overlap = len(ca1 & ca2) / max(1, min(len(ca1), len(ca2)))
        if ca_overlap > 0.7: stats["high_addr_char_overlap"] += 1
        
        num1 = extract_numbers(n1 + " " + a1)
        num2 = extract_numbers(n2 + " " + a2)
        if num1 & num2: stats["shared_numbers"] += 1
        
        if len(num1 & num2) >= 1:
            # Check if at least one shared number is reasonably long (e.g., >2 digits)
            long_nums = {n for n in (num1 & num2) if len(n) >= 3}
            if long_nums:
                stats["has_useful_address_digits"] += 1
        
        if n1 == n2: stats["exact_name_match"] += 1
        
        gen_terms = {"restaurant", "hospital", "clinic", "shop", "store", "school"}
        if any(g in n1 for g in gen_terms): stats["generic_name"] += 1
        
        pin1 = re.search(r'\b\d{5,6}\b', a1)
        pin2 = re.search(r'\b\d{5,6}\b', a2)
        if pin1 and pin2 and pin1.group() == pin2.group():
            stats["shared_pincode"] += 1
            
        ph1 = re.search(r'\b\d{10}\b', a1)
        ph2 = re.search(r'\b\d{10}\b', a2)
        if ph1 and ph2 and ph1.group() == ph2.group():
            stats["shared_10digit_phone"] += 1
            
    print("\nPre-TopK Misses Analysis:")
    for k, v in stats.items():
        if k == "total_misses": continue
        pct = (v / len(pre_blocker_misses)) * 100
        print(f"{k}: {v} ({pct:.2f}%)")
        
    os.makedirs("results/phase5", exist_ok=True)
    with open("results/phase5/pre_topk_misses_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
