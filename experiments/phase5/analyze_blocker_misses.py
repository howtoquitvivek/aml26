#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
import time
import collections
import re

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import extract_lr_features, query_custom_s1
from sklearn.linear_model import LogisticRegression

def train_frozen_model():
    with open("experiments/dev_s1_ids.txt") as f:
        dev_ids = {line.strip() for line in f if line.strip()}
    val_gt = rdb.load_gt_for_ids(dev_ids)
    s1_data = rdb.load_s1_records(dev_ids)
    
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    records = []
    
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
        
        pass1 = {}
        needed_encoded = set()
        
        for sid, s1 in country_s1.items():
            all_cands, ranked_budget, _ = query_custom_s1(s1, 25, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            budget_eids = []
            for i in ranked_budget:
                enc = cand_ids[i]
                needed_encoded.add(enc)
                budget_eids.append(rdb.decode_id(enc))
            pass1[sid] = budget_eids
            
        needed_eids_str = {rdb.decode_id(enc) for enc in needed_encoded}
        del idx_tuple
        gc.collect()
        
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
                        
        for sid, eids in pass1.items():
            s1 = country_s1[sid]
            gt_set = val_gt.get(sid, set())
            for cid in eids:
                cm = cand_meta.get(cid)
                if not cm: continue
                vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                records.append({
                    "vec": vec,
                    "label": 1 if cid in gt_set else 0
                })
        del pass1, cand_meta
        gc.collect()
        
    X_train = np.array([r["vec"] for r in records], dtype=np.float32)
    y_train = np.array([r["label"] for r in records], dtype=bool)
    
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
    clf.fit(X_train, y_train)
    return clf, 0.8

def extract_numbers(s):
    return set(re.findall(r'\d+', s))

def count_non_ascii(s):
    return sum(1 for c in s if ord(c) > 127)

def main():
    print("1. Training frozen model...")
    clf, best_t = train_frozen_model()
    
    print("2. Loading validation IDs...")
    with open("experiments/val_s1_ids.txt") as f:
        target_ids = {line.strip() for line in f if line.strip()}
        
    val_gt = rdb.load_gt_for_ids(target_ids)
    s1_data = rdb.load_s1_records(target_ids)
    
    missed_by_blocker = []
    recovered_by_blocker = []
    
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    print("3. Querying blocker to split GT pairs...")
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
            
            _, ranked_budget, _ = query_custom_s1(s1, 25, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            
            budget_eids = set()
            for i in ranked_budget:
                budget_eids.add(rdb.decode_id(cand_ids[i]))
                
            for gt in gt_set:
                if gt in budget_eids:
                    recovered_by_blocker.append((sid, gt, country))
                else:
                    missed_by_blocker.append((sid, gt, country))
                    
        del idx_tuple, cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e
        gc.collect()

    print(f"Total GT pairs: {len(missed_by_blocker) + len(recovered_by_blocker)}")
    print(f"Missed by blocker: {len(missed_by_blocker)}")
    print(f"Recovered by blocker: {len(recovered_by_blocker)}")
    
    needed_eids = set()
    for sid, gt, _ in missed_by_blocker: needed_eids.add(gt)
    for sid, gt, _ in recovered_by_blocker: needed_eids.add(gt)
    
    print("4. Parsing S2/S3 for GT bodies...")
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
                    
    print("5. Scoring recovered pairs with LR...")
    lr_rejected = []
    lr_accepted = []
    
    X_recovered = []
    recovered_valid = []
    
    for sid, gt, country in recovered_by_blocker:
        if gt not in cand_meta: continue
        s1 = s1_data[sid]
        cm = cand_meta[gt]
        vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], gt, country)
        X_recovered.append(vec)
        recovered_valid.append((sid, gt, country))
        
    if X_recovered:
        probs = clf.predict_proba(np.array(X_recovered))[:, 1]
        for i, (sid, gt, country) in enumerate(recovered_valid):
            if probs[i] >= best_t:
                lr_accepted.append((sid, gt, country))
            else:
                lr_rejected.append((sid, gt, country))
                
    print("6. Analyzing blocker misses...")
    
    stats = {
        "cross_script": 0,
        "high_name_char_overlap": 0,
        "high_addr_char_overlap": 0,
        "shared_numbers": 0,
        "shared_pincode": 0,
        "shared_10digit_phone": 0,
        "exact_name_match": 0,
        "generic_name": 0,
        "india_misses": 0,
        "us_misses": 0
    }
    
    for sid, gt, country in missed_by_blocker:
        if gt not in cand_meta: continue
        s1 = s1_data[sid]
        cm = cand_meta[gt]
        
        if country == "India": stats["india_misses"] += 1
        else: stats["us_misses"] += 1
        
        n1, a1 = s1["name"].lower(), s1["address"].lower()
        n2, a2 = cm["name"].lower(), cm["address"].lower()
        
        na1 = count_non_ascii(n1)
        na2 = count_non_ascii(n2)
        if (na1 > 3 and na2 == 0) or (na2 > 3 and na1 == 0):
            stats["cross_script"] += 1
            
        c1, c2 = set(n1), set(n2)
        u = len(c1 | c2)
        char_jac = len(c1 & c2) / u if u > 0 else 0
        if char_jac > 0.7: stats["high_name_char_overlap"] += 1
        
        ca1, ca2 = set(a1), set(a2)
        ua = len(ca1 | ca2)
        achar_jac = len(ca1 & ca2) / ua if ua > 0 else 0
        if achar_jac > 0.7: stats["high_addr_char_overlap"] += 1
        
        num1 = extract_numbers(n1 + " " + a1)
        num2 = extract_numbers(n2 + " " + a2)
        shared_nums = num1 & num2
        if shared_nums:
            stats["shared_numbers"] += 1
            if any(len(x) == 6 for x in shared_nums):
                stats["shared_pincode"] += 1
            if any(len(x) == 10 for x in shared_nums):
                stats["shared_10digit_phone"] += 1
                
        if n1 == n2: stats["exact_name_match"] += 1
        
        if len(n1.split()) == 1 or len(n1) < 5:
            stats["generic_name"] += 1
            
    report = {
        "total_gt_pairs": len(missed_by_blocker) + len(recovered_by_blocker),
        "total_missed": len(missed_by_blocker),
        "total_rejected": len(lr_rejected),
        "total_accepted": len(lr_accepted),
        "miss_stats": {k: round(v / len(missed_by_blocker), 4) for k, v in stats.items()}
    }
    
    print("\n--- ANALYSIS ---")
    print(json.dumps(report, indent=2))
    
    os.makedirs("results/phase5", exist_ok=True)
    with open("results/phase5/miss_analysis.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
