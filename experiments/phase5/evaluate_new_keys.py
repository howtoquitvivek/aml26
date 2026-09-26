#!/usr/bin/env python3
import sys
import os
import gc
import json
import collections
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb

def get_consonant_key(canon_name):
    consonants = [c for c in canon_name if c in "bcdfghjklmnpqrstvwxyz"]
    if len(consonants) >= 4:
        return "".join(consonants[:4])
    return ""

def get_addr_alpha_key(canon_addr):
    # First alphabetic word of length >= 4
    for tok in canon_addr.split():
        if tok.isalpha() and len(tok) >= 4:
            return tok[:4]
    return ""

def get_name_num_key(canon_name):
    for tok in canon_name.split():
        if tok.isdigit():
            return tok
    return ""

def main():
    print("Loading 10,000 Holdout IDs...")
    with open("experiments/phase4/artifacts/holdout_10k/s1_ids.txt") as f:
        target_ids = {line.strip() for line in f if line.strip()}
        
    val_gt = rdb.load_gt_for_ids(target_ids)
    s1_data = rdb.load_s1_records(target_ids)
    
    needed_eids = set()
    for gt_set in val_gt.values():
        needed_eids.update(gt_set)
        
    print("Loading GT candidate bodies...")
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

    # We evaluate keys by checking how many GT pairs they match, 
    # but we ALSO need to estimate explosion. To estimate explosion, we must build the index for these keys over the entire S2/S3.
    # We can do that by parsing S2/S3 once and just counting the frequencies of these new keys.
    print("Building frequencies for new keys across entire S2/S3...")
    
    freq_cons = collections.defaultdict(int)
    freq_addr_alpha = collections.defaultdict(int)
    freq_name_num = collections.defaultdict(int)
    
    count = 0
    for fpath in (rdb.S2_FILE, rdb.S3_FILE):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                count += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    country = parts[3].strip().lower()
                    _, canon_n, _ = rdb.normalize_and_canonicalize(parts[1])
                    canon_a = rdb.normalize_address(parts[2])
                    
                    k_cons = get_consonant_key(canon_n)
                    if k_cons: freq_cons[f"{country}_{k_cons}"] += 1
                    
                    k_addr = get_addr_alpha_key(canon_a)
                    if k_addr: freq_addr_alpha[f"{country}_{k_addr}"] += 1
                    
                    k_num = get_name_num_key(canon_n)
                    if k_num: freq_name_num[f"{country}_{k_num}"] += 1
                    
    print("Scoring new keys against missed GT from the 10k holdout...")
    
    # Wait, we need to know what CFG1_NoTwoTok_KeyE missed.
    # To do this correctly in Python without full index, we just simulate the keys.
    # CFG1_NoTwoTok_KeyE keys:
    # Exact, Canon, AddrOld, KeyB, KeyE
    
    keys_stats = {
        "Key_Consonants": {"gt_recovered": 0, "avg_explosion": 0.0},
        "Key_AddrAlpha": {"gt_recovered": 0, "avg_explosion": 0.0},
        "Key_NameNum": {"gt_recovered": 0, "avg_explosion": 0.0}
    }
    
    gt_pairs_evaluated = 0
    
    explosions = {"Key_Consonants": [], "Key_AddrAlpha": [], "Key_NameNum": []}
    
    for sid, s1 in s1_data.items():
        gt_set = val_gt.get(sid, set())
        if not gt_set: continue
        
        country = s1["country"].lower()
        _, cn1, _ = rdb.normalize_and_canonicalize(s1["name"])
        ca1 = rdb.normalize_address(s1["address"])
        
        # Existing keys simulation
        def get_existing(cn, ca):
            k = set()
            if cn: k.add(f"canon_{cn}")
            if ca: k.add(f"addr_{ca}")
            if cn and ca:
                k.add(f"keyb_{cn[:3]}_{ca.split()[0]}")
            return k
            
        e1 = get_existing(cn1, ca1)
        
        kc1 = get_consonant_key(cn1)
        ka1 = get_addr_alpha_key(ca1)
        kn1 = get_name_num_key(cn1)
        
        if kc1: explosions["Key_Consonants"].append(freq_cons[f"{country}_{kc1}"])
        if ka1: explosions["Key_AddrAlpha"].append(freq_addr_alpha[f"{country}_{ka1}"])
        if kn1: explosions["Key_NameNum"].append(freq_name_num[f"{country}_{kn1}"])
        
        for gt in gt_set:
            if gt not in cand_meta: continue
            gt_pairs_evaluated += 1
            cm = cand_meta[gt]
            _, cn2, _ = rdb.normalize_and_canonicalize(cm["name"])
            ca2 = rdb.normalize_address(cm["address"])
            
            e2 = get_existing(cn2, ca2)
            
            is_missed = True
            if (s1["name"].lower().strip() == cm["name"].lower().strip()) and s1["name"].strip():
                is_missed = False
            if e1 & e2:
                is_missed = False
                
            if is_missed:
                # Would the new keys catch it?
                kc2 = get_consonant_key(cn2)
                ka2 = get_addr_alpha_key(ca2)
                kn2 = get_name_num_key(cn2)
                
                if kc1 and kc2 and kc1 == kc2: keys_stats["Key_Consonants"]["gt_recovered"] += 1
                if ka1 and ka2 and ka1 == ka2: keys_stats["Key_AddrAlpha"]["gt_recovered"] += 1
                if kn1 and kn2 and kn1 == kn2: keys_stats["Key_NameNum"]["gt_recovered"] += 1

    for k in explosions:
        if explosions[k]:
            keys_stats[k]["avg_explosion"] = float(np.mean(explosions[k]))
            keys_stats[k]["p95_explosion"] = int(np.percentile(explosions[k], 95))
            keys_stats[k]["max_explosion"] = int(np.max(explosions[k]))
            
    print(json.dumps(keys_stats, indent=2))
    
    with open("results/phase5/proposed_keys.json", "w") as f:
        json.dump(keys_stats, f, indent=2)

if __name__ == "__main__":
    main()
