#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
from typing import Dict, Set, List
import collections

sys.path.insert(0, os.path.abspath("."))
from src.normalization import normalize_and_canonicalize, normalize_address, extract_numeric_tokens
import experiments.phase3.run_dev_benchmark as rdb

# Define Key E: Address-Only Key
def get_key_e(addr: str) -> Set[str]:
    if not addr: return set()
    tokens = normalize_address(addr).split()
    nums = [t for t in tokens if t.isdigit()]
    alphas = [t for t in tokens if t.isalpha() and len(t) >= 4 and t not in rdb.GENERIC_ADDR_TERMS]
    
    keys = set()
    # 1. Two numeric tokens (e.g., "9" and "10" -> "addr#9#10")
    if len(nums) >= 2:
        keys.add(f"addr#{nums[0]}#{nums[1]}")
    # 2. One numeric + One salient alpha (e.g., "404" and "ahmedabad" -> "addr@404@ahmedabad")
    if len(nums) >= 1 and len(alphas) >= 1:
        keys.add(f"addr@{nums[0]}@{alphas[0]}")
    return keys

def main():
    print("Loading dev IDs...")
    with open(rdb.DEV_IDS_FILE) as f:
        dev_ids = {line.strip() for line in f if line.strip()}

    s1_data = rdb.load_s1_records(dev_ids)
    gt_links = rdb.load_gt_for_ids(dev_ids)

    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    # Trackers
    total_gt = 0
    base_rec = 0
    no2tok_rec = 0
    new_rec = 0
    
    base_cands = []
    no2tok_cands = []
    new_cands = []

    for country in unique_countries:
        print(f"\nProcessing {country}...")
        
        # Build indexes
        cand_ids = rdb.array.array("I")
        idx_exact = collections.defaultdict(lambda: rdb.array.array("I"))
        idx_canon = collections.defaultdict(lambda: rdb.array.array("I"))
        idx_two_tok = collections.defaultdict(lambda: rdb.array.array("I"))
        idx_addr_old = collections.defaultdict(lambda: rdb.array.array("I"))
        idx_key_b = collections.defaultdict(lambda: rdb.array.array("I"))
        idx_key_e = collections.defaultdict(lambda: rdb.array.array("I"))
        
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
                        if len(clean_toks) >= 2: idx_two_tok[f"{clean_toks[0]} {clean_toks[1]}"].append(rec_idx)
                        
                        if addr and clean_toks:
                            old_nums = extract_numeric_tokens(addr)
                            tok_p = clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0]
                            for n in old_nums:
                                if len(n) >= 2:
                                    idx_addr_old[f"{n}_{tok_p}"].append(rec_idx)
                                    
                        for k in rdb.get_key_b(canon_name, nums):
                            idx_key_b[k].append(rec_idx)
                            
                        # NEW KEY E
                        for k in get_key_e(addr):
                            idx_key_e[k].append(rec_idx)

        print(f"  Indexed {len(cand_ids)} records. Querying...")
        
        country_s1 = {sid: s1 for sid, s1 in s1_data.items() if s1["country"] == country}
        
        for sid, s1 in country_s1.items():
            gt_set = gt_links.get(sid, set())
            total_gt += len(gt_set)
            
            norm_name, canon_name, clean_toks = normalize_and_canonicalize(s1["name"])
            nums, alphas = rdb.extract_address_features(s1["address"])
            
            c_exact = set(idx_exact.get(norm_name, ()))
            c_canon = set(idx_canon.get(canon_name, ()))
            
            c_two = set()
            if len(clean_toks) >= 2:
                c_two = set(idx_two_tok.get(f"{clean_toks[0]} {clean_toks[1]}", ()))
                
            c_addr = set()
            if s1["address"] and clean_toks:
                old_nums = extract_numeric_tokens(s1["address"])
                tok_p = clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0]
                for n in old_nums:
                    if len(n) >= 2:
                        c_addr.update(idx_addr_old.get(f"{n}_{tok_p}", ()))
                        
            c_key_b = set()
            for k in rdb.get_key_b(canon_name, nums):
                c_key_b.update(idx_key_b.get(k, ()))
                
            c_key_e = set()
            for k in get_key_e(s1["address"]):
                c_key_e.update(idx_key_e.get(k, ()))
                
            # ABLATIONS
            cfg_base = c_exact | c_canon | c_two | c_addr | c_key_b
            cfg_no2tok = c_exact | c_canon | c_addr | c_key_b
            cfg_new = cfg_no2tok | c_key_e
            
            base_cands.append(len(cfg_base))
            no2tok_cands.append(len(cfg_no2tok))
            new_cands.append(len(cfg_new))
            
            base_eids = {rdb.decode_id(cand_ids[i]) for i in cfg_base}
            no2tok_eids = {rdb.decode_id(cand_ids[i]) for i in cfg_no2tok}
            new_eids = {rdb.decode_id(cand_ids[i]) for i in cfg_new}
            
            base_rec += len(gt_set & base_eids)
            no2tok_rec += len(gt_set & no2tok_eids)
            new_rec += len(gt_set & new_eids)
            
        del cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_e
        gc.collect()

    def stats(arr, rec, gt):
        arr = np.array(arr)
        return {
            "recovered": rec,
            "missed": gt - rec,
            "recall": round(rec/max(1, gt), 4),
            "avg_cands": round(float(arr.mean()), 1),
            "p95_cands": round(float(np.percentile(arr, 95)), 1),
            "p99_cands": round(float(np.percentile(arr, 99)), 1),
            "max_cands": int(arr.max())
        }

    out = {
        "total_gt": total_gt,
        "CFG1_Base": stats(base_cands, base_rec, total_gt),
        "CFG1_NoTwoTok": stats(no2tok_cands, no2tok_rec, total_gt),
        "CFG_New_AddrOnly": stats(new_cands, new_rec, total_gt)
    }
    
    print("\n--- RESULTS ---")
    print(json.dumps(out, indent=2))
    
    with open("results/phase3/ablation_new_blocking.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
