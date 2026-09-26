#!/usr/bin/env python3
import sys
import os
import collections
import random
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import get_key_e

def get_consonants(word):
    return "".join(c for c in word.lower() if c.isalpha() and c not in "aeiouyh")

def get_key_p3_detailed(name, addr):
    name_toks = [t for t in rdb.normalize_address(name).split() if t.isalpha()]
    addr_toks = [t for t in rdb.normalize_address(addr).split() if t.isalpha()]
    
    name_toks.sort(key=len, reverse=True)
    addr_toks.sort(key=len, reverse=True)
    
    nums = [t for t in rdb.normalize_address(addr).split() if t.isdigit() and len(t) >= 3]
    
    p3_keys = set()
    p3n_keys = set()
    
    for nt in name_toks[:2]:
        if len(nt) >= 5:
            nc = get_consonants(nt)
            if len(nc) >= 3:
                for at in addr_toks[:2]:
                    if len(at) >= 5:
                        ac = get_consonants(at)
                        if len(ac) >= 3:
                            p3_keys.add(f"P3#{nc}#{ac}")
                for n in nums[:2]:
                    p3n_keys.add(f"P3N#{nc}#{n}")
                    
    return p3_keys, p3n_keys

def main():
    print("Loading 5,000 S1 validation sample...")
    random.seed(42)
    with open("experiments/val_s1_ids.txt") as f:
        all_s1_ids = sorted([line.strip() for line in f if line.strip()])
    
    test_ids = set(random.sample(all_s1_ids, 5000))
    val_gt = rdb.load_gt_for_ids(test_ids)
    s1_data = rdb.load_s1_records(test_ids)
    
    countries = sorted({r["country"] for r in s1_data.values()})
    
    cand_counts_p3 = []
    cand_counts_p3n = []
    
    p3n_digits_cands = collections.defaultdict(list)
    p3n_digits_rec = collections.defaultdict(int)
    
    recovered_by_p3 = set()
    recovered_by_p3n = set()
    recovered_by_both = set()
    
    baseline_recovered = set()
    total_gt = 0
    
    posting_size_p3 = collections.defaultdict(int)
    posting_size_p3n = collections.defaultdict(int)
    
    for country in countries:
        print(f"Processing {country}...")
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        
        idx_exact = collections.defaultdict(list)
        idx_canon = collections.defaultdict(list)
        idx_addr_old = collections.defaultdict(list)
        idx_key_b = collections.defaultdict(list)
        idx_key_e = collections.defaultdict(list)
        
        idx_p3 = collections.defaultdict(list)
        idx_p3n = collections.defaultdict(list)
        
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
                            
                        p3_keys, p3n_keys = get_key_p3_detailed(name, addr)
                        for k in p3_keys:
                            idx_p3[k].append(rec_idx)
                            posting_size_p3[k] += 1
                        for k in p3n_keys:
                            idx_p3n[k].append(rec_idx)
                            posting_size_p3n[k] += 1
                            
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
            total_gt += len(gt_set)
            
            norm_name, canon_name, clean_toks = rdb.normalize_and_canonicalize(s1["name"])
            c_base = set(idx_exact.get(norm_name, [])) | set(idx_canon.get(canon_name, []))
            if clean_toks:
                old_nums = rdb.extract_numeric_tokens(s1["address"])
                tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
                for n in old_nums:
                    if len(n) >= 2:
                        c_base.update(idx_addr_old.get(f"{n}_{tok_p}", []))
            nums, alphas = rdb.extract_address_features(s1["address"])
            for k in rdb.get_key_b(canon_name, nums):
                c_base.update(idx_key_b.get(k, []))
            for k in get_key_e(s1["address"]):
                c_base.update(idx_key_e.get(k, []))
                
            retrieved_base = {cand_ids_arr[i] for i in c_base}
            for gt_enc in (gt_enc_set & retrieved_base):
                baseline_recovered.add((sid, gt_enc))
                
            c_p3 = set()
            c_p3n = set()
            p3_keys, p3n_keys = get_key_p3_detailed(s1["name"], s1["address"])
            
            for k in p3_keys:
                c_p3.update(idx_p3.get(k, []))
            for k in p3n_keys:
                c_p3n.update(idx_p3n.get(k, []))
                
            cand_counts_p3.append(len(c_p3))
            cand_counts_p3n.append(len(c_p3n))
            
            retrieved_p3 = {cand_ids_arr[i] for i in c_p3}
            retrieved_p3n = {cand_ids_arr[i] for i in c_p3n}
            
            rec_p3 = gt_enc_set & retrieved_p3
            rec_p3n = gt_enc_set & retrieved_p3n
            
            for gt_enc in rec_p3:
                recovered_by_p3.add((sid, gt_enc))
            for gt_enc in rec_p3n:
                recovered_by_p3n.add((sid, gt_enc))
            for gt_enc in (rec_p3 & rec_p3n):
                recovered_by_both.add((sid, gt_enc))
                
            for k in p3n_keys:
                n_str = k.split("#")[2]
                digits = len(n_str)
                if digits == 3: d_key = "3"
                elif digits == 4: d_key = "4"
                elif digits == 5: d_key = "5"
                else: d_key = "6+"
                
                c_k = set(idx_p3n.get(k, []))
                p3n_digits_cands[d_key].append(len(c_k))
                
                rec_k = gt_enc_set & {cand_ids_arr[i] for i in c_k}
                for gt_enc in rec_k:
                    if (sid, gt_enc) not in baseline_recovered:
                        p3n_digits_rec[d_key] += 1
                        
    print("\n" + "="*50)
    print("DIAGNOSTIC REPORT")
    print("="*50)
    
    print(f"\nTotal GT Links: {total_gt}")
    print(f"Baseline Recovered: {len(baseline_recovered)}")
    
    def print_stats(name, counts, recovered_set):
        print(f"\n--- {name} Family ---")
        print(f"Total candidates: {sum(counts):,}")
        print(f"Avg candidates/S1: {np.mean(counts):.1f}")
        print(f"Median: {np.median(counts):.1f}")
        print(f"P95: {np.percentile(counts, 95):.1f}")
        print(f"P99: {np.percentile(counts, 99):.1f}")
        print(f"Max: {np.max(counts):,}")
        print(f"Total GT recovered: {len(recovered_set)}")
        new_rec = len(recovered_set - baseline_recovered)
        print(f"Baseline-missed GT recovered: {new_rec}")
        
    print_stats("P3 (Name+Address Signature)", cand_counts_p3, recovered_by_p3)
    print_stats("P3N (Name Signature + Address Number)", cand_counts_p3n, recovered_by_p3n)
    
    print("\n--- Overlap ---")
    print(f"Recovered by both: {len(recovered_by_both)}")
    new_overlap = len(recovered_by_both - baseline_recovered)
    print(f"Baseline-missed recovered by both: {new_overlap}")
    
    print("\n--- P3N Breakdown by Digits ---")
    for d in ["3", "4", "5", "6+"]:
        cands = sum(p3n_digits_cands[d]) if d in p3n_digits_cands else 0
        rec = p3n_digits_rec[d]
        print(f"{d}-digit numbers: {cands:,} total cands, {rec} new GT links")
        
    print("\n--- Top 10 Collision-Causing Keys (P3) ---")
    top_p3 = sorted(posting_size_p3.items(), key=lambda x: x[1], reverse=True)[:10]
    for k, v in top_p3:
        print(f"{k}: {v:,} records")
        
    print("\n--- Top 10 Collision-Causing Keys (P3N) ---")
    top_p3n = sorted(posting_size_p3n.items(), key=lambda x: x[1], reverse=True)[:10]
    for k, v in top_p3n:
        print(f"{k}: {v:,} records")

if __name__ == "__main__":
    main()
