#!/usr/bin/env python3
import sys
import os
import collections
import random
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import get_key_e
import src.normalization as norm

GENERIC_WORDS = norm.LEGAL_SUFFIX_SET | rdb.GENERIC_ADDR_TERMS

def get_consonants(word):
    return "".join(c for c in word.lower() if c.isalpha() and c not in "aeiouyh")

def get_keys(name, addr):
    name_toks_orig = [t for t in rdb.normalize_address(name).split() if t.isalpha()]
    addr_toks_orig = [t for t in rdb.normalize_address(addr).split() if t.isalpha()]
    
    name_toks_filt = [t for t in name_toks_orig if t not in GENERIC_WORDS]
    addr_toks_filt = [t for t in addr_toks_orig if t not in GENERIC_WORDS]
    
    name_toks_orig.sort(key=len, reverse=True)
    addr_toks_orig.sort(key=len, reverse=True)
    name_toks_filt.sort(key=len, reverse=True)
    addr_toks_filt.sort(key=len, reverse=True)
    
    addr_norm = rdb.normalize_address(addr).split()
    nums_3 = [t for t in addr_norm if t.isdigit() and len(t) >= 3]
    nums_4 = [t for t in addr_norm if t.isdigit() and len(t) >= 4]
    nums_5 = [t for t in addr_norm if t.isdigit() and len(t) >= 5]
    nums_6 = [t for t in addr_norm if t.isdigit() and len(t) >= 6]
    
    p3_orig, p3_filt = set(), set()
    p3n_3, p3n_4, p3n_5, p3n_6 = set(), set(), set(), set()
    
    for nt in name_toks_orig[:2]:
        if len(nt) >= 5:
            nc = get_consonants(nt)
            if len(nc) >= 3:
                for at in addr_toks_orig[:2]:
                    if len(at) >= 5:
                        ac = get_consonants(at)
                        if len(ac) >= 3: p3_orig.add(f"P3#{nc}#{ac}")
                for n in nums_3[:2]: p3n_3.add(f"P3N#{nc}#{n}")
                for n in nums_4[:2]: p3n_4.add(f"P3N#{nc}#{n}")
                for n in nums_5[:2]: p3n_5.add(f"P3N#{nc}#{n}")
                for n in nums_6[:2]: p3n_6.add(f"P3N#{nc}#{n}")
                
    for nt in name_toks_filt[:2]:
        if len(nt) >= 5:
            nc = get_consonants(nt)
            if len(nc) >= 3:
                for at in addr_toks_filt[:2]:
                    if len(at) >= 5:
                        ac = get_consonants(at)
                        if len(ac) >= 3: p3_filt.add(f"P3#{nc}#{ac}")
                        
    return p3_orig, p3_filt, p3n_3, p3n_4, p3n_5, p3n_6

def main():
    print("Loading 5,000 S1 validation sample...")
    random.seed(42)
    with open("experiments/val_s1_ids.txt") as f:
        all_s1_ids = sorted([line.strip() for line in f if line.strip()])
    test_ids = set(random.sample(all_s1_ids, 5000))
    val_gt = rdb.load_gt_for_ids(test_ids)
    s1_data = rdb.load_s1_records(test_ids)
    countries = sorted({r["country"] for r in s1_data.values()})
    
    configs = [
        "A. BASELINE",
        "B. ORIGINAL P3",
        "C. P3 + GENERIC-TOKEN FILTER",
        "D. P3 + FREQ FILTER (50)",
        "D. P3 + FREQ FILTER (100)",
        "D. P3 + FREQ FILTER (250)",
        "D. P3 + FREQ FILTER (500)",
        "D. P3 + FREQ FILTER (1000)",
        "E. P3 + GENERIC + FREQ (100)",
        "E. P3 + GENERIC + FREQ (250)",
        "E. P3 + GENERIC + FREQ (500)",
        "F. P3N ABLATION (3+ digits)",
        "F. P3N ABLATION (4+ digits)",
        "F. P3N ABLATION (5+ digits)",
        "F. P3N ABLATION (6+ digits)"
    ]
    
    results = {c: {"cands": [], "recovered_sid_gt": set()} for c in configs}
    
    total_gt = 0
    t_start = time.time()
    
    for country in countries:
        print(f"Processing {country}...")
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        
        idx_base = collections.defaultdict(list)
        idx_p3_orig = collections.defaultdict(list)
        idx_p3_filt = collections.defaultdict(list)
        idx_p3n = {3: collections.defaultdict(list), 4: collections.defaultdict(list), 5: collections.defaultdict(list), 6: collections.defaultdict(list)}
        
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
                        idx_base[f"E#{norm_name}"].append(rec_idx)
                        idx_base[f"C#{canon_name}"].append(rec_idx)
                        
                        nums, alphas = rdb.extract_address_features(addr)
                        if clean_toks:
                            tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
                            for n in nums:
                                if len(n) >= 2: idx_base[f"A#{n}_{tok_p}"].append(rec_idx)
                        for k in rdb.get_key_b(canon_name, nums): idx_base[f"B#{k}"].append(rec_idx)
                        for k in get_key_e(addr): idx_base[f"K#{k}"].append(rec_idx)
                        
                        p3_orig, p3_filt, p3n_3, p3n_4, p3n_5, p3n_6 = get_keys(name, addr)
                        for k in p3_orig: idx_p3_orig[k].append(rec_idx)
                        for k in p3_filt: idx_p3_filt[k].append(rec_idx)
                        for k in p3n_3: idx_p3n[3][k].append(rec_idx)
                        for k in p3n_4: idx_p3n[4][k].append(rec_idx)
                        for k in p3n_5: idx_p3n[5][k].append(rec_idx)
                        for k in p3n_6: idx_p3n[6][k].append(rec_idx)
                        
        print(f"Indexed {len(cand_ids_arr)} records for {country}")
        
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
            total_gt += len(gt_set)
            
            c_base = set()
            norm_name, canon_name, clean_toks = rdb.normalize_and_canonicalize(s1["name"])
            c_base.update(idx_base.get(f"E#{norm_name}", []))
            c_base.update(idx_base.get(f"C#{canon_name}", []))
            if clean_toks:
                old_nums = rdb.extract_numeric_tokens(s1["address"])
                tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
                for n in old_nums:
                    if len(n) >= 2: c_base.update(idx_base.get(f"A#{n}_{tok_p}", []))
            nums, alphas = rdb.extract_address_features(s1["address"])
            for k in rdb.get_key_b(canon_name, nums): c_base.update(idx_base.get(f"B#{k}", []))
            for k in get_key_e(s1["address"]): c_base.update(idx_base.get(f"K#{k}", []))
            
            p3_orig, p3_filt, p3n_3, p3n_4, p3n_5, p3n_6 = get_keys(s1["name"], s1["address"])
            
            def eval_config(cfg_name, extra_cands):
                c_all = c_base | extra_cands
                results[cfg_name]["cands"].append(len(c_all))
                retrieved_encs = {cand_ids_arr[i] for i in c_all}
                rec = gt_enc_set & retrieved_encs
                for gt_enc in rec:
                    results[cfg_name]["recovered_sid_gt"].add((sid, gt_enc))
                    
            eval_config("A. BASELINE", set())
            
            c_p3_orig = set()
            for k in p3_orig: c_p3_orig.update(idx_p3_orig.get(k, []))
            eval_config("B. ORIGINAL P3", c_p3_orig)
            
            c_p3_filt = set()
            for k in p3_filt: c_p3_filt.update(idx_p3_filt.get(k, []))
            eval_config("C. P3 + GENERIC-TOKEN FILTER", c_p3_filt)
            
            for T in [50, 100, 250, 500, 1000]:
                c_freq = set()
                for k in p3_orig:
                    pl = idx_p3_orig.get(k, [])
                    if len(pl) <= T: c_freq.update(pl)
                eval_config(f"D. P3 + FREQ FILTER ({T})", c_freq)
                
            for T in [100, 250, 500]:
                c_freq_filt = set()
                for k in p3_filt:
                    pl = idx_p3_filt.get(k, [])
                    if len(pl) <= T: c_freq_filt.update(pl)
                eval_config(f"E. P3 + GENERIC + FREQ ({T})", c_freq_filt)
                
            for n_dig in [3, 4, 5, 6]:
                c_p3n = set()
                p3n_keys = p3n_3 if n_dig == 3 else (p3n_4 if n_dig == 4 else (p3n_5 if n_dig == 5 else p3n_6))
                for k in p3n_keys:
                    c_p3n.update(idx_p3n[n_dig].get(k, []))
                eval_config(f"F. P3N ABLATION ({n_dig}+ digits)", c_p3n)
                
    total_time = time.time() - t_start
    
    print("\n" + "="*80)
    print("P3.2 EXPERIMENT RESULTS")
    print("="*80)
    
    base_rec = results["A. BASELINE"]["recovered_sid_gt"]
    p3_orig_rec = results["B. ORIGINAL P3"]["recovered_sid_gt"]
    
    print(f"{'Configuration':<30} | {'Recall':<6} | {'RecGT':<5} | {'NewGT':<5} | {'Cand(M)':<7} | {'Avg/S1':<6} | {'Median':<6} | {'P95':<6} | {'P99':<6} | {'Max':<7} | {'Tradeoff'}")
    print("-" * 120)
    
    for cfg in configs:
        res = results[cfg]
        cands = res["cands"]
        tot_cands = sum(cands)
        rec_set = res["recovered_sid_gt"]
        
        recall = len(rec_set) / total_gt if total_gt > 0 else 0
        new_gt = len(rec_set - base_rec)
        avg = np.mean(cands)
        med = np.median(cands)
        p95 = np.percentile(cands, 95)
        p99 = np.percentile(cands, 99)
        max_c = np.max(cands)
        
        tradeoff = new_gt / (tot_cands / 1e6) if tot_cands > 0 else 0
        
        print(f"{cfg:<30} | {recall*100:>5.2f}% | {len(rec_set):>5} | {new_gt:>5} | {tot_cands/1e6:>7.2f} | {avg:>6.0f} | {med:>6.0f} | {p95:>6.0f} | {p99:>6.0f} | {max_c:>7.0f} | {tradeoff:>6.1f}")
        
    print(f"\nRuntime: {total_time:.1f}s")

if __name__ == "__main__":
    main()
