#!/usr/bin/env python3
import sys
import os
import collections
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import get_key_e
import src.normalization as norm

GENERIC_WORDS = norm.LEGAL_SUFFIX_SET | rdb.GENERIC_ADDR_TERMS

def get_consonants(word):
    return "".join(c for c in word.lower() if c.isalpha() and c not in "aeiouyh")

def get_p3_2_keys(name, addr):
    name_toks = [t for t in rdb.normalize_address(name).split() if t.isalpha() and t not in GENERIC_WORDS]
    addr_toks = [t for t in rdb.normalize_address(addr).split() if t.isalpha() and t not in GENERIC_WORDS]
    
    name_toks.sort(key=len, reverse=True)
    addr_toks.sort(key=len, reverse=True)
    
    p3 = set()
    for nt in name_toks[:2]:
        if len(nt) >= 5:
            nc = get_consonants(nt)
            if len(nc) >= 3:
                for at in addr_toks[:2]:
                    if len(at) >= 5:
                        ac = get_consonants(at)
                        if len(ac) >= 3:
                            p3.add(f"P3#{nc}#{ac}")
    return p3

def rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def peak_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    print("Loading FULL 220,683 S1 validation set...")
    with open("experiments/val_s1_ids.txt") as f:
        all_s1_ids = set([line.strip() for line in f if line.strip()])
    
    val_gt = rdb.load_gt_for_ids(all_s1_ids)
    s1_data = rdb.load_s1_records(all_s1_ids)
    countries = sorted({r["country"] for r in s1_data.values()})
    
    total_gt = sum(len(v) for v in val_gt.values())
    
    results = {
        "A. BASELINE": {"cands": [], "recovered_sid_gt": set(), "country_gt": collections.defaultdict(int), "country_rec": collections.defaultdict(int)},
        "B. P3.2": {"cands": [], "recovered_sid_gt": set(), "country_gt": collections.defaultdict(int), "country_rec": collections.defaultdict(int)}
    }
    
    t_start = time.time()
    
    for country in countries:
        print(f"\nProcessing {country}...")
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        
        idx_base = collections.defaultdict(list)
        idx_p3_raw = collections.defaultdict(list)
        
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
                        
                        p3_keys = get_p3_2_keys(name, addr)
                        for k in p3_keys: idx_p3_raw[k].append(rec_idx)
                        
        print(f"Indexed {len(cand_ids_arr)} records for {country}")
        
        # Apply Frequency Threshold 100
        idx_p3 = {k: v for k, v in idx_p3_raw.items() if len(v) <= 100}
        del idx_p3_raw
        
        print(f"P3.2 keys after thresholding: {len(idx_p3)}")
        
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
            
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
            
            p3_keys = get_p3_2_keys(s1["name"], s1["address"])
            c_p3 = set()
            for k in p3_keys: c_p3.update(idx_p3.get(k, []))
            
            c_all = c_base | c_p3
            
            # Record Baseline
            results["A. BASELINE"]["cands"].append(len(c_base))
            rec_base = gt_enc_set & {cand_ids_arr[i] for i in c_base}
            for gt_enc in rec_base: results["A. BASELINE"]["recovered_sid_gt"].add((sid, gt_enc))
            results["A. BASELINE"]["country_gt"][country] += len(gt_set)
            results["A. BASELINE"]["country_rec"][country] += len(rec_base)
            
            # Record P3.2
            results["B. P3.2"]["cands"].append(len(c_all))
            rec_p32 = gt_enc_set & {cand_ids_arr[i] for i in c_all}
            for gt_enc in rec_p32: results["B. P3.2"]["recovered_sid_gt"].add((sid, gt_enc))
            results["B. P3.2"]["country_gt"][country] += len(gt_set)
            results["B. P3.2"]["country_rec"][country] += len(rec_p32)
            
    total_time = time.time() - t_start
    
    print("\n" + "="*80)
    print("FULL VALIDATION RESULTS")
    print("="*80)
    
    base = results["A. BASELINE"]
    p32 = results["B. P3.2"]
    
    base_rec = base["recovered_sid_gt"]
    p32_rec = p32["recovered_sid_gt"]
    
    base_miss = total_gt - len(base_rec)
    p32_miss = total_gt - len(p32_rec)
    new_recovered = len(p32_rec - base_rec)
    
    print(f"Total validation S1: {len(all_s1_ids)}")
    print(f"Total GT links: {total_gt}")
    print(f"Baseline blocker recall: {len(base_rec)/total_gt*100:.2f}%")
    print(f"P3.2 blocker recall: {len(p32_rec)/total_gt*100:.2f}%")
    print(f"GT links generated by baseline: {len(base_rec)}")
    print(f"GT links generated by P3.2: {len(p32_rec)}")
    print(f"Completely missed GT links under baseline: {base_miss}")
    print(f"Completely missed GT links under P3.2: {p32_miss}")
    print(f"Number of previously completely missed links recovered: {new_recovered}")
    
    base_cands = base["cands"]
    p32_cands = p32["cands"]
    print(f"Total candidate pairs baseline: {sum(base_cands)}")
    print(f"Total candidate pairs P3.2: {sum(p32_cands)}")
    print(f"Average candidates/S1 (P3.2): {np.mean(p32_cands):.1f}")
    print(f"Median (P3.2): {np.median(p32_cands):.1f}")
    print(f"P95 (P3.2): {np.percentile(p32_cands, 95):.1f}")
    print(f"P99 (P3.2): {np.percentile(p32_cands, 99):.1f}")
    print(f"Maximum (P3.2): {np.max(p32_cands)}")
    print(f"Candidate-volume increase: {sum(p32_cands) - sum(base_cands)}")
    print(f"Runtime: {total_time:.1f}s")
    print(f"Peak RSS: {peak_rss_mb():.1f} MB")
    
    print("\nCountry-Level Recall:")
    for country in countries:
        c_gt = base["country_gt"][country]
        if c_gt > 0:
            base_r = base["country_rec"][country] / c_gt
            p32_r = p32["country_rec"][country] / c_gt
            print(f"  {country}: Baseline {base_r*100:.2f}% -> P3.2 {p32_r*100:.2f}% (+{(p32_r-base_r)*100:.2f}%)")

    # Pass 2: Analysis of Recovered Links
    print("\nAnalyzing recovered link slices...")
    new_rec = p32_rec - base_rec
    
    needed_encs = {gt_enc for sid, gt_enc in new_rec}
    needed_cids = {rdb.decode_id(enc) for enc in needed_encs}
    
    cand_meta = {}
    for fpath in ("dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    cid = parts[0].strip()
                    if cid in needed_cids:
                        cand_meta[cid] = {"name": parts[1].strip(), "address": parts[2].strip()}
                if len(cand_meta) >= len(needed_cids):
                    break
                    
    def compute_char_jaccard(s1, s2):
        set1, set2 = set(str(s1).lower()), set(str(s2).lower())
        if not set1 or not set2: return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def extract_numbers(s):
        import re
        return set(re.findall(r'\b\d+\b', str(s)))

    def count_non_ascii(s):
        return sum(1 for c in str(s) if ord(c) > 127)

    cross_script_cnt = 0
    high_name_cnt = 0
    high_addr_cnt = 0
    shared_num_cnt = 0
    
    for sid, gt_enc in new_rec:
        cid = rdb.decode_id(gt_enc)
        s1 = s1_data[sid]
        cm = cand_meta.get(cid)
        if not cm: continue
        
        s1_non_ascii = count_non_ascii(s1["name"]) + count_non_ascii(s1["address"])
        cand_non_ascii = count_non_ascii(cm["name"]) + count_non_ascii(cm["address"])
        if (s1_non_ascii > 0) != (cand_non_ascii > 0): cross_script_cnt += 1
        
        name_j = compute_char_jaccard(s1["name"], cm["name"])
        addr_j = compute_char_jaccard(s1["address"], cm["address"])
        if name_j >= 0.7: high_name_cnt += 1
        if addr_j >= 0.7: high_addr_cnt += 1
        
        s1_nums = extract_numbers(s1["address"])
        cand_nums = extract_numbers(cm["address"])
        if len(s1_nums & cand_nums) > 0: shared_num_cnt += 1
        
    print(f"Slice Analysis for {len(new_rec)} newly recovered links:")
    print(f"  Cross-script/Transliteration: {cross_script_cnt}")
    print(f"  High name-character overlap (>=0.7): {high_name_cnt}")
    print(f"  High address-character overlap (>=0.7): {high_addr_cnt}")
    print(f"  Shared numbers in address: {shared_num_cnt}")

if __name__ == "__main__":
    main()
