#!/usr/bin/env python3
import sys
import os
import json
import time
import random
import collections
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
                        if len(ac) >= 3: p3.add(f"P3#{nc}#{ac}")
    return p3

def ngrams(s, n):
    s = f"^{str(s).lower()}$"
    return {s[i:i+n] for i in range(len(s)-n+1)}

def jaccard(set1, set2):
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def is_cross_script(s1, s2):
    def count_non_ascii(s): return sum(1 for c in str(s) if ord(c) > 127)
    s1_na = count_non_ascii(s1)
    s2_na = count_non_ascii(s2)
    return (s1_na > 0) != (s2_na > 0)

def main():
    print("Loading 5,000 S1 validation sample...")
    random.seed(42)
    with open("experiments/val_s1_ids.txt") as f:
        all_s1_ids = sorted([line.strip() for line in f if line.strip()])
    
    test_ids = set(random.sample(all_s1_ids, 5000))
    val_gt = rdb.load_gt_for_ids(test_ids)
    s1_data = rdb.load_s1_records(test_ids)
    countries = sorted({r["country"] for r in s1_data.values()})
    
    total_gt = sum(len(v) for v in val_gt.values())
    missed_gt_pairs = []
    
    print("Identifying missed links...")
    for country in countries:
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
                        
                        for k in get_p3_2_keys(name, addr): idx_p3_raw[k].append(rec_idx)
                        
        idx_p3 = {k: v for k, v in idx_p3_raw.items() if len(v) <= 100}
        del idx_p3_raw
        
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            if not gt_set: continue
            
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
            
            for k in get_p3_2_keys(s1["name"], s1["address"]): c_base.update(idx_p3.get(k, []))
            
            recovered = gt_enc_set & {cand_ids_arr[i] for i in c_base}
            missed_encs = gt_enc_set - recovered
            
            for enc in missed_encs:
                missed_gt_pairs.append((sid, rdb.decode_id(enc), country))
                
    print(f"Out of {total_gt} GT links, P3.2 completely missed {len(missed_gt_pairs)}")
    
    needed_cids = {cid for _, cid, _ in missed_gt_pairs}
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
                if len(cand_meta) >= len(needed_cids): break
                
    print("\nAnalyzing Missed Links...")
    
    stats = collections.defaultdict(list)
    cross_script_cnt = 0
    
    for sid, cid, country in missed_gt_pairs:
        s1 = s1_data[sid]
        cm = cand_meta.get(cid)
        if not cm: continue
        
        n1, a1 = str(s1["name"]).lower(), str(s1["address"]).lower()
        n2, a2 = str(cm["name"]).lower(), str(cm["address"]).lower()
        
        # Cross script check
        if is_cross_script(s1["name"], cm["name"]) or is_cross_script(s1["address"], cm["address"]):
            cross_script_cnt += 1
            
        stats["name_char_jac"].append(jaccard(set(n1), set(n2)))
        stats["addr_char_jac"].append(jaccard(set(a1), set(a2)))
        stats["name_3gram_jac"].append(jaccard(ngrams(n1, 3), ngrams(n2, 3)))
        stats["addr_3gram_jac"].append(jaccard(ngrams(a1, 3), ngrams(a2, 3)))
        stats["name_4gram_jac"].append(jaccard(ngrams(n1, 4), ngrams(n2, 4)))
        stats["addr_4gram_jac"].append(jaccard(ngrams(a1, 4), ngrams(a2, 4)))
        stats["name_tok_jac"].append(jaccard(set(n1.split()), set(n2.split())))
        stats["addr_tok_jac"].append(jaccard(set(a1.split()), set(a2.split())))
        
        nums1 = set(rdb.extract_numeric_tokens(s1["address"]))
        nums2 = set(rdb.extract_numeric_tokens(cm["address"]))
        stats["num_overlap"].append(1 if (nums1 & nums2) else 0)
        
    print(f"\n--- Diagnostic Results for {len(missed_gt_pairs)} Missed Links ---")
    print(f"Cross-script/Transliteration: {cross_script_cnt} ({cross_script_cnt/max(1, len(missed_gt_pairs))*100:.1f}%)")
    
    for k, v in stats.items():
        v = np.array(v)
        print(f"  {k:<15} - Avg: {np.mean(v):.3f} | Med: {np.median(v):.3f} | >0.5: {np.mean(v > 0.5)*100:.1f}%")

if __name__ == "__main__":
    main()
