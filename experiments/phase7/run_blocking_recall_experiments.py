import sys
import os
import json
import time
import collections
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import get_key_e

def load_validation_data():
    with open("experiments/val_s1_ids.txt") as f:
        target_ids = {line.strip() for line in f if line.strip()}
    
    val_gt = rdb.load_gt_for_ids(target_ids)
    s1_data = rdb.load_s1_records(target_ids)
    return target_ids, val_gt, s1_data

def build_index_for_country(country, config, s2_file="dataset/train/train_source2.tsv", s3_file="dataset/train/train_source3.tsv"):
    idx_exact = collections.defaultdict(list)
    idx_canon = collections.defaultdict(list)
    idx_addr_old = collections.defaultdict(list)
    idx_key_b = collections.defaultdict(list)
    idx_key_e = collections.defaultdict(list)
    
    # Extensions for experiments
    idx_zip = collections.defaultdict(list)
    idx_city = collections.defaultdict(list)
    
    cand_ids_arr = []
    
    for fpath in (s2_file, s3_file):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or parts[3].strip() != country:
                    continue
                
                eid = parts[0].strip()
                name = parts[1].strip()
                addr = parts[2].strip()
                
                rec_idx = len(cand_ids_arr)
                cand_ids_arr.append(rdb.encode_id(eid))
                
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
                    
                # Extracted address features for Exp C
                if len(nums) > 0 and len(nums[-1]) >= 4:
                    # simplistic zip code approximation
                    idx_zip[nums[-1]].append(rec_idx)
                if alphas:
                    idx_city[alphas[-1]].append(rec_idx)
                    
    # Prune
    for idx in (idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_zip, idx_city):
        keys_to_prune = [k for k, v in idx.items() if len(v) > 500]
        for k in keys_to_prune:
            idx[k] = idx[k][:500]
            
    return cand_ids_arr, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_zip, idx_city


def run_experiment(s1_data, val_gt, config_mode, K=100):
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    total_gt = 0
    recovered_gt = 0
    all_cand_counts = []
    zero_cand_s1s = 0
    missed_links = []
    
    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        if not country_s1: continue
        
        idx_tuple = build_index_for_country(country, config_mode)
        cand_ids_arr, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e, idx_zip, idx_city = idx_tuple
        
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            total_gt += len(gt_set)
            
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
                
            c_zip = set()
            if len(nums) > 0 and len(nums[-1]) >= 4:
                c_zip.update(idx_zip.get(nums[-1], ()))
                
            # Candidate Logic based on config_mode
            if config_mode == "BASELINE":
                # Current logic
                all_cands = c_exact | c_canon | c_addr | c_key_b | c_key_e
                
                weights = {}
                for i in all_cands:
                    w = 0
                    if i in c_exact: w += 40
                    if i in c_canon: w += 30
                    if i in c_addr:  w += 20
                    if i in c_key_b: w += 10
                    if i in c_key_e: w += 2
                    weights[i] = w
                    
                ranked = sorted(list(all_cands), key=lambda i: (-weights[i], rdb.decode_id(cand_ids_arr[i])))
                final_cands = ranked[:K]
                
            elif config_mode == "EXP_A_UNION":
                # NAME candidates UNION ADDRESS candidates
                # A large name set must not suppress address
                c_name = c_exact | c_canon
                c_address = c_addr | c_key_b | c_key_e
                
                # To prevent suppression, give equal slots or rank them interleaving?
                # Actually, the user says "remove the logic 'if name >= N, don't search address'".
                # But in baseline we don't have that! We just have K=100.
                # If we take Top K names, and Top K addresses, we guarantee address isn't suppressed!
                # Let's take K/2 names and K/2 addresses?
                # Or just K names + K addresses.
                ranked_names = sorted(list(c_name), key=lambda i: (-30, rdb.decode_id(cand_ids_arr[i])))
                ranked_addrs = sorted(list(c_address), key=lambda i: (-20, rdb.decode_id(cand_ids_arr[i])))
                
                budget_names = ranked_names[:K]
                budget_addrs = ranked_addrs[:K]
                
                final_cands = list(set(budget_names) | set(budget_addrs))
                
            elif config_mode == "EXP_B_NO_SUPPRESS":
                # In baseline, the only suppression is the K=100 limit. Let's remove the limit entirely!
                all_cands = c_exact | c_canon | c_addr | c_key_b | c_key_e
                final_cands = list(all_cands)
                
            elif config_mode == "EXP_C_ADD_ZIP":
                all_cands = c_exact | c_canon | c_addr | c_key_b | c_key_e | c_zip
                weights = {}
                for i in all_cands:
                    w = 0
                    if i in c_exact: w += 40
                    if i in c_canon: w += 30
                    if i in c_addr:  w += 20
                    if i in c_key_b: w += 10
                    if i in c_zip: w += 5
                    if i in c_key_e: w += 2
                    weights[i] = w
                ranked = sorted(list(all_cands), key=lambda i: (-weights[i], rdb.decode_id(cand_ids_arr[i])))
                final_cands = ranked[:K]
            
            all_cand_counts.append(len(final_cands))
            if len(final_cands) == 0:
                zero_cand_s1s += 1
                
            gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
            retrieved_encs = {cand_ids_arr[i] for i in final_cands}
            recovered = len(gt_enc_set & retrieved_encs)
            recovered_gt += recovered
            
            for gt in gt_set:
                if rdb.encode_id(gt) not in retrieved_encs:
                    missed_links.append((sid, gt))
                    
    all_cand_counts = np.array(all_cand_counts)
    stats = {
        "recall": recovered_gt / total_gt if total_gt > 0 else 0,
        "total_cands": int(np.sum(all_cand_counts)),
        "avg_cands": float(np.mean(all_cand_counts)),
        "median": float(np.median(all_cand_counts)),
        "p95": float(np.percentile(all_cand_counts, 95)),
        "p99": float(np.percentile(all_cand_counts, 99)),
        "max": int(np.max(all_cand_counts)),
        "zero_cands": zero_cand_s1s,
        "missed": len(missed_links)
    }
    return stats, missed_links

def main():
    print("Loading validation data...")
    target_ids, val_gt, s1_data = load_validation_data()
    print(f"Validation S1: {len(target_ids)}")
    
    results = {}
    
    print("\n--- BASELINE ---")
    base_stats, base_missed = run_experiment(s1_data, val_gt, "BASELINE", K=100)
    print(json.dumps(base_stats, indent=2))
    results["BASELINE"] = base_stats
    
    print("\n--- EXP B (NO SUPPRESSION: K=unlimited) ---")
    exp_b_stats, exp_b_missed = run_experiment(s1_data, val_gt, "EXP_B_NO_SUPPRESS")
    print(json.dumps(exp_b_stats, indent=2))
    results["EXP_B_NO_SUPPRESS"] = exp_b_stats
    
    print("\n--- EXP A (NAME UNION ADDRESS, up to K=100 each) ---")
    exp_a_stats, exp_a_missed = run_experiment(s1_data, val_gt, "EXP_A_UNION", K=100)
    print(json.dumps(exp_a_stats, indent=2))
    results["EXP_A_UNION"] = exp_a_stats
    
    print("\n--- EXP C (ADD ZIP/PIN KEY) ---")
    exp_c_stats, exp_c_missed = run_experiment(s1_data, val_gt, "EXP_C_ADD_ZIP", K=100)
    print(json.dumps(exp_c_stats, indent=2))
    results["EXP_C_ADD_ZIP"] = exp_c_stats

    with open("experiments/phase7/blocking_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
