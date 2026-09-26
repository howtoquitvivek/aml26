import sys
import os
import gc
import json
import time

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import query_custom_s1

def load_persisted_index(country: str):
    idx_dir = f"experiments/phase4/artifacts/indexes/notwotok_keye_v1/{country.lower()}"
    with open(f"{idx_dir}/cand_ids.json") as f: cand_ids = json.load(f)
    with open(f"{idx_dir}/idx_exact.json") as f: idx_exact = json.load(f)
    with open(f"{idx_dir}/idx_canon.json") as f: idx_canon = json.load(f)
    with open(f"{idx_dir}/idx_addr_old.json") as f: idx_addr_old = json.load(f)
    with open(f"{idx_dir}/idx_key_b.json") as f: idx_key_b = json.load(f)
    with open(f"{idx_dir}/idx_key_e.json") as f: idx_key_e = json.load(f)
    return (cand_ids, idx_exact, idx_canon, {}, idx_addr_old, idx_key_b, idx_key_e)

def main():
    with open("experiments/val_s1_ids.txt") as f:
        target_ids = [line.strip() for line in f if line.strip()]
    
    target_ids_set = set(target_ids)
    val_gt = rdb.load_gt_for_ids(target_ids_set)
    s1_data = rdb.load_s1_records(target_ids_set)
    
    survived_gt = 0
    generated_gt = 0
    total_gt = 0
    
    for country in ["India", "US"]:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        if not country_s1: continue
        
        idx_tuple = load_persisted_index(country)
        cand_ids = idx_tuple[0]
        
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            if not gt_set: continue
            total_gt += len(gt_set)
            
            all_cands, ranked_budget, _ = query_custom_s1(s1, 100, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            
            gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
            retrieved_encs = {cand_ids[i] for i in all_cands}
            generated_gt += len(gt_enc_set & retrieved_encs)
            
            survived_encs = {cand_ids[i] for i in ranked_budget}
            survived_gt += len(gt_enc_set & survived_encs)
            
        del idx_tuple
        gc.collect()

    print(f"Total GT: {total_gt}")
    print(f"Generated before budget: {generated_gt}")
    print(f"Survived Top-100 Budget: {survived_gt}")
    print(f"Lost due to budget: {generated_gt - survived_gt}")

if __name__ == "__main__":
    main()
