#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
import time
import collections
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb

def get_key_e(addr: str) -> set:
    nums, alphas = rdb.extract_address_features(addr)
    keys = set()
    if len(nums) >= 2:
        keys.add(f"addr#{nums[0]}#{nums[1]}")
    if len(nums) >= 1 and len(alphas) >= 1:
        keys.add(f"addr@{nums[0]}@{alphas[0]}")
    return keys

def build_index_for_country(country: str, out_dir: str):
    t0 = time.time()
    def mk(): return collections.defaultdict(list)
    idx_exact = mk()
    idx_canon = mk()
    idx_addr_old = mk()
    idx_key_b = mk()
    idx_key_e = mk()
    
    cand_ids_arr = []
    
    print(f"[{country}] Building CFG1_NoTwoTok_KeyE index...")
    
    records_processed = 0
    for fpath in (rdb.S2_FILE, rdb.S3_FILE):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or parts[3].strip() != country:
                    continue
                
                records_processed += 1
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
                    
    # Save to disk
    print(f"[{country}] Saving {records_processed} candidates to disk...")
    os.makedirs(out_dir, exist_ok=True)
    
    def save_json(data, name):
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(data, f)
            
    save_json(cand_ids_arr, "cand_ids")
    save_json(idx_exact, "idx_exact")
    save_json(idx_canon, "idx_canon")
    save_json(idx_addr_old, "idx_addr_old")
    save_json(idx_key_b, "idx_key_b")
    save_json(idx_key_e, "idx_key_e")
    
    # Check sizes
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    
    build_time = time.time() - t0
    
    # Calculate bytes on disk
    total_bytes = sum(os.path.getsize(os.path.join(out_dir, f"{n}.json")) for n in 
                      ["cand_ids", "idx_exact", "idx_canon", "idx_addr_old", "idx_key_b", "idx_key_e"])
                      
    return {
        "records_processed": records_processed,
        "build_time_sec": round(build_time, 2),
        "peak_rss_mb": round(rss, 2),
        "disk_size_mb": round(total_bytes / (1024*1024), 2),
        "keys": {
            "exact": len(idx_exact),
            "canon": len(idx_canon),
            "addr_old": len(idx_addr_old),
            "key_b": len(idx_key_b),
            "key_e": len(idx_key_e),
        }
    }

def main():
    base_dir = "experiments/phase4/artifacts/indexes/notwotok_keye_v1"
    os.makedirs(base_dir, exist_ok=True)
    
    metadata = {
        "index_version": "phase4_notwotok_keye_v1",
        "source_dataset": "dataset/train/train_source2.tsv & train_source3.tsv",
        "blocking_configuration": "CFG1_NoTwoTok_KeyE",
        "serialization_format": "json",
        "creation_timestamp": datetime.now().astimezone().isoformat(),
        "partitions": {}
    }
    
    for country in ["India", "US"]:
        out_dir = os.path.join(base_dir, country.lower())
        stats = build_index_for_country(country, out_dir)
        metadata["partitions"][country] = stats
        
        # Free memory before next country
        gc.collect()
        
    with open(os.path.join(base_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("\n--- INDEX BUILDING COMPLETE ---")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
