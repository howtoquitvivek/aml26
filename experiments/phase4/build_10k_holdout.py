#!/usr/bin/env python3
import sys
import os
import json
import collections
import random
import time
from datetime import datetime



def main():
    print("Building 10k Holdout...")
    
    # 1. Load validation pool
    with open("experiments/val_s1_ids.txt", "r") as f:
        val_pool = {line.strip() for line in f if line.strip()}
        
    # 2. Load existing 1.5k dev IDs to exclude them
    with open("experiments/dev_s1_ids.txt", "r") as f:
        dev_ids = {line.strip() for line in f if line.strip()}
        
    available_pool = val_pool - dev_ids
    print(f"Total Validation Pool: {len(val_pool)}")
    print(f"Dev IDs to exclude: {len(dev_ids)}")
    print(f"Available for holdout: {len(available_pool)}")
    
    # 3. Load GT to get match counts
    s1_gt_counts = collections.defaultdict(int)
    with open("dataset/train/train_ground_truth.tsv", "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                s1_gt_counts[parts[0]] += 1
                
    # 4. Load S1 to get countries
    s1_countries = {}
    with open("dataset/train/train_source1.tsv", "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                sid = parts[0]
                if sid in available_pool:
                    s1_countries[sid] = parts[3].strip()
                    
    # 5. Build records for stratification
    # Must match format expected by src.split_data.stratified_sample
    available_records = []
    for sid in available_pool:
        if sid in s1_countries:
            available_records.append({
                "id": sid,
                "country": s1_countries[sid],
                "match_count": s1_gt_counts[sid]
            })
            
    # 6. Stratify
    print("Stratifying...")
    # The existing stratified_sample splits into target sizes. We need exactly 10,000.
    # Actually, stratified_sample is built to split an entire dataset into train/val.
    # It takes (records, val_fraction, seed).
    # val_fraction = 10000 / len(available_records)
    fraction = 10000 / len(available_records)
    
    # Let's write a targeted exact-size stratifier instead, to guarantee exactly 10k
    groups = collections.defaultdict(list)
    for r in available_records:
        mc = r["match_count"]
        bin_c = min(mc, 5)
        groups[(r["country"], bin_c)].append(r["id"])
        
    random.seed(1337)
    holdout_ids = []
    for key, group in groups.items():
        # proportion
        k_size = int(round(len(group) / len(available_records) * 10000))
        # safety
        k_size = min(k_size, len(group))
        sampled = random.sample(group, k_size)
        holdout_ids.extend(sampled)
        
    # Adjust to exactly 10k if rounding caused issues
    while len(holdout_ids) > 10000:
        holdout_ids.pop()
    while len(holdout_ids) < 10000:
        # just pick randomly from what's left
        left = list(available_pool - set(holdout_ids))
        holdout_ids.append(random.choice(left))
        
    print(f"Generated {len(holdout_ids)} holdout IDs.")
    
    # Validate intersection
    intersect = set(holdout_ids) & dev_ids
    if intersect:
        print(f"CRITICAL ERROR: Holdout contains {len(intersect)} dev IDs!")
        sys.exit(1)
        
    # Stats
    final_countries = collections.defaultdict(int)
    final_bins = collections.defaultdict(int)
    for sid in holdout_ids:
        c = s1_countries[sid]
        mc = s1_gt_counts[sid]
        final_countries[c] += 1
        final_bins[min(mc, 5)] += 1
        
    metadata = {
        "sample_size": len(holdout_ids),
        "random_seed": 1337,
        "source_validation_split": "experiments/val_s1_ids.txt",
        "country_distribution": final_countries,
        "match_count_bin_distribution": final_bins,
        "creation_timestamp": datetime.utcnow().isoformat() + "Z",
        "excluded_dev_sample": "experiments/dev_s1_ids.txt",
        "excluded_count": len(dev_ids),
        "leakage_check_passed": len(intersect) == 0
    }
    
    with open("experiments/phase4/artifacts/holdout_10k/s1_ids.txt", "w") as f:
        for sid in sorted(holdout_ids):
            f.write(sid + "\n")
            
    with open("experiments/phase4/artifacts/holdout_10k/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Holdout successfully saved to experiments/phase4/artifacts/holdout_10k/")

if __name__ == "__main__":
    main()
