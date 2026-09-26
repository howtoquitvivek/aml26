#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) Script for Amazon ML Challenge 2026.
Memory-safe streaming execution for multi-million row TSVs.
"""

import os
import json
import collections
import pandas as pd
import numpy as np

DATA_DIR = "dataset"
EXPERIMENTS_DIR = "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

SOURCE_FILES = {
    "train_source1": os.path.join(DATA_DIR, "train", "train_source1.tsv"),
    "train_source2": os.path.join(DATA_DIR, "train", "train_source2.tsv"),
    "train_source3": os.path.join(DATA_DIR, "train", "train_source3.tsv"),
    "test_source1": os.path.join(DATA_DIR, "test", "test_source1.tsv"),
    "test_source2": os.path.join(DATA_DIR, "test", "test_source2.tsv"),
    "test_source3": os.path.join(DATA_DIR, "test", "test_source3.tsv"),
}
GROUND_TRUTH_FILE = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")

def analyze_source_file(file_path, name):
    print(f"--- Analyzing {name} ({file_path}) ---")
    chunksize = 200_000
    
    total_rows = 0
    missing_counts = collections.defaultdict(int)
    blank_name_count = 0
    short_name_count = 0  # len <= 3
    blank_addr_count = 0
    short_addr_count = 0  # len <= 3
    
    country_counts = collections.Counter()
    name_lengths = []
    addr_lengths = []
    
    # Track ID duplicates efficiently using a set or bloom filter; with 5M strings, a python set takes ~200MB
    id_set = set()
    dup_ids = 0
    
    # Process in chunks to stay low on memory
    for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False):
        total_rows += len(chunk)
        
        # Columns & missing
        for col in chunk.columns:
            missing_counts[col] += (chunk[col] == "").sum()
            
        # Duplicate IDs
        ids = chunk["entity_id"].values
        before_len = len(id_set)
        id_set.update(ids)
        dup_ids += len(ids) - (len(id_set) - before_len)
        
        # Country
        country_counts.update(chunk["country"].value_counts().to_dict())
        
        # Lengths
        names = chunk["business_name"].fillna("").astype(str)
        addrs = chunk["business_address"].fillna("").astype(str)
        
        n_lens = names.str.len()
        a_lens = addrs.str.len()
        
        blank_name_count += (n_lens == 0).sum()
        short_name_count += ((n_lens > 0) & (n_lens <= 3)).sum()
        
        blank_addr_count += (a_lens == 0).sum()
        short_addr_count += ((a_lens > 0) & (a_lens <= 3)).sum()
        
        # Subsample for quantile calculation (sample ~50k rows total across the file)
        sample_rate = max(1, len(chunk) // 10_000)
        name_lengths.extend(n_lens.iloc[::sample_rate].tolist())
        addr_lengths.extend(a_lens.iloc[::sample_rate].tolist())

    del id_set
    
    name_lens_arr = np.array(name_lengths)
    addr_lens_arr = np.array(addr_lengths)
    
    stats = {
        "file_name": name,
        "total_rows": int(total_rows),
        "columns": ["entity_id", "business_name", "business_address", "country"],
        "duplicate_ids": int(dup_ids),
        "missing_values": dict(missing_counts),
        "country_distribution": dict(country_counts),
        "business_name_stats": {
            "blank_count": int(blank_name_count),
            "short_count_len_le_3": int(short_name_count),
            "min_len": int(name_lens_arr.min()) if len(name_lens_arr) else 0,
            "mean_len": float(round(name_lens_arr.mean(), 2)) if len(name_lens_arr) else 0,
            "median_len": float(np.median(name_lens_arr)) if len(name_lens_arr) else 0,
            "p25_len": float(np.percentile(name_lens_arr, 25)) if len(name_lens_arr) else 0,
            "p75_len": float(np.percentile(name_lens_arr, 75)) if len(name_lens_arr) else 0,
            "max_len": int(name_lens_arr.max()) if len(name_lens_arr) else 0,
        },
        "business_address_stats": {
            "blank_count": int(blank_addr_count),
            "short_count_len_le_3": int(short_addr_count),
            "min_len": int(addr_lens_arr.min()) if len(addr_lens_arr) else 0,
            "mean_len": float(round(addr_lens_arr.mean(), 2)) if len(addr_lens_arr) else 0,
            "median_len": float(np.median(addr_lens_arr)) if len(addr_lens_arr) else 0,
            "p25_len": float(np.percentile(addr_lens_arr, 25)) if len(addr_lens_arr) else 0,
            "p75_len": float(np.percentile(addr_lens_arr, 75)) if len(addr_lens_arr) else 0,
            "max_len": int(addr_lens_arr.max()) if len(addr_lens_arr) else 0,
        }
    }
    return stats

def analyze_ground_truth():
    print(f"--- Analyzing Ground Truth ({GROUND_TRUTH_FILE}) ---")
    total_s1 = 0
    match_count_dist = collections.Counter()
    s2_match_counts = collections.Counter()
    s3_match_counts = collections.Counter()
    
    matched_s2_ids = collections.Counter()
    matched_s3_ids = collections.Counter()
    
    both_s2_s3 = 0
    s2_only = 0
    s3_only = 0
    singletons = 0
    
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            total_s1 += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1].strip():
                match_count_dist[0] += 1
                singletons += 1
                continue
                
            m_str = parts[1].strip()
            m_ids = [m.strip() for m in m_str.split(",") if m.strip()]
            n_matches = len(m_ids)
            match_count_dist[n_matches] += 1
            
            s2_ids = [m for m in m_ids if m.startswith("S2-")]
            s3_ids = [m for m in m_ids if m.startswith("S3-")]
            
            s2_match_counts[len(s2_ids)] += 1
            s3_match_counts[len(s3_ids)] += 1
            
            for s2 in s2_ids:
                matched_s2_ids[s2] += 1
            for s3 in s3_ids:
                matched_s3_ids[s3] += 1
                
            if s2_ids and s3_ids:
                both_s2_s3 += 1
            elif s2_ids:
                s2_only += 1
            elif s3_ids:
                s3_only += 1

    s2_multi_s1 = sum(1 for c in matched_s2_ids.values() if c > 1)
    s3_multi_s1 = sum(1 for c in matched_s3_ids.values() if c > 1)
    
    gt_stats = {
        "total_source1_entities": total_s1,
        "singletons_count": singletons,
        "singletons_percentage": float(round((singletons / total_s1) * 100, 4)),
        "non_singletons_count": total_s1 - singletons,
        "non_singletons_percentage": float(round(((total_s1 - singletons) / total_s1) * 100, 4)),
        "both_s2_and_s3_count": both_s2_s3,
        "s2_only_count": s2_only,
        "s3_only_count": s3_only,
        "match_count_distribution": {int(k): int(v) for k, v in sorted(match_count_dist.items())},
        "s2_matches_per_s1_distribution": {int(k): int(v) for k, v in sorted(s2_match_counts.items())},
        "s3_matches_per_s1_distribution": {int(k): int(v) for k, v in sorted(s3_match_counts.items())},
        "total_unique_s2_matched": len(matched_s2_ids),
        "total_unique_s3_matched": len(matched_s3_ids),
        "s2_entities_matching_multiple_s1": s2_multi_s1,
        "s3_entities_matching_multiple_s1": s3_multi_s1,
    }
    return gt_stats

def default_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

def main():
    results = {"sources": {}, "ground_truth": {}}
    for name, path in SOURCE_FILES.items():
        results["sources"][name] = analyze_source_file(path, name)
        
    results["ground_truth"] = analyze_ground_truth()
    
    out_json = os.path.join(EXPERIMENTS_DIR, "eda_statistics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=default_json)
    print(f"Saved EDA statistics to {out_json}")

if __name__ == "__main__":
    main()
