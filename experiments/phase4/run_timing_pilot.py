#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
import time
import collections
import random
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import (
    extract_lr_features, query_custom_s1, FEATURE_NAMES
)
from sklearn.linear_model import LogisticRegression
import resource

def create_5k_subset():
    t0 = time.time()
    with open("experiments/val_s1_ids.txt", "r") as f:
        val_pool = {line.strip() for line in f if line.strip()}
    with open("experiments/dev_s1_ids.txt", "r") as f:
        dev_ids = {line.strip() for line in f if line.strip()}
        
    available_pool = val_pool - dev_ids
    
    s1_gt_counts = collections.defaultdict(int)
    with open("dataset/train/train_ground_truth.tsv", "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                s1_gt_counts[parts[0]] += 1
                
    s1_countries = {}
    with open("dataset/train/train_source1.tsv", "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                sid = parts[0]
                if sid in available_pool:
                    s1_countries[sid] = parts[3].strip()
                    
    available_records = []
    for sid in available_pool:
        if sid in s1_countries:
            available_records.append({
                "id": sid,
                "country": s1_countries[sid],
                "match_count": s1_gt_counts[sid]
            })
            
    groups = collections.defaultdict(list)
    for r in available_records:
        mc = r["match_count"]
        bin_c = min(mc, 5)
        groups[(r["country"], bin_c)].append(r["id"])
        
    random.seed(9999) # Different from 10k holdout
    pilot_ids = []
    for key, group in groups.items():
        k_size = int(round(len(group) / len(available_records) * 5000))
        k_size = min(k_size, len(group))
        sampled = random.sample(group, k_size)
        pilot_ids.extend(sampled)
        
    while len(pilot_ids) > 5000:
        pilot_ids.pop()
    while len(pilot_ids) < 5000:
        left = list(available_pool - set(pilot_ids))
        pilot_ids.append(random.choice(left))
        
    intersect = set(pilot_ids) & dev_ids
    if intersect:
        raise ValueError("CRITICAL ERROR: Pilot contains dev IDs!")
        
    final_countries = collections.defaultdict(int)
    final_bins = collections.defaultdict(int)
    for sid in pilot_ids:
        c = s1_countries[sid]
        mc = s1_gt_counts[sid]
        final_countries[c] += 1
        final_bins[min(mc, 5)] += 1
        
    metadata = {
        "sample_size": len(pilot_ids),
        "random_seed": 9999,
        "country_distribution": final_countries,
        "match_count_bin_distribution": final_bins,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "excluded_dev_sample": "experiments/dev_s1_ids.txt",
        "leakage_check_passed": len(intersect) == 0
    }
    
    os.makedirs("experiments/phase4/artifacts/timing_pilot_5k", exist_ok=True)
    with open("experiments/phase4/artifacts/timing_pilot_5k/s1_ids.txt", "w") as f:
        for sid in sorted(pilot_ids):
            f.write(sid + "\n")
            
    with open("experiments/phase4/artifacts/timing_pilot_5k/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    return pilot_ids, time.time() - t0

def load_persisted_index(country: str):
    idx_dir = f"experiments/phase4/artifacts/indexes/notwotok_keye_v1/{country.lower()}"
    if not os.path.exists(idx_dir):
        raise FileNotFoundError(f"Missing index for {country} at {idx_dir}")
    with open(f"{idx_dir}/cand_ids.json") as f: cand_ids = json.load(f)
    with open(f"{idx_dir}/idx_exact.json") as f: idx_exact = json.load(f)
    with open(f"{idx_dir}/idx_canon.json") as f: idx_canon = json.load(f)
    with open(f"{idx_dir}/idx_addr_old.json") as f: idx_addr_old = json.load(f)
    with open(f"{idx_dir}/idx_key_b.json") as f: idx_key_b = json.load(f)
    with open(f"{idx_dir}/idx_key_e.json") as f: idx_key_e = json.load(f)
    idx_two_tok = {}
    return (cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_e)

def main():
    print("1. Creating 5k Timing Pilot Subset...")
    pilot_ids, _ = create_5k_subset()
    
    t_start = time.time()
    
    times = {
        "index_load": 0.0,
        "candidate_gen": 0.0,
        "feature_extract": 0.0,
        "lr_scoring": 0.0,
        "eval_output": 0.0
    }
    
    s1_data = rdb.load_s1_records(pilot_ids)
    val_gt = rdb.load_gt_for_ids(pilot_ids)
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    records = []
    blocker_stats = []
    
    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        
        t0 = time.time()
        idx_tuple = load_persisted_index(country)
        times["index_load"] += time.time() - t0
        
        cand_ids = idx_tuple[0]
        pass1 = {}
        needed_encoded = set()
        
        t0 = time.time()
        for sid, s1 in country_s1.items():
            all_cands, ranked_budget, _ = query_custom_s1(s1, 25, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            blocker_stats.append(len(all_cands))
            
            budget_eids = []
            for i in ranked_budget:
                enc = cand_ids[i]
                needed_encoded.add(enc)
                budget_eids.append(rdb.decode_id(enc))
            pass1[sid] = budget_eids
        times["candidate_gen"] += time.time() - t0
        
        needed_eids_str = {rdb.decode_id(enc) for enc in needed_encoded}
        
        del idx_tuple
        gc.collect()
        
        t0 = time.time()
        cand_meta = {}
        for fpath in (rdb.S2_FILE, rdb.S3_FILE):
            with open(fpath, encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4:
                        eid = parts[0].strip()
                        if eid in needed_eids_str:
                            cand_meta[eid] = {"name": parts[1].strip(), "address": parts[2].strip()}
                    if len(cand_meta) >= len(needed_eids_str):
                        break
                        
        for sid, eids in pass1.items():
            s1 = country_s1[sid]
            gt_set = val_gt.get(sid, set())
            for cid in eids:
                cm = cand_meta.get(cid)
                if not cm: continue
                vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                records.append({
                    "sid": sid,
                    "cid": cid,
                    "vec": vec,
                    "label": 1 if cid in gt_set else 0,
                    "gt_set": gt_set,
                })
        times["feature_extract"] += time.time() - t0
        
        del pass1, cand_meta
        gc.collect()
        
    t0 = time.time()
    X_all = np.array([r["vec"] for r in records])
    y_all = np.array([r["label"] for r in records])
    
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
    if len(X_all) > 0:
        clf.fit(X_all, y_all)
        probs = clf.predict_proba(X_all)[:, 1]
    times["lr_scoring"] += time.time() - t0
    
    t0 = time.time()
    times["eval_output"] += time.time() - t0
    
    t_total = time.time() - t_start
    throughput = len(pilot_ids) / t_total
    
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    
    res = {
        "times": times,
        "total_wall_sec": round(t_total, 2),
        "s1_per_sec": round(throughput, 2),
        "peak_rss_mb": round(peak_rss, 1),
        "candidates": {
            "total": sum(blocker_stats),
            "avg": round(float(np.mean(blocker_stats)), 2),
            "p95": int(np.percentile(blocker_stats, 95)),
            "p99": int(np.percentile(blocker_stats, 99)),
            "max": int(np.max(blocker_stats))
        }
    }
    
    print("\n--- TIMING PILOT RESULTS ---")
    print(json.dumps(res, indent=2))
    
    full_count = 220683
    eta_sec = full_count / throughput
    print(f"\nEstimated Full Runtime: {eta_sec/3600:.2f} hours ({eta_sec:.0f} seconds)")

if __name__ == "__main__":
    main()
