#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
import time
import collections

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import (
    extract_lr_features, query_custom_s1, evaluate_threshold, FEATURE_NAMES
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import resource

def load_persisted_index(country: str):
    idx_dir = f"experiments/phase4/artifacts/indexes/notwotok_keye_v1/{country.lower()}"
    if not os.path.exists(idx_dir):
        raise FileNotFoundError(f"Missing index for {country} at {idx_dir}")
        
    print(f"[{country}] Loading persisted index...")
    t0 = time.time()
    with open(f"{idx_dir}/cand_ids.json") as f: cand_ids = json.load(f)
    with open(f"{idx_dir}/idx_exact.json") as f: idx_exact = json.load(f)
    with open(f"{idx_dir}/idx_canon.json") as f: idx_canon = json.load(f)
    with open(f"{idx_dir}/idx_addr_old.json") as f: idx_addr_old = json.load(f)
    with open(f"{idx_dir}/idx_key_b.json") as f: idx_key_b = json.load(f)
    with open(f"{idx_dir}/idx_key_e.json") as f: idx_key_e = json.load(f)
    idx_two_tok = {}
    print(f"[{country}] Index loaded in {time.time()-t0:.1f}s")
    
    return (cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_e)

def main():
    print("Loading 10,000 Holdout IDs...")
    with open("experiments/phase4/artifacts/holdout_10k/s1_ids.txt") as f:
        target_ids = {line.strip() for line in f if line.strip()}
        
    t_start = time.time()
    
    val_gt = rdb.load_gt_for_ids(target_ids)
    s1_data = rdb.load_s1_records(target_ids)
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    records = []
    blocker_stats = []
    
    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        idx_tuple = load_persisted_index(country)
        cand_ids = idx_tuple[0]
        
        pass1 = {}
        needed_encoded = set()
        
        print(f"[{country}] Querying blocker for {len(country_s1)} entities...")
        for sid, s1 in country_s1.items():
            all_cands, ranked_budget, _ = query_custom_s1(s1, 25, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            
            gt_set = val_gt.get(sid, set())
            gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
            retrieved_encs = {cand_ids[i] for i in all_cands}
            recovered = len(gt_enc_set & retrieved_encs)
            
            blocker_stats.append({
                "sid": sid,
                "cand_count": len(all_cands),
                "budget_count": len(ranked_budget),
                "recovered": recovered,
                "total_gt": len(gt_set)
            })
            
            budget_eids = []
            for i in ranked_budget:
                enc = cand_ids[i]
                needed_encoded.add(enc)
                budget_eids.append(rdb.decode_id(enc))
            pass1[sid] = budget_eids
            
        needed_eids_str = {rdb.decode_id(enc) for enc in needed_encoded}
        
        del idx_tuple
        gc.collect()
        
        cand_meta = {}
        print(f"[{country}] Parsing S2/S3 for {len(needed_eids_str)} candidate bodies...")
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
                        
        print(f"[{country}] Extracting LR features...")
        for sid, eids in pass1.items():
            s1 = country_s1[sid]
            gt_set = val_gt.get(sid, set())
            for cid in eids:
                cm = cand_meta.get(cid)
                if not cm: continue
                
                vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                label = 1 if cid in gt_set else 0
                
                records.append({
                    "sid": sid,
                    "cid": cid,
                    "vec": vec,
                    "label": label,
                    "gt_set": gt_set,
                })
                
        del pass1, cand_meta
        gc.collect()

    cands_list = [s["cand_count"] for s in blocker_stats]
    total_recovered = sum(s["recovered"] for s in blocker_stats)
    total_gt = sum(s["total_gt"] for s in blocker_stats)
    blocking_recall = total_recovered / max(1, total_gt)
    
    b_metrics = {
        "blocking_recall": round(blocking_recall, 6),
        "total_candidates": sum(cands_list),
        "avg_candidates": round(float(np.mean(cands_list)), 2),
        "median_candidates": int(np.median(cands_list)),
        "p95_candidates": int(np.percentile(cands_list, 95)),
        "p99_candidates": int(np.percentile(cands_list, 99)),
        "max_candidates": int(np.max(cands_list)),
        "final_topk_count": len(records),
    }
    
    print(f"\n--- BLOCKER STATS (10K HOLDOUT) ---")
    print(f"Recall: {blocking_recall:.4f} ({total_recovered}/{total_gt})")
    print(f"Candidates: {b_metrics['total_candidates']} (Avg {b_metrics['avg_candidates']})")
    
    print("\n--- 3-FOLD CV LOGISTIC REGRESSION ---")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    unique_sids = np.array(sorted(list(target_ids)))
    
    lr_probs = np.zeros(len(records))
    y_true = np.array([r["label"] for r in records])
    
    print(f"Training LR on {len(records)} pairs (Pos: {sum(y_true)}, Neg: {len(y_true)-sum(y_true)})")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_sids)):
        train_sids = set(unique_sids[train_idx])
        val_sids = set(unique_sids[val_idx])
        
        train_mask = [r["sid"] in train_sids for r in records]
        val_mask = [r["sid"] in val_sids for r in records]
        
        X_train = np.array([r["vec"] for i, r in enumerate(records) if train_mask[i]])
        y_train = y_true[train_mask]
        
        X_val = np.array([r["vec"] for i, r in enumerate(records) if val_mask[i]])
        
        clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
        if len(X_train) > 0:
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)[:, 1]
            lr_probs[np.where(val_mask)[0]] = probs
            
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    best_t = 0.5
    best_f05 = 0.0
    lr_thresh_results = {}
    for t in thresholds:
        m = evaluate_threshold(records, lr_probs, t, target_ids, val_gt)
        lr_thresh_results[t] = m
        if m["macro_f0_5"] > best_f05:
            best_f05 = m["macro_f0_5"]
            best_t = t
            
    print(f"\nLR Best: Thresh {best_t} -> F0.5 = {best_f05:.4f}")
    
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    
    final_output = {
        "blocker_metrics": b_metrics,
        "lr_best_threshold": best_t,
        "lr_best_metrics": lr_thresh_results[best_t],
        "lr_thresholds": lr_thresh_results,
        "runtime_sec": round(time.time() - t_start, 1),
        "peak_rss_mb": round(peak_rss, 1)
    }
    
    os.makedirs("results/phase4/holdout_10k", exist_ok=True)
    with open("results/phase4/holdout_10k/evaluation.json", "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\nResults successfully saved.")

if __name__ == "__main__":
    main()
