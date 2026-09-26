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
from experiments.phase4.ablate_lr_blocking import extract_lr_features, query_custom_s1, evaluate_threshold, FEATURE_NAMES
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import resource

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
    print("Loading 10,000 Holdout IDs...")
    with open("experiments/phase4/artifacts/holdout_10k/s1_ids.txt") as f:
        target_ids = {line.strip() for line in f if line.strip()}
        
    t_start = time.time()
    
    val_gt = rdb.load_gt_for_ids(target_ids)
    s1_data = rdb.load_s1_records(target_ids)
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    master_records = []
    master_blocker_stats = {} 

    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        idx_tuple = load_persisted_index(country)
        cand_ids = idx_tuple[0]
        
        needed_encoded = set()
        
        print(f"[{country}] Querying blocker for Top-100 candidates...")
        for sid, s1 in country_s1.items():
            all_cands, ranked_budget, _ = query_custom_s1(s1, 100, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            
            budget_eids = []
            for i in ranked_budget:
                enc = cand_ids[i]
                needed_encoded.add(enc)
                budget_eids.append(rdb.decode_id(enc))
                
            master_blocker_stats[sid] = {
                "total_cands": len(all_cands),
                "top100_eids": budget_eids
            }
            
        needed_eids_str = {rdb.decode_id(enc) for enc in needed_encoded}
        del idx_tuple
        gc.collect()
        
        cand_meta = {}
        print(f"[{country}] Parsing {len(needed_eids_str)} candidate bodies...")
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
                        
        print(f"[{country}] Extracting features...")
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            eids = master_blocker_stats[sid]["top100_eids"]
            for rank, cid in enumerate(eids):
                cm = cand_meta.get(cid)
                if not cm: continue
                
                vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                label = 1 if cid in gt_set else 0
                
                master_records.append({
                    "sid": sid,
                    "cid": cid,
                    "vec": vec,
                    "label": label,
                    "rank": rank,
                })
                
        del cand_meta
        gc.collect()

    K_VALUES = [25, 50, 75, 100]
    results_out = {}
    
    unique_sids = np.array(sorted(list(target_ids)))
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_splits = list(kf.split(unique_sids))

    for K in K_VALUES:
        print(f"\n======================================")
        print(f"       Evaluating Top-{K} Budget")
        print(f"======================================")
        
        records = [r for r in master_records if r["rank"] < K]
        
        cands_list = [min(K, s["total_cands"]) for s in master_blocker_stats.values()]
        
        total_recovered = 0
        total_gt = 0
        gt_pairs_surviving = 0
        
        for sid, s1 in s1_data.items():
            gt_set = val_gt.get(sid, set())
            total_gt += len(gt_set)
            
            topK_eids = set(master_blocker_stats[sid]["top100_eids"][:K])
            recovered = len(gt_set & topK_eids)
            total_recovered += recovered
            gt_pairs_surviving += recovered
            
        blocking_recall = total_recovered / max(1, total_gt)
        
        b_metrics = {
            "blocking_recall": round(blocking_recall, 6),
            "total_candidates": sum(cands_list),
            "avg_candidates": round(float(np.mean(cands_list)), 2),
            "median_candidates": int(np.median(cands_list)),
            "p95_candidates": int(np.percentile(cands_list, 95)),
            "p99_candidates": int(np.percentile(cands_list, 99)),
            "max_candidates": int(np.max(cands_list)),
            "gt_pairs_surviving": gt_pairs_surviving,
            "final_topk_count": len(records),
        }
        
        print(f"Blocker Recall: {b_metrics['blocking_recall']} | Survived GT: {gt_pairs_surviving}/{total_gt}")
        print(f"Budget fed to ML: {b_metrics['total_candidates']} (Avg {b_metrics['avg_candidates']}/S1)")
        
        lr_probs = np.zeros(len(records))
        y_true = np.array([r["label"] for r in records])
        
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
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
                if len(X_val) > 0:
                    probs = clf.predict_proba(X_val)[:, 1]
                    lr_probs[np.where(val_mask)[0]] = probs
                
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        best_t = 0.5
        best_f05 = 0.0
        best_m = None
        
        for t in thresholds:
            m = evaluate_threshold(records, lr_probs, t, target_ids, val_gt)
            if m["macro_f0_5"] > best_f05:
                best_f05 = m["macro_f0_5"]
                best_t = t
                best_m = m
                
        print(f"LR Best: Thresh {best_t} -> F0.5 = {best_f05:.4f}")
        
        results_out[f"K_{K}"] = {
            "blocker_metrics": b_metrics,
            "lr_best_threshold": best_t,
            "metrics": best_m
        }
        
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    
    final_output = {
        "ablation_results": results_out,
        "runtime_sec": round(time.time() - t_start, 1),
        "peak_rss_mb": round(peak_rss, 1)
    }
    
    with open("results/phase5/topk_ablation/results.json", "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\nAblation complete. Results saved.")

if __name__ == "__main__":
    main()
