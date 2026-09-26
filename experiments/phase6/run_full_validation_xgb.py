#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
import time
import collections
import random

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import extract_lr_features, query_custom_s1
from xgboost import XGBClassifier
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

def train_frozen_model():
    print("Training Frozen XGBoost Model on 1,500-S1 Clean Dev Set...")
    with open("experiments/phase6/artifacts/clean_dev_1500/clean_dev_ids.txt") as f:
        dev_ids = {line.strip() for line in f if line.strip()}
    
    val_gt = rdb.load_gt_for_ids(dev_ids)
    s1_data = rdb.load_s1_records(dev_ids)
    
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    records = []
    
    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        idx_tuple = load_persisted_index(country)
        cand_ids = idx_tuple[0]
        
        pass1 = {}
        needed_encoded = set()
        
        for sid, s1 in country_s1.items():
            all_cands, ranked_budget, _ = query_custom_s1(s1, 100, *idx_tuple, "CFG1_NoTwoTok_KeyE")
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
                    "vec": vec,
                    "label": 1 if cid in gt_set else 0
                })
        del pass1, cand_meta
        gc.collect()
        
    X_train = np.array([r["vec"] for r in records], dtype=np.float32)
    y_train = np.array([r["label"] for r in records], dtype=bool)
    
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', scale_pos_weight=pos_weight)
    clf.fit(X_train, y_train)
    print("Frozen XGBoost model trained.")
    return clf, 0.90  # Replace this value if cross-validation on clean dev yields a different optimal threshold

def main():
    dry_run = "--dry-run" in sys.argv
    print(f"Starting Validation Run (Dry Run: {dry_run})...")
    
    clf, best_t = train_frozen_model()
    
    t_start = time.time()
    
    with open("experiments/val_s1_ids.txt") as f:
        target_ids = [line.strip() for line in f if line.strip()]
        
    if dry_run:
        random.seed(42)
        target_ids = random.sample(target_ids, 200)
        
    target_ids_set = set(target_ids)
    num_s1 = len(target_ids)
    sid_to_int = {sid: i for i, sid in enumerate(target_ids)}
    
    val_gt = rdb.load_gt_for_ids(target_ids_set)
    gt_sizes = np.zeros(num_s1, dtype=int)
    for sid, gt in val_gt.items():
        if sid in sid_to_int:
            gt_sizes[sid_to_int[sid]] = len(gt)
            
    s1_data = rdb.load_s1_records(target_ids_set)
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    all_vecs = []
    all_labels = []
    all_sids = []
    
    blocker_stats = []
    
    final_output_metrics = {
        "accepted_matches": 0,
        "predicted_singletons": 0,
        "duplicate_output_ids": 0,
        "invalid_output_ids": 0
    }
    
    times = {
        "index_load": 0.0,
        "candidate_gen": 0.0,
        "feature_extract": 0.0,
        "lr_scoring": 0.0
    }
    
    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        if not country_s1: continue
        
        t0 = time.time()
        idx_tuple = load_persisted_index(country)
        cand_ids = idx_tuple[0]
        times["index_load"] += time.time() - t0
        
        chunk_size = 20000
        country_sids = list(country_s1.keys())
        
        for chunk_idx in range(0, len(country_sids), chunk_size):
            chunk = country_sids[chunk_idx:chunk_idx+chunk_size]
            
            pass1 = {}
            needed_encoded = set()
            
            t0 = time.time()
            for sid in chunk:
                s1 = country_s1[sid]
                all_cands, ranked_budget, _ = query_custom_s1(s1, 100, *idx_tuple, "CFG1_NoTwoTok_KeyE")
                
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
            times["candidate_gen"] += time.time() - t0
            
            t0 = time.time()
            needed_eids_str = {rdb.decode_id(enc) for enc in needed_encoded}
            
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
                
                seen = set()
                for cid in eids:
                    if cid in seen:
                        final_output_metrics["duplicate_output_ids"] += 1
                    seen.add(cid)
                    if cid not in cand_meta:
                        final_output_metrics["invalid_output_ids"] += 1
                        continue
                        
                    cm = cand_meta[cid]
                    vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                    all_vecs.append(vec)
                    all_labels.append(1 if cid in gt_set else 0)
                    all_sids.append(sid_to_int[sid])
                    
            times["feature_extract"] += time.time() - t0
            
            del pass1, cand_meta, needed_eids_str
            gc.collect()
            
        del idx_tuple
        gc.collect()
        
    t0 = time.time()
    X_val = np.array(all_vecs, dtype=np.float32)
    y_val = np.array(all_labels, dtype=bool)
    sids_val = np.array(all_sids, dtype=np.int32)
    
    print(f"Scoring {X_val.shape[0]} candidates...")
    probs = clf.predict_proba(X_val)[:, 1]
    preds = probs >= best_t
    
    tp_mask = y_val & preds
    fp_mask = (~y_val) & preds
    
    tp_counts = np.bincount(sids_val[tp_mask], minlength=num_s1)
    fp_counts = np.bincount(sids_val[fp_mask], minlength=num_s1)
    
    times["lr_scoring"] += time.time() - t0
    
    fn_counts = gt_sizes - tp_counts
    
    pred_pos = tp_counts + fp_counts
    prec = np.zeros(num_s1)
    valid_prec = pred_pos > 0
    prec[valid_prec] = tp_counts[valid_prec] / pred_pos[valid_prec]
    prec[~valid_prec] = 1.0
    
    rec = np.zeros(num_s1)
    valid_rec = gt_sizes > 0
    rec[valid_rec] = tp_counts[valid_rec] / gt_sizes[valid_rec]
    rec[~valid_rec] = 1.0
    
    f05 = np.zeros(num_s1)
    valid_f05 = (prec > 0) | (rec > 0)
    f05[valid_f05] = (1.25 * prec[valid_f05] * rec[valid_f05]) / (0.25 * prec[valid_f05] + rec[valid_f05])
    f05[~valid_f05] = 0.0
    
    is_singleton = gt_sizes == 1
    singleton_correct = is_singleton & (tp_counts == 1) & (fp_counts == 0)
    
    final_output_metrics["accepted_matches"] = int(np.sum(pred_pos))
    final_output_metrics["predicted_singletons"] = int(np.sum(pred_pos == 1))
    final_output_metrics["average_final_candidates"] = round(float(np.mean(pred_pos)), 2)
    
    # blocker metrics
    cands_list = [s["cand_count"] for s in blocker_stats]
    total_recovered = sum(s["recovered"] for s in blocker_stats)
    total_gt = sum(s["total_gt"] for s in blocker_stats)
    blocking_recall = total_recovered / max(1, total_gt)
    
    cands_np = np.array(cands_list)
    b_metrics = {
        "blocking_recall": round(float(blocking_recall), 6),
        "total_candidates": sum(cands_list),
        "avg_candidates": round(float(np.mean(cands_list)), 2),
        "median_candidates": int(np.median(cands_list)),
        "p95_candidates": int(np.percentile(cands_list, 95)),
        "p99_candidates": int(np.percentile(cands_list, 99)),
        "max_candidates": int(np.max(cands_list)),
        "s1_over_1k": int(np.sum(cands_np > 1000)),
        "s1_over_100_before_topk": int(np.sum(cands_np > 100)),
        "s1_reaching_exactly_100_fed_to_lr": int(np.sum([s["budget_count"] == 100 for s in blocker_stats])),
        "s1_over_5k": int(np.sum(cands_np > 5000)),
        "s1_over_10k": int(np.sum(cands_np > 10000)),
        "s1_over_50k": int(np.sum(cands_np > 50000)),
        "final_topk_count": len(all_vecs)
    }
    
    eval_metrics = {
        "macro_f0_5": round(float(np.mean(f05)), 6),
        "macro_precision": round(float(np.mean(prec)), 6),
        "macro_recall": round(float(np.mean(rec)), 6),
        "tp": int(np.sum(tp_counts)),
        "fp": int(np.sum(fp_counts)),
        "fn": int(np.sum(fn_counts)),
        "singleton_acc": round(float(np.sum(singleton_correct) / max(1, np.sum(is_singleton))), 6)
    }
    
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    
    res = {
        "metrics": eval_metrics,
        "blocker_metrics": b_metrics,
        "final_output_metrics": final_output_metrics,
        "resource_metrics": {
            "times": {k: round(v, 2) for k,v in times.items()},
            "total_runtime": round(time.time() - t_start, 2),
            "peak_rss_mb": round(peak_rss, 1)
        }
    }
    
    print("\n--- RESULTS ---")
    print(json.dumps(res, indent=2))
    
    prefix = "clean_" if not dry_run else "dryrun_"
    os.makedirs("results/phase6", exist_ok=True)
    with open(f"results/phase6/{prefix}evaluation.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
