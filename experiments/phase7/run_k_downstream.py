import sys
import os
import json
import time
import collections
import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import query_custom_s1, extract_lr_features
from experiments.phase7.run_k_optimization import load_persisted_index, load_validation_data
from src.evaluation import evaluate_predictions

def evaluate_downstream(k, s1_results, val_gt, total_gt, baseline_results=None):
    total_cands = 0
    cand_counts = []
    
    predictions = {}
    
    for sid, res in s1_results.items():
        cands = res["cands"][:k]
        probs = res["probs"][:k]
        
        cand_counts.append(len(cands))
        
        # Threshold = 0.90
        preds = {cands[i] for i in range(len(cands)) if probs[i] >= 0.90}
        predictions[sid] = preds
        
    cand_counts = np.array(cand_counts)
    
    eval_summary = evaluate_predictions(val_gt, predictions)
    
    stats = {
        "total_cands": int(np.sum(cand_counts)),
        "avg_cands": float(np.mean(cand_counts)),
        "tp": eval_summary["aggregate_tp"],
        "fp": eval_summary["aggregate_fp"],
        "fn": eval_summary["aggregate_fn"],
        "precision": eval_summary["macro_precision"],
        "recall": eval_summary["macro_recall"],
        "f05": eval_summary["macro_f0_5"],
        "singleton_accuracy": eval_summary["singleton_accuracy"],
        "singletons_total": eval_summary["singletons_total"],
        "singletons_correct": eval_summary["singletons_correct"],
        "total_source1_entities": eval_summary["total_source1_entities"],
        "predicted_empty": sum(1 for sid, preds in predictions.items() if not preds),
        "zero_candidates": sum(1 for c in cand_counts if c == 0)
    }
    
    if baseline_results:
        stats["delta_precision"] = stats["precision"] - baseline_results["precision"]
        stats["delta_recall"] = stats["recall"] - baseline_results["recall"]
        stats["delta_f05"] = stats["f05"] - baseline_results["f05"]
        stats["add_tp"] = stats["tp"] - baseline_results["tp"]
        stats["add_fp"] = stats["fp"] - baseline_results["fp"]
        stats["add_fn"] = stats["fn"] - baseline_results["fn"]
        
    return stats

def main():
    print("Loading validation data...")
    target_ids, val_gt, s1_data = load_validation_data()
    total_gt = sum(len(v) for v in val_gt.values() if v)
    
    print("Loading XGBoost Model...")
    clf = XGBClassifier()
    clf.load_model("experiments/phase6/artifacts/final_xgb_model.ubj")
    
    print("Loading S2/S3 Metadata...")
    # Load S2 and S3 directly to get metadata for feature extraction
    cand_meta = {}
    for fpath in ("dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    cand_meta[parts[0].strip()] = {"name": parts[1].strip(), "address": parts[2].strip()}
                    
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    s1_results = {}
    
    t_start = time.time()
    
    # Max K to evaluate is 300
    MAX_K = 300
    
    for country in unique_countries:
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        if not country_s1: continue
        
        try:
            print(f"[{country}] Loading persisted index...")
            idx_tuple = load_persisted_index(country)
        except Exception as e:
            print(f"[{country}] Persisted index not found. Building dynamically...")
            from experiments.phase7.run_blocking_recall_experiments import build_index_for_country
            idx_tuple_full = build_index_for_country(country, "BASELINE")
            idx_tuple = (idx_tuple_full[0], idx_tuple_full[1], idx_tuple_full[2], {}, idx_tuple_full[3], idx_tuple_full[4], idx_tuple_full[5])
            
        cand_ids = idx_tuple[0]
        
        print(f"[{country}] Extracting features and scoring...")
        for sid, s1 in country_s1.items():
            all_cands, ranked_budget, _ = query_custom_s1(s1, MAX_K, *idx_tuple, "CFG1_NoTwoTok_KeyE")
            
            ranked_encs = [cand_ids[i] for i in ranked_budget]
            decoded_cands = [rdb.decode_id(e) for e in ranked_encs]
            
            features = []
            valid_cands = []
            for cid in decoded_cands:
                cm = cand_meta.get(cid)
                if not cm: continue
                vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                features.append(vec)
                valid_cands.append(cid)
                
            if not features:
                s1_results[sid] = {"cands": [], "probs": []}
                continue
                
            X = np.array(features, dtype=np.float32)
            probs = clf.predict_proba(X)[:, 1]
            
            s1_results[sid] = {"cands": valid_cands, "probs": probs.tolist()}
            
    t_end = time.time()
    runtime = t_end - t_start
    print(f"Feature Extraction & Scoring took {runtime:.2f}s")
    
    results = {}
    
    print("\n--- Evaluating K=100 (BASELINE) ---")
    base_stats = evaluate_downstream(100, s1_results, val_gt, total_gt)
    results["K=100"] = base_stats
    print(json.dumps(base_stats, indent=2))
    
    for k in [150, 200, 300]:
        print(f"\n--- Evaluating K={k} ---")
        stats = evaluate_downstream(k, s1_results, val_gt, total_gt, base_stats)
        results[f"K={k}"] = stats
        print(json.dumps(stats, indent=2))
        
    with open("experiments/phase7/k_downstream_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
