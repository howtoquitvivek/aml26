"""
End-to-End Pipeline Evaluation for Amazon ML Challenge 2026.
Evaluates the final pipeline against the validation set.
"""
import os
import json
import time
import xgboost as xgb
import numpy as np
import experiments.phase3.run_dev_benchmark as rdb
from src.candidate_generation import CandidateGenerator
from src.feature_extraction import extract_features

def evaluate_pipeline(model_path: str, threshold: float = 0.90, max_k: int = 300):
    t0 = time.time()
    
    print(f"Loading XGBoost model from {model_path}...")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    
    print("Loading FULL S1 validation set...")
    with open("experiments/val_s1_ids.txt") as f:
        all_s1_ids = set([line.strip() for line in f if line.strip()])
        
    val_gt = rdb.load_gt_for_ids(all_s1_ids)
    s1_data = rdb.load_s1_records(all_s1_ids)
    
    print("Loading candidate metadata...")
    cand_meta = {}
    for fpath in ("dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"):
        prefix = "S2" if "source2" in fpath else "S3"
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    cand_meta[f"{prefix}#{parts[0].strip()}"] = {
                        "name": parts[1].strip(),
                        "address": parts[2].strip(),
                        "country": parts[3].strip(),
                        "is_s2": (prefix == "S2")
                    }
                    
    print("Evaluating pipeline...")
    cg = CandidateGenerator("experiments/final_hardening/index")
    
    s1_items = list(s1_data.items())
    s1_items.sort(key=lambda x: x[1]["country"])
    
    predictions = {}
    total_cands = 0
    total_cands_retained = 0
    
    for i, (sid, s1) in enumerate(s1_items):
        if i > 0 and i % 10000 == 0:
            print(f"Processed {i} queries...")
            
        cands = cg.get_candidates(s1["name"], s1["address"], s1["country"])
        total_cands += len(cands)
        
        if not cands:
            predictions[sid] = set()
            continue
            
        cands_list = list(cands)
        
        # In a real system, we'd have a heuristic fast-ranker here to trim to max_k.
        # But this script is just final evaluation over generated candidates.
        # If we have > max_k candidates, we sort them deterministically by a fast score
        # to ensure K truncation is consistent.
        if len(cands_list) > max_k:
            from src.baseline_matcher import compute_pair_features, compute_deterministic_score
            fast_scores = []
            for c in cands_list:
                cand = cand_meta[c]
                feat = compute_pair_features(
                    s1["name"], s1["address"], s1["country"],
                    cand["name"], cand["address"], cand["country"]
                )
                fast_scores.append((c, compute_deterministic_score(feat, "balanced")))
            fast_scores.sort(key=lambda x: x[1], reverse=True)
            cands_list = [x[0] for x in fast_scores[:max_k]]
            
        total_cands_retained += len(cands_list)
        
        features = []
        for c in cands_list:
            cand = cand_meta[c]
            feat = extract_features(
                s1["name"], s1["address"], s1["country"],
                cand["name"], cand["address"], cand["country"],
                cand["is_s2"]
            )
            features.append(feat)
            
        X = np.array(features, dtype=np.float32)
        probs = clf.predict_proba(X)[:, 1]
        
        pred_cands = set()
        for c, prob in zip(cands_list, probs):
            if prob >= threshold:
                pred_cands.add(c.split("#")[1]) # Return raw ID
                
        predictions[sid] = pred_cands
        
    t1 = time.time()
    print(f"Evaluation took {t1 - t0:.2f} seconds.")
    print(f"Total candidates from index: {total_cands} (avg {total_cands/len(s1_items):.1f})")
    print(f"Total candidates after K={max_k}: {total_cands_retained} (avg {total_cands_retained/len(s1_items):.1f})")
    
    print("\nComputing metrics...")
    from src.evaluation import compute_entity_metrics
    metrics = compute_entity_metrics(val_gt, predictions)
    
    print(f"TP: {metrics['TP']}")
    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Macro F0.5: {metrics['f05_macro']:.4f}")
    
    with open("experiments/final_hardening/evaluation_results.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "avg_candidates_pre_k": total_cands/len(s1_items),
            "avg_candidates_post_k": total_cands_retained/len(s1_items),
            "threshold": threshold,
            "max_k": max_k
        }, f, indent=2)

if __name__ == "__main__":
    evaluate_pipeline("experiments/final_hardening/artifacts/xgb_100000.ubj")
