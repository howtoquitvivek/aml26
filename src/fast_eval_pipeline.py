import os
import json
import time
import multiprocessing as mp
import numpy as np
import xgboost as xgb
from functools import partial

# We will fork, so global variables will be copy-on-write
GLOBAL_CLF = None
GLOBAL_CAND_META = None
GLOBAL_CG = None
GLOBAL_CG_MR = None

def init_worker():
    # Force sklearn/xgboost to use single thread inside workers to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def process_chunk(s1_chunk, threshold, max_k):
    from src.feature_extraction import extract_features
    from src.baseline_matcher import compute_pair_features, compute_deterministic_score
    
    results = {}
    total_cands = 0
    total_retained = 0
    
    for sid, s1 in s1_chunk:
        cands_p3 = GLOBAL_CG.get_candidates(s1["name"], s1["address"], s1["country"])
        cands_mr = GLOBAL_CG_MR.get_candidates(s1["name"], s1["address"], s1["country"])
        
        cands = cands_p3 | cands_mr
        total_cands += len(cands)
        
        if not cands:
            results[sid] = set()
            continue
            
        cands_list = list(cands)
        
        if len(cands_list) > max_k:
            fast_scores = []
            for c in cands_list:
                cand = GLOBAL_CAND_META.get(c)
                if not cand: continue
                feat = compute_pair_features(
                    s1["name"], s1["address"], s1["country"],
                    cand["name"], cand["address"], cand["country"]
                )
                fast_scores.append((c, compute_deterministic_score(feat, "balanced")))
            fast_scores.sort(key=lambda x: x[1], reverse=True)
            cands_list = [x[0] for x in fast_scores[:max_k]]
            
        total_retained += len(cands_list)
        
        features = []
        valid_cands = []
        for c in cands_list:
            cand = GLOBAL_CAND_META.get(c)
            if not cand: continue
            feat = extract_features(
                s1["name"], s1["address"], s1["country"],
                cand["name"], cand["address"], cand["country"],
                cand["is_s2"]
            )
            features.append(feat)
            valid_cands.append(c)
            
        if not features:
            results[sid] = set()
            continue
            
        X = np.array(features, dtype=np.float32)
        probs = GLOBAL_CLF.predict_proba(X)[:, 1]
        
        # Store ALL probabilities so we can test thresholds
        results[sid] = []
        for c, prob in zip(valid_cands, probs):
            results[sid].append((c.split("#")[1], prob))
            
    return results, total_cands, total_retained

def load_country_metadata(country):
    print(f"Loading candidate metadata for {country}...")
    cand_meta = {}
    for fpath in ("dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"):
        prefix = "S2" if "source2" in fpath else "S3"
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3].strip() == country:
                    cand_meta[f"{prefix}#{parts[0].strip()}"] = {
                        "name": parts[1].strip(),
                        "address": parts[2].strip(),
                        "country": parts[3].strip(),
                        "is_s2": (prefix == "S2")
                    }
    return cand_meta

def evaluate_fast(model_path: str, ids_file: str = "experiments/dev_s1_ids.txt"):
    global GLOBAL_CLF, GLOBAL_CAND_META, GLOBAL_CG, GLOBAL_CG_MR
    import experiments.phase3.run_dev_benchmark as rdb
    from src.candidate_generation import CandidateGenerator
    from src.candidate_generation_mr import MRCandidateGenerator
    from src.evaluation import compute_entity_metrics
    import gc
    
    mp.set_start_method("fork", force=True)
    
    t0 = time.time()
    
    print(f"Loading XGBoost model from {model_path}...")
    GLOBAL_CLF = xgb.XGBClassifier()
    GLOBAL_CLF.load_model(model_path)
    GLOBAL_CLF.set_params(n_jobs=1)
    
    print(f"Loading S1 validation set from {ids_file}...")
    with open(ids_file) as f:
        all_s1_ids = set([line.strip() for line in f if line.strip()])
        
    val_gt = rdb.load_gt_for_ids(all_s1_ids)
    s1_data = rdb.load_s1_records(all_s1_ids)
    
    GLOBAL_CG = CandidateGenerator("experiments/final_hardening/index")
    GLOBAL_CG_MR = MRCandidateGenerator("experiments/final_hardening/index")
    
    all_predictions_with_probs = collections.defaultdict(list)
    global_total_cands = 0
    global_total_retained = 0
    
    for country in ["India", "US"]:
        print(f"\n=== Processing {country} ===")
        GLOBAL_CAND_META = load_country_metadata(country)
        GLOBAL_CG.load_country(country)
        GLOBAL_CG_MR.load_country(country)
        
        country_s1 = [(sid, s1) for sid, s1 in s1_data.items() if s1["country"] == country]
        print(f"Found {len(country_s1)} S1 queries for {country}")
        
        chunk_size = 500
        chunks = [country_s1[i:i + chunk_size] for i in range(0, len(country_s1), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            res_dict, t_cands, t_retained = process_chunk(chunk, 0.0, 300)
            for k, v in res_dict.items():
                all_predictions_with_probs[k].extend(v)
            global_total_cands += t_cands
            global_total_retained += t_retained
            print(f"  Completed chunk {i+1}/{len(chunks)}... (cands={t_cands})")
            import sys
            sys.stdout.flush()
            
        print(f"Finished {country}.")
        
        GLOBAL_CAND_META.clear()
        GLOBAL_CG.idx_base.clear()
        GLOBAL_CG_MR.idx_base.clear()
        gc.collect()
        
    t1 = time.time()
    print(f"\nEvaluation took {t1 - t0:.2f} seconds.")
    
    from src.evaluation import evaluate_predictions
    
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.92, 0.94, 0.96, 0.98]
    best_f05 = 0.0
    best_t = 0.90
    
    for t in thresholds:
        print(f"\nComputing metrics for threshold {t}...")
        t_preds = {}
        for sid, cands_probs in all_predictions_with_probs.items():
            t_preds[sid] = {c for c, p in cands_probs if p >= t}
            
        metrics = evaluate_predictions(val_gt, t_preds)
        print(f"TP: {metrics['aggregate_tp']}, FP: {metrics['aggregate_fp']}, FN: {metrics['aggregate_fn']}")
        print(f"Precision: {metrics['macro_precision']:.4f}")
        print(f"Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F0.5: {metrics['macro_f0_5']:.4f}")
        
        if metrics['macro_f0_5'] > best_f05:
            best_f05 = metrics['macro_f0_5']
            best_t = t

    print(f"\nBest threshold is {best_t} with F0.5 = {best_f05:.4f}")

if __name__ == "__main__":
    import argparse
    import collections
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="experiments/final_hardening/artifacts/xgb_100000.ubj")
    parser.add_argument("--ids", type=str, default="experiments/dev_s1_ids.txt")
    args = parser.parse_args()
    
    evaluate_fast(args.model, args.ids)
