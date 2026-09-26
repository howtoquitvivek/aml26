"""
Matcher training script for Amazon ML Challenge 2026.
Trains an XGBoost matcher using deterministic features and a large deterministic sample.
"""
import argparse
import os
import random
import time
import json
import xgboost as xgb
import experiments.phase3.run_dev_benchmark as rdb
from src.feature_extraction import extract_features, FEATURE_NAMES
from src.candidate_generation import CandidateGenerator

def train_matcher(train_s1_size: int = 100000):
    t0 = time.time()
    
    print(f"Loading {train_s1_size} training S1 records...")
    with open("experiments/val_s1_ids.txt") as f:
        val_s1_ids = set([line.strip() for line in f if line.strip()])
        
    # We load S1 from train dataset
    s1_all = []
    with open("dataset/train/train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                sid = parts[0].strip()
                if sid not in val_s1_ids:
                    s1_all.append({
                        "id": sid,
                        "name": parts[1].strip(),
                        "address": parts[2].strip(),
                        "country": parts[3].strip()
                    })
                    
    random.seed(42)
    random.shuffle(s1_all)
    s1_train = s1_all[:train_s1_size]
    s1_train.sort(key=lambda x: x["country"])
    s1_train_ids = {r["id"] for r in s1_train}
    print(f"Selected {len(s1_train)} train S1 entities.")
    
    print("Loading Ground Truth for Training...")
    gt_map = rdb.load_gt_for_ids(s1_train_ids)
    
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
                    
    print("Generating training candidates...")
    cg = CandidateGenerator("experiments/final_hardening/index")
    
    features = []
    labels = []
    
    # We need to cap candidates per S1 so memory doesn't explode. Max 300 candidates per S1.
    for i, s1 in enumerate(s1_train):
        if i > 0 and i % 10000 == 0:
            print(f"  Generated features for {i} S1 records...")
            
        cands = cg.get_candidates(s1["name"], s1["address"], s1["country"])
        gt_cands = gt_map.get(s1["id"], set())
        gt_cands = {f"S2#{gt}" for gt in gt_cands} | {f"S3#{gt}" for gt in gt_cands}
        
        # We ensure GT is always included even if blocker missed it!
        cands.update(gt_cands)
        
        cands_list = list(cands)
        if len(cands_list) > 300:
            cands_list = random.sample(cands_list, 300)
            cands_list = list(set(cands_list) | gt_cands) # Make sure GT is there
            
        for cid in cands_list:
            if cid not in cand_meta: continue
            cand = cand_meta[cid]
            feat = extract_features(
                s1["name"], s1["address"], s1["country"],
                cand["name"], cand["address"], cand["country"],
                cand["is_s2"]
            )
            label = 1 if cid in gt_cands else 0
            features.append(feat)
            labels.append(label)
            
    print(f"Total candidate pairs for training: {len(features)}")
    
    import numpy as np
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    pos_count = np.sum(y)
    neg_count = len(y) - pos_count
    spw = neg_count / max(1, pos_count)
    
    print(f"Positives: {pos_count}, Negatives: {neg_count}, SPW: {spw:.2f}")
    
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    
    print("Training XGBoost...")
    clf.fit(X, y)
    
    os.makedirs("experiments/final_hardening/artifacts", exist_ok=True)
    out_path = f"experiments/final_hardening/artifacts/xgb_{train_s1_size}.ubj"
    clf.save_model(out_path)
    
    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f} seconds.")
    print(f"Model saved to {out_path}")
    
    # Feature importances
    imp = clf.feature_importances_
    print("Feature Importances:")
    for name, importance in zip(FEATURE_NAMES, imp):
        print(f"  {name}: {importance:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100000)
    args = parser.parse_args()
    train_matcher(args.size)
