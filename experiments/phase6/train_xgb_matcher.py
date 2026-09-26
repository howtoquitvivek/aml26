#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
import time
import collections
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from src.baseline_matcher import compute_pair_features, compute_deterministic_score
from src.evaluation import compute_entity_metrics
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

FEATURE_NAMES = [
    "name_exact",
    "canon_exact",
    "name_tok_jac",
    "name_char_jac",
    "name_len_diff",
    "addr_exact",
    "addr_missing",
    "addr_tok_jac",
    "addr_char_jac",
    "addr_num_jac",
    "addr_len_diff",
    "postal_overlap",
    "salient_addr_jac",
    "is_s2",
    "country_match"
]

def extract_lr_features(s1_name, s1_addr, cand_name, cand_addr, cand_id, country):
    base = compute_pair_features(s1_name, s1_addr, country, cand_name, cand_addr, country)
    
    s1_nums, s1_alphas = rdb.extract_address_features(s1_addr)
    cand_nums, cand_alphas = rdb.extract_address_features(cand_addr)
    
    s1_postal = {n for n in s1_nums if len(n) in (5, 6)}
    cand_postal = {n for n in cand_nums if len(n) in (5, 6)}
    postal_overlap = 1.0 if s1_postal and cand_postal and not s1_postal.isdisjoint(cand_postal) else 0.0
    
    s1_alpha_set = set(s1_alphas)
    cand_alpha_set = set(cand_alphas)
    if not s1_alpha_set or not cand_alpha_set:
        salient_overlap = 0.0
    else:
        salient_overlap = len(s1_alpha_set & cand_alpha_set) / len(s1_alpha_set | cand_alpha_set)
        
    is_s2 = 1.0 if cand_id.startswith("S2") else 0.0
    
    return [
        base["name_exact"],
        base["name_canonical_exact"],
        base["name_token_jaccard"],
        base["name_char_jaccard"],
        base["name_len_diff_ratio"],
        base["addr_exact"],
        base["addr_missing"],
        base["addr_token_jaccard"],
        base["addr_char_jaccard"],
        base["addr_numeric_jaccard"],
        base["addr_len_diff_ratio"],
        postal_overlap,
        salient_overlap,
        is_s2,
        base["country_match"]
    ]

def collect_dataset(target_ids: Set[str], budget: int):
    t0 = time.time()
    s1_data = rdb.load_s1_records(target_ids)
    val_gt = rdb.load_gt_for_ids(target_ids)
    
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    records = []
    
    for country in unique_countries:
        print(f"Collecting features for {country}...")
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_c, idx_key_d = rdb.build_country_index(country)
        
        pass1 = {}
        needed_encoded = set()
        
        for sid, s1 in country_s1.items():
            cfg1, _, _, ranked_budget, _ = rdb.query_s1(
                s1, budget, cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_c, idx_key_d
            )
            budget_eids = []
            for i in ranked_budget:
                enc = cand_ids[i]
                needed_encoded.add(enc)
                budget_eids.append(rdb.decode_id(enc))
            pass1[sid] = budget_eids
            
        needed_eids_str = {rdb.decode_id(enc) for enc in needed_encoded}
        del cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_c, idx_key_d
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
                
                # Base features for baseline evaluation
                base_feats = compute_pair_features(s1["name"], s1["address"], country, cm["name"], cm["address"], country)
                base_score = compute_deterministic_score(base_feats, "conservative_precision")
                
                vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                label = 1 if cid in gt_set else 0
                
                records.append({
                    "sid": sid,
                    "cid": cid,
                    "vec": vec,
                    "label": label,
                    "gt_set": gt_set,
                    "base_score": base_score
                })
                
        del pass1, cand_meta
        gc.collect()
        
    print(f"Collection finished in {time.time()-t0:.1f}s")
    return records

def evaluate_threshold(records, probs, threshold, target_sids, gt_map):
    per_entity = {sid: {"gt_set": gt_map.get(sid, set()), "pred_set": set()} for sid in target_sids}
    for i, r in enumerate(records):
        sid = r["sid"]
        if sid in per_entity and probs[i] >= threshold:
            per_entity[sid]["pred_set"].add(r["cid"])
            
    metrics = [compute_entity_metrics(d["gt_set"], d["pred_set"]) for d in per_entity.values()]
    f05 = np.mean([m["f0_5"] for m in metrics])
    prec = np.mean([m["precision"] for m in metrics])
    rec = np.mean([m["recall"] for m in metrics])
    tp = sum(m["tp"] for m in metrics)
    fp = sum(m["fp"] for m in metrics)
    fn = sum(m["fn"] for m in metrics)
    sing_tot = sum(1 for m in metrics if m["is_singleton"])
    sing_cor = sum(1 for m in metrics if m.get("singleton_correct", False))
    
    return {
        "macro_f0_5": round(float(f05), 6),
        "macro_precision": round(float(prec), 6),
        "macro_recall": round(float(rec), 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "singleton_acc": round(float(sing_cor / max(1, sing_tot)), 6)
    }

def main():
    print("Loading 1500 Clean Dev IDs...")
    with open("experiments/phase6/artifacts/clean_dev_1500/clean_dev_ids.txt") as f:
        dev_ids = {line.strip() for line in f if line.strip()}
        
    chk_ids = set(list(dev_ids)[:20])
    all_targets = dev_ids
    
    records = collect_dataset(all_targets, budget=100)
    val_gt = rdb.load_gt_for_ids(all_targets)
    
    dev_records = [r for r in records if r["sid"] in dev_ids]
    chk_records = [r for r in records if r["sid"] in chk_ids]
    
    print("\n--- BASELINE MATCHER RE-EVALUATION (Threshold 0.70) ---")
    base_probs = np.array([r["base_score"] for r in dev_records])
    base_eval = evaluate_threshold(dev_records, base_probs, 0.70, dev_ids, val_gt)
    print(f"F0.5: {base_eval['macro_f0_5']:.4f} | TP: {base_eval['tp']} | FP: {base_eval['fp']} | FN: {base_eval['fn']}")
    
    print("\n--- 3-FOLD CV XGBOOST CLASSIFIER ON DEV SET ---")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    unique_sids = np.array(sorted(list(dev_ids)))
    
    lr_probs = np.zeros(len(dev_records))
    y_true = np.array([r["label"] for r in dev_records])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_sids)):
        train_sids = set(unique_sids[train_idx])
        val_sids = set(unique_sids[val_idx])
        
        train_mask = [r["sid"] in train_sids for r in dev_records]
        val_mask = [r["sid"] in val_sids for r in dev_records]
        
        X_train = np.array([r["vec"] for i, r in enumerate(dev_records) if train_mask[i]])
        y_train = y_true[train_mask]
        
        X_val = np.array([r["vec"] for i, r in enumerate(dev_records) if val_mask[i]])
        
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', scale_pos_weight=pos_weight)
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_val)[:, 1]
        
        val_indices = np.where(val_mask)[0]
        lr_probs[val_indices] = probs
        
    print(f"Total Candidate Pairs Evaluated: {len(y_true)}")
    print(f"Positive Labels (GT Match): {sum(y_true)}")
    print(f"Negative Labels (False Cand): {len(y_true) - sum(y_true)}")
    
    print("\nEvaluating Multiple Thresholds...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_t = 0.5
    best_f05 = 0.0
    results = {}
    
    for t in thresholds:
        metrics = evaluate_threshold(dev_records, lr_probs, t, dev_ids, val_gt)
        results[t] = metrics
        print(f"  Thresh {t:.1f}: F0.5={metrics['macro_f0_5']:.4f}, P={metrics['macro_precision']:.4f}, R={metrics['macro_recall']:.4f}, TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
        if metrics['macro_f0_5'] > best_f05:
            best_f05 = metrics['macro_f0_5']
            best_t = t
            
    print(f"\nBEST LR MATCHING: Threshold {best_t:.1f} -> F0.5 = {best_f05:.4f}")
    
    print("\n--- FINAL MODEL TRAINING ---")
    X_full = np.array([r["vec"] for r in dev_records])
    y_full = np.array([r["label"] for r in dev_records])
    
    pos_weight = (len(y_full) - sum(y_full)) / sum(y_full)
    final_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', scale_pos_weight=pos_weight)
    final_clf.fit(X_full, y_full)
    
    print("Learned Feature Importances:")
    for feat, weight in zip(FEATURE_NAMES, final_clf.feature_importances_):
        print(f"  {feat:<15}: {weight:.4f}")
        
    if chk_records:
        print("\n--- 20-S1 CORRECTNESS CHECK EVALUATION ---")
        X_chk = np.array([r["vec"] for r in chk_records])
        chk_probs = final_clf.predict_proba(X_chk)[:, 1]
        chk_eval = evaluate_threshold(chk_records, chk_probs, best_t, chk_ids, val_gt)
        print(f"20-S1 LR Eval (Thresh {best_t:.1f}): F0.5={chk_eval['macro_f0_5']:.4f}, TP={chk_eval['tp']}, FP={chk_eval['fp']}, FN={chk_eval['fn']}")

    with open("experiments/phase6/xgb_matcher_results_clean.json", "w") as f:
        json.dump({
            "baseline": base_eval,
            "xgb_best_threshold": best_t,
            "xgb_best_metrics": results[best_t],
            "xgb_thresholds": results,
            "feature_importances": {f: round(float(w), 4) for f, w in zip(FEATURE_NAMES, final_clf.feature_importances_)}
        }, f, indent=2)

if __name__ == "__main__":
    main()
