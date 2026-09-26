#!/usr/bin/env python3
import sys
import os
import json
import time
import collections
import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath("."))
import experiments.phase3.run_dev_benchmark as rdb
from experiments.phase4.ablate_lr_blocking import extract_lr_features, get_key_e
from src.evaluation import evaluate_predictions
import src.normalization as norm

GENERIC_WORDS = norm.LEGAL_SUFFIX_SET | rdb.GENERIC_ADDR_TERMS

def get_consonants(word):
    return "".join(c for c in word.lower() if c.isalpha() and c not in "aeiouyh")

def get_p3_2_keys(name, addr):
    name_toks = [t for t in rdb.normalize_address(name).split() if t.isalpha() and t not in GENERIC_WORDS]
    addr_toks = [t for t in rdb.normalize_address(addr).split() if t.isalpha() and t not in GENERIC_WORDS]
    name_toks.sort(key=len, reverse=True)
    addr_toks.sort(key=len, reverse=True)
    p3 = set()
    for nt in name_toks[:2]:
        if len(nt) >= 5:
            nc = get_consonants(nt)
            if len(nc) >= 3:
                for at in addr_toks[:2]:
                    if len(at) >= 5:
                        ac = get_consonants(at)
                        if len(ac) >= 3: p3.add(f"P3#{nc}#{ac}")
    return p3

def peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    print("Loading FULL 220,683 S1 validation set...")
    with open("experiments/val_s1_ids.txt") as f:
        all_s1_ids = set([line.strip() for line in f if line.strip()])
        
    val_gt = rdb.load_gt_for_ids(all_s1_ids)
    s1_data = rdb.load_s1_records(all_s1_ids)
    countries = sorted({r["country"] for r in s1_data.values()})
    
    print("Loading XGBoost Model...")
    clf = XGBClassifier()
    clf.load_model("experiments/phase6/artifacts/final_xgb_model.ubj")
    
    print("Loading Candidate Meta for feature extraction...")
    cand_meta = {}
    for fpath in ("dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    cand_meta[parts[0].strip()] = {"name": parts[1].strip(), "address": parts[2].strip()}

    MAX_K = 300
    predictions = {}
    
    total_cands_before_k = 0
    total_cands_after_k = 0
    
    newly_recovered_gt = set()
    newly_recovered_survived_k = set()
    newly_recovered_accepted = set()
    newly_recovered_rejected = set()
    
    t_start = time.time()
    
    for country in countries:
        print(f"\nProcessing {country}...")
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        if not country_s1: continue
        
        idx_base = collections.defaultdict(list)
        idx_p3_raw = collections.defaultdict(list)
        
        cand_ids_arr = []
        
        for fpath in ("dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"):
            with open(fpath, encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4 and parts[3].strip() == country:
                        cid = parts[0].strip()
                        name = parts[1].strip()
                        addr = parts[2].strip()
                        
                        rec_idx = len(cand_ids_arr)
                        cand_ids_arr.append(rdb.encode_id(cid))
                        
                        norm_name, canon_name, clean_toks = rdb.normalize_and_canonicalize(name)
                        idx_base[f"E#{norm_name}"].append(rec_idx)
                        idx_base[f"C#{canon_name}"].append(rec_idx)
                        
                        nums, alphas = rdb.extract_address_features(addr)
                        if clean_toks:
                            tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
                            for n in nums:
                                if len(n) >= 2: idx_base[f"A#{n}_{tok_p}"].append(rec_idx)
                        for k in rdb.get_key_b(canon_name, nums): idx_base[f"B#{k}"].append(rec_idx)
                        for k in get_key_e(addr): idx_base[f"K#{k}"].append(rec_idx)
                        
                        p3_keys = get_p3_2_keys(name, addr)
                        for k in p3_keys: idx_p3_raw[k].append(rec_idx)
                        
        print(f"Indexed {len(cand_ids_arr)} records for {country}")
        idx_p3 = {k: v for k, v in idx_p3_raw.items() if len(v) <= 100}
        del idx_p3_raw
        
        print("Querying and Scoring...")
        for sid, s1 in country_s1.items():
            gt_set = val_gt.get(sid, set())
            gt_enc_set = {rdb.encode_id(gt) for gt in gt_set}
            
            c_exact, c_canon, c_addr, c_key_b, c_key_e = set(), set(), set(), set(), set()
            
            norm_name, canon_name, clean_toks = rdb.normalize_and_canonicalize(s1["name"])
            c_exact.update(idx_base.get(f"E#{norm_name}", []))
            c_canon.update(idx_base.get(f"C#{canon_name}", []))
            if clean_toks:
                old_nums = rdb.extract_numeric_tokens(s1["address"])
                tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
                for n in old_nums:
                    if len(n) >= 2: c_addr.update(idx_base.get(f"A#{n}_{tok_p}", []))
            nums, alphas = rdb.extract_address_features(s1["address"])
            for k in rdb.get_key_b(canon_name, nums): c_key_b.update(idx_base.get(f"B#{k}", []))
            for k in get_key_e(s1["address"]): c_key_e.update(idx_base.get(f"K#{k}", []))
            
            p3_keys = get_p3_2_keys(s1["name"], s1["address"])
            c_p3 = set()
            for k in p3_keys: c_p3.update(idx_p3.get(k, []))
            
            c_base = c_exact | c_canon | c_addr | c_key_b | c_key_e
            c_all = c_base | c_p3
            total_cands_before_k += len(c_all)
            
            base_recovered = gt_enc_set & {cand_ids_arr[i] for i in c_base}
            p3_recovered = gt_enc_set & {cand_ids_arr[i] for i in c_all}
            new_gt = p3_recovered - base_recovered
            for gt_enc in new_gt:
                newly_recovered_gt.add((sid, gt_enc))
                
            weights = {}
            for i in c_all:
                w = 0
                if i in c_exact: w += 40
                if i in c_canon: w += 30
                if i in c_addr:  w += 20
                if i in c_key_b: w += 10
                if i in c_key_e: w += 2
                if w == 0 and i in c_p3: w += 1 
                weights[i] = w
                
            ranked_all = sorted(list(c_all), key=lambda i: (-weights[i], rdb.decode_id(cand_ids_arr[i])))
            ranked_budget = ranked_all[:MAX_K]
            total_cands_after_k += len(ranked_budget)
            
            budget_encs = {cand_ids_arr[i] for i in ranked_budget}
            for gt_enc in new_gt:
                if gt_enc in budget_encs:
                    newly_recovered_survived_k.add((sid, gt_enc))
                    
            decoded_cands = [rdb.decode_id(cand_ids_arr[i]) for i in ranked_budget]
            features = []
            valid_cands = []
            for cid in decoded_cands:
                cm = cand_meta.get(cid)
                if not cm: continue
                vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                features.append(vec)
                valid_cands.append(cid)
                
            preds = set()
            if features:
                X = np.array(features, dtype=np.float32)
                probs = clf.predict_proba(X)[:, 1]
                for i, prob in enumerate(probs):
                    if prob >= 0.90:
                        preds.add(valid_cands[i])
                        
            predictions[sid] = preds
            
            for gt_enc in new_gt:
                if gt_enc in budget_encs:
                    cid = rdb.decode_id(gt_enc)
                    if cid in preds:
                        newly_recovered_accepted.add((sid, gt_enc))
                    else:
                        newly_recovered_rejected.add((sid, gt_enc))
                        
    runtime = time.time() - t_start
    
    print("\n--- Evaluating Predictions ---")
    eval_summary = evaluate_predictions(val_gt, predictions)
    
    stats = {
        "macro_f0_5": eval_summary["macro_f0_5"],
        "macro_precision": eval_summary["macro_precision"],
        "macro_recall": eval_summary["macro_recall"],
        "tp": eval_summary["aggregate_tp"],
        "fp": eval_summary["aggregate_fp"],
        "fn": eval_summary["aggregate_fn"],
        "singleton_accuracy": eval_summary["singleton_accuracy"],
        "total_cands_before_k": total_cands_before_k,
        "total_cands_after_k": total_cands_after_k,
        "avg_cands_after_k": total_cands_after_k / len(s1_data),
        "predicted_matches": sum(len(preds) for preds in predictions.values()),
        "predicted_empty": sum(1 for preds in predictions.values() if not preds),
        "runtime": runtime,
        "peak_rss": peak_rss_mb(),
        "newly_recovered_total": len(newly_recovered_gt),
        "newly_recovered_survived_k": len(newly_recovered_survived_k),
        "newly_recovered_accepted": len(newly_recovered_accepted),
        "newly_recovered_rejected": len(newly_recovered_rejected)
    }
    
    base_f05 = 0.8216
    base_prec = 0.8800
    base_rec = 0.7276
    base_tp = 554499
    base_fp = 46164
    base_fn = 209093
    
    print("\n--- COMPARISON TO BASELINE ---")
    print(f"Macro F0.5: {stats['macro_f0_5']:.4f} (Change: {stats['macro_f0_5'] - base_f05:+.4f})")
    print(f"Precision: {stats['macro_precision']:.4f} (Change: {stats['macro_precision'] - base_prec:+.4f})")
    print(f"Recall: {stats['macro_recall']:.4f} (Change: {stats['macro_recall'] - base_rec:+.4f})")
    print(f"TP: {stats['tp']} (Change: {stats['tp'] - base_tp:+d})")
    print(f"FP: {stats['fp']} (Change: {stats['fp'] - base_fp:+d})")
    print(f"FN: {stats['fn']} (Change: {stats['fn'] - base_fn:+d})")
    
    print("\n--- NEWLY RECOVERABLE LINKS ANALYSIS ---")
    print(f"Total newly recovered by P3.2 blocker: {stats['newly_recovered_total']}")
    print(f"  -> Survived K=300 budget: {stats['newly_recovered_survived_k']}")
    print(f"  -> Accepted by XGBoost: {stats['newly_recovered_accepted']}")
    print(f"  -> Rejected by XGBoost: {stats['newly_recovered_rejected']}")
    
    with open("experiments/phase7/p3_2_downstream_results.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
