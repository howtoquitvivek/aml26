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
from src.baseline_matcher import compute_pair_features
from src.evaluation import compute_entity_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import resource

FEATURE_NAMES = [
    "name_exact", "canon_exact", "name_tok_jac", "name_char_jac", "name_len_diff",
    "addr_exact", "addr_missing", "addr_tok_jac", "addr_char_jac", "addr_num_jac",
    "addr_len_diff", "postal_overlap", "salient_addr_jac", "is_s2", "country_match"
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
        base["name_exact"], base["name_canonical_exact"], base["name_token_jaccard"],
        base["name_char_jaccard"], base["name_len_diff_ratio"], base["addr_exact"],
        base["addr_missing"], base["addr_token_jaccard"], base["addr_char_jaccard"],
        base["addr_numeric_jaccard"], base["addr_len_diff_ratio"], postal_overlap,
        salient_overlap, is_s2, base["country_match"]
    ]

def get_key_e(addr: str) -> Set[str]:
    nums, alphas = rdb.extract_address_features(addr)
    keys = set()
    if len(nums) >= 2:
        keys.add(f"addr#{nums[0]}#{nums[1]}")
    if len(nums) >= 1 and len(alphas) >= 1:
        keys.add(f"addr@{nums[0]}@{alphas[0]}")
    return keys

def build_custom_index(country: str, config: str):
    def mk(): return collections.defaultdict(list)
    idx_exact = mk()
    idx_canon = mk()
    idx_two_tok = mk()
    idx_addr_old = mk()
    idx_key_b = mk()
    idx_key_e = mk()
    
    cand_ids_arr = []
    
    for fpath in (rdb.S2_FILE, rdb.S3_FILE):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or parts[3].strip() != country:
                    continue
                
                eid = parts[0].strip()
                name = parts[1].strip()
                addr = parts[2].strip()
                
                rec_idx = len(cand_ids_arr)
                cand_ids_arr.append(rdb.encode_id(eid))
                
                norm_name, canon_name, clean_toks = rdb.normalize_and_canonicalize(name)
                
                idx_exact[norm_name].append(rec_idx)
                idx_canon[canon_name].append(rec_idx)
                
                if config == "CFG1":
                    if len(clean_toks) >= 2:
                        idx_two_tok[f"{clean_toks[0]} {clean_toks[1]}"].append(rec_idx)
                
                nums, alphas = rdb.extract_address_features(addr)
                if clean_toks:
                    tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
                    for n in nums:
                        if len(n) >= 2:
                            idx_addr_old[f"{n}_{tok_p}"].append(rec_idx)
                
                for k in rdb.get_key_b(canon_name, nums):
                    idx_key_b[k].append(rec_idx)
                    
                if config == "CFG1_NoTwoTok_KeyE":
                    for k in get_key_e(addr):
                        idx_key_e[k].append(rec_idx)
                        
    return (cand_ids_arr, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_e)

def query_custom_s1(s1, budget, cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_e, config):
    norm_name, canon_name, clean_toks = rdb.normalize_and_canonicalize(s1["name"])
    
    c_exact = set(idx_exact.get(norm_name, ()))
    c_canon = set(idx_canon.get(canon_name, ()))
    
    c_two = set()
    if config == "CFG1" and len(clean_toks) >= 2:
        c_two = set(idx_two_tok.get(f"{clean_toks[0]} {clean_toks[1]}", ()))
        
    c_addr = set()
    if clean_toks:
        old_nums = rdb.extract_numeric_tokens(s1["address"])
        tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
        for n in old_nums:
            if len(n) >= 2:
                c_addr.update(idx_addr_old.get(f"{n}_{tok_p}", ()))
                
    nums, alphas = rdb.extract_address_features(s1["address"])
    c_key_b = set()
    for k in rdb.get_key_b(canon_name, nums):
        c_key_b.update(idx_key_b.get(k, ()))
        
    c_key_e = set()
    if config == "CFG1_NoTwoTok_KeyE":
        for k in get_key_e(s1["address"]):
            c_key_e.update(idx_key_e.get(k, ()))
            
    # Combine
    all_cands = c_exact | c_canon | c_two | c_addr | c_key_b | c_key_e
    
    # Compute Weights
    W_EXACT = 40
    W_CANON = 30
    W_ADDR = 20
    W_KEY_B = 10
    W_KEY_E = 2 
    
    weights = {}
    for i in all_cands:
        w = 0
        if i in c_exact: w += W_EXACT
        if i in c_canon: w += W_CANON
        if i in c_two:   w += W_CANON # Matches old run_dev_benchmark logic where two_tok merged into c_canon
        if i in c_addr:  w += W_ADDR
        if i in c_key_b: w += W_KEY_B
        if i in c_key_e: w += W_KEY_E
        weights[i] = w
        
    ranked_all = sorted(list(all_cands), key=lambda i: (-weights[i], rdb.decode_id(cand_ids[i])))
    ranked_budget = ranked_all[:budget]
    
    return all_cands, ranked_budget, weights

def collect_dataset(target_ids: Set[str], budget: int, config: str):
    t0 = time.time()
    s1_data = rdb.load_s1_records(target_ids)
    val_gt = rdb.load_gt_for_ids(target_ids)
    
    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    records = []
    blocker_stats = []
    
    for country in unique_countries:
        print(f"  Building index for {country}...")
        country_s1 = {sid: r for sid, r in s1_data.items() if r["country"] == country}
        cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_e = build_custom_index(country, config)
        
        pass1 = {}
        needed_encoded = set()
        
        for sid, s1 in country_s1.items():
            all_cands, ranked_budget, _ = query_custom_s1(
                s1, budget, cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_e, config
            )
            
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
        del cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_e
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
        
    print(f"  Collection finished in {time.time()-t0:.1f}s")
    return records, blocker_stats


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
    print("Loading 1500 Dev IDs...")
    with open(rdb.DEV_IDS_FILE) as f:
        dev_ids = {line.strip() for line in f if line.strip()}
    all_targets = dev_ids
    val_gt = rdb.load_gt_for_ids(all_targets)
    
    configs = ["CFG1", "CFG1_NoTwoTok", "CFG1_NoTwoTok_KeyE"]
    results = {}
    
    for config in configs:
        t_start = time.time()
        print(f"\n=============================================")
        print(f"EVALUATING {config}")
        print(f"=============================================")
        records, blocker_stats = collect_dataset(all_targets, budget=25, config=config)
        
        # Calculate Blocker metrics
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
        
        print(f"Blocker Recall: {blocking_recall:.4f} ({total_recovered}/{total_gt})")
        print(f"Total Cands Before Budget: {b_metrics['total_candidates']} (Avg {b_metrics['avg_candidates']})")
        print(f"P95: {b_metrics['p95_candidates']} | P99: {b_metrics['p99_candidates']} | Max: {b_metrics['max_candidates']}")
        
        # LR 3-Fold CV
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        unique_sids = np.array(sorted(list(dev_ids)))
        
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
                
                val_indices = np.where(val_mask)[0]
                lr_probs[val_indices] = probs
                
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        best_t = 0.5
        best_f05 = 0.0
        lr_thresh_results = {}
        for t in thresholds:
            m = evaluate_threshold(records, lr_probs, t, dev_ids, val_gt)
            lr_thresh_results[t] = m
            if m["macro_f0_5"] > best_f05:
                best_f05 = m["macro_f0_5"]
                best_t = t
                
        print(f"LR Best: Thresh {best_t} -> F0.5 = {best_f05:.4f}")
        
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        
        results[config] = {
            "blocker_metrics": b_metrics,
            "lr_best_threshold": best_t,
            "lr_best_metrics": lr_thresh_results[best_t],
            "runtime_sec": round(time.time() - t_start, 1),
            "peak_rss_mb": round(peak_rss, 1)
        }
        
    with open("experiments/phase4/ablation_lr_blocking_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
