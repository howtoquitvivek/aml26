#!/usr/bin/env python3
import sys
import os
import gc
import json
import numpy as np
import time
import collections
import resource
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath("."))
from src.blocking import BlockingIndex
import experiments.phase3.run_dev_benchmark as rdb
from src.normalization import get_canonical_name_key, get_name_tokens, extract_numeric_tokens
from experiments.phase4.ablate_lr_blocking import extract_lr_features, query_custom_s1, get_key_e

def get_peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def build_test_index_for_country(country: str, cand_meta: dict):
    """Builds the inverted indices for a specific country using cand_meta."""
    t0 = time.time()
    print(f"[{country}] Building in-memory BlockingIndex for {len(cand_meta)} records...")
    
    cand_ids = []
    idx_exact = collections.defaultdict(list)
    idx_canon = collections.defaultdict(list)
    idx_addr_old = collections.defaultdict(list)
    idx_key_b = collections.defaultdict(list)
    idx_key_e = collections.defaultdict(list)
    
    for i, (eid, meta) in enumerate(cand_meta.items()):
        encoded_eid = rdb.encode_id(eid)
        cand_ids.append(encoded_eid)
        
        name = meta["name"]
        addr = meta["address"]
        
        # Exact Name
        norm_name = name.lower().strip()
        if norm_name: idx_exact[norm_name].append(i)
        
        # Canon Name
        canon_name = get_canonical_name_key(name, strip_legal=True)
        if canon_name and canon_name != norm_name:
            idx_canon[canon_name].append(i)
            
        # Addr Old (numeric + first_tok)
        tokens = get_name_tokens(name, strip_legal=True, min_len=2)
        if addr and tokens:
            nums = extract_numeric_tokens(addr)
            first_tok = tokens[0][:4] if len(tokens[0]) >= 4 else tokens[0]
            for num in nums:
                if len(num) >= 2:
                    idx_addr_old[f"{num}_{first_tok}"].append(i)
                    
        # Key B: 4-char prefix + all nums
        if len(canon_name) >= 4:
            prefix = canon_name[:4]
            idx_key_b[prefix].append(i)
            nums = extract_numeric_tokens(addr)
            for n in nums:
                if len(n) >= 2:
                    idx_key_b[f"{prefix}_{n}"].append(i)
                    
        # Key E: Addr Token + char prefix
        if len(canon_name) >= 4 and addr:
            addr_toks = get_name_tokens(addr, strip_legal=False, min_len=3)
            prefix = canon_name[:4]
            for tok in addr_toks:
                idx_key_e[f"{tok}_{prefix}"].append(i)
                
    # Prune large blocks
    for idx in (idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e):
        keys_to_prune = [k for k, v in idx.items() if len(v) > 500]
        for k in keys_to_prune:
            idx[k] = idx[k][:500]
            
    print(f"[{country}] Index built in {time.time()-t0:.1f}s")
    return cand_ids, dict(idx_exact), dict(idx_canon), dict(idx_addr_old), dict(idx_key_b), dict(idx_key_e)

def query_candidates_inline(s1, budget, cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e):
    _, ranked_budget, _ = query_custom_s1(s1, budget, cand_ids, idx_exact, idx_canon, {}, idx_addr_old, idx_key_b, idx_key_e, "CFG1_NoTwoTok_KeyE")
    
    dedup = []
    for c_idx in ranked_budget:
        eid = rdb.decode_id(cand_ids[c_idx])
        dedup.append(eid)
    return dedup

def main():
    dry_run = "--dry-run" in sys.argv
    max_s1 = 10000 if dry_run else None
    print(f"Starting Test Inference (Dry Run: {dry_run}, Max S1: {max_s1})")
    
    t_start = time.time()
    
    print("Loading XGBoost Model...")
    clf = XGBClassifier()
    clf.load_model("experiments/phase6/artifacts/final_xgb_model.ubj")
    
    # Pre-parse S1 to know which countries exist and to chunk
    print("Scanning Test S1...")
    s1_countries = collections.defaultdict(list)
    s1_metadata = {}
    with open("dataset/test/test_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                sid, name, addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                country = country.lower()
                s1_countries[country].append(sid)
                s1_metadata[sid] = {"name": name, "address": addr, "country": country}
                if max_s1 and len(s1_metadata) >= max_s1:
                    break
    
    os.makedirs("output", exist_ok=True)
    out_matching = open("output/matching_results.tsv", "w", encoding="utf-8")
    out_matching.write("source1_entity_id\tmatched_entity_ids\n")
    
    out_candidates = open("output/candidate_pairs.tsv", "w", encoding="utf-8")
    out_candidates.write("source1_entity_id\tcandidate_entity_ids\n")
    
    total_s1 = 0
    total_tp = 0
    total_cands = 0
    
    for country in sorted(s1_countries.keys()):
        s1_ids = s1_countries[country]
        if not s1_ids: continue
        
        print(f"\n--- Processing Country: {country.upper()} ({len(s1_ids)} S1s) ---")
        
        # Load S2/S3 cand_meta for this country ONLY
        print(f"[{country}] Loading S2/S3 Metadata...")
        cand_meta = {}
        for fpath in ("dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"):
            with open(fpath, encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4 and parts[3].strip().lower() == country:
                        cand_meta[parts[0].strip()] = {"name": parts[1].strip(), "address": parts[2].strip()}
                        
        cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e = build_test_index_for_country(country, cand_meta)
        
        print(f"[{country}] Generating candidates and scoring...")
        for sid in s1_ids:
            s1 = s1_metadata[sid]
            cands = query_candidates_inline(s1, 100, cand_ids, idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e)
            
            # Write candidates
            out_candidates.write(f"{sid}\t{','.join(cands)}\n")
            total_cands += len(cands)
            
            if not cands:
                out_matching.write(f"{sid}\t\n")
                total_s1 += 1
                continue
                
            features = []
            valid_cands = []
            for cid in cands:
                cm = cand_meta.get(cid)
                if not cm: continue
                vec = extract_lr_features(s1["name"], s1["address"], cm["name"], cm["address"], cid, country)
                features.append(vec)
                valid_cands.append(cid)
                
            if not features:
                out_matching.write(f"{sid}\t\n")
                total_s1 += 1
                continue
                
            X = np.array(features, dtype=np.float32)
            probs = clf.predict_proba(X)[:, 1]
            
            matched = [valid_cands[i] for i in range(len(valid_cands)) if probs[i] >= 0.90]
            
            # Remove duplicates just in case
            seen = set()
            dedup_matched = []
            for m in matched:
                if m not in seen:
                    seen.add(m)
                    dedup_matched.append(m)
                    
            out_matching.write(f"{sid}\t{','.join(dedup_matched)}\n")
            total_tp += len(dedup_matched)
            total_s1 += 1
            
            if total_s1 % 5000 == 0:
                print(f"  Processed {total_s1} S1s...")
                
        # Cleanup memory before next country
        del cand_meta
        del idx_exact, idx_canon, idx_addr_old, idx_key_b, idx_key_e
        gc.collect()
        
    out_matching.close()
    out_candidates.close()
    
    print("\n--- INFERENCE COMPLETE ---")
    print(f"Total S1s processed: {total_s1}")
    print(f"Total Matches Predicted: {total_tp}")
    print(f"Total Candidates Extracted: {total_cands}")
    print(f"Runtime: {time.time() - t_start:.2f}s")
    print(f"Peak RAM: {get_peak_rss():.1f} MB")

if __name__ == "__main__":
    main()
