import argparse
import collections
import gc
import json
import os
import time
import xgboost as xgb
import numpy as np

from src.index_builder import build_all_indexes
from src.index_builder_mr import build_all_indexes as build_all_indexes_mr
from src.candidate_generation import CandidateGenerator
from src.candidate_generation_mr import MRCandidateGenerator
from src.feature_extraction import extract_features

GLOBAL_CLF = None
GLOBAL_CAND_META = {}
GLOBAL_CG = None
GLOBAL_CG_MR = None

def process_chunk(chunk, threshold, max_k):
    results = {}
    candidates = {}
    for sid, s1 in chunk:
        cands = GLOBAL_CG.get_candidates(s1["name"], s1["address"], s1["country"])
        mr_cands_set = GLOBAL_CG_MR.get_candidates(s1["name"], s1["address"], s1["country"])
        cands.update(mr_cands_set)
        
        valid_cands = []
        for c in cands:
            if c in GLOBAL_CAND_META:
                valid_cands.append(c)
                
        # Random sample if above max_k, prioritizing MR candidates
        if len(valid_cands) > max_k:
            import random
            mr_cands = [c for c in valid_cands if c in mr_cands_set]
            other_cands = [c for c in valid_cands if c not in mr_cands_set]
            if len(mr_cands) >= max_k:
                valid_cands = random.sample(mr_cands, max_k)
            else:
                valid_cands = mr_cands + random.sample(other_cands, max_k - len(mr_cands))
                
        features = []
        for c in valid_cands:
            cand = GLOBAL_CAND_META[c]
            feat = extract_features(
                s1["name"], s1["address"], s1["country"],
                cand["name"], cand["address"], cand["country"],
                cand["is_s2"]
            )
            features.append(feat)
            
        c_ids = [c.split("#")[1] for c in valid_cands]
        candidates[sid] = c_ids
        
        if not features:
            results[sid] = set()
            continue
            
        X = np.array(features, dtype=np.float32)
        probs = GLOBAL_CLF.predict_proba(X)[:, 1]
        
        pred_cands = set()
        for c, prob in zip(valid_cands, probs):
            if prob >= threshold:
                pred_cands.add(c.split("#")[1])
                
        results[sid] = pred_cands
        
    return results, candidates

def load_country_metadata(s2_path, s3_path, country):
    print(f"Loading candidate metadata for {country}...")
    cand_meta = {}
    for fpath in (s2_path, s3_path):
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

def run_inference(s1_path, s2_path, s3_path, index_dir, out_csv, cand_csv, model_path, threshold=0.98, max_k=300):
    global GLOBAL_CLF, GLOBAL_CAND_META, GLOBAL_CG, GLOBAL_CG_MR
    
    t0 = time.time()
    
    # Parse S1 records by country first to know what indexes we need
    print(f"Parsing S1 test records from {s1_path}...")
    s1_by_country = collections.defaultdict(list)
    with open(s1_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                sid, name, addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                s1_by_country[country].append((sid, {
                    "name": name,
                    "address": addr,
                    "country": country
                }))
                
    # 1. Build indexes if they don't exist or are incomplete
    missing_indexes = False
    if not os.path.exists(index_dir):
        missing_indexes = True
    else:
        if not os.path.exists(os.path.join(index_dir, "index_manifest.json")):
            missing_indexes = True
        if not os.path.exists(os.path.join(index_dir, "index_manifest_mr.json")):
            missing_indexes = True
            
        for country in s1_by_country.keys():
            if not os.path.exists(os.path.join(index_dir, f"cand_ids_{country}.json")): missing_indexes = True
            if not os.path.exists(os.path.join(index_dir, f"idx_{country}.pkl")): missing_indexes = True
            if not os.path.exists(os.path.join(index_dir, f"mr_cand_ids_{country}.json")): missing_indexes = True
            if not os.path.exists(os.path.join(index_dir, f"mr_idx_{country}.pkl")): missing_indexes = True

    if missing_indexes:
        print(f"Indexes in {index_dir} missing or incomplete. Building indexes...")
        os.makedirs(index_dir, exist_ok=True)
        build_all_indexes(s2_path, s3_path, index_dir)
        build_all_indexes_mr(s2_path, s3_path, index_dir)
    else:
        print(f"Using complete existing indexes in {index_dir}...")
        
    print(f"Loading XGBoost model from {model_path}...")
    GLOBAL_CLF = xgb.XGBClassifier()
    GLOBAL_CLF.load_model(model_path)
    GLOBAL_CLF.set_params(n_jobs=1)
    
    GLOBAL_CG = CandidateGenerator(index_dir)
    GLOBAL_CG_MR = MRCandidateGenerator(index_dir)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    
    output_f = open(out_csv, "w", encoding="utf-8")
    output_f.write("source1_entity_id\tmatched_entity_ids\n")
    
    if cand_csv:
        os.makedirs(os.path.dirname(os.path.abspath(cand_csv)), exist_ok=True)
        cand_f = open(cand_csv, "w", encoding="utf-8")
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    else:
        cand_f = None
    
    total_processed = 0
    total_found = 0
    
    for country, country_s1 in s1_by_country.items():
        print(f"\n=== Processing {country} ({len(country_s1)} queries) ===")
        GLOBAL_CAND_META = load_country_metadata(s2_path, s3_path, country)
        GLOBAL_CG.load_country(country)
        GLOBAL_CG_MR.load_country(country)
        
        chunk_size = 500
        chunks = [country_s1[i:i + chunk_size] for i in range(0, len(country_s1), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            res_dict, cand_dict = process_chunk(chunk, threshold, max_k)
            for sid, matched in res_dict.items():
                m_str = ",".join(matched)
                output_f.write(f"{sid}\t{m_str}\n")
                if cand_f:
                    c_str = ",".join(cand_dict[sid])
                    cand_f.write(f"{sid}\t{c_str}\n")
                total_processed += 1
                total_found += len(matched)
                
            if (i+1) % 10 == 0 or (i+1) == len(chunks):
                print(f"  Completed {min((i+1)*chunk_size, len(country_s1))}/{len(country_s1)} queries for {country}...")
                
        print(f"Finished {country}.")
        
        GLOBAL_CAND_META.clear()
        GLOBAL_CG.idx_base.clear()
        GLOBAL_CG_MR.idx_base.clear()
        gc.collect()
        
    output_f.close()
    if cand_f:
        cand_f.close()
    t1 = time.time()
    print(f"\nInference complete in {t1 - t0:.2f} seconds.")
    print(f"Processed {total_processed} queries. Found {total_found} links.")
    print(f"Output written to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=str, default="dataset/test/test_source1.tsv")
    parser.add_argument("--s2", type=str, default="dataset/test/test_source2.tsv")
    parser.add_argument("--s3", type=str, default="dataset/test/test_source3.tsv")
    parser.add_argument("--index_dir", type=str, default="experiments/final_hardening/test_index")
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--cand", type=str, default=None)
    parser.add_argument("--model", type=str, default="experiments/final_hardening/artifacts/xgb_15000.ubj")
    parser.add_argument("--threshold", type=float, default=0.98)
    args = parser.parse_args()
    
    run_inference(args.s1, args.s2, args.s3, args.index_dir, args.out, args.cand, args.model, args.threshold)
