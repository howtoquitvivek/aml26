"""
Gap analysis script.
Identifies which GT links are completely missed by the generated candidates.
"""
import json
import time
import experiments.phase3.run_dev_benchmark as rdb
from src.candidate_generation import CandidateGenerator

def main():
    print("Loading FULL S1 validation set...")
    with open("experiments/val_s1_ids.txt") as f:
        all_s1_ids = set([line.strip() for line in f if line.strip()])
        
    val_gt = rdb.load_gt_for_ids(all_s1_ids)
    s1_data = rdb.load_s1_records(all_s1_ids)
    
    print("Loading candidate generator...")
    cg = CandidateGenerator("experiments/final_hardening/index")
    
    missed_gts = []
    
    s1_items = list(s1_data.items())
    s1_items.sort(key=lambda x: x[1]["country"])
    
    current_ctry = None
    reverse_map = {}
    
    for i, (sid, s1) in enumerate(s1_items):
        if i > 0 and i % 1000 == 0:
            print(f"Processed {i} S1 records...")
            
        gt_set = val_gt.get(sid, set())
        if not gt_set: continue
        
        c_ctry = s1["country"]
        if c_ctry != current_ctry:
            cg.load_country(c_ctry)
            current_ctry = c_ctry
            print(f"Building reverse map for {c_ctry}...")
            t0 = time.time()
            reverse_map = {cid.split("#")[1]: idx for idx, cid in enumerate(cg.cand_ids)}
            print(f"Built reverse map in {time.time()-t0:.2f}s")
            
        cands_indices = cg.get_candidate_indices(s1["name"], s1["address"], s1["country"])
        
        target_indices = set()
        for gt in gt_set:
            if gt in reverse_map:
                target_indices.add(reverse_map[gt])
                
        missed_indices = target_indices - cands_indices
        
        if missed_indices:
            for gt in gt_set:
                if gt in reverse_map and reverse_map[gt] in missed_indices:
                    missed_gts.append({
                        "s1_id": sid,
                        "gt_id": gt,
                        "s1_name": s1["name"],
                        "s1_address": s1["address"],
                        "country": s1["country"]
                    })
                elif gt not in reverse_map:
                    # Should be impossible if index contains all dataset
                    pass
                
    print(f"Total completely missed GT links: {len(missed_gts)}")
    with open("experiments/final_hardening/missed_gt.json", "w") as f:
        json.dump(missed_gts, f, indent=2)

if __name__ == "__main__":
    main()
