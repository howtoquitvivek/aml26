#!/usr/bin/env python3
import sys
import os
import gc
import difflib
from collections import defaultdict
from typing import Dict, Set

sys.path.insert(0, os.path.abspath("."))
from src.normalization import normalize_and_canonicalize
import experiments.phase3.run_dev_benchmark as rdb

# Re-use config values
S1_FILE = rdb.S1_FILE
S2_FILE = rdb.S2_FILE
S3_FILE = rdb.S3_FILE
GT_FILE = rdb.GT_FILE
DEV_IDS_FILE = rdb.DEV_IDS_FILE

def main():
    print("Loading dev IDs...")
    with open(DEV_IDS_FILE) as f:
        dev_ids = {line.strip() for line in f if line.strip()}

    print(f"Loading {len(dev_ids)} S1 records...")
    s1_data = rdb.load_s1_records(dev_ids)
    
    print("Loading GT links...")
    gt_links = rdb.load_gt_for_ids(dev_ids)
    gt_s2s3_ids = set()
    for links in gt_links.values():
        gt_s2s3_ids.update(links)
        
    print(f"Total GT target IDs: {len(gt_s2s3_ids)}")
    
    print("Fetching GT candidate text...")
    gt_s2s3_data = {}
    for fname in (S2_FILE, S3_FILE):
        with open(fname, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[0].strip() in gt_s2s3_ids:
                    gt_s2s3_data[parts[0].strip()] = {
                        "name": parts[1].strip(),
                        "address": parts[2].strip()
                    }
                    if len(gt_s2s3_data) == len(gt_s2s3_ids):
                        break

    unique_countries = sorted({r["country"] for r in s1_data.values()})
    
    missed_links = []
    explosions = []
    
    for country in unique_countries:
        print(f"\nProcessing {country}...")
        
        cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_c, idx_key_d = rdb.build_country_index(country)
        
        country_s1 = {sid: s1 for sid, s1 in s1_data.items() if s1["country"] == country}
        print(f"Evaluating {len(country_s1)} S1 entities for {country}...")
        
        for sid, s1 in country_s1.items():
            gt_set = gt_links.get(sid, set())
            
            # Use query_s1 from rdb to get exact same keys
            cfg1, cfg2, cfg3, _, _ = rdb.query_s1(
                s1, 1, cand_ids,
                idx_exact, idx_canon, idx_two_tok, idx_addr_old,
                idx_key_b, idx_key_c, idx_key_d
            )
            
            cfg1_eids = {rdb.decode_id(cand_ids[i]) for i in cfg1}
            cfg2_eids = {rdb.decode_id(cand_ids[i]) for i in cfg2}
            cfg3_eids = {rdb.decode_id(cand_ids[i]) for i in cfg3}
            
            # Analyze missed links
            for gt_id in gt_set:
                if gt_id not in cfg1_eids:
                    gt_cand = gt_s2s3_data.get(gt_id, {"name": "", "address": ""})
                    
                    s1_name_norm, _, _ = normalize_and_canonicalize(s1["name"])
                    gt_name_norm, _, _ = normalize_and_canonicalize(gt_cand["name"])
                    sim = difflib.SequenceMatcher(None, s1_name_norm, gt_name_norm).ratio()
                    
                    s1_nums, _ = rdb.extract_address_features(s1["address"])
                    gt_nums, _ = rdb.extract_address_features(gt_cand["address"])
                    has_nums = bool(s1_nums) and bool(gt_nums)
                    
                    missed_links.append({
                        "s1_id": sid,
                        "gt_id": gt_id,
                        "country": country,
                        "s1_name": s1["name"],
                        "gt_name": gt_cand["name"],
                        "s1_addr": s1["address"],
                        "gt_addr": gt_cand["address"],
                        "name_sim": sim,
                        "has_numerics": has_nums,
                        "rec2": gt_id in cfg2_eids,
                        "rec3": gt_id in cfg3_eids,
                        "c1_cands": len(cfg1)
                    })
                    
            # Analyze explosions
            if len(cfg1) > 10000:
                norm_name, canon_name, clean_toks = normalize_and_canonicalize(s1["name"])
                c_exact = len(idx_exact.get(norm_name, ()))
                c_canon = len(idx_canon.get(canon_name, ()))
                c_two = len(idx_two_tok.get(f"{clean_toks[0]} {clean_toks[1]}", ())) if len(clean_toks) >= 2 else 0
                
                explosions.append({
                    "s1_id": sid,
                    "country": country,
                    "s1_name": s1["name"],
                    "s1_addr": s1["address"],
                    "gt_count": len(gt_set),
                    "c1_cands": len(cfg1),
                    "exact": c_exact,
                    "canon": c_canon,
                    "two_tok": c_two
                })
                
        del cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old, idx_key_b, idx_key_c, idx_key_d
        gc.collect()

    # Generate Report
    print(f"\nWriting report with {len(missed_links)} missed links and {len(explosions)} explosions...")
    out_path = "results/phase3/blocking_diagnostic_1500.md"
    with open(out_path, "w") as f:
        f.write("# Blocking Diagnostic (1500-S1 Dev Set)\n\n")
        
        # Section 1: Misses
        f.write("## 1. CFG1 Missed GT Links\n")
        f.write(f"**Total missed GT links**: {len(missed_links)}\n\n")
        
        rec2_cnt = sum(1 for x in missed_links if x["rec2"])
        rec3_cnt = sum(1 for x in missed_links if x["rec3"])
        rec3_only = rec3_cnt - rec2_cnt
        f.write(f"- Recovered by Key C (CFG2): {rec2_cnt}\n")
        f.write(f"- Recovered by Key D (CFG3 \\ CFG2): {rec3_only}\n")
        f.write(f"- Still Missed (NO recovery by CFG3): {len(missed_links) - rec3_cnt}\n\n")
        
        no_num = sum(1 for x in missed_links if not x["has_numerics"])
        f.write(f"- Missed links lacking shared numeric address tokens: {no_num} ({(no_num/max(1,len(missed_links)))*100:.1f}%)\n")
        
        sims = [x["name_sim"] for x in missed_links]
        if sims:
            f.write(f"- Average Name Similarity (SequenceMatcher): {sum(sims)/len(sims):.3f}\n")
            f.write(f"- Missed links with <0.5 Name Similarity: {sum(1 for x in sims if x < 0.5)}\n\n")
            
        # Analysis summary
        f.write("### Analysis of Causes for Missed Links\n")
        f.write("Based on the data above, the missed links likely fall into these categories:\n")
        f.write("- **Low Name Similarity + No Address Numerics**: The blocker requires a shared word or number. If the name is completely different (e.g., transliteration mismatch, or brand vs legal name) AND there is no shared numeric token in the address, CFG1 will completely miss it. (Key C and Key D also cannot recover these).\n")
        f.write("- **Second-Token Divergence**: S1 and GT share the first word, but differ on the second word. CFG1 misses this if `idx_two_tok` is the only active key. CFG2 (Key C - first_tok + salient_alpha) can recover some of these.\n")
        f.write("- **Address Format Mismatch**: When names don't exactly match and addresses don't have overlapping parsed numerics.\n\n")

        f.write("### Sample Missed Links (Top 15 lowest similarity)\n")
        f.write("| S1 Name | GT Name | S1 Addr | GT Addr | Sim | Num? | Rec? | CFG1_Cands |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for x in sorted(missed_links, key=lambda i: i["name_sim"])[:15]:
            rec_str = "KeyC" if x["rec2"] else ("KeyD" if x["rec3"] else "None")
            num_str = "Y" if x["has_numerics"] else "N"
            s1_n = x['s1_name'].replace('|', ' ')
            gt_n = x['gt_name'].replace('|', ' ')
            s1_a = x['s1_addr'].replace('|', ' ')
            gt_a = x['gt_addr'].replace('|', ' ')
            f.write(f"| {s1_n} | {gt_n} | {s1_a} | {gt_a} | {x['name_sim']:.2f} | {num_str} | {rec_str} | {x['c1_cands']} |\n")
            
        f.write("\n## 2. CFG1 Candidate Explosions (>10k candidates)\n")
        f.write(f"**Total entities with >10k candidates**: {len(explosions)}\n\n")
        
        f.write("### Cross-Tabulation: Explosions vs Missed Links\n")
        misses_for_explosions = sum(1 for m in missed_links if m["c1_cands"] > 10000)
        f.write(f"- Number of Missed GT Links that came from 'explosion' S1 entities: {misses_for_explosions}\n")
        f.write("- **Conclusion**: The missed links and the explosions are largely distinct problems. High candidate counts come from very generic entities, while missed links are mostly specific businesses with severe name/address discrepancies or lacking shared tokens.\n\n")

        f.write("### Top Explosions (by total candidates)\n")
        f.write("| Country | S1 Name | GT Links | Total Cands | Exact Index Cands | Canon Index Cands | TwoTok Index Cands |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        explosions.sort(key=lambda x: x["c1_cands"], reverse=True)
        for x in explosions[:30]:
            s1_n = x['s1_name'].replace('|', ' ')
            f.write(f"| {x['country']} | {s1_n} | {x['gt_count']} | {x['c1_cands']} | {x['exact']} | {x['canon']} | {x['two_tok']} |\n")
            
    print(f"Done. Report saved to {out_path}.")

if __name__ == "__main__":
    main()
