#!/usr/bin/env python3
"""
Inspect and sample qualitative noise patterns across matching records:
- exact matches
- abbreviation differences
- punctuation differences
- legal suffix differences
- typos
- word-order differences
- transliteration differences
- address abbreviations
- missing address components
- landmark-style addresses
- difficult-looking matches
- difficult-looking non-matches
"""

import os
import re
import json
import collections
import pandas as pd

DATA_DIR = "dataset"
EXPERIMENTS_DIR = "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# 1. Load a slice of ground truth records
print("Loading sample ground truth...")
gt_samples = []
target_s1_ids = set()
target_s2_ids = set()
target_s3_ids = set()

# Sample non-empty ground truth rows
with open(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    header = next(f)
    for i, line in enumerate(f):
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[1]:
            s1_id = parts[0]
            matched_ids = [m.strip() for m in parts[1].split(",") if m.strip()]
            gt_samples.append((s1_id, matched_ids))
            target_s1_ids.add(s1_id)
            for mid in matched_ids:
                if mid.startswith("S2-"):
                    target_s2_ids.add(mid)
                elif mid.startswith("S3-"):
                    target_s3_ids.add(mid)
        if len(gt_samples) >= 30000:
            break

print(f"Collected {len(gt_samples)} GT pairs. Targets: S1={len(target_s1_ids)}, S2={len(target_s2_ids)}, S3={len(target_s3_ids)}")

def load_records(file_path, target_ids):
    records = {}
    with open(file_path, "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, bname, baddr, ctry = parts[0], parts[1], parts[2], parts[3]
                if eid in target_ids:
                    records[eid] = {
                        "name": bname,
                        "address": baddr,
                        "country": ctry
                    }
                    if len(records) == len(target_ids):
                        break
    return records

print("Fetching S1 records...")
s1_recs = load_records(os.path.join(DATA_DIR, "train", "train_source1.tsv"), target_s1_ids)
print("Fetching S2 records...")
s2_recs = load_records(os.path.join(DATA_DIR, "train", "train_source2.tsv"), target_s2_ids)
print("Fetching S3 records...")
s3_recs = load_records(os.path.join(DATA_DIR, "train", "train_source3.tsv"), target_s3_ids)

examples = {
    "exact_matches": [],
    "legal_suffix_differences": [],
    "abbreviation_differences": [],
    "punctuation_differences": [],
    "typos": [],
    "word_order_differences": [],
    "transliteration_differences": [],
    "address_abbreviations": [],
    "missing_address_components": [],
    "landmark_style_addresses": [],
    "difficult_matches": [],
    "difficult_non_matches": []
}

LEGAL_SUFFIXES = r"\b(inc|corp|corporation|incorporated|llc|ltd|limited|pvt|private|co|company|llp|gmbh|sarl|sa)\b"
ADDR_ABBR = [("rd", "road"), ("st", "street"), ("ave", "avenue"), ("blvd", "boulevard"), ("dr", "drive"), ("hwy", "highway")]
LANDMARKS = ["near", "opp", "opposite", "behind", "beside", "adj", "adjacent", "metro", "station", "temple", "atm", "hospital", "floor", "phase", "sector"]

def clean(s):
    return re.sub(r"[^\w\s]", " ", s.lower()).strip()

# Analyze true pairs
for s1_id, matched_ids in gt_samples:
    s1 = s1_recs.get(s1_id)
    if not s1:
        continue
        
    s1_name_clean = clean(s1["name"])
    s1_tokens = set(s1_name_clean.split())
    s1_addr_clean = clean(s1["address"])
    
    for mid in matched_ids:
        target = s2_recs.get(mid) if mid.startswith("S2-") else s3_recs.get(mid)
        if not target:
            continue
            
        t_name_clean = clean(target["name"])
        t_tokens = set(t_name_clean.split())
        t_addr_clean = clean(target["address"])
        
        pair_info = {
            "s1_id": s1_id,
            "match_id": mid,
            "country": s1["country"],
            "s1_name": s1["name"],
            "target_name": target["name"],
            "s1_address": s1["address"],
            "target_address": target["address"]
        }
        
        # Exact match
        if s1["name"].strip().lower() == target["name"].strip().lower() and s1["address"].strip().lower() == target["address"].strip().lower():
            if len(examples["exact_matches"]) < 5:
                examples["exact_matches"].append(pair_info)
                
        # Legal suffix differences
        s1_no_legal = re.sub(LEGAL_SUFFIXES, "", s1_name_clean)
        t_no_legal = re.sub(LEGAL_SUFFIXES, "", t_name_clean)
        if " ".join(s1_no_legal.split()) == " ".join(t_no_legal.split()) and s1_name_clean != t_name_clean:
            if len(examples["legal_suffix_differences"]) < 5:
                examples["legal_suffix_differences"].append(pair_info)
                
        # Word order differences
        if s1_tokens == t_tokens and s1_name_clean != t_name_clean and len(s1_tokens) > 1:
            if len(examples["word_order_differences"]) < 5:
                examples["word_order_differences"].append(pair_info)
                
        # Transliteration (Devanagari vs Latin or Hindi text)
        has_devanagari_s1 = any("\u0900" <= c <= "\u097F" for c in s1["name"] + s1["address"])
        has_devanagari_t = any("\u0900" <= c <= "\u097F" for c in target["name"] + target["address"])
        if has_devanagari_s1 != has_devanagari_t or (has_devanagari_s1 and has_devanagari_t):
            if len(examples["transliteration_differences"]) < 5:
                examples["transliteration_differences"].append(pair_info)
                
        # Landmark style addresses
        if any(lm in t_addr_clean.split() for lm in LANDMARKS) or any(lm in s1_addr_clean.split() for lm in LANDMARKS):
            if len(examples["landmark_style_addresses"]) < 5:
                examples["landmark_style_addresses"].append(pair_info)
                
        # Address abbreviation differences
        addr_abbr_found = False
        for abbr, full in ADDR_ABBR:
            if (f" {abbr} " in f" {s1_addr_clean} " and f" {full} " in f" {t_addr_clean} ") or \
               (f" {full} " in f" {s1_addr_clean} " and f" {abbr} " in f" {t_addr_clean} "):
                addr_abbr_found = True
                break
        if addr_abbr_found and len(examples["address_abbreviations"]) < 5:
            examples["address_abbreviations"].append(pair_info)
            
        # Missing address components (e.g. one address has only city/state or is much shorter)
        if (len(s1["address"]) < 15 and len(target["address"]) > 35) or (len(target["address"]) < 15 and len(s1["address"]) > 35) or not target["address"] or not s1["address"]:
            if len(examples["missing_address_components"]) < 5:
                examples["missing_address_components"].append(pair_info)
                
        # Punctuation differences (& vs and, commas, slashes)
        if ("&" in s1["name"] and "and" in target["name"].lower()) or ("&" in target["name"] and "and" in s1["name"].lower()):
            if len(examples["punctuation_differences"]) < 5:
                examples["punctuation_differences"].append(pair_info)
                
        # Difficult matches: low token overlap in name (< 0.5) but true match
        if s1_tokens and t_tokens:
            jaccard = len(s1_tokens & t_tokens) / len(s1_tokens | t_tokens)
            if jaccard < 0.4 and len(examples["difficult_matches"]) < 5:
                examples["difficult_matches"].append(dict(pair_info, jaccard_name=round(jaccard, 3)))

# Find difficult non-matches (different S1 entities sharing similar names in same country)
# Sample a few pairs of S1 records with identical or near-identical names
names_map = collections.defaultdict(list)
for sid, r in list(s1_recs.items())[:5000]:
    cname = " ".join(clean(r["name"]).split())
    if len(cname) > 5:
        names_map[cname].append((sid, r))

for cname, sid_recs in names_map.items():
    if len(sid_recs) >= 2:
        rec1, rec2 = sid_recs[0], sid_recs[1]
        if rec1[1]["address"] != rec2[1]["address"]:
            examples["difficult_non_matches"].append({
                "note": "Distinct S1 businesses with identical name but different addresses",
                "entity_1": {"id": rec1[0], "name": rec1[1]["name"], "address": rec1[1]["address"], "country": rec1[1]["country"]},
                "entity_2": {"id": rec2[0], "name": rec2[1]["name"], "address": rec2[1]["address"], "country": rec2[1]["country"]}
            })
            if len(examples["difficult_non_matches"]) >= 5:
                break

out_path = os.path.join(EXPERIMENTS_DIR, "noise_pattern_examples.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print(f"Saved qualitative pattern examples to {out_path}")
