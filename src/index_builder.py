"""
Index Builder for Amazon ML Challenge 2026.
Builds deterministic inverted indexes per country to allow fast candidate generation while keeping RAM usage safe.
"""
import collections
import json
import os
import time
import pickle
from src.normalization import normalize_and_canonicalize, extract_address_features, extract_numeric_tokens, normalize_address

def get_consonants(word):
    return "".join(c for c in word.lower() if c.isalpha() and c not in "aeiouyh")

GENERIC_WORDS = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "co", "company", "llp", "pllc", "lp", "pc", "pvt", "private", "sarl", "sas", "sasu", "sci", "sa", "eurl", "snc", "gie",
    "road", "street", "avenue", "boulevard", "drive", "highway", "lane", "court",
    "circle", "parkway", "terrace", "place", "square", "way", "apartment", "suite",
    "floor", "building", "number", "opposite", "near", "adjacent", "sector", "phase",
    "khasra", "rue", "route", "rd", "st", "ave", "dr", "apt", "ste", "bldg", "no",
    "opp", "nr", "adj", "sec", "ph", "kh", "rte"
}

def get_p3_2_keys(name, addr):
    name_toks = [t for t in normalize_address(name).split() if t.isalpha() and t not in GENERIC_WORDS]
    addr_toks = [t for t in normalize_address(addr).split() if t.isalpha() and t not in GENERIC_WORDS]
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

def get_key_b(canon_name, nums):
    tokens = canon_name.split()
    if not tokens: return []
    res = []
    t = tokens[0]
    if len(t) >= 4:
        for n in nums:
            if len(n) >= 2: res.append(f"{t}_{n}")
    return res

def get_key_e(addr):
    addr = normalize_address(addr)
    tokens = [t for t in addr.split() if t not in GENERIC_WORDS and t.isalpha()]
    res = []
    for t in tokens:
        if len(t) >= 6:
            c = get_consonants(t)
            if len(c) >= 4: res.append(c)
    return res

def build_index_for_country(s2_path: str, s3_path: str, out_dir: str, country: str):
    idx_base = collections.defaultdict(list)
    idx_p3_raw = collections.defaultdict(list)
    cand_arr = []
    
    for file_idx, fpath in enumerate([s2_path, s3_path]):
        prefix = "S2" if file_idx == 0 else "S3"
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3].strip() == country:
                    cid = parts[0].strip()
                    name = parts[1].strip()
                    addr = parts[2].strip()
                    
                    rec_idx = len(cand_arr)
                    cand_arr.append(f"{prefix}#{cid}")
                    
                    norm_name, canon_name, clean_toks = normalize_and_canonicalize(name)
                    idx_base[f"E#{norm_name}"].append(rec_idx)
                    idx_base[f"C#{canon_name}"].append(rec_idx)
                    
                    nums, _ = extract_address_features(addr)
                    if clean_toks:
                        tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
                        for n in nums:
                            if len(n) >= 2: idx_base[f"A#{n}_{tok_p}"].append(rec_idx)
                    
                    for k in get_key_b(canon_name, nums):
                        idx_base[f"B#{k}"].append(rec_idx)
                    
                    for k in get_key_e(addr):
                        idx_base[f"K#{k}"].append(rec_idx)
                        
                    p3_keys = get_p3_2_keys(name, addr)
                    for k in p3_keys:
                        idx_p3_raw[k].append(rec_idx)
                        
    print(f"[{country}] Filter P3 keys by freq...")
    idx_p3 = {k: v for k, v in idx_p3_raw.items() if len(v) <= 100}
    del idx_p3_raw
    
    for k, v in idx_p3.items():
        idx_base[k].extend(v)
        
    print(f"[{country}] Saving...")
    with open(os.path.join(out_dir, f"cand_ids_{country}.json"), "w") as f:
        json.dump(cand_arr, f)
        
    with open(os.path.join(out_dir, f"idx_{country}.pkl"), "wb") as f:
        pickle.dump(dict(idx_base), f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return len(cand_arr), len(idx_base)

def build_all_indexes(s2_path: str, s3_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    
    countries = set()
    for fpath in [s2_path, s3_path]:
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    countries.add(parts[3].strip())
                    
    countries = sorted(list(countries))
    print(f"Found {len(countries)} countries: {countries}")
    
    manifest = {
        "version": "1.0",
        "timestamp": t0,
        "key_families": ["E", "C", "A", "B", "K", "P3"],
        "frequency_thresholds": {"P3": 100},
        "countries": {}
    }
    
    for c in countries:
        cands, keys = build_index_for_country(s2_path, s3_path, out_dir, c)
        manifest["countries"][c] = {"candidates": cands, "keys": keys}
        
    t1 = time.time()
    manifest["runtime_seconds"] = t1 - t0
    
    with open(os.path.join(out_dir, "index_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"All indexes built in {t1 - t0:.2f} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2", type=str, default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3", type=str, default="dataset/train/train_source3.tsv")
    parser.add_argument("--out", type=str, default="experiments/final_hardening/index")
    args = parser.parse_args()
    build_all_indexes(args.s2, args.s3, args.out)
