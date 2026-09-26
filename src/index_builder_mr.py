"""
Miss-Recovery Index Builder.
Builds an additional index strictly for recovering pairs missed by P3.2.
"""
import collections
import json
import os
import time
import pickle
from src.normalization import normalize_and_canonicalize, normalize_address

GENERIC_WORDS = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "co", "company", "llp", "pllc", "lp", "pc", "pvt", "private", "sarl", "sas", "sasu", "sci", "sa", "eurl", "snc", "gie",
    "road", "street", "avenue", "boulevard", "drive", "highway", "lane", "court",
    "circle", "parkway", "terrace", "place", "square", "way", "apartment", "suite",
    "floor", "building", "number", "opposite", "near", "adjacent", "sector", "phase",
    "khasra", "rue", "route", "rd", "st", "ave", "dr", "apt", "ste", "bldg", "no",
    "opp", "nr", "adj", "sec", "ph", "kh", "rte"
}

def get_mr_keys(name: str, addr: str):
    _, _, clean_toks = normalize_and_canonicalize(name)
    addr_toks = [t for t in normalize_address(addr).split() if t.isalnum() and t not in GENERIC_WORDS]
    
    name_toks = clean_toks[:3]
    addr_toks = addr_toks[:3]
    
    keys = set()
    for nt in name_toks:
        if len(nt) >= 3:
            for at in addr_toks:
                if len(at) >= 3:
                    keys.add(f"MR1#{nt}#{at}")
                    
    # Also just index the first rare name token if it's long enough
    for nt in name_toks:
        if len(nt) >= 5:
            keys.add(f"MR2#{nt}")
            
    # And first rare address token
    for at in addr_toks:
        if len(at) >= 6:
            keys.add(f"MR3#{at}")
            
    return keys

def build_index_for_country(s2_path: str, s3_path: str, out_dir: str, country: str):
    idx_raw = collections.defaultdict(list)
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
                    
                    for k in get_mr_keys(name, addr):
                        idx_raw[k].append(rec_idx)
                        
    print(f"[{country}] Filter MR keys by freq...")
    # Keep MR1 if <= 100, MR2 if <= 50, MR3 if <= 50
    idx_final = collections.defaultdict(list)
    for k, v in idx_raw.items():
        if k.startswith("MR1#") and len(v) <= 100:
            idx_final[k] = v
        elif k.startswith("MR2#") and len(v) <= 50:
            idx_final[k] = v
        elif k.startswith("MR3#") and len(v) <= 50:
            idx_final[k] = v
            
    del idx_raw
        
    print(f"[{country}] Saving {len(idx_final)} keys...")
    with open(os.path.join(out_dir, f"mr_cand_ids_{country}.json"), "w") as f:
        json.dump(cand_arr, f)
        
    with open(os.path.join(out_dir, f"mr_idx_{country}.pkl"), "wb") as f:
        pickle.dump(dict(idx_final), f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return len(cand_arr), len(idx_final)

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
        "key_families": ["MR1", "MR2", "MR3"],
        "countries": {}
    }
    
    for c in countries:
        cands, keys = build_index_for_country(s2_path, s3_path, out_dir, c)
        manifest["countries"][c] = {"candidates": cands, "keys": keys}
        
    t1 = time.time()
    manifest["runtime_seconds"] = t1 - t0
    
    with open(os.path.join(out_dir, "mr_index_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"All MR indexes built in {t1 - t0:.2f} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2", type=str, default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3", type=str, default="dataset/train/train_source3.tsv")
    parser.add_argument("--out", type=str, default="experiments/final_hardening/index")
    args = parser.parse_args()
    build_all_indexes(args.s2, args.s3, args.out)
