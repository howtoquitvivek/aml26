"""
Candidate Generation Module for Amazon ML Challenge 2026.
Generates candidates from the deterministic indexes.
"""
import os
import json
import pickle
from typing import Dict, List, Set, Tuple
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

class CandidateGenerator:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.manifest = None
        with open(os.path.join(index_dir, "index_manifest.json"), "r") as f:
            self.manifest = json.load(f)
            
        self.loaded_country = None
        self.cand_ids = []
        self.idx_base = {}

    def load_country(self, country: str):
        if self.loaded_country == country:
            return
            
        print(f"Loading indexes for {country}...")
        self.cand_ids = []
        self.idx_base = {}
        
        id_file = os.path.join(self.index_dir, f"cand_ids_{country}.json")
        idx_file = os.path.join(self.index_dir, f"idx_{country}.pkl")
        
        with open(id_file, "r") as f:
            self.cand_ids = json.load(f)
            
        with open(idx_file, "rb") as f:
            self.idx_base = pickle.load(f)
            
        self.loaded_country = country

    def get_candidates(self, s1_name: str, s1_addr: str, s1_ctry: str) -> Set[str]:
        if s1_ctry not in self.manifest["countries"]:
            return set()
            
        self.load_country(s1_ctry)
        
        cands = set()
        
        norm_name, canon_name, clean_toks = normalize_and_canonicalize(s1_name)
        def add_bounded(key, limit=500):
            lst = self.idx_base.get(key, [])
            if len(lst) <= limit: cands.update(lst)
            
        add_bounded(f"E#{norm_name}")
        add_bounded(f"C#{canon_name}")
        
        if clean_toks:
            old_nums = extract_numeric_tokens(s1_addr)
            tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
            for n in old_nums:
                if len(n) >= 2: add_bounded(f"A#{n}_{tok_p}")
                
        nums, _ = extract_address_features(s1_addr)
        for k in get_key_b(canon_name, nums):
            add_bounded(f"B#{k}")
            
        for k in get_key_e(s1_addr):
            add_bounded(f"K#{k}")
            
        p3_keys = get_p3_2_keys(s1_name, s1_addr)
        for k in p3_keys:
            cands.update(self.idx_base.get(k, [])) # P3 is already bounded to 100 in index
            
        return {self.cand_ids[idx] for idx in cands}

    def get_candidate_indices(self, s1_name: str, s1_addr: str, s1_ctry: str) -> Set[int]:
        if s1_ctry not in self.manifest["countries"]:
            return set()
            
        self.load_country(s1_ctry)
        
        cands = set()
        
        norm_name, canon_name, clean_toks = normalize_and_canonicalize(s1_name)
        def add_bounded(key, limit=500):
            lst = self.idx_base.get(key, [])
            if len(lst) <= limit: cands.update(lst)
            
        add_bounded(f"E#{norm_name}")
        add_bounded(f"C#{canon_name}")
        
        if clean_toks:
            old_nums = extract_numeric_tokens(s1_addr)
            tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
            for n in old_nums:
                if len(n) >= 2: add_bounded(f"A#{n}_{tok_p}")
                
        nums, _ = extract_address_features(s1_addr)
        for k in get_key_b(canon_name, nums):
            add_bounded(f"B#{k}")
            
        for k in get_key_e(s1_addr):
            add_bounded(f"K#{k}")
            
        p3_keys = get_p3_2_keys(s1_name, s1_addr)
        for k in p3_keys:
            cands.update(self.idx_base.get(k, []))
            
        return cands
