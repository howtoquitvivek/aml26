"""
Miss-Recovery Candidate Generation Module.
"""
import os
import json
import pickle
from typing import Set
from src.normalization import normalize_and_canonicalize, normalize_address
from src.index_builder_mr import GENERIC_WORDS

class MRCandidateGenerator:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.manifest = None
        with open(os.path.join(index_dir, "mr_index_manifest.json"), "r") as f:
            self.manifest = json.load(f)
            
        self.loaded_country = None
        self.cand_ids = []
        self.idx_base = {}

    def load_country(self, country: str):
        if self.loaded_country == country:
            return
            
        print(f"[MR] Loading indexes for {country}...")
        self.cand_ids = []
        self.idx_base = {}
        
        id_file = os.path.join(self.index_dir, f"mr_cand_ids_{country}.json")
        idx_file = os.path.join(self.index_dir, f"mr_idx_{country}.pkl")
        
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
        
        _, _, clean_toks = normalize_and_canonicalize(s1_name)
        addr_toks = [t for t in normalize_address(s1_addr).split() if t.isalnum() and t not in GENERIC_WORDS]
        
        name_toks = clean_toks[:3]
        addr_toks = addr_toks[:3]
        
        for nt in name_toks:
            if len(nt) >= 3:
                for at in addr_toks:
                    if len(at) >= 3:
                        lst = self.idx_base.get(f"MR1#{nt}#{at}", [])
                        if len(lst) <= 500: cands.update(lst)
                        
        for nt in name_toks:
            if len(nt) >= 5:
                lst = self.idx_base.get(f"MR2#{nt}", [])
                if len(lst) <= 500: cands.update(lst)
                
        for at in addr_toks:
            if len(at) >= 6:
                lst = self.idx_base.get(f"MR3#{at}", [])
                if len(lst) <= 500: cands.update(lst)
                
        return {self.cand_ids[idx] for idx in cands}
