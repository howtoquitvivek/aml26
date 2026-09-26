"""
Matcher feature extraction module for Amazon ML Challenge 2026.
Defines deterministic feature schemas and ensures consistent ordered output.
"""
import math
from typing import Dict, List, Set, Tuple
from src.normalization import (
    normalize_name,
    get_canonical_name_key,
    get_name_tokens,
    normalize_address,
    get_address_tokens,
    extract_numeric_tokens,
    get_character_ngrams,
    normalize_and_canonicalize
)

FEATURE_SCHEMA_VERSION = 2

FEATURE_NAMES = [
    "name_exact",
    "name_canonical_exact",
    "name_tok_jac",
    "name_char_jac",
    "name_len_diff",
    "addr_exact",
    "addr_missing",
    "addr_tok_jac",
    "addr_char_jac",
    "addr_num_jac",
    "addr_len_diff",
    "postal_overlap",
    "salient_addr_jac",
    "is_s2",
    "country_match"
]

def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b: return 0.0
    if not set_a or not set_b: return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0

def extract_features(s1_name: str, s1_addr: str, s1_ctry: str, 
                     cand_name: str, cand_addr: str, cand_ctry: str, is_s2: bool) -> List[float]:
    
    country_match = float(s1_ctry == cand_ctry)
    
    n1, c1, _ = normalize_and_canonicalize(s1_name)
    n2, c2, _ = normalize_and_canonicalize(cand_name)
    
    name_exact = float(n1 == n2 and n1 != "")
    name_canonical_exact = float(c1 == c2 and c1 != "")
    
    t1 = set(n1.split())
    t2 = set(n2.split())
    name_tok_jac = jaccard_similarity(t1, t2)
    name_char_jac = jaccard_similarity(set(n1), set(n2))
    name_len_diff = abs(len(n1) - len(n2)) / max(1, max(len(n1), len(n2)))
    
    a1 = normalize_address(s1_addr)
    a2 = normalize_address(cand_addr)
    addr_exact = float(a1 == a2 and a1 != "")
    addr_missing = float(not a1 or not a2)
    
    at1 = set(a1.split())
    at2 = set(a2.split())
    addr_tok_jac = jaccard_similarity(at1, at2)
    addr_char_jac = jaccard_similarity(set(a1), set(a2))
    
    num1 = extract_numeric_tokens(s1_addr)
    num2 = extract_numeric_tokens(cand_addr)
    addr_num_jac = jaccard_similarity(num1, num2)
    
    addr_len_diff = abs(len(a1) - len(a2)) / max(1, max(len(a1), len(a2)))
    
    # Postal overlap (approximated as 6 digit strings in India, 5 in US/FR)
    p1 = {n for n in num1 if len(n) in (5, 6)}
    p2 = {n for n in num2 if len(n) in (5, 6)}
    postal_overlap = jaccard_similarity(p1, p2)
    
    # Salient address tokens (len >= 5)
    sat1 = {t for t in at1 if len(t) >= 5 and not t.isdigit()}
    sat2 = {t for t in at2 if len(t) >= 5 and not t.isdigit()}
    salient_addr_jac = jaccard_similarity(sat1, sat2)
    
    return [
        name_exact,
        name_canonical_exact,
        name_tok_jac,
        name_char_jac,
        name_len_diff,
        addr_exact,
        addr_missing,
        addr_tok_jac,
        addr_char_jac,
        addr_num_jac,
        addr_len_diff,
        postal_overlap,
        salient_addr_jac,
        float(is_s2),
        country_match
    ]
