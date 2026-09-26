"""
Deterministic baseline matcher for Amazon ML Challenge 2026.

Computes interpretable similarity features between Source 1 and candidate S2/S3 records:
- Name features: exact normalized match, canonical match, token Jaccard, character 3-gram Jaccard, length ratio
- Address features: exact match, token Jaccard, character 3-gram Jaccard, numeric token overlap, missing indicator
- Country feature: equality indicator

Provides transparent weighted scoring and threshold evaluation against validation ground truth.
"""

from typing import Dict, List, Set, Tuple, Optional
from src.normalization import (
    normalize_name,
    get_canonical_name_key,
    get_name_tokens,
    normalize_address,
    get_address_tokens,
    extract_numeric_tokens,
    get_character_ngrams,
)


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_ctry: str,
    cand_name: str,
    cand_addr: str,
    cand_ctry: str,
) -> Dict[str, float]:
    """Compute interpretable similarity features between an S1 entity and a candidate."""
    # Country
    country_match = 1.0 if s1_ctry == cand_ctry else 0.0

    # Name features
    norm_s1 = normalize_name(s1_name, strip_legal=False)
    norm_cand = normalize_name(cand_name, strip_legal=False)
    name_exact = 1.0 if norm_s1 and norm_s1 == norm_cand else 0.0

    canon_s1 = get_canonical_name_key(s1_name, strip_legal=True)
    canon_cand = get_canonical_name_key(cand_name, strip_legal=True)
    name_canonical_exact = 1.0 if canon_s1 and canon_s1 == canon_cand else 0.0

    tok_s1 = set(get_name_tokens(s1_name, strip_legal=True, min_len=2))
    tok_cand = set(get_name_tokens(cand_name, strip_legal=True, min_len=2))
    name_token_jaccard = jaccard_similarity(tok_s1, tok_cand)

    char_s1 = get_character_ngrams(norm_s1, n=3)
    char_cand = get_character_ngrams(norm_cand, n=3)
    name_char_jaccard = jaccard_similarity(char_s1, char_cand)

    len_s1 = len(norm_s1)
    len_cand = len(norm_cand)
    name_len_diff_ratio = abs(len_s1 - len_cand) / max(len_s1, len_cand, 1)

    # Address features
    norm_a1 = normalize_address(s1_addr)
    norm_ac = normalize_address(cand_addr)
    addr_missing = 1.0 if not norm_ac else 0.0

    if not norm_ac:
        addr_exact = 0.0
        addr_token_jaccard = 0.0
        addr_char_jaccard = 0.0
        addr_numeric_jaccard = 0.0
        addr_len_diff_ratio = 1.0
    else:
        addr_exact = 1.0 if norm_a1 and norm_a1 == norm_ac else 0.0
        atok_s1 = set(get_address_tokens(s1_addr))
        atok_cand = set(get_address_tokens(cand_addr))
        addr_token_jaccard = jaccard_similarity(atok_s1, atok_cand)

        achar_s1 = get_character_ngrams(norm_a1, n=3)
        achar_cand = get_character_ngrams(norm_ac, n=3)
        addr_char_jaccard = jaccard_similarity(achar_s1, achar_cand)

        nums_s1 = extract_numeric_tokens(s1_addr)
        nums_cand = extract_numeric_tokens(cand_addr)
        addr_numeric_jaccard = jaccard_similarity(nums_s1, nums_cand)

        alen_s1 = len(norm_a1)
        alen_cand = len(norm_ac)
        addr_len_diff_ratio = abs(alen_s1 - alen_cand) / max(alen_s1, alen_cand, 1)

    return {
        "country_match": country_match,
        "name_exact": name_exact,
        "name_canonical_exact": name_canonical_exact,
        "name_token_jaccard": name_token_jaccard,
        "name_char_jaccard": name_char_jaccard,
        "name_len_diff_ratio": name_len_diff_ratio,
        "addr_missing": addr_missing,
        "addr_exact": addr_exact,
        "addr_token_jaccard": addr_token_jaccard,
        "addr_char_jaccard": addr_char_jaccard,
        "addr_numeric_jaccard": addr_numeric_jaccard,
        "addr_len_diff_ratio": addr_len_diff_ratio,
    }


def compute_deterministic_score(
    features: Dict[str, float],
    config_name: str = "balanced",
) -> float:
    """Compute transparent weighted score for a candidate pair."""
    # If countries do not match, zero probability
    if features["country_match"] == 0.0:
        return 0.0

    # Strong exact match shortcut: if exact name matches
    if features["name_exact"] == 1.0 or features["name_canonical_exact"] == 1.0:
        if features["addr_missing"] == 1.0:
            return 0.85
        if features["addr_exact"] == 1.0:
            return 1.0
        # If address has non-zero token overlap or numeric overlap
        if features["addr_token_jaccard"] >= 0.20 or features["addr_numeric_jaccard"] >= 0.30:
            return 0.90 + 0.10 * features["addr_token_jaccard"]
        else:
            # Name matches but address completely disagrees -> potential branch/franchise
            return 0.65

    if config_name == "name_heavy":
        # 75% name, 25% address
        name_subscore = 0.50 * features["name_token_jaccard"] + 0.50 * features["name_char_jaccard"]
        if features["addr_missing"] == 1.0:
            addr_subscore = 0.50  # neutral imputation
        else:
            addr_subscore = 0.60 * features["addr_token_jaccard"] + 0.40 * features["addr_numeric_jaccard"]
        return 0.75 * name_subscore + 0.25 * addr_subscore

    elif config_name == "balanced":
        # 55% name, 45% address
        name_subscore = 0.50 * features["name_token_jaccard"] + 0.50 * features["name_char_jaccard"]
        if features["addr_missing"] == 1.0:
            addr_subscore = 0.50
        else:
            addr_subscore = 0.50 * features["addr_token_jaccard"] + 0.50 * features["addr_numeric_jaccard"]
        return 0.55 * name_subscore + 0.45 * addr_subscore

    elif config_name == "conservative_precision":
        # Requires high agreement on both name and address
        name_subscore = 0.60 * features["name_token_jaccard"] + 0.40 * features["name_char_jaccard"]
        if features["addr_missing"] == 1.0:
            addr_subscore = 0.35  # penalized missing
        else:
            addr_subscore = 0.50 * features["addr_token_jaccard"] + 0.50 * features["addr_numeric_jaccard"]
        return 0.50 * name_subscore + 0.50 * addr_subscore

    else:
        raise ValueError(f"Unknown config: {config_name}")
