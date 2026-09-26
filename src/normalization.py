"""
Text normalization module for Amazon ML Challenge 2026: Business Entity Resolution.

Supports:
- Multilingual Unicode normalization (preserving English, Devanagari, French accents)
- Punctuation and whitespace cleaning
- Domain name handling (e.g. 'company.com' -> 'company')
- Legal suffix extraction and normalization across US, India, and France
- Address standardization (common street types, directional abbreviations)
- Numeric token extraction (house numbers, PIN codes, postal codes)
"""

import re
import unicodedata
from typing import List, Set, Tuple

# Domain suffixes to recognize in business names
DOMAIN_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|org|net|co|io|in|co\.in|fr|biz|info|us)(?:/.*)?$",
    re.IGNORECASE,
)

# Legal suffixes across US, India, France
LEGAL_SUFFIX_SET = {
    # US / General
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "co", "company", "llp", "pllc", "lp", "pc",
    # India
    "pvt", "private",
    # France
    "sarl", "sas", "sasu", "sci", "sa", "eurl", "snc", "gie"
}

# Regex to strip legal tokens at word boundaries
LEGAL_REGEX = re.compile(
    r"\b(" + "|".join(sorted(LEGAL_SUFFIX_SET, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Address abbreviations mapping (standardized to canonical form)
ADDRESS_ABBR_MAP = {
    # Road types
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "hwy": "highway",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pkwy": "parkway",
    "ter": "terrace",
    "terr": "terrace",
    "pl": "place",
    "sq": "square",
    "way": "way",
    # Unit / Building types
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "no": "number",
    # Indian context
    "opp": "opposite",
    "nr": "near",
    "adj": "adjacent",
    "sec": "sector",
    "ph": "phase",
    "kh": "khasra",
    # French context
    "r": "rue",
    "rte": "route",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters while preserving accents and non-Latin scripts."""
    if not text:
        return ""
    # NFKC normalizes compatibility characters (e.g. ligatures, full-width) while keeping base scripts
    return unicodedata.normalize("NFKC", text)


# Match punctuation while preserving letters, numbers, Devanagari (including matras), and combining diacritics
PUNCT_REGEX = re.compile(r"[^\w\s\u0900-\u097F\u0300-\u036F]")


def normalize_name(text: str, strip_legal: bool = False) -> str:
    """Normalize business name:
    - Lowercase & Unicode normalize
    - Check for domain pattern (e.g. example.com -> example)
    - Replace ampersands with 'and'
    - Remove punctuation (preserving Devanagari and accents)
    - Optionally strip legal suffixes
    - Normalize whitespace
    """
    if not text:
        return ""

    text = normalize_unicode(text).lower().strip()

    # Check for domain-like business name (common in S3)
    domain_match = DOMAIN_PATTERN.match(text)
    if domain_match:
        text = domain_match.group(1).replace("-", " ")

    # Standardize ampersand
    text = re.sub(r"&", " and ", text)

    # Remove non-alphanumeric punctuation
    text = PUNCT_REGEX.sub(" ", text)

    if strip_legal:
        text = LEGAL_REGEX.sub(" ", text)

    # Normalize whitespace
    return " ".join(text.split())


def normalize_and_canonicalize(text: str) -> Tuple[str, str, List[str]]:
    """High-speed single-pass normalization, canonicalization, and tokenization:
    - Unicode & lowercase normalization
    - Domain pattern extraction
    - Ampersand expansion
    - Punctuation removal
    - Legal suffix filtering via fast set lookup
    Returns (norm_name, canon_name, clean_tokens)
    """
    if not text:
        return "", "", []

    text_norm = normalize_unicode(text).lower().strip()
    domain_match = DOMAIN_PATTERN.match(text_norm)
    if domain_match:
        text_norm = domain_match.group(1).replace("-", " ")
    if "&" in text_norm:
        text_norm = text_norm.replace("&", " and ")

    text_norm = PUNCT_REGEX.sub(" ", text_norm)
    toks = text_norm.split()
    norm_name = " ".join(toks)
    clean_toks = [t for t in toks if t not in LEGAL_SUFFIX_SET]
    canon_name = " ".join(sorted(clean_toks))
    return norm_name, canon_name, clean_toks


def get_name_tokens(text: str, strip_legal: bool = False, min_len: int = 1) -> List[str]:
    """Get list of normalized name tokens."""
    _, _, clean_toks = normalize_and_canonicalize(text)
    if strip_legal:
        return [t for t in clean_toks if len(t) >= min_len]
    norm = normalize_name(text, strip_legal=False)
    return [t for t in norm.split() if len(t) >= min_len]


def get_canonical_name_key(text: str, strip_legal: bool = True) -> str:
    """Get sorted token canonical key (handles word order permutations)."""
    _, canon, _ = normalize_and_canonicalize(text)
    return canon


def normalize_address(text: str) -> str:
    """Normalize address:
    - Lowercase & Unicode normalize
    - Replace punctuation with spaces
    - Standardize common address abbreviations
    - Normalize whitespace
    """
    if not text:
        return ""

    text = normalize_unicode(text).lower().strip()
    # Replace punctuation
    text = PUNCT_REGEX.sub(" ", text)

    tokens = text.split()
    standardized = [ADDRESS_ABBR_MAP.get(t, t) for t in tokens]
    return " ".join(standardized)


def get_address_tokens(text: str) -> List[str]:
    """Get list of standardized address tokens."""
    norm = normalize_address(text)
    return norm.split()


def extract_numeric_tokens(text: str) -> Set[str]:
    """Extract numeric sequences from text (useful for PIN codes, building numbers)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def get_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if not text:
        return set()
    cleaned = "".join(text.split())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}

GENERIC_ADDR_TERMS = {
    "road", "street", "avenue", "boulevard", "drive", "highway", "lane", "court",
    "circle", "parkway", "terrace", "place", "square", "way", "apartment", "suite",
    "floor", "building", "number", "opposite", "near", "adjacent", "sector", "phase",
    "khasra", "rue", "route"
}

def extract_address_features(addr: str) -> Tuple[List[str], List[str]]:
    if not addr:
        return [], []
    tokens = normalize_address(addr).split()
    nums   = [t for t in tokens if t.isdigit() and len(t) >= 2]
    postal = [n for n in nums if len(n) in (5, 6)]
    other  = [n for n in nums if len(n) not in (5, 6)]
    salient_alphas = [
        t for t in tokens
        if t.isalpha() and len(t) >= 4 and t not in GENERIC_ADDR_TERMS
    ]
    return postal + other, salient_alphas
