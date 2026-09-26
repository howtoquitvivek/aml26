"""
Blocking and Candidate Generation module for Amazon ML Challenge 2026.

Designed for memory-safe execution on ~10.3M records:
- Country-partitioned inverted indices
- Exact normalized name blocking
- Canonical token blocking (order-invariant, legal-suffix invariant)
- Character n-gram / prefix-based blocking for typo resilience
- Address-assisted blocking using numeric tokens and distinctive tokens
- Unified multi-strategy blocking with union and candidate ranking
"""

import collections
import time
from typing import Dict, List, Set, Tuple, Optional, Iterable
from src.normalization import (
    normalize_name,
    normalize_and_canonicalize,
    get_canonical_name_key,
    get_name_tokens,
    normalize_address,
    extract_numeric_tokens,
    get_character_ngrams,
)


class BlockingIndex:
    """In-memory multi-strategy blocking index over Source 2 and Source 3 records."""

    def __init__(self, max_block_size: int = 500):
        self.max_block_size = max_block_size

        # Inverted indices partitioned by country
        # (country, exact_norm_name) -> list of entity_ids
        self.exact_name_index = collections.defaultdict(list)

        # (country, canonical_token_key) -> list of entity_ids
        self.canonical_name_index = collections.defaultdict(list)

        # (country, first_2_tokens) -> list of entity_ids
        self.token_prefix_index = collections.defaultdict(list)

        # (country, 4char_prefix) -> list of entity_ids
        self.char_prefix_index = collections.defaultdict(list)

        # (country, address_numeric_token + name_initial) -> list of entity_ids
        self.address_numeric_index = collections.defaultdict(list)

        # Metadata tracking
        self.total_indexed_records = 0

    def add_record(self, entity_id: str, name: str, address: str, country: str):
        """Index a single Source 2 or Source 3 record."""
        self.total_indexed_records += 1

        norm_name, canon_name, tokens = normalize_and_canonicalize(name)

        # 1. Exact normalized name
        if norm_name:
            self.exact_name_index[(country, norm_name)].append(entity_id)

        # 2. Canonical token key
        if canon_name and canon_name != norm_name:
            self.canonical_name_index[(country, canon_name)].append(entity_id)

        # 3. Token prefix (first 2 tokens if >= 2 tokens)
        if len(tokens) >= 2:
            two_tokens = " ".join(tokens[:2])
            self.token_prefix_index[(country, two_tokens)].append(entity_id)

        # 4. Character prefix (first 4 characters of canonical key)
        if len(canon_name) >= 4:
            prefix4 = canon_name[:4]
            self.char_prefix_index[(country, prefix4)].append(entity_id)

        # 5. Address numeric token + first name token
        if address and tokens:
            nums = extract_numeric_tokens(address)
            first_tok = tokens[0][:4] if len(tokens[0]) >= 4 else tokens[0]
            for num in nums:
                if len(num) >= 2:  # Ignore single digits
                    addr_key = (country, f"{num}_{first_tok}")
                    self.address_numeric_index[addr_key].append(entity_id)

    def prune_large_blocks(self):
        """Prune inverted list blocks that exceed max_block_size to avoid low-selectivity explosion."""
        for idx in [
            self.exact_name_index,
            self.canonical_name_index,
            self.token_prefix_index,
            self.char_prefix_index,
            self.address_numeric_index,
        ]:
            keys_to_prune = [k for k, v in idx.items() if len(v) > self.max_block_size]
            for k in keys_to_prune:
                # Keep only a sample or truncate
                idx[k] = idx[k][: self.max_block_size]

    def get_candidates_exact(self, name: str, country: str) -> Set[str]:
        """Candidates from exact normalized name."""
        norm_name = normalize_name(name, strip_legal=False)
        return set(self.exact_name_index.get((country, norm_name), []))

    def get_candidates_canonical(self, name: str, country: str) -> Set[str]:
        """Candidates from canonical sorted name tokens."""
        canon_name = get_canonical_name_key(name, strip_legal=True)
        return set(self.canonical_name_index.get((country, canon_name), []))

    def get_candidates_token_prefix(self, name: str, country: str) -> Set[str]:
        """Candidates from first two tokens."""
        tokens = get_name_tokens(name, strip_legal=True, min_len=2)
        if len(tokens) >= 2:
            two_tokens = " ".join(tokens[:2])
            return set(self.token_prefix_index.get((country, two_tokens), []))
        return set()

    def get_candidates_char_prefix(self, name: str, country: str) -> Set[str]:
        """Candidates from character prefix."""
        canon_name = get_canonical_name_key(name, strip_legal=True)
        if len(canon_name) >= 4:
            return set(self.char_prefix_index.get((country, canon_name[:4]), []))
        return set()

    def get_candidates_address_assisted(self, name: str, address: str, country: str) -> Set[str]:
        """Candidates from address numeric token + name prefix."""
        candidates = set()
        tokens = get_name_tokens(name, strip_legal=True, min_len=2)
        if address and tokens:
            nums = extract_numeric_tokens(address)
            first_tok = tokens[0][:4] if len(tokens[0]) >= 4 else tokens[0]
            for num in nums:
                if len(num) >= 2:
                    addr_key = (country, f"{num}_{first_tok}")
                    candidates.update(self.address_numeric_index.get(addr_key, []))
        return candidates

    def get_candidates_combined(
        self,
        name: str,
        address: str,
        country: str,
        include_char_prefix: bool = True,
        include_address: bool = True,
        max_candidates: int = 150,
    ) -> Set[str]:
        """High-recall union of multiple blocking strategies."""
        candidates = set()

        # Tier 1: High precision / high selectivity
        candidates.update(self.get_candidates_exact(name, country))
        candidates.update(self.get_candidates_canonical(name, country))
        candidates.update(self.get_candidates_token_prefix(name, country))

        # Tier 2: Address assisted
        if include_address and len(candidates) < max_candidates:
            addr_cands = self.get_candidates_address_assisted(name, address, country)
            candidates.update(addr_cands)

        # Tier 3: Character prefix (only if candidates are few to recover typos)
        if include_char_prefix and len(candidates) < 10:
            char_cands = self.get_candidates_char_prefix(name, country)
            # Add up to limit
            for c in char_cands:
                candidates.add(c)
                if len(candidates) >= max_candidates:
                    break

        return candidates
