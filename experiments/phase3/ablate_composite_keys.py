"""
Phase 3.1 Composite Blocking Keys Ablation on Sample 20.
Evaluates:
- Exact Name
- Canonical Name
- Address-Assisted
- Key A: first significant name token + postal code / numeric anchor
- Key B: canonical_name[:4] + postal code / numeric anchor
- Key C: first significant name token + salient address token
- Key D: canonical_name[:3] + salient address token OR postal code

Evaluates individual performance and unions against Ground Truth links on --sample 20.
"""

import os
import sys
import re
import array
import collections
from typing import Dict, List, Set, Tuple
import numpy as np

DATA_DIR = "dataset"
EXPERIMENTS_DIR = "experiments"
S1_FILE = os.path.join(DATA_DIR, "train", "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train", "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train", "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
VAL_IDS_FILE = os.path.join(EXPERIMENTS_DIR, "val_s1_ids.txt")

sys.path.insert(0, os.path.abspath("."))
from src.normalization import (
    normalize_and_canonicalize,
    normalize_address,
    extract_numeric_tokens,
)

GENERIC_ADDR_TERMS = {
    "road", "street", "avenue", "drive", "lane", "highway", "court",
    "circle", "parkway", "terrace", "place", "square", "way", "boulevard",
    "apartment", "suite", "floor", "building", "number", "opposite", "near",
    "adjacent", "sector", "phase", "khasra", "rue", "route", "post", "box",
    "pobox", "st", "rd", "ave", "dr", "ln", "blvd", "apt", "ste", "bldg", "no"
}


def extract_address_features(addr: str) -> Tuple[List[str], List[str]]:
    """
    Extract:
    1. Postal / numeric anchors: 5-digit/6-digit postal codes preferred, or general numeric tokens >= 2 digits
    2. Salient alpha tokens: alphabetic tokens with length >= 4, excluding generic address terms
    """
    if not addr:
        return [], []
    
    norm_addr = normalize_address(addr)
    tokens = norm_addr.split()
    
    # 1. Numerics
    nums = [t for t in tokens if t.isdigit() and len(t) >= 2]
    # Sort with preference for 5 or 6 digit codes (postal/PIN codes)
    postal_codes = [n for n in nums if len(n) in (5, 6)]
    other_nums = [n for n in nums if len(n) not in (5, 6)]
    ordered_nums = postal_codes + other_nums

    # 2. Salient alpha tokens
    salient_alphas = [
        t for t in tokens
        if t.isalpha() and len(t) >= 4 and t not in GENERIC_ADDR_TERMS
    ]
    
    return ordered_nums, salient_alphas


def get_key_a(clean_tokens: List[str], nums: List[str]) -> Set[str]:
    """Key A: first significant name token + postal code / numeric anchor"""
    if not clean_tokens or not nums:
        return set()
    first_tok = clean_tokens[0]
    keys = set()
    for n in nums[:2]:  # Use top numeric anchors
        keys.add(f"{first_tok}#{n}")
    return keys


def get_key_b(canon_name: str, nums: List[str]) -> Set[str]:
    """Key B: canonical_name[:4] + postal code / numeric anchor"""
    if len(canon_name) < 4 or not nums:
        return set()
    p4 = canon_name[:4]
    keys = set()
    for n in nums[:2]:
        keys.add(f"{p4}#{n}")
    return keys


def get_key_c(clean_tokens: List[str], alphas: List[str]) -> Set[str]:
    """Key C: first significant name token + salient address token"""
    if not clean_tokens or not alphas:
        return set()
    first_tok = clean_tokens[0]
    keys = set()
    for a in alphas[:3]:  # Top salient locality/street tokens
        keys.add(f"{first_tok}@{a}")
    return keys


def get_key_d(canon_name: str, nums: List[str], alphas: List[str]) -> Set[str]:
    """Key D: canonical_name[:3] + salient address token OR postal code"""
    if len(canon_name) < 3:
        return set()
    p3 = canon_name[:3]
    keys = set()
    for n in nums[:2]:
        keys.add(f"{p3}#{n}")
    for a in alphas[:2]:
        keys.add(f"{p3}@{a}")
    return keys


def main():
    print("Loading validation IDs...")
    with open(VAL_IDS_FILE, "r", encoding="utf-8") as f:
        val_s1_ids = {line.strip() for line in f if line.strip()}

    print("Loading validation S1 records...")
    val_records = {}
    with open(S1_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0].strip() in val_s1_ids:
                val_records[parts[0].strip()] = {
                    "name": parts[1].strip(),
                    "address": parts[2].strip(),
                    "country": parts[3].strip(),
                }

    print("Loading ground truth...")
    val_gt = collections.defaultdict(set)
    with open(GT_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0].strip()
            if sid in val_s1_ids:
                if len(parts) > 1 and parts[1].strip():
                    val_gt[sid] = {m.strip() for m in parts[1].split(",") if m.strip()}
                else:
                    val_gt[sid] = set()

    # Dynamic country partitioning
    country_partitions = collections.defaultdict(dict)
    for sid, r in val_records.items():
        country_partitions[r["country"]][sid] = r

    unique_countries = sorted(country_partitions.keys())
    sample_size = 10  # 10 per country = 20 S1 entities total
    sample_s1 = {}
    for c in unique_countries:
        s_keys = list(country_partitions[c].keys())[:sample_size]
        for k in s_keys:
            sample_s1[k] = country_partitions[c][k]

    print(f"Sample size: {len(sample_s1)} S1 entities across countries: {unique_countries}")
    total_gt_pairs = sum(len(val_gt[sid]) for sid in sample_s1)
    print(f"Total Ground-Truth links in sample: {total_gt_pairs}")

    # Build index for each country
    for country in unique_countries:
        print(f"\n--- Indexing Source 2 & 3 for {country} ---")
        cand_ids = []
        idx_exact = collections.defaultdict(lambda: array.array("I"))
        idx_canon = collections.defaultdict(lambda: array.array("I"))
        idx_two_tok = collections.defaultdict(lambda: array.array("I"))
        idx_addr_old = collections.defaultdict(lambda: array.array("I"))
        idx_key_a = collections.defaultdict(lambda: array.array("I"))
        idx_key_b = collections.defaultdict(lambda: array.array("I"))
        idx_key_c = collections.defaultdict(lambda: array.array("I"))
        idx_key_d = collections.defaultdict(lambda: array.array("I"))

        for fpath in [S2_FILE, S3_FILE]:
            with open(fpath, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4 and parts[3].strip() == country:
                        eid = parts[0].strip()
                        name = parts[1].strip()
                        addr = parts[2].strip()

                        rec_idx = len(cand_ids)
                        cand_ids.append(eid)

                        norm_name, canon_name, clean_toks = normalize_and_canonicalize(name)
                        nums, alphas = extract_address_features(addr)

                        # Existing
                        if norm_name:
                            idx_exact[norm_name].append(rec_idx)
                        if canon_name and canon_name != norm_name:
                            idx_canon[canon_name].append(rec_idx)
                        if len(clean_toks) >= 2:
                            idx_two_tok[f"{clean_toks[0]} {clean_toks[1]}"].append(rec_idx)
                        if addr and clean_toks:
                            old_nums = extract_numeric_tokens(addr)
                            tok_p = clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0]
                            for n in old_nums:
                                if len(n) >= 2:
                                    idx_addr_old[f"{n}_{tok_p}"].append(rec_idx)

                        # Composite Keys
                        for k in get_key_a(clean_toks, nums):
                            idx_key_a[k].append(rec_idx)
                        for k in get_key_b(canon_name, nums):
                            idx_key_b[k].append(rec_idx)
                        for k in get_key_c(clean_toks, alphas):
                            idx_key_c[k].append(rec_idx)
                        for k in get_key_d(canon_name, nums, alphas):
                            idx_key_d[k].append(rec_idx)

        print(f"Indexed {len(cand_ids):,} candidate records for {country}.")

        # Evaluate S1 entities for this country
        c_s1_keys = [sid for sid in sample_s1 if sample_s1[sid]["country"] == country]

        results = {
            "exact": {"counts": [], "recovered": set()},
            "canon": {"counts": [], "recovered": set()},
            "addr": {"counts": [], "recovered": set()},
            "key_a": {"counts": [], "recovered": set()},
            "key_b": {"counts": [], "recovered": set()},
            "key_c": {"counts": [], "recovered": set()},
            "key_d": {"counts": [], "recovered": set()},
            "existing": {"counts": [], "recovered": set()},
            "existing_plus_a": {"counts": [], "recovered": set()},
            "existing_plus_b": {"counts": [], "recovered": set()},
            "existing_plus_c": {"counts": [], "recovered": set()},
            "existing_plus_d": {"counts": [], "recovered": set()},
            "existing_plus_all": {"counts": [], "recovered": set()},
        }

        for sid in c_s1_keys:
            r = sample_s1[sid]
            gt = val_gt[sid]
            name = r["name"]
            addr = r["address"]

            norm_name, canon_name, clean_toks = normalize_and_canonicalize(name)
            nums, alphas = extract_address_features(addr)

            # Lookups
            c_exact = set(idx_exact.get(norm_name, []))
            c_canon = set(idx_canon.get(canon_name, []))
            if len(clean_toks) >= 2:
                c_canon.update(idx_two_tok.get(f"{clean_toks[0]} {clean_toks[1]}", []))
            
            c_addr = set()
            if addr and clean_toks:
                old_nums = extract_numeric_tokens(addr)
                tok_p = clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0]
                for n in old_nums:
                    if len(n) >= 2:
                        c_addr.update(idx_addr_old.get(f"{n}_{tok_p}", []))

            c_key_a = set()
            for k in get_key_a(clean_toks, nums):
                c_key_a.update(idx_key_a.get(k, []))

            c_key_b = set()
            for k in get_key_b(canon_name, nums):
                c_key_b.update(idx_key_b.get(k, []))

            c_key_c = set()
            for k in get_key_c(clean_toks, alphas):
                c_key_c.update(idx_key_c.get(k, []))

            c_key_d = set()
            for k in get_key_d(canon_name, nums, alphas):
                c_key_d.update(idx_key_d.get(k, []))

            c_existing = c_exact | c_canon | c_addr
            c_ext_a = c_existing | c_key_a
            c_ext_b = c_existing | c_key_b
            c_ext_c = c_existing | c_key_c
            c_ext_d = c_existing | c_key_d
            c_ext_all = c_existing | c_key_a | c_key_b | c_key_c | c_key_d

            strat_map = {
                "exact": c_exact,
                "canon": c_canon,
                "addr": c_addr,
                "key_a": c_key_a,
                "key_b": c_key_b,
                "key_c": c_key_c,
                "key_d": c_key_d,
                "existing": c_existing,
                "existing_plus_a": c_ext_a,
                "existing_plus_b": c_ext_b,
                "existing_plus_c": c_ext_c,
                "existing_plus_d": c_ext_d,
                "existing_plus_all": c_ext_all,
            }

            for sk, cset in strat_map.items():
                c_eids = {cand_ids[idx] for idx in cset}
                found_gt = gt.intersection(c_eids)
                results[sk]["counts"].append(len(cset))
                for m in found_gt:
                    results[sk]["recovered"].add((sid, m))

        # Store country results
        country_partitions[country]["results"] = results

    # Global aggregation
    print("\n" + "=" * 80)
    print("COMPOSITE BLOCKING KEYS EVALUATION (20 S1 SAMPLE)")
    print("=" * 80)

    strat_display = [
        ("Exact Name", "exact"),
        ("Canonical Name", "canon"),
        ("Address-Assisted", "addr"),
        ("Key A (1st tok + Postal/Num)", "key_a"),
        ("Key B (4-char prefix + Postal/Num)", "key_b"),
        ("Key C (1st tok + Salient Addr)", "key_c"),
        ("Key D (3-char prefix + Salient/Postal)", "key_d"),
        ("Existing (Exact ∪ Canon ∪ Addr)", "existing"),
        ("Existing + Key A", "existing_plus_a"),
        ("Existing + Key B", "existing_plus_b"),
        ("Existing + Key C", "existing_plus_c"),
        ("Existing + Key D", "existing_plus_d"),
        ("Existing + (A ∪ B ∪ C ∪ D)", "existing_plus_all"),
    ]

    print(f"\n{'Strategy':<42} | {'Recall':<8} | {'Total Pairs':<11} | {'Avg/S1':<8} | {'Median':<6} | {'P95':<6} | {'P99':<6} | {'Max':<6}")
    print("-" * 110)

    for label, sk in strat_display:
        all_counts = []
        all_recovered = set()
        for country in unique_countries:
            res = country_partitions[country]["results"][sk]
            all_counts.extend(res["counts"])
            all_recovered.update(res["recovered"])
        
        arr = np.array(all_counts)
        tot_cands = int(arr.sum())
        avg_c = float(arr.mean())
        med = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mx = int(arr.max())
        rec = (len(all_recovered) / total_gt_pairs) * 100.0 if total_gt_pairs > 0 else 0.0

        print(f"{label:<42} | {rec:6.2f}% | {tot_cands:<11,d} | {avg_c:8.2f} | {med:6.1f} | {p95:6.1f} | {p99:6.1f} | {mx:6d}")

    # Detailed link recovery analysis
    existing_rec = set()
    for country in unique_countries:
        existing_rec.update(country_partitions[country]["results"]["existing"]["recovered"])

    all_gt_set = set()
    for sid in sample_s1:
        for m in val_gt[sid]:
            all_gt_set.add((sid, m))

    missed_by_existing = all_gt_set - existing_rec
    print("\n" + "=" * 80)
    print("MARGINAL GROUND TRUTH LINK RECOVERY ANALYSIS")
    print("=" * 80)
    print(f"Total Ground-Truth Links in Sample: {len(all_gt_set)}")
    print(f"Recovered by Existing (Exact ∪ Canon ∪ Addr): {len(existing_rec)} ({len(existing_rec)/len(all_gt_set)*100:.2f}%)")
    print(f"Missed by Existing: {len(missed_by_existing)} ({len(missed_by_existing)/len(all_gt_set)*100:.2f}%)")
    print("-" * 80)

    for label, sk in [
        ("Key A (1st tok + Postal/Num)", "key_a"),
        ("Key B (4-char prefix + Postal/Num)", "key_b"),
        ("Key C (1st tok + Salient Addr)", "key_c"),
        ("Key D (3-char prefix + Salient/Postal)", "key_d"),
        ("Combined Composite (A ∪ B ∪ C ∪ D)", "existing_plus_all"),
    ]:
        rec_set = set()
        for country in unique_countries:
            rec_set.update(country_partitions[country]["results"][sk]["recovered"])
        
        if sk == "existing_plus_all":
            recovered_from_missed = rec_set - existing_rec
            total_after = len(existing_rec | rec_set)
        else:
            recovered_from_missed = rec_set.intersection(missed_by_existing)
            total_after = len(existing_rec | rec_set)
        
        print(f"{label:<45}: Recovered {len(recovered_from_missed):2d} missed GT links -> New Total: {total_after:2d}/{len(all_gt_set)} ({total_after/len(all_gt_set)*100:.2f}%)")


if __name__ == "__main__":
    main()
