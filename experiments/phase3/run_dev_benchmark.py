#!/usr/bin/env python3
"""
experiments/phase3/run_dev_benchmark.py

Development benchmark for Amazon ML Challenge 2026 Entity Resolution.

Creates a fixed, reusable 1,500-S1 development set from the existing
validation split and runs blocking + matching evaluation with
country-partitioned, reusable inverted indices.

Key design choices:
  - Entity IDs stored as uint32 (bit31=source, bits0-30=numeric ID):
    ~4 bytes/record vs ~70 bytes Python string -> ~17x memory saving.
  - Index is built ONCE per country, queried for ALL S1 entities in
    that partition. No per-S1 index rebuilds.
  - 2-pass per country: Pass 1 = index + blocking eval, Pass 2 = text
    fetch for top-K candidates, then matching eval.
  - Correctness check against known 20-S1 diagnostic values runs first.
    If any check fails the script exits(1) before the dev run.

Blocking configs (identical to run_matcher_diagnostics.py):
  CFG1 = Exact | Canonical | TwoToken | AddrOld | KeyB   [primary]
  CFG2 = CFG1 | KeyC
  CFG3 = CFG2 | KeyD

Usage:
  python experiments/run_dev_benchmark.py
  python experiments/run_dev_benchmark.py --skip-check
  python experiments/run_dev_benchmark.py --sample-size 500 --seed 99
  python experiments/run_dev_benchmark.py --budget 50 --threshold 0.65
  python experiments/run_dev_benchmark.py --resample   # force new sample

Output:
  experiments/dev_s1_ids.txt           (saved once, reused on subsequent runs)
  results/phase3/dev_benchmark_results.json
"""

import argparse
import array
import collections
import gc
import json
import os
import resource
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath("."))
from src.normalization import (
    normalize_and_canonicalize,
    normalize_address,
    extract_numeric_tokens,
)
from src.baseline_matcher import compute_pair_features, compute_deterministic_score
from src.evaluation import compute_entity_metrics

# ── File paths ──────────────────────────────────────────────────────────────────
DATA_DIR        = "dataset"
EXPERIMENTS_DIR = "experiments"
S1_FILE  = os.path.join(DATA_DIR, "train", "train_source1.tsv")
S2_FILE  = os.path.join(DATA_DIR, "train", "train_source2.tsv")
S3_FILE  = os.path.join(DATA_DIR, "train", "train_source3.tsv")
GT_FILE  = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
VAL_IDS_FILE = os.path.join(EXPERIMENTS_DIR, "val_s1_ids.txt")
DEV_IDS_FILE = os.path.join(EXPERIMENTS_DIR, "dev_s1_ids.txt")

# ── Correctness check: known values from matcher_diagnostic_20s1.md ─────────────
# Produced by run_matcher_diagnostics.py / ablate_composite_keys.py on the
# 20-S1 sample at budget=25, threshold=0.70, conservative_precision.
CHECK_BUDGET    = 25
CHECK_THRESHOLD = 0.70
CHECK_CONFIG    = "conservative_precision"
KNOWN = {
    "total_gt_links": 51,
    "cfg1_recovered": 33,
    "cfg2_recovered": 37,
    "cfg3_recovered": 38,
    "agg_tp":         25,
    "agg_fp":         10,
    "macro_f0_5":     0.5397,
}
F05_TOL = 0.0005   # +/- 0.05% tolerance for floating-point F0.5 comparison

# ── Blocking signal weights (identical to run_matcher_diagnostics.py) ────────────
W_EXACT = 16
W_CANON =  8   # covers canonical key + two-token prefix
W_ADDR  =  4   # address-numeric (old blocker)
W_KEY_B =  2
W_KEY_C =  1
W_KEY_D =  1

# ── Generic address terms excluded from salient-token extraction ─────────────────
GENERIC_ADDR_TERMS = {
    "road", "street", "avenue", "drive", "lane", "highway", "court",
    "circle", "parkway", "terrace", "place", "square", "way", "boulevard",
    "apartment", "suite", "floor", "building", "number", "opposite", "near",
    "adjacent", "sector", "phase", "khasra", "rue", "route", "post", "box",
    "pobox", "st", "rd", "ave", "dr", "ln", "blvd", "apt", "ste", "bldg", "no",
}


# ==============================================================================
# UINT32 ID ENCODING
#
# Entity IDs: "S2-NNNNNNNNN" or "S3-NNNNNNNNN" (9-digit numeric part).
# Encoding: uint32 where bit 31 = source (0=S2, 1=S3), bits 0-30 = numeric.
# Max numeric = 999_999_999 < 2^30 = 1_073_741_824 -> fits safely.
# Memory: ~4 bytes/record vs ~70 bytes (Python str + list slot) -> ~17x saving.
# At 10.3M records split into two country passes: peak ~25 MB vs ~430 MB.
# ==============================================================================

def encode_id(eid: str) -> int:
    """Encode 'S2-NNNNNNNNN' or 'S3-NNNNNNNNN' to a uint32 integer."""
    source_bit = 0 if eid[1] == "2" else 1
    return (source_bit << 31) | int(eid[3:])


def decode_id(enc: int) -> str:
    """Decode a uint32 back to the entity ID string."""
    src = "2" if (enc >> 31) == 0 else "3"
    return f"S{src}-{enc & 0x7FFFFFFF}"


# ==============================================================================
# ADDRESS FEATURE EXTRACTION -- identical to run_matcher_diagnostics.py
# ==============================================================================

def extract_address_features(addr: str) -> Tuple[List[str], List[str]]:
    """
    Return (ordered_nums, salient_alphas):
      ordered_nums   -- numeric tokens >=2 digits, postal/PIN codes first
      salient_alphas -- alphabetic tokens >=4 chars, excluding generic terms
    """
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


# ==============================================================================
# COMPOSITE BLOCKING KEY GENERATORS -- identical to run_matcher_diagnostics.py
# ==============================================================================

def get_key_b(canon: str, nums: List[str]) -> Set[str]:
    """Key B: canon[:4] + '#' + numeric anchor."""
    if len(canon) < 4 or not nums:
        return set()
    p4 = canon[:4]
    return {f"{p4}#{n}" for n in nums[:2]}


def get_key_c(clean_toks: List[str], alphas: List[str]) -> Set[str]:
    """Key C: first significant name token + '@' + salient address alpha."""
    if not clean_toks or not alphas:
        return set()
    return {f"{clean_toks[0]}@{a}" for a in alphas[:3]}


def get_key_d(canon: str, nums: List[str], alphas: List[str]) -> Set[str]:
    """Key D: canon[:3] + '#' + numeric  OR  canon[:3] + '@' + salient alpha."""
    if len(canon) < 3:
        return set()
    p3   = canon[:3]
    keys = {f"{p3}#{n}" for n in nums[:2]}
    keys.update({f"{p3}@{a}" for a in alphas[:2]})
    return keys


# ==============================================================================
# MEMORY HELPERS
# ==============================================================================

def rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ==============================================================================
# MATCH-COUNT BIN -- identical to src/split_data.py
# ==============================================================================

def get_match_bin(n: int) -> str:
    if n == 0: return "0_singleton"
    if n == 1: return "1_match"
    if n == 2: return "2_matches"
    if n == 3: return "3_matches"
    if n == 4: return "4_matches"
    return "5_plus_matches"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_val_ids() -> Set[str]:
    with open(VAL_IDS_FILE, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def load_s1_records(target_ids: Set[str]) -> Dict[str, dict]:
    """Stream S1 file; collect name/address/country for target_ids only."""
    records: Dict[str, dict] = {}
    with open(S1_FILE, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                sid = parts[0].strip()
                if sid in target_ids:
                    records[sid] = {
                        "name":    parts[1].strip(),
                        "address": parts[2].strip(),
                        "country": parts[3].strip(),
                    }
                    if len(records) == len(target_ids):
                        break
    return records


def load_gt_for_ids(target_ids: Set[str]) -> Dict[str, Set[str]]:
    """
    Stream GT file; collect ground-truth sets for target_ids.
    Singletons (empty matched_entity_ids) get gt[sid] = set().
    IDs absent from the GT file also get set() (defensive fallback).
    """
    gt: Dict[str, Set[str]] = {}
    with open(GT_FILE, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0].strip()
            if sid in target_ids:
                if len(parts) > 1 and parts[1].strip():
                    gt[sid] = {m.strip() for m in parts[1].split(",") if m.strip()}
                else:
                    gt[sid] = set()
                if len(gt) == len(target_ids):
                    break
    # Fallback for any IDs not found in the GT file
    for sid in target_ids:
        if sid not in gt:
            gt[sid] = set()
    return gt


# ==============================================================================
# 20-S1 CHECK SAMPLE SELECTION
# Replicates run_matcher_diagnostics.py EXACTLY:
#   scan S1_FILE in file order, take first 10 India + first 10 US in val set.
# ==============================================================================

def load_check_sample_ids() -> Set[str]:
    val_s1_ids = load_val_ids()
    country_counts: Dict[str, int] = collections.defaultdict(int)
    selected: List[str] = []

    with open(S1_FILE, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                sid     = parts[0].strip()
                country = parts[3].strip()
                if sid in val_s1_ids and country_counts[country] < 10:
                    selected.append(sid)
                    country_counts[country] += 1
            if all(country_counts.get(c, 0) >= 10 for c in ["India", "US"]):
                break

    return set(selected)


# ==============================================================================
# STRATIFIED DEV SAMPLE
# ==============================================================================

def select_dev_sample(n: int = 1500, seed: int = 42) -> List[str]:
    """
    Select n S1 entities from the validation split, stratified by
    (country, match_count_bin). Each stratum is sampled proportional to
    its share of total validation entities. Returns a sorted list of IDs.
    """
    print("  Sampling: loading val IDs...")
    val_s1_ids = load_val_ids()
    total_val  = len(val_s1_ids)

    print(f"  Sampling: loading S1 country labels ({total_val:,} val entities)...")
    id_country: Dict[str, str] = {}
    with open(S1_FILE, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                sid = parts[0].strip()
                if sid in val_s1_ids:
                    id_country[sid] = parts[3].strip()
                    if len(id_country) == total_val:
                        break

    print("  Sampling: loading match counts from ground truth...")
    id_match_count: Dict[str, int] = collections.defaultdict(int)
    with open(GT_FILE, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0].strip()
            if sid in val_s1_ids:
                if len(parts) > 1 and parts[1].strip():
                    id_match_count[sid] = len(
                        [m for m in parts[1].split(",") if m.strip()]
                    )

    # Build strata; sort each list for determinism before sampling
    strata: Dict[str, List[str]] = collections.defaultdict(list)
    for sid in val_s1_ids:
        country = id_country.get(sid, "Unknown")
        m_bin   = get_match_bin(id_match_count.get(sid, 0))
        strata[f"{country}__{m_bin}"].append(sid)
    for k in strata:
        strata[k].sort()

    rng      = np.random.RandomState(seed)
    selected: List[str] = []

    print(f"  Sampling: selecting {n} entities across {len(strata)} strata...")
    for stratum_key, ids in sorted(strata.items()):
        stratum_n = max(1, round(n * len(ids) / total_val))
        stratum_n = min(stratum_n, len(ids))
        chosen_idx = rng.choice(len(ids), size=stratum_n, replace=False)
        selected.extend(ids[i] for i in sorted(chosen_idx))

    selected.sort()
    print(f"  Sampling: selected {len(selected)} entities")
    return selected


# ==============================================================================
# INDEX CONSTRUCTION -- one call per country; indices reused for all S1 queries
# ==============================================================================

def build_country_index(country: str):
    """
    Build 7 inverted indices for all S2+S3 records of `country`.

    cand_ids: array.array('I') of uint32-encoded entity IDs.
              Lookup: cand_ids[rec_idx] -> decode_id() -> entity ID string.
              Memory: ~4 bytes/record vs ~70 bytes Python str -> ~17x saving.

    The 7 index dicts map blocking key string -> array.array('I') of rec indices.
    This is the same data structure as run_matcher_diagnostics.py.

    Returns:
        (cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old,
         idx_key_b, idx_key_c, idx_key_d)
    """
    cand_ids = array.array("I")   # uint32-encoded entity IDs

    def mk():
        return collections.defaultdict(lambda: array.array("I"))

    idx_exact    = mk()
    idx_canon    = mk()
    idx_two_tok  = mk()
    idx_addr_old = mk()
    idx_key_b    = mk()
    idx_key_c    = mk()
    idx_key_d    = mk()

    for fpath in (S2_FILE, S3_FILE):
        with open(fpath, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3].strip() == country:
                    eid  = parts[0].strip()
                    name = parts[1].strip()
                    addr = parts[2].strip()

                    rec_idx = len(cand_ids)
                    cand_ids.append(encode_id(eid))

                    norm_name, canon_name, clean_toks = normalize_and_canonicalize(name)
                    nums, alphas = extract_address_features(addr)

                    if norm_name:
                        idx_exact[norm_name].append(rec_idx)
                    if canon_name and canon_name != norm_name:
                        idx_canon[canon_name].append(rec_idx)
                    if len(clean_toks) >= 2:
                        idx_two_tok[f"{clean_toks[0]} {clean_toks[1]}"].append(rec_idx)
                    if addr and clean_toks:
                        old_nums = extract_numeric_tokens(addr)
                        tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4
                                 else clean_toks[0])
                        for n in old_nums:
                            if len(n) >= 2:
                                idx_addr_old[f"{n}_{tok_p}"].append(rec_idx)
                    for k in get_key_b(canon_name, nums):
                        idx_key_b[k].append(rec_idx)
                    for k in get_key_c(clean_toks, alphas):
                        idx_key_c[k].append(rec_idx)
                    for k in get_key_d(canon_name, nums, alphas):
                        idx_key_d[k].append(rec_idx)

    return (cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old,
            idx_key_b, idx_key_c, idx_key_d)


# ==============================================================================
# S1 QUERY -- identical logic to run_matcher_diagnostics.py
# ==============================================================================

def query_s1(s1: dict, budget: int,
             cand_ids: array.array,
             idx_exact, idx_canon, idx_two_tok, idx_addr_old,
             idx_key_b, idx_key_c, idx_key_d):
    """
    Query all 7 indices for one S1 record.

    Returns:
        cfg1, cfg2, cfg3   -- sets of integer record indices
        ranked_budget      -- list of top-`budget` record indices from cfg3,
                              sorted: weight desc, then decode_id(cand_ids[i]) asc.
                              The string-based tie-break is identical to the old
                              code's cand_ids[idx_val] string sort for 9-digit IDs.
        weights            -- dict {record_index: signal_weight}
    """
    norm_name, canon_name, clean_toks = normalize_and_canonicalize(s1["name"])
    nums, alphas = extract_address_features(s1["address"])

    c_exact = set(idx_exact.get(norm_name, ()))

    c_canon = set(idx_canon.get(canon_name, ()))
    if len(clean_toks) >= 2:
        c_canon.update(idx_two_tok.get(f"{clean_toks[0]} {clean_toks[1]}", ()))

    c_addr: Set[int] = set()
    if s1["address"] and clean_toks:
        old_nums = extract_numeric_tokens(s1["address"])
        tok_p = (clean_toks[0][:4] if len(clean_toks[0]) >= 4 else clean_toks[0])
        for n in old_nums:
            if len(n) >= 2:
                c_addr.update(idx_addr_old.get(f"{n}_{tok_p}", ()))

    c_key_b: Set[int] = set()
    for k in get_key_b(canon_name, nums):
        c_key_b.update(idx_key_b.get(k, ()))

    c_key_c: Set[int] = set()
    for k in get_key_c(clean_toks, alphas):
        c_key_c.update(idx_key_c.get(k, ()))

    c_key_d: Set[int] = set()
    for k in get_key_d(canon_name, nums, alphas):
        c_key_d.update(idx_key_d.get(k, ()))

    cfg1 = c_exact | c_canon | c_addr | c_key_b
    cfg2 = cfg1 | c_key_c
    cfg3 = cfg2 | c_key_d

    # Compute blocking signal weight for each candidate in cfg3
    weights: Dict[int, int] = {}
    for i in cfg3:
        w = 0
        if i in c_exact:  w += W_EXACT
        if i in c_canon:  w += W_CANON
        if i in c_addr:   w += W_ADDR
        if i in c_key_b:  w += W_KEY_B
        if i in c_key_c:  w += W_KEY_C
        if i in c_key_d:  w += W_KEY_D
        weights[i] = w

    # Sort: weight desc, then entity ID string asc (tie-break, matches old code)
    ranked_all    = sorted(cfg3, key=lambda i: (-weights[i], decode_id(cand_ids[i])))
    ranked_budget = ranked_all[:budget]

    return cfg1, cfg2, cfg3, ranked_budget, weights


# ==============================================================================
# MAIN EVALUATION LOOP
# ==============================================================================

def run_evaluation(sample_s1: Dict[str, dict],
                   val_gt:    Dict[str, Set[str]],
                   budget:    int,
                   threshold: float,
                   matcher_config: str,
                   label: str = "EVAL") -> dict:
    """
    Run blocking + matching evaluation on sample_s1.
    Processes one country partition at a time; releases index memory between them.
    Returns a result dict with all required metrics.
    """
    t_start          = time.time()
    unique_countries = sorted({r["country"] for r in sample_s1.values()})

    # Accumulators (accumulate across all country partitions)
    per_entity: Dict[str, dict] = {}
    cfg1_counts: List[int] = []
    cfg2_counts: List[int] = []
    cfg3_counts: List[int] = []
    cfg1_total_gt  = 0
    cfg1_recovered = 0
    cfg2_recovered = 0
    cfg3_recovered = 0

    for country in unique_countries:
        country_s1 = {sid: r for sid, r in sample_s1.items()
                      if r["country"] == country}
        n_s1 = len(country_s1)
        print(f"\n  [{label}] -- {country}: {n_s1} S1 entities --")

        # ---- Index build --------------------------------------------------------
        t0 = time.time()
        (cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old,
         idx_key_b, idx_key_c, idx_key_d) = build_country_index(country)
        n_indexed = len(cand_ids)
        t_idx = time.time() - t0
        cand_ids_mb = n_indexed * 4 / 1024 / 1024
        print(f"  [{label}] Indexed {n_indexed:,} records in {t_idx:.1f}s"
              f" | cand_ids={cand_ids_mb:.1f} MB"
              f" | RSS={rss_mb():.0f} MB  peak={peak_rss_mb():.0f} MB")

        # ---- Pass 1: query all S1 in this country → blocking stats -------------
        pass1: Dict[str, dict] = {}       # sid -> {gt_set, budget_eids}
        needed_encoded: Set[int] = set()  # encoded IDs to fetch text for

        for sid, s1 in country_s1.items():
            gt_set = val_gt[sid]
            cfg1, cfg2, cfg3, ranked_budget, _ = query_s1(
                s1, budget, cand_ids,
                idx_exact, idx_canon, idx_two_tok, idx_addr_old,
                idx_key_b, idx_key_c, idx_key_d,
            )

            # Decode to entity ID strings for GT intersection (blocking recall)
            cfg1_eids = {decode_id(cand_ids[i]) for i in cfg1}
            cfg2_eids = {decode_id(cand_ids[i]) for i in cfg2}
            cfg3_eids = {decode_id(cand_ids[i]) for i in cfg3}

            cfg1_total_gt  += len(gt_set)
            cfg1_recovered += len(gt_set & cfg1_eids)
            cfg2_recovered += len(gt_set & cfg2_eids)
            cfg3_recovered += len(gt_set & cfg3_eids)

            cfg1_counts.append(len(cfg1))
            cfg2_counts.append(len(cfg2))
            cfg3_counts.append(len(cfg3))

            # Decode budget candidates; collect encoded IDs for text fetch
            budget_eids: List[str] = []
            for i in ranked_budget:
                enc = cand_ids[i]
                needed_encoded.add(enc)
                budget_eids.append(decode_id(enc))

            pass1[sid] = {
                "gt_set":      gt_set,
                "budget_eids": budget_eids,   # ordered list, length <= budget
            }

        # ---- Free index memory before Pass 2 -----------------------------------
        n_needed = len(needed_encoded)
        needed_eids_str: Set[str] = {decode_id(enc) for enc in needed_encoded}
        del cand_ids, idx_exact, idx_canon, idx_two_tok, idx_addr_old
        del idx_key_b, idx_key_c, idx_key_d, needed_encoded
        gc.collect()
        print(f"  [{label}] Pass 1 done. Fetching text for {n_needed:,} candidates"
              f" | RSS={rss_mb():.0f} MB  peak={peak_rss_mb():.0f} MB")

        # ---- Pass 2: fetch candidate text from disk ----------------------------
        # Scan S2 then S3 sequentially. Use early exit once all needed IDs found.
        # No country filter: all needed IDs came from this country's index.
        cand_meta: Dict[str, dict] = {}
        for fpath in (S2_FILE, S3_FILE):
            with open(fpath, encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4:
                        eid = parts[0].strip()
                        if eid in needed_eids_str:
                            cand_meta[eid] = {
                                "name":    parts[1].strip(),
                                "address": parts[2].strip(),
                            }
                    if len(cand_meta) >= len(needed_eids_str):
                        break   # all needed IDs found; stop scanning this file

        del needed_eids_str
        print(f"  [{label}] Pass 2 done. Loaded {len(cand_meta):,} candidate texts"
              f" | RSS={rss_mb():.0f} MB  peak={peak_rss_mb():.0f} MB")

        # ---- Matching evaluation -----------------------------------------------
        for sid, res in pass1.items():
            s1       = country_s1[sid]
            gt_set   = res["gt_set"]
            pred_set: Set[str] = set()

            for cid in res["budget_eids"]:
                cm = cand_meta.get(cid)
                if cm is None:
                    continue   # defensive; should not happen
                feats = compute_pair_features(
                    s1["name"], s1["address"], country,
                    cm["name"], cm["address"], country,
                )
                score = compute_deterministic_score(feats, matcher_config)
                if score >= threshold:
                    pred_set.add(cid)

            per_entity[sid] = compute_entity_metrics(gt_set, pred_set)

        del pass1, cand_meta
        gc.collect()
        print(f"  [{label}] {country} complete"
              f" | RSS={rss_mb():.0f} MB  peak={peak_rss_mb():.0f} MB")

    # ---- Aggregate results -----------------------------------------------------
    t_total = time.time() - t_start

    def blocking_stats(counts: List[int], recovered: int, total_gt: int) -> dict:
        arr = np.array(counts, dtype=np.int64)
        return {
            "total_candidates": int(arr.sum()),
            "avg_per_s1":       round(float(arr.mean()), 2),
            "median_per_s1":    round(float(np.median(arr)), 1),
            "p95_per_s1":       round(float(np.percentile(arr, 95)), 1),
            "p99_per_s1":       round(float(np.percentile(arr, 99)), 1),
            "max_per_s1":       int(arr.max()),
            "total_gt_links":   total_gt,
            "recovered":        recovered,
            "blocking_recall":  round(recovered / total_gt, 6) if total_gt > 0 else 0.0,
            "missed":           total_gt - recovered,
        }

    f05_arr  = np.array([m["f0_5"]      for m in per_entity.values()])
    prec_arr = np.array([m["precision"] for m in per_entity.values()])
    rec_arr  = np.array([m["recall"]    for m in per_entity.values()])

    agg_tp = sum(m["tp"] for m in per_entity.values())
    agg_fp = sum(m["fp"] for m in per_entity.values())
    agg_fn = sum(m["fn"] for m in per_entity.values())

    singleton_total   = sum(1 for m in per_entity.values() if m["is_singleton"])
    singleton_correct = sum(
        1 for m in per_entity.values() if m.get("singleton_correct", False)
    )
    non_singleton_f05 = [
        m["f0_5"] for m in per_entity.values() if not m["is_singleton"]
    ]

    country_dist = dict(collections.Counter(
        r["country"] for r in sample_s1.values()
    ))
    bin_dist = dict(collections.Counter(
        get_match_bin(len(val_gt.get(sid, set()))) for sid in sample_s1
    ))

    return {
        "sample_info": {
            "n_entities":             len(sample_s1),
            "country_distribution":   country_dist,
            "match_bin_distribution": dict(sorted(bin_dist.items())),
            "singleton_count":        singleton_total,
        },
        "blocking_cfg1": blocking_stats(cfg1_counts, cfg1_recovered, cfg1_total_gt),
        "blocking_cfg2": blocking_stats(cfg2_counts, cfg2_recovered, cfg1_total_gt),
        "blocking_cfg3": blocking_stats(cfg3_counts, cfg3_recovered, cfg1_total_gt),
        "matching": {
            "config": {
                "budget":         budget,
                "threshold":      threshold,
                "matcher_config": matcher_config,
            },
            "macro_f0_5":           round(float(f05_arr.mean()), 6),
            "macro_precision":      round(float(prec_arr.mean()), 6),
            "macro_recall":         round(float(rec_arr.mean()), 6),
            "aggregate_tp":         agg_tp,
            "aggregate_fp":         agg_fp,
            "aggregate_fn":         agg_fn,
            "singleton_count":      singleton_total,
            "singleton_correct":    singleton_correct,
            "singleton_accuracy":   round(
                singleton_correct / singleton_total, 6
            ) if singleton_total > 0 else 1.0,
            "non_singleton_macro_f0_5": round(
                float(np.mean(non_singleton_f05)), 6
            ) if non_singleton_f05 else 0.0,
            "per_entity_f0_5_distribution": {
                "min": round(float(f05_arr.min()), 4),
                "p25": round(float(np.percentile(f05_arr, 25)), 4),
                "p50": round(float(np.percentile(f05_arr, 50)), 4),
                "p75": round(float(np.percentile(f05_arr, 75)), 4),
                "p95": round(float(np.percentile(f05_arr, 95)), 4),
                "max": round(float(f05_arr.max()), 4),
            },
        },
        "runtime": {
            "total_seconds": round(t_total, 1),
            "peak_rss_mb":   round(peak_rss_mb(), 1),
        },
    }


# ==============================================================================
# CORRECTNESS CHECK
# ==============================================================================

def run_correctness_check() -> bool:
    """
    Run evaluation on the exact 20-S1 sample used by run_matcher_diagnostics.py
    at budget=25, threshold=0.70, conservative_precision, and verify the results
    against known values from matcher_diagnostic_20s1.md.

    Returns True iff all checks pass. Exits(1) on failure (caller should not
    proceed to the dev benchmark if this returns False).
    """
    print("\n" + "=" * 62)
    print("CORRECTNESS CHECK")
    print(f"  Parameters: budget={CHECK_BUDGET}  threshold={CHECK_THRESHOLD}"
          f"  config={CHECK_CONFIG}")
    print("  Replicating run_matcher_diagnostics.py on 20-S1 sample")
    print("=" * 62)

    check_ids = load_check_sample_ids()
    sample_s1 = load_s1_records(check_ids)
    val_gt    = load_gt_for_ids(check_ids)

    total_gt     = sum(len(v) for v in val_gt.values())
    country_dist = collections.Counter(r["country"] for r in sample_s1.values())
    print(f"  Sample: {len(sample_s1)} entities"
          + "".join(f"  {c}={n}" for c, n in sorted(country_dist.items()))
          + f"  total_GT_links={total_gt}")

    result = run_evaluation(
        sample_s1, val_gt,
        budget=CHECK_BUDGET,
        threshold=CHECK_THRESHOLD,
        matcher_config=CHECK_CONFIG,
        label="CHECK",
    )

    b1 = result["blocking_cfg1"]
    b2 = result["blocking_cfg2"]
    b3 = result["blocking_cfg3"]
    m  = result["matching"]

    # (label, actual, expected, float_tol_or_None)
    checks = [
        ("total_gt_links", b1["total_gt_links"], KNOWN["total_gt_links"], None),
        ("cfg1_recovered", b1["recovered"],      KNOWN["cfg1_recovered"], None),
        ("cfg2_recovered", b2["recovered"],      KNOWN["cfg2_recovered"], None),
        ("cfg3_recovered", b3["recovered"],      KNOWN["cfg3_recovered"], None),
        ("aggregate_tp",   m["aggregate_tp"],    KNOWN["agg_tp"],         None),
        ("aggregate_fp",   m["aggregate_fp"],    KNOWN["agg_fp"],         None),
        ("macro_f0_5",     m["macro_f0_5"],      KNOWN["macro_f0_5"],     F05_TOL),
    ]

    all_ok = True
    print("\n  +----------------------------------------------------------+")
    print(  "  | Metric                  Expected   Actual     Status    |")
    print(  "  +----------------------------------------------------------+")
    for name, actual, expected, tol in checks:
        if tol is None:
            ok      = (actual == expected)
            exp_str = f"{expected:>8d}"
            act_str = f"{actual:>8d}"
        else:
            ok      = abs(actual - expected) <= tol
            exp_str = f"{expected:>8.4f}"
            act_str = f"{actual:>8.4f}"
        status = "PASS" if ok else "FAIL ***"
        if not ok:
            all_ok = False
        print(f"  | {name:<22}  {exp_str}   {act_str}    {status:8s}|")
    print("  +----------------------------------------------------------+")

    if all_ok:
        print("\n  CORRECTNESS CHECK PASSED -- proceeding to dev benchmark.\n")
    else:
        print("\n  CORRECTNESS CHECK FAILED.")
        print("  The new implementation disagrees with known 20-S1 values.")
        print("  Investigate before running the full dev benchmark.\n")

    return all_ok


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Amazon ML 2026 -- Development Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-check",  action="store_true",
                   help="Skip 20-S1 correctness check (not recommended)")
    p.add_argument("--sample-size", type=int,   default=1500,
                   help="Number of S1 entities in dev sample")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for stratified sampling")
    p.add_argument("--budget",      type=int,   default=25,
                   help="Candidate budget per S1 for matching stage")
    p.add_argument("--threshold",   type=float, default=0.70,
                   help="Matcher score threshold")
    p.add_argument("--config",      type=str,   default="conservative_precision",
                   help="Matcher scoring config name")
    p.add_argument("--output",      type=str,
                   default="results/phase3/dev_benchmark_results.json",
                   help="Output JSON path")
    p.add_argument("--no-save-ids", action="store_true",
                   help="Do not write dev IDs to experiments/dev_s1_ids.txt")
    p.add_argument("--resample",    action="store_true",
                   help="Ignore existing dev_s1_ids.txt and resample")
    return p.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args    = parse_args()
    t_wall  = time.time()

    print("=" * 62)
    print("Amazon ML 2026 -- Development Benchmark")
    print("=" * 62)
    print(f"  sample_size : {args.sample_size}")
    print(f"  seed        : {args.seed}")
    print(f"  budget      : {args.budget}")
    print(f"  threshold   : {args.threshold}")
    print(f"  config      : {args.config}")
    print(f"  skip_check  : {args.skip_check}")
    print()

    # ---- Step 1: Correctness check --------------------------------------------
    if not args.skip_check:
        passed = run_correctness_check()
        if not passed:
            sys.exit(1)
    else:
        print("  [WARNING] Correctness check skipped (--skip-check).\n")

    # ---- Step 2: Select or load dev sample ------------------------------------
    print("=" * 62)
    print("DEV SAMPLE")
    print("=" * 62)

    ids_exist = os.path.exists(DEV_IDS_FILE)
    if ids_exist and not args.resample:
        print(f"  Loading existing IDs from {DEV_IDS_FILE}")
        with open(DEV_IDS_FILE, encoding="utf-8") as f:
            dev_ids = [ln.strip() for ln in f if ln.strip()]
        if len(dev_ids) != args.sample_size:
            print(f"  WARNING: file has {len(dev_ids)} IDs but --sample-size="
                  f"{args.sample_size}. Re-sampling.")
            dev_ids = select_dev_sample(args.sample_size, args.seed)
        else:
            print(f"  Loaded {len(dev_ids)} dev IDs")
    else:
        if args.resample and ids_exist:
            print(f"  --resample: ignoring existing {DEV_IDS_FILE}")
        dev_ids = select_dev_sample(args.sample_size, args.seed)

    if not args.no_save_ids:
        with open(DEV_IDS_FILE, "w", encoding="utf-8") as f:
            f.writelines(sid + "\n" for sid in dev_ids)
        print(f"  Saved {len(dev_ids)} IDs -> {DEV_IDS_FILE}")

    # ---- Step 3: Load S1 records and GT ---------------------------------------
    dev_ids_set = set(dev_ids)
    print(f"\n  Loading records for {len(dev_ids_set)} dev entities...")
    sample_s1 = load_s1_records(dev_ids_set)
    val_gt    = load_gt_for_ids(dev_ids_set)

    missing = dev_ids_set - set(sample_s1)
    if missing:
        print(f"  WARNING: {len(missing)} dev IDs not found in S1 file.")

    total_gt     = sum(len(v) for v in val_gt.values())
    country_dist = collections.Counter(r["country"] for r in sample_s1.values())
    print(f"  {len(sample_s1)} records | GT links: {total_gt}")
    for c, n in sorted(country_dist.items()):
        print(f"    {c}: {n}")

    # ---- Step 4: Run dev benchmark --------------------------------------------
    print("\n" + "=" * 62)
    print("DEV BENCHMARK")
    print("=" * 62)

    result = run_evaluation(
        sample_s1, val_gt,
        budget=args.budget,
        threshold=args.threshold,
        matcher_config=args.config,
        label="DEV",
    )

    # ---- Step 5: Print summary ------------------------------------------------
    si = result["sample_info"]
    b1 = result["blocking_cfg1"]
    b2 = result["blocking_cfg2"]
    b3 = result["blocking_cfg3"]
    m  = result["matching"]
    rt = result["runtime"]

    print("\n" + "=" * 62)
    print("RESULTS SUMMARY")
    print("=" * 62)

    print(f"\n  Sample    : {si['n_entities']} entities"
          + " | " + " | ".join(
              f"{c}: {n}" for c, n in sorted(si["country_distribution"].items())))
    print(f"  Singletons: {si['singleton_count']}")
    print(f"  GT links  : {b1['total_gt_links']}")

    print(f"\n  BLOCKING  (candidate counts before budget cut)")
    print(f"  {'Config':<6}  {'Recall':>8}  {'Recov':>7}  {'Missed':>7}"
          f"  {'Avg/S1':>7}  {'Median':>7}  {'P95':>7}  {'P99':>7}  {'Max':>8}")
    for tag, bk in [("CFG1", b1), ("CFG2", b2), ("CFG3", b3)]:
        print(f"  {tag:<6}  {bk['blocking_recall']:>8.4f}"
              f"  {bk['recovered']:>7d}  {bk['missed']:>7d}"
              f"  {bk['avg_per_s1']:>7.1f}  {bk['median_per_s1']:>7.1f}"
              f"  {bk['p95_per_s1']:>7.1f}  {bk['p99_per_s1']:>7.1f}"
              f"  {bk['max_per_s1']:>8d}")

    print(f"\n  MATCHING  (budget={args.budget}  threshold={args.threshold}"
          f"  config={args.config})")
    print(f"    Macro F0.5          : {m['macro_f0_5']:.4f}")
    print(f"    Macro Precision     : {m['macro_precision']:.4f}")
    print(f"    Macro Recall        : {m['macro_recall']:.4f}")
    print(f"    Agg TP / FP / FN   : {m['aggregate_tp']} / "
          f"{m['aggregate_fp']} / {m['aggregate_fn']}")
    print(f"    Singleton accuracy  : {m['singleton_accuracy']:.4f}"
          f"  ({m['singleton_correct']}/{m['singleton_count']})")
    print(f"    Non-singleton F0.5  : {m['non_singleton_macro_f0_5']:.4f}")
    d = m["per_entity_f0_5_distribution"]
    print(f"    Per-entity F0.5     : "
          f"min={d['min']:.3f}  p25={d['p25']:.3f}  p50={d['p50']:.3f}"
          f"  p75={d['p75']:.3f}  p95={d['p95']:.3f}  max={d['max']:.3f}")

    print(f"\n  RUNTIME : {rt['total_seconds']:.1f}s"
          f" | Peak RSS: {rt['peak_rss_mb']:.0f} MB")
    print(f"  WALL    : {time.time() - t_wall:.1f}s"
          f" (incl. correctness check + sampling)")

    # ---- Step 6: Save JSON ----------------------------------------------------
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results -> {args.output}")


if __name__ == "__main__":
    main()
