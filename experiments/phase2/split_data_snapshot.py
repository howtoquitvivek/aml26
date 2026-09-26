"""
Module to create a leak-free, stratified validation split for Amazon ML Challenge 2026.

Partitioning Strategy:
- Split strictly at the Source 1 entity level (S1).
- Every ground-truth link belonging to an S1 entity stays with that S1 entity.
- Stratified by (country, match_count_bin) to ensure identical distribution of:
    * Countries (US vs India)
    * Singletons (0 matches) vs Non-singletons
    * Number of true matches (0, 1, 2, 3, 4, 5+)
- Validation fraction: 10% (~220,682 S1 entities), leaving 90% (~1,986,139 S1 entities) for training.
- Fully reproducible using random_state=42.
"""

import os
import json
import collections
import numpy as np
import pandas as pd

DATA_DIR = "dataset"
EXPERIMENTS_DIR = "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

S1_FILE = os.path.join(DATA_DIR, "train", "train_source1.tsv")
GT_FILE = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")

VAL_FRACTION = 0.10
RANDOM_SEED = 42


def get_match_bin(n_matches: int) -> str:
    """Bin match count into discrete strata."""
    if n_matches == 0:
        return "0_singleton"
    elif n_matches == 1:
        return "1_match"
    elif n_matches == 2:
        return "2_matches"
    elif n_matches == 3:
        return "3_matches"
    elif n_matches == 4:
        return "4_matches"
    else:
        return "5_plus_matches"


def create_stratified_split(val_fraction: float = VAL_FRACTION, seed: int = RANDOM_SEED):
    print("Loading Source 1 entities and countries...")
    s1_countries = {}
    with open(S1_FILE, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                s1_countries[parts[0]] = parts[3]

    print("Loading Ground Truth match counts...")
    s1_match_counts = collections.defaultdict(int)
    with open(GT_FILE, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                m_ids = [m.strip() for m in parts[1].split(",") if m.strip()]
                s1_match_counts[s1_id] = len(m_ids)
            else:
                s1_match_counts[s1_id] = 0

    print("Forming stratification strata...")
    strata = collections.defaultdict(list)
    for s1_id, country in s1_countries.items():
        n_matches = s1_match_counts.get(s1_id, 0)
        m_bin = get_match_bin(n_matches)
        stratum_key = f"{country}__{m_bin}"
        strata[stratum_key].append(s1_id)

    rng = np.random.RandomState(seed)
    train_ids = []
    val_ids = []

    strata_stats = {}
    for stratum_key, entity_ids in sorted(strata.items()):
        entity_ids = sorted(entity_ids)  # Deterministic base order
        perm = rng.permutation(len(entity_ids))
        val_size = int(round(len(entity_ids) * val_fraction))
        
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        s_val_ids = [entity_ids[i] for i in val_idx]
        s_train_ids = [entity_ids[i] for i in train_idx]

        val_ids.extend(s_val_ids)
        train_ids.extend(s_train_ids)

        strata_stats[stratum_key] = {
            "total": len(entity_ids),
            "train": len(s_train_ids),
            "val": len(s_val_ids),
            "val_pct": round(len(s_val_ids) / len(entity_ids) * 100, 2),
        }

    train_set = set(train_ids)
    val_set = set(val_ids)

    # Verification checks
    assert len(train_set & val_set) == 0, "FATAL: Intersection found between train and val!"
    assert len(train_set | val_set) == len(s1_countries), "FATAL: Entity count mismatch!"

    print(f"Total S1 entities: {len(s1_countries):,}")
    print(f"Train S1 entities: {len(train_ids):,} ({len(train_ids)/len(s1_countries)*100:.2f}%)")
    print(f"Val S1 entities:   {len(val_ids):,} ({len(val_ids)/len(s1_countries)*100:.2f}%)")

    # Save ID lists
    train_path = os.path.join(EXPERIMENTS_DIR, "train_s1_ids.txt")
    val_path = os.path.join(EXPERIMENTS_DIR, "val_s1_ids.txt")

    print(f"Saving train IDs to {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for eid in sorted(train_ids):
            f.write(f"{eid}\n")

    print(f"Saving val IDs to {val_path}...")
    with open(val_path, "w", encoding="utf-8") as f:
        for eid in sorted(val_ids):
            f.write(f"{eid}\n")

    # Aggregate validation split statistics
    def compute_split_stats(ids):
        countries = collections.Counter()
        match_counts = collections.Counter()
        singletons = 0
        for sid in ids:
            c = s1_countries[sid]
            countries[c] += 1
            cnt = s1_match_counts.get(sid, 0)
            match_counts[cnt] += 1
            if cnt == 0:
                singletons += 1
        
        tot = len(ids)
        return {
            "total_s1": tot,
            "countries": {k: {"count": v, "pct": round(v / tot * 100, 2)} for k, v in sorted(countries.items())},
            "singletons": {"count": singletons, "pct": round(singletons / tot * 100, 4)},
            "match_count_distribution": {int(k): int(v) for k, v in sorted(match_counts.items())},
        }

    summary = {
        "val_fraction": val_fraction,
        "random_seed": seed,
        "train_set": compute_split_stats(train_ids),
        "val_set": compute_split_stats(val_ids),
        "strata_breakdown": strata_stats,
    }

    summary_path = os.path.join(EXPERIMENTS_DIR, "validation_split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved split summary to {summary_path}")

    return summary


if __name__ == "__main__":
    create_stratified_split()
