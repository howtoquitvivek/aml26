import os
import random
import collections
import numpy as np

# Load train IDs
with open("experiments/train_s1_ids.txt") as f:
    train_ids = {line.strip() for line in f if line.strip()}

# Load old dev IDs to ensure disjointness
with open("experiments/dev_s1_ids.txt") as f:
    old_dev_ids = {line.strip() for line in f if line.strip()}

# Load countries
s1_countries = {}
with open("dataset/train/train_source1.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4 and parts[0] in train_ids:
            s1_countries[parts[0]] = parts[3]

# Load match counts
s1_match_counts = collections.defaultdict(int)
with open("dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        s1_id = parts[0].strip()
        if s1_id in train_ids:
            if len(parts) > 1 and parts[1].strip():
                m_ids = [m.strip() for m in parts[1].split(",") if m.strip()]
                s1_match_counts[s1_id] = len(m_ids)
            else:
                s1_match_counts[s1_id] = 0

def get_match_bin(n):
    if n == 0: return "0_singleton"
    elif n == 1: return "1_match"
    elif n == 2: return "2_matches"
    elif n == 3: return "3_matches"
    elif n == 4: return "4_matches"
    else: return "5_plus_matches"

strata = collections.defaultdict(list)
for s1_id, country in s1_countries.items():
    if s1_id in old_dev_ids:
        continue # Should be disjoint anyway, but enforcing it
    m_bin = get_match_bin(s1_match_counts.get(s1_id, 0))
    strata[f"{country}__{m_bin}"].append(s1_id)

rng = np.random.RandomState(42)
target_size = 1500
total_pool = sum(len(v) for v in strata.values())

clean_dev_ids = []
for k, v in sorted(strata.items()):
    v_sorted = sorted(v)
    perm = rng.permutation(len(v_sorted))
    # Stratified proportion
    frac = len(v_sorted) / total_pool
    size = int(round(frac * target_size))
    # Safety check if size > len(v)
    size = min(size, len(v_sorted))
    
    selected_indices = perm[:size]
    for idx in selected_indices:
        clean_dev_ids.append(v_sorted[idx])

# Fix off-by-one due to rounding
if len(clean_dev_ids) < target_size:
    print(f"Adding {target_size - len(clean_dev_ids)} random extra records...")
    all_unselected = []
    selected_set = set(clean_dev_ids)
    for k, v in strata.items():
        all_unselected.extend([x for x in v if x not in selected_set])
    all_unselected_sorted = sorted(all_unselected)
    extra = rng.choice(all_unselected_sorted, size=(target_size - len(clean_dev_ids)), replace=False)
    clean_dev_ids.extend(extra)
elif len(clean_dev_ids) > target_size:
    print(f"Removing {len(clean_dev_ids) - target_size} records...")
    clean_dev_ids = clean_dev_ids[:target_size]

clean_dev_set = set(clean_dev_ids)

# Load validation IDs to check intersection
with open("experiments/val_s1_ids.txt") as f:
    val_ids = {line.strip() for line in f if line.strip()}

print(f"Generated clean dev set: {len(clean_dev_set)}")
print(f"Intersection with val: {len(clean_dev_set & val_ids)}")
print(f"Intersection with old dev: {len(clean_dev_set & old_dev_ids)}")

with open("experiments/phase6/artifacts/clean_dev_1500/clean_dev_ids.txt", "w") as f:
    for eid in sorted(clean_dev_ids):
        f.write(f"{eid}\n")
print("Saved to experiments/phase6/artifacts/clean_dev_1500/clean_dev_ids.txt")
