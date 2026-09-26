"""
Evaluation module for Amazon ML Challenge 2026: Business Entity Resolution.

Implements the official macro F_0.5 metric specified in .ps/ps.md:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Metric Rules from Problem Statement:
- Computed per Source 1 entity, then macro-averaged across ALL Source 1 entities in the evaluation set.
- Singletons are included in the average:
    * truth = [] and prediction = []  => score = 1.0 (Precision=1.0, Recall=1.0, F0.5=1.0)
    * truth = [] and prediction != [] => score = 0.0 (Precision=0.0, Recall=0.0, F0.5=0.0)
    * truth != [] and prediction = [] => score = 0.0 (Precision=0.0, Recall=0.0, F0.5=0.0)
- Example from .ps/ps.md:
    truth = [S2-00047, S3-00812]
    prediction = [S2-00047, S2-00193, S3-00812]
    Precision = 2/3, Recall = 1.0 => F_0.5 = 0.7142857...
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np


def compute_entity_metrics(
    true_ids: Union[Iterable[str], Set[str]],
    pred_ids: Union[Iterable[str], Set[str]],
) -> Dict[str, Union[float, int, bool]]:
    """Compute precision, recall, and F_0.5 for a single Source 1 entity.

    Args:
        true_ids: Set or list of true matching entity IDs (S2/S3).
        pred_ids: Set or list of predicted matching entity IDs (S2/S3).

    Returns:
        dict with precision, recall, f0_5, tp, fp, fn, is_singleton, singleton_correct.
    """
    true_set = set(true_ids)
    pred_set = set(pred_ids)

    # Singleton cases
    if not true_set and not pred_set:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f0_5": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "is_singleton": True,
            "singleton_correct": True,
        }

    if not true_set and pred_set:
        # False merge on singleton
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f0_5": 0.0,
            "tp": 0,
            "fp": len(pred_set),
            "fn": 0,
            "is_singleton": True,
            "singleton_correct": False,
        }

    if true_set and not pred_set:
        # Missed all matches on non-singleton
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f0_5": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(true_set),
            "is_singleton": False,
            "singleton_correct": False,
        }

    # Non-empty truth and prediction
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if tp == 0:
        f0_5 = 0.0
    else:
        # F_0.5 formula: (1.25 * P * R) / (0.25 * P + R)
        denominator = 0.25 * precision + recall
        f0_5 = (1.25 * precision * recall) / denominator if denominator > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f0_5": float(f0_5),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "is_singleton": False,
        "singleton_correct": False,
    }


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    return_details: bool = False,
) -> Dict[str, Union[float, int, dict]]:
    """Evaluate predictions against ground truth across all Source 1 entities.

    Args:
        ground_truth: Dict mapping source1_entity_id -> set of true matching IDs.
        predictions: Dict mapping source1_entity_id -> set of predicted matching IDs.
        return_details: If True, include per-entity breakdown in result dict.

    Returns:
        dict with macro_f0_5, macro_precision, macro_recall, singleton metrics, etc.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        raise ValueError("Ground truth dictionary is empty.")

    f0_5_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    singleton_count = 0
    singleton_correct_count = 0
    non_singleton_f0_5: List[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_entity_details = {}

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        m = compute_entity_metrics(true_set, pred_set)

        f0_5_scores.append(m["f0_5"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])

        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]

        if m["is_singleton"]:
            singleton_count += 1
            if m["singleton_correct"]:
                singleton_correct_count += 1
        else:
            non_singleton_f0_5.append(m["f0_5"])

        if return_details:
            per_entity_details[s1_id] = m

    summary = {
        "macro_f0_5": float(np.mean(f0_5_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "total_source1_entities": total_entities,
        "singletons_total": singleton_count,
        "singletons_correct": singleton_correct_count,
        "singleton_accuracy": float(singleton_correct_count / singleton_count) if singleton_count > 0 else 0.0,
        "non_singleton_entities": len(non_singleton_f0_5),
        "non_singleton_macro_f0_5": float(np.mean(non_singleton_f0_5)) if non_singleton_f0_5 else 0.0,
        "aggregate_tp": total_tp,
        "aggregate_fp": total_fp,
        "aggregate_fn": total_fn,
    }

    if return_details:
        summary["entity_details"] = per_entity_details

    return summary


def load_tsv_mapping(tsv_path: str, is_ground_truth: bool = False) -> Dict[str, Set[str]]:
    """Load a TSV file (matching_results or ground_truth) into {s1_id: set_of_ids}."""
    mapping = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = next(f, None)
        for line in f:
            line_str = line.rstrip("\n")
            if not line_str.strip():
                continue
            parts = line_str.split("\t")
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                ids = {x.strip() for x in parts[1].split(",") if x.strip()}
            else:
                ids = set()
            mapping[s1_id] = ids
    return mapping


def evaluate_tsv_files(
    ground_truth_tsv: str,
    matching_tsv: str,
    return_details: bool = False,
) -> Dict[str, Union[float, int, dict]]:
    """Convenience function to evaluate matching_results.tsv against ground_truth.tsv."""
    gt = load_tsv_mapping(ground_truth_tsv, is_ground_truth=True)
    pred = load_tsv_mapping(matching_tsv, is_ground_truth=False)
    return evaluate_predictions(gt, pred, return_details=return_details)
