"""
Unit tests for src/evaluation.py using stdlib unittest.
Verifies exact official macro F_0.5 rules, singleton behavior, and competition examples.
"""

import math
import unittest
from src.evaluation import compute_entity_metrics, evaluate_predictions


class TestEvaluation(unittest.TestCase):
    def test_singleton_correct(self):
        """truth = [] and prediction = [] => entity score = 1.0."""
        res = compute_entity_metrics([], [])
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f0_5"], 1.0)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertTrue(res["is_singleton"])
        self.assertTrue(res["singleton_correct"])

    def test_singleton_false_merge(self):
        """truth = [] and prediction != [] => entity score = 0.0."""
        res = compute_entity_metrics([], ["S2-00001"])
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["f0_5"], 0.0)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 0)
        self.assertTrue(res["is_singleton"])
        self.assertFalse(res["singleton_correct"])

    def test_non_singleton_missed_completely(self):
        """truth != [] and prediction = [] => entity score = 0.0."""
        res = compute_entity_metrics(["S2-00001", "S3-00002"], [])
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["f0_5"], 0.0)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 2)
        self.assertFalse(res["is_singleton"])
        self.assertFalse(res["singleton_correct"])

    def test_competition_example_from_problem_statement(self):
        """
        Example from .ps/ps.md:
          truth: [S2-00047, S3-00812]
          prediction: [S2-00047, S2-00193, S3-00812]
          Precision = 2/3, Recall = 1.0
          F_0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = 5/7 ≈ 0.714
        """
        truth = ["S2-00047", "S3-00812"]
        pred = ["S2-00047", "S2-00193", "S3-00812"]
        res = compute_entity_metrics(truth, pred)

        self.assertAlmostEqual(res["precision"], 2 / 3, places=5)
        self.assertAlmostEqual(res["recall"], 1.0, places=5)
        expected_f0_5 = (1.25 * (2 / 3) * 1.0) / (0.25 * (2 / 3) + 1.0)  # exactly 5/7
        self.assertAlmostEqual(res["f0_5"], expected_f0_5, places=5)
        self.assertEqual(round(res["f0_5"], 3), 0.714)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 0)

    def test_perfect_match(self):
        """truth and prediction match perfectly."""
        truth = ["S2-0001", "S3-0002"]
        pred = ["S3-0002", "S2-0001"]
        res = compute_entity_metrics(truth, pred)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f0_5"], 1.0)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)

    def test_zero_overlap_non_empty(self):
        """Non-empty prediction with completely disjoint matches."""
        res = compute_entity_metrics(["S2-0001"], ["S2-9999", "S3-8888"])
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["f0_5"], 0.0)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 2)
        self.assertEqual(res["fn"], 1)

    def test_macro_evaluation(self):
        """Test macro averaging across multiple diverse entities."""
        ground_truth = {
            "S1-1": set(),  # singleton
            "S1-2": {"S2-00047", "S3-00812"},  # competition example
            "S1-3": {"S2-00001"},  # wrong prediction
        }
        predictions = {
            "S1-1": set(),  # correctly predicted singleton -> score = 1.0
            "S1-2": {"S2-00047", "S2-00193", "S3-00812"},  # score = 5/7
            "S1-3": {"S2-99999"},  # score = 0.0
        }
        summary = evaluate_predictions(ground_truth, predictions)

        expected_macro = (1.0 + (5 / 7) + 0.0) / 3  # (12/7)/3 = 4/7 ≈ 0.57142857
        self.assertAlmostEqual(summary["macro_f0_5"], expected_macro, places=5)
        self.assertEqual(summary["total_source1_entities"], 3)
        self.assertEqual(summary["singletons_total"], 1)
        self.assertEqual(summary["singletons_correct"], 1)
        self.assertEqual(summary["singleton_accuracy"], 1.0)
        self.assertEqual(summary["aggregate_tp"], 2)
        self.assertEqual(summary["aggregate_fp"], 2)  # 1 from S1-2, 1 from S1-3
        self.assertEqual(summary["aggregate_fn"], 1)  # 1 from S1-3


if __name__ == "__main__":
    unittest.main()
