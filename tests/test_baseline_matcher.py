"""
Unit tests for src/baseline_matcher.py.
"""

import unittest
from src.baseline_matcher import (
    jaccard_similarity,
    compute_pair_features,
    compute_deterministic_score,
)


class TestBaselineMatcher(unittest.TestCase):
    def test_jaccard_similarity(self):
        s1 = {"a", "b", "c"}
        s2 = {"b", "c", "d"}
        self.assertAlmostEqual(jaccard_similarity(s1, s2), 2 / 4)
        self.assertEqual(jaccard_similarity(set(), set()), 1.0)
        self.assertEqual(jaccard_similarity({"a"}, set()), 0.0)

    def test_pair_features_exact(self):
        feats = compute_pair_features(
            s1_name="Apex Summit Inc",
            s1_addr="105 Elm Street, Morganton, NC",
            s1_ctry="US",
            cand_name="Apex Summit",
            cand_addr="105 ELM ST, MORGANTON, NC",
            cand_ctry="US",
        )
        self.assertEqual(feats["country_match"], 1.0)
        self.assertEqual(feats["name_canonical_exact"], 1.0)
        self.assertEqual(feats["addr_exact"], 1.0)
        self.assertEqual(feats["addr_numeric_jaccard"], 1.0)

        score = compute_deterministic_score(feats, "balanced")
        self.assertGreaterEqual(score, 0.95)

    def test_pair_features_cross_country(self):
        feats = compute_pair_features(
            s1_name="Apex Summit Inc",
            s1_addr="105 Elm Street, Morganton, NC",
            s1_ctry="US",
            cand_name="Apex Summit Inc",
            cand_addr="105 Elm Street, Morganton, NC",
            cand_ctry="India",
        )
        self.assertEqual(feats["country_match"], 0.0)
        score = compute_deterministic_score(feats, "balanced")
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
