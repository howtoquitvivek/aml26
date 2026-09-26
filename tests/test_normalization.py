"""
Unit tests for src/normalization.py.
"""

import unittest
from src.normalization import (
    normalize_unicode,
    normalize_name,
    get_canonical_name_key,
    normalize_address,
    extract_numeric_tokens,
    get_character_ngrams,
)


class TestNormalization(unittest.TestCase):
    def test_unicode_preservation(self):
        # Hindi / Devanagari
        hindi = "एसएस फूड प्राइवेट लिमिटेड"
        self.assertEqual(normalize_name(hindi), "एसएस फूड प्राइवेट लिमिटेड")

        # French accents
        french = "Marina École France Sàrl"
        norm = normalize_name(french)
        self.assertIn("école", norm)
        self.assertIn("sàrl", norm)

    def test_domain_name_handling(self):
        domain = "maurewilliamscolombier.com"
        self.assertEqual(normalize_name(domain), "maurewilliamscolombier")

        url = "http://www.crystal-staffing.org"
        self.assertEqual(normalize_name(url), "crystal staffing")

    def test_legal_suffix_and_canonical_key(self):
        # Permutation and legal suffix
        name1 = "Crystal Staffing Solutions LLC"
        name2 = "Llc Crystal Staffing Solutions"
        name3 = "Solutions Crystal Staffing Inc"

        key1 = get_canonical_name_key(name1, strip_legal=True)
        key2 = get_canonical_name_key(name2, strip_legal=True)
        key3 = get_canonical_name_key(name3, strip_legal=True)

        self.assertEqual(key1, "crystal solutions staffing")
        self.assertEqual(key1, key2)
        self.assertEqual(key1, key3)

    def test_address_abbreviations(self):
        addr1 = "105 ELM ST, MORGANTON, NC"
        addr2 = "105 Elm Street, Morganton, NC"

        norm1 = normalize_address(addr1)
        norm2 = normalize_address(addr2)

        self.assertIn("street", norm1)
        self.assertEqual(norm1, norm2)

    def test_numeric_token_extraction(self):
        addr = "Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad"
        nums = extract_numeric_tokens(addr)
        self.assertTrue({"684", "989", "9487203"}.issubset(nums))

    def test_character_ngrams(self):
        text = "abcde"
        ngrams = get_character_ngrams(text, n=3)
        self.assertEqual(ngrams, {"abc", "bcd", "cde"})


if __name__ == "__main__":
    unittest.main()
