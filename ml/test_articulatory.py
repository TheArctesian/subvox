"""Chimera-derived articulator values. Fixtures verify code, never anatomical or sensor accuracy."""
import hashlib
import json
import math
import unittest
from subvox_ml import chimera_tables
from subvox_ml.articulatory import (ALPHABETS, CHIMERA_JAW_BINS, VOWEL_JAW, chimera_jaw, chimera_jaw_comparison,
                                    chimera_key, chimera_values, cohort_entropy)

# The app's training inventory: Subvox/Linguistics/ArticulatoryInventory.swift (27 consonants, 33 vowels).
INVENTORY = ("p b m f v θ ð t d s z n l ʃ ʒ tʃ dʒ k ɡ ŋ ɹ w j h ɾ ʔ r "
             "i ɪ ɛ æ ɑ ɔ ʌ ʊ u ɝ eɪ aɪ ɔɪ aʊ oʊ ə ɚ ɒ ɑː ɜː əʊ e o a iː uː ɔː ɛː ɜ ɐ ɪə eə ʊə").split()
PREREGISTERED = ("phone", "jaw4", "jaw_round_labial", "observable", "viseme_std")


class ArticulatoryTests(unittest.TestCase):
    def test_preregistered_alphabets_give_the_same_units_as_before_chimera_was_added(self):
        table = {a: {p: repr(ALPHABETS[a](p)) for p in INVENTORY} for a in PREREGISTERED}
        fingerprint = hashlib.sha256(json.dumps(table, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        # Computed from the package at 257bb27, before any Chimera code existed.
        self.assertEqual(fingerprint, "8b2f40dd49257219baf487438b73021ee267107371b823a8c44dc6dadb420a20")
        self.assertEqual(list(ALPHABETS)[:len(PREREGISTERED)], list(PREREGISTERED))

    def test_every_inventory_phone_has_a_chimera_symbol_and_a_jaw_chimera_unit(self):
        self.assertEqual(len(INVENTORY), 60)
        for phone in INVENTORY:
            chimera_key(phone)
            self.assertIn(ALPHABETS["jaw_chimera"](phone), {"high", "mid", "low", "closed", "open"})

    def test_symbol_rules_strip_length_take_diphthong_onsets_and_map_script_g(self):
        self.assertEqual(chimera_key("ɡ"), "g")
        self.assertEqual(chimera_key("iː"), "i")
        self.assertEqual(chimera_key("ɚ"), "ɝ")
        self.assertEqual(chimera_key("ɜː"), "ɝ")
        self.assertEqual(chimera_key("aɪ"), "a")
        self.assertEqual(chimera_key("əʊ"), "ə")
        self.assertEqual(chimera_key("ɔ̃"), "ɔ")
        self.assertEqual(chimera_key("tʰ"), "t")

    def test_unknown_phones_raise_instead_of_borrowing_a_neighbour(self):
        with self.assertRaises(ValueError):
            chimera_key("ɕ")

    def test_consonant_keys_chimera_leaves_to_coarticulation_stay_empty(self):
        self.assertEqual(chimera_values("p"), dict(jaw=None, lipAp=-0.25, lipProt=None, lipWidth=None))
        self.assertIsNone(chimera_values("t")["jaw"])
        self.assertEqual(chimera_values("ɡ")["jaw"], chimera_tables.CONSONANTS["g"][0])

    def test_vowel_bins_follow_the_fixed_cut_offs_and_consonants_keep_jaw4(self):
        self.assertEqual([limit for limit, _ in CHIMERA_JAW_BINS[:-1]], [0.6, 1.1])
        for phone in VOWEL_JAW:
            jaw = chimera_values(phone)["jaw"]
            expected = "high" if jaw < 0.6 else "mid" if jaw < 1.1 else "low"
            self.assertEqual(chimera_jaw(phone), expected, phone)
        self.assertEqual(chimera_jaw("p"), "closed")
        self.assertEqual(chimera_jaw("w"), "open")

    def test_comparison_lists_every_vowel_with_bucket_ranges_and_disagreements(self):
        report = chimera_jaw_comparison()
        self.assertEqual(report["chimera_rev"], chimera_tables.CHIMERA_REV)
        self.assertEqual({r["phone"] for r in report["vowels"]}, set(VOWEL_JAW))
        self.assertEqual(sum(r["vowels"] for r in report["bucket_ranges"].values()), len(VOWEL_JAW))
        for row in report["vowels"]:
            self.assertEqual(row["phone"] in report["disagreements"], row["bucket"] != row["chimera_bin"])
        self.assertTrue(all(math.isfinite(r["chimera_jaw"]) for r in report["vowels"]))

    def test_cohort_entropy_runs_on_jaw_chimera(self):
        language = dict(entries={"bat": [dict(ipa=["b", "æ", "t"])], "pit": [dict(ipa=["p", "ɪ", "t"])],
                                 "boot": [dict(ipa=["b", "uː", "t"])]})
        result = cohort_entropy(language, "jaw_chimera")
        self.assertEqual(result["words"], 3)
        # "pit" and "boot" are both closed-high-closed: one cohort of two.
        self.assertEqual(result["unit_strings"], 2)


if __name__ == "__main__":
    unittest.main()
