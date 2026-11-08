import unittest
from subvox_ml import ipa
from subvox_ml.phone_check import align, check, mispronounced


class Alignment(unittest.TestCase):
    def test_substitution_deletion_insertion(self):
        self.assertEqual(align(list("abc"), list("axcd")),
                         [("match", "a", "a"), ("sub", "b", "x"), ("match", "c", "c"), ("ins", None, "d")])
        self.assertEqual(align(list("ab"), list("b")), [("del", "a", None), ("match", "b", "b")])


class PhoneCheck(unittest.TestCase):
    def test_learner_phone_intervals_are_the_produced_phones(self):
        # A learner says "tu" with /u/ for /y/: Whisper hears the right word, the phone recogniser does not.
        intervals = [dict(phone="t", start=0.0, end=.08), dict(phone="u", start=.08, end=.3)]
        result = check("tu", "fr", "tu", intervals)
        self.assertEqual(result["source"], "phone-intervals")
        self.assertEqual(result["produced_phones"], ["t", "u"])
        self.assertEqual(mispronounced(result), [("y", "u")])
        self.assertTrue(result["trainable"], "Text agreed, so the take trains on the produced phones")

    def test_whisper_disagreeing_with_the_text_is_practice_only(self):
        result = check("你好", "zh", "你们", None)
        self.assertFalse(result["trainable"])
        self.assertEqual(result["source"], "whisper-words")

    def test_rejects_symbols_outside_the_shared_set(self):
        with self.assertRaises(ValueError):
            check("tu", "fr", "tu", [dict(phone="ʘ", start=0, end=.1)])

    @unittest.skipUnless(ipa.espeak_available(), "needs espeak-ng")
    def test_wrong_word_shows_up_in_whisper_word_phones(self):
        result = check("la rue", "fr", "la roue")
        self.assertEqual(mispronounced(result), [("y", "u")])
        self.assertFalse(result["trainable"], "One of two words differs: 50% < 85%")


if __name__ == "__main__":
    unittest.main()
