"""Hand-checked IPA for each language (docs/next-phase-plan.md, item 1)."""
import unittest
from subvox_ml import ipa


@unittest.skipUnless(ipa.espeak_available(), "espeak-ng not installed (brew install espeak-ng)")
class EspeakLanguages(unittest.TestCase):
    def test_french_liaison_comes_from_the_whole_phrase(self):
        self.assertEqual(ipa.phonemize("je suis", "fr"), [["ʒ", "ə"], ["s", "ɥ", "i"]])
        self.assertEqual(ipa.phones("tu es", "fr"), ["t", "y", "ɛ"])
        self.assertEqual(ipa.phones("il est", "fr"), ["i", "l", "ɛ"])
        self.assertEqual(ipa.phones("vous êtes", "fr"), ["v", "u", "z", "ɛ", "t"])  # liaison z
        self.assertEqual(ipa.phones("ils ont", "fr"), ["i", "l", "z", "ɔ̃"])

    def test_french_gn_is_palatal_but_ni_stays_n_j(self):
        self.assertEqual(ipa.phones("agneau", "fr"), ["a", "ɲ", "o"])
        self.assertEqual(ipa.phones("champignon", "fr"), ["ʃ", "ɑ̃", "p", "i", "ɲ", "ɔ̃"])
        self.assertEqual(ipa.phones("panier", "fr"), ["p", "a", "n", "j", "e"])

    def test_french_nasal_vowels_and_rounded_front_vowels(self):
        self.assertEqual(ipa.phones("un bon vin blanc", "fr"), ["œ̃", "b", "ɔ̃", "v", "ɛ̃", "b", "l", "ɑ̃"])
        self.assertEqual(ipa.phones("la rue", "fr"), ["l", "a", "ʁ", "y"])
        self.assertEqual(ipa.phones("la roue", "fr"), ["l", "a", "ʁ", "u"])

    def test_spanish_five_vowels_and_trill(self):
        self.assertEqual(ipa.phones("casa mesa piso luna", "es"), list("kasamesapisoluna"))
        self.assertEqual(ipa.phones("el perro", "es"), ["e", "l", "p", "e", "r", "o"])
        self.assertEqual(ipa.phones("ñandú", "es"), ["ɲ", "a", "n", "d", "u"])

    def test_spanish_diphthongs_have_glides(self):
        self.assertEqual(ipa.phones("soy", "es"), ["s", "o", "j"])
        self.assertEqual(ipa.phones("hay", "es"), ["a", "j"])
        self.assertEqual(ipa.phones("causa", "es"), ["k", "a", "w", "s", "a"])
        self.assertEqual(ipa.phones("bueno", "es"), ["b", "w", "e", "n", "o"])
        self.assertEqual(ipa.phones("piano", "es"), ["p", "j", "a", "n", "o"])
        self.assertEqual(ipa.phones("Europa", "es"), ["e", "w", "ɾ", "o", "p", "a"])

    def test_english_fallback_lands_in_the_shared_set(self):
        self.assertEqual(ipa.phones("the cat sat", "en"), ["ð", "ə", "k", "æ", "t", "s", "æ", "t"])
        for text in ("thinking about you", "bird father", "she measures the church judge"):
            self.assertTrue(all(p in ipa.INVENTORY for p in ipa.phones(text, "en")), text)


class Mandarin(unittest.TestCase):
    def test_characters_with_more_than_one_reading_follow_the_phrase(self):
        self.assertEqual([s["pinyin"] for s in ipa.mandarin_syllables("银行")], ["yin2", "hang2"])
        self.assertEqual([s["pinyin"] for s in ipa.mandarin_syllables("行走")], ["xing2", "zou3"])
        self.assertEqual([s["pinyin"] for s in ipa.mandarin_syllables("长江")], ["chang2", "jiang1"])
        self.assertEqual([s["pinyin"] for s in ipa.mandarin_syllables("长大")], ["zhang3", "da4"])

    def test_pinyin_to_ipa(self):
        self.assertEqual(ipa.phonemize("你好", "zh"), [["n", "i"], ["x", "a", "u"]])
        self.assertEqual(ipa.phonemize("是字", "zh"), [["ʂ", "ɻ̩"], ["ts", "ɹ̩"]])
        self.assertEqual(ipa.phonemize("女学", "zh"), [["n", "y"], ["ɕ", "ɥ", "e"]])
        self.assertEqual(ipa.phonemize("吃", "zh"), [["ʈʂʰ", "ɻ̩"]])

    def test_labels_keep_tones_and_the_model_side_is_toneless(self):
        syllables = ipa.mandarin_syllables("你好")
        self.assertEqual([s["tone"] for s in syllables], [3, 3])
        self.assertEqual(ipa.strip_tones(["ni3", "hao3"]), ["ni", "hao"])

    def test_traditional_characters_read_the_same(self):
        self.assertEqual([s["pinyin"] for s in ipa.mandarin_syllables("舉頭望明月")], ["ju3", "tou2", "wang4", "ming2", "yue4"])
        self.assertEqual([s["pinyin"] for s in ipa.mandarin_syllables("銀行")], ["yin2", "hang2"])
        self.assertEqual(ipa.pinyin_marks("我們去銀行。"), "wǒ men qù yín háng")

    def test_tone_restoration_uses_phrase_context(self):
        # Citation tones (no third-tone sandhi), as in the labels. Homophones such as yinhang
        # (银行 / 引航) are ambiguous without characters; the dictionary placeholder can't settle those.
        self.assertEqual(ipa.restore_tones(["peng", "you"]), ["peng2", "you3"])
        self.assertEqual(ipa.restore_tones(["bei", "jing"]), ["bei3", "jing1"])
        self.assertEqual(ipa.restore_tones(["zhong", "yao"]), ["zhong4", "yao4"])


class SharedSet(unittest.TestCase):
    def test_arpabet_compatibility_path(self):
        self.assertEqual(ipa.arpabet_to_ipa("HH AH0 L OW1"), ["h", "ə", "l", "oʊ"])
        self.assertEqual(ipa.arpabet_to_ipa("B ER1 D"), ["b", "ɝ", "d"])
        self.assertEqual(len(ipa.ARPABET), 39)
        self.assertTrue(set(ipa.ARPABET.values()) <= set(ipa.INVENTORY))

    def test_inventory_records_which_languages_use_each_phone(self):
        self.assertEqual(ipa.INVENTORY["y"], ("fr", "zh"))
        self.assertIn("en", ipa.INVENTORY["p"])
        self.assertEqual(ipa.INVENTORY["ʁ"], ("fr",))

    def test_language_tags(self):
        self.assertEqual(ipa.validate_language_tag("fr-CA"), "fr")
        self.assertEqual(ipa.validate_language_tag("zh-Hant"), "zh")
        with self.assertRaises(ValueError):
            ipa.validate_language_tag("de")


if __name__ == "__main__":
    unittest.main()
