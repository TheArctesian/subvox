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

    def test_french_liaison_only_where_natives_make_it(self):
        def say(t):
            return " ".join("".join(w) for w in ipa.phonemize(t, "fr"))
        self.assertEqual(say("tu vas au travail"), "ty va o tʁavaj")          # no liaison after a verb
        self.assertEqual(say("je voudrais un café"), "ʒə vudʁɛ œ̃ kafe")
        self.assertEqual(say("Tenait en son bec"), "tənɛ ɑ̃ sɔ̃ bɛk")
        self.assertEqual(say("trois heures et quart"), "tʁwaz œʁ e kaʁ")      # never before et
        self.assertEqual(say("les enfants"), "lez ɑ̃fɑ̃")                       # obligatory after a determiner
        self.assertEqual(say("vous êtes"), "vuz ɛt")
        self.assertEqual(say("un bon ami"), "œ̃ bɔn ami")                     # bon loses its nasal
        self.assertEqual(say("mon ami"), "mɔ̃n ami")
        self.assertEqual(say("comment allez-vous"), "kɔmɑ̃t alevu")
        self.assertEqual(say("vas-y"), "vazi")
        self.assertEqual(say("Rouen"), "ʁwɑ̃")
        self.assertEqual(say("nous habitions à Nice"), "nuz abitjɔ̃ a nis")
        self.assertEqual(say("Qui suivent"), "ki sɥiv")
        self.assertEqual(say("j'en ai deux"), "ʒɑ̃n e dø")
        self.assertEqual(say("les haricots"), "le aʁiko")                      # h aspiré: no liaison
        self.assertEqual(say("les hommes"), "lez ɔm")
        self.assertEqual(say("des roses envolées"), "de ʁoz ɑ̃vole")          # no doubled z
        self.assertEqual(say("je suis allé"), "ʒə sɥi ale")
        self.assertEqual(say("quelques amis"), "kɛlkəz ami")
        self.assertEqual(say("vingt et un"), "vɛ̃t e œ̃")
        self.assertEqual(say("mon ancien ami"), "mɔ̃n ɑ̃sjɛn ami")

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

    def test_mexican_spanish_has_seseo_and_yeismo(self):
        self.assertEqual(ipa.phones("zapato", "es", "MX"), ["s", "a", "p", "a", "t", "o"])
        self.assertEqual(ipa.phones("calle", "es", "MX"), ["k", "a", "ʝ", "e"])
        self.assertEqual(ipa.phones("zapato", "es"), ["θ", "a", "p", "a", "t", "o"])

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

    def test_everyday_readings_pypinyin_gets_wrong(self):
        self.assertEqual(ipa.pinyin_marks("两只狗"), "liǎng zhī gǒu")      # measure word, not "only"
        self.assertEqual(ipa.pinyin_marks("只要"), "zhǐ yào")
        self.assertEqual(ipa.pinyin_marks("他跑得很快"), "tā pǎo de hěn kuài")
        self.assertEqual(ipa.pinyin_marks("我得走了"), "wǒ děi zǒu le")
        self.assertEqual(ipa.pinyin_marks("拿着手机"), "ná zhe shǒu jī")
        self.assertEqual(ipa.pinyin_marks("我睡着了"), "wǒ shuì zháo le")
        self.assertEqual(ipa.pinyin_marks("一本书"), "yī běn shū")        # citation tones by default

    def test_lexical_neutral_tones(self):
        self.assertEqual(ipa.pinyin_marks("谢谢"), "xiè xie")
        self.assertEqual(ipa.pinyin_marks("休息"), "xiū xi")
        self.assertEqual(ipa.pinyin_marks("我有一个哥哥"), "wǒ yǒu yī gè gē ge")

    def test_dictionary_readings_are_kept_where_rules_dont_apply(self):
        self.assertEqual(ipa.pinyin_marks("获得"), "huò dé")
        self.assertEqual(ipa.pinyin_marks("我得到了"), "wǒ dé dào le")
        self.assertEqual(ipa.pinyin_marks("着急"), "zháo jí")
        self.assertEqual(ipa.pinyin_marks("东西南北"), "dōng xī nán běi")
        self.assertEqual(ipa.pinyin_marks("休息", neutral=False), "xiū xī")

    def test_tone_change_drills_show_spoken_tones(self):
        self.assertEqual(ipa.pinyin_marks("一天", sandhi=True), "yì tiān")
        self.assertEqual(ipa.pinyin_marks("一个", sandhi=True), "yí gè")
        self.assertEqual(ipa.pinyin_marks("十一点", sandhi=True), "shí yī diǎn")
        self.assertEqual(ipa.pinyin_marks("不对", sandhi=True), "bú duì")
        self.assertEqual(ipa.pinyin_marks("一月", sandhi=True), "yī yuè")
        self.assertEqual(ipa.pinyin_marks("看一看", sandhi=True), "kàn yi kàn")

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
