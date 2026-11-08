import collections
import unittest

import community as c

LEXICON = {w: p.split() for w, p in {
    "the": "DH AH", "cat": "K AE T", "sat": "S AE T", "on": "AA N", "mat": "M AE T", "a": "AH", "dog": "D AO G",
    "ran": "R AE N", "far": "F AA R", "away": "AH W EY", "we": "W IY", "saw": "S AO", "big": "B IH G", "red": "R EH D",
    "sun": "S AH N", "rise": "R AY Z", "i": "AY", "am": "AE M", "glad": "G L AE D", "you": "Y UW", "came": "K EY M",
    "thee": "DH IY", "love": "L AH V", "it": "IH T", "is": "IH Z", "cold": "K OW L D", "today": "T AH D EY",
    "color": "K AH L ER", "was": "W AA Z", "bright": "B R AY T", "my": "M AY", "home": "HH OW M", "here": "HH IY R",
    "go": "G OW", "now": "N AW", "please": "P L IY Z", "stop": "S T AA P", "soon": "S UW N", "very": "V EH R IY",
}.items()}

SOURCE = dict(source_id="test-book", gutenberg_id="1", title="Test Book", author="A. Writer", author_death_year="1900",
              first_published="1880", kind="prose", translator="", translator_death_year="", allow_archaic="0")


class FilterTests(unittest.TestCase):
    def reason(self, sentence, archaic=False):
        return c.reject_reason(c.normalise_spelling(sentence), LEXICON, archaic)

    def test_easy_sentences_pass(self):
        self.assertIsNone(self.reason("The cat sat on the mat."))
        self.assertIsNone(self.reason("I am glad you came!"))

    def test_each_rule_names_its_reason(self):
        self.assertEqual(self.reason("The cat sat."), "length")
        self.assertEqual(self.reason('"The cat sat on the mat."'), "dialogue")
        self.assertEqual(self.reason("The cat sat on 3 mats."), "numeral")
        self.assertEqual(self.reason("Mr. Cat sat on the mat."), "abbreviation")
        self.assertEqual(self.reason("The cat sat on the Mat today."), "name")
        self.assertEqual(self.reason("I love thee very much."), "archaic")
        self.assertEqual(self.reason("The cat sat on the zebra."), "not in lexicon")
        self.assertEqual(self.reason("the cat sat on the mat."), "punctuation")

    def test_archaic_allowlist_and_us_spelling(self):
        self.assertIsNone(self.reason("I love thee, it is cold.", archaic=True))
        self.assertEqual(c.normalise_spelling("The colour was bright to-day."), "The color was bright today.")
        self.assertIsNone(self.reason("The colour was bright today."))

    def test_reading_level_rejects_long_words(self):
        long = dict(LEXICON, extraordinary="IH K S T R AO R D AH N EH R IY".split(), unbelievably="AH N B AH L IY V AH B L IY".split())
        self.assertEqual(c.reject_reason("The extraordinary unbelievably extraordinary cat.", long), "reading level")

    def test_prompt_id_matches_app_identity(self):
        self.assertEqual(c.prompt_id("The cat, sat!"), c.prompt_id("the CAT sat"))
        import hashlib
        self.assertEqual(c.prompt_id("The cat, sat!"), hashlib.sha256(b"the cat sat").hexdigest())


class SourceTests(unittest.TestCase):
    def test_public_domain_rule_for_both_countries(self):
        self.assertEqual(c.public_domain_problems(SOURCE, year=2026), [])
        self.assertTrue(c.public_domain_problems(dict(SOURCE, author_death_year="1956"), year=2026))
        self.assertFalse(c.public_domain_problems(dict(SOURCE, author_death_year="1955"), year=2026))
        self.assertTrue(c.public_domain_problems(dict(SOURCE, first_published="1931"), year=2026))
        self.assertTrue(c.public_domain_problems(dict(SOURCE, translator="Someone", translator_death_year=""), year=2026))

    def test_curated_sources_are_public_domain_and_unique(self):
        sources = c.load_sources()
        self.assertGreaterEqual(len(sources), 20)
        self.assertEqual(len({s["source_id"] for s in sources}), len(sources))
        self.assertEqual({s["source_id"]: p for s in sources if (p := c.public_domain_problems(s, year=2026))}, {})


class BuildTests(unittest.TestCase):
    TEXT = ("The cat sat on the mat. The dog ran far away. "
            "We saw the big red sun rise. I am glad you came! "
            + "It is cold today. " * 0
            + " ".join(["Filler sentence with zebra words here."] * 8) + " "
            "My home is here, go now. Please stop the dog soon. The color was very bright today. "
            "We saw a cat on the mat. I love the big red dog. You came home very soon.")

    def build(self, texts=None, sources=None, existing=frozenset()):
        sources = sources or [SOURCE]
        return c.build(sources, LEXICON, texts or {"test-book": self.TEXT}, existing=existing, year=2026)

    def test_passages_keep_order_and_split_on_gaps(self):
        accepted = [(0, "a"), (1, "b"), (2, "c"), (12, "d"), (13, "e")]
        groups = c.passages_of(SOURCE, accepted, gap=6, limit=2)
        self.assertEqual([g["sentences"] for g in groups], [["a", "b"], ["c"], ["d", "e"]])
        self.assertEqual(groups[0]["passage_id"], "test-book-0000")

    def test_build_is_deterministic_and_sets_are_disjoint(self):
        prompts, calibration, held, report = self.build()
        again = self.build()
        self.assertEqual((prompts, calibration, held), again[:3])
        training_ids = {p["prompt_id"] for p in prompts}
        self.assertFalse(training_ids & {p["prompt_id"] for p in calibration})
        self.assertFalse(training_ids & {p["prompt_id"] for p in held})
        self.assertFalse({p["passage_id"] for p in prompts} & {p["passage_id"] for p in held})
        self.assertGreater(len(calibration), 0)
        self.assertEqual(report["rejections"]["not in lexicon"], 8)

    def test_existing_bank_sentences_are_never_reused(self):
        prompts, calibration, held, report = self.build(existing={c.prompt_id("The cat sat on the mat.")})
        everything = prompts + calibration + held
        self.assertNotIn(c.prompt_id("The cat sat on the mat."), {p["prompt_id"] for p in everything})
        self.assertEqual(report["rejections"]["duplicate or existing bank sentence"], 1)

    def test_build_refuses_sources_still_in_copyright(self):
        with self.assertRaisesRegex(ValueError, "not public domain"):
            self.build(sources=[dict(SOURCE, author_death_year="1970")])

    def test_heldout_is_about_ten_percent(self):
        share = sum(c.heldout(f"book-{i:04d}") for i in range(5000)) / 5000
        self.assertAlmostEqual(share, .1, delta=.015)


class AssignTests(unittest.TestCase):
    def passage(self, pid, pairs, words=4, held=False):
        return dict(passage_id=pid, heldout=held, prompts=[dict(diphones=pairs, words=words)])

    def test_assignment_fills_gaps_skips_heldout_and_recorded(self):
        passages = [self.passage("common", ["A B"] * 4), self.passage("rare", ["C D", "E F"]),
                    self.passage("held", ["X Y", "Z W"], held=True), self.passage("done", ["P Q", "R S"])]
        chosen = c.assign(passages, {"A B": 500}, n=2, recorded={"done"})
        self.assertEqual([p["passage_id"] for p in chosen], ["rare", "common"])

    def test_picks_complement_each_other(self):
        passages = [self.passage("x1", ["C D"]), self.passage("x2", ["C D"]), self.passage("y", ["E F"])]
        chosen = c.assign(passages, {}, n=2)
        self.assertEqual({p["passage_id"] for p in chosen}, {"x1", "y"}, "A second copy of the same gap loses to a new gap")


if __name__ == "__main__":
    unittest.main()
