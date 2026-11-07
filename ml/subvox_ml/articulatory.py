"""Articulatory features per phone and the coarse alphabets built from them.

Jaw height, lip rounding, labial closure and closure type are what a jaw-coupled earbud can
plausibly observe. Voicing, nasality and tongue place are kept only as negative controls: a
measured transfer for them means an alignment or leakage bug, not a capable sensor."""
from __future__ import annotations
import math
from collections import Counter, defaultdict

# Diphthongs and centring vowels take the height of their onset, where the jaw sets first.
VOWEL_JAW = {**{v: "high" for v in ("i", "iː", "ɪ", "u", "uː", "ʊ", "ɪə", "ʊə")},
             **{v: "mid" for v in ("e", "ɛ", "ɛː", "ʌ", "ə", "ɚ", "ɜ", "ɜː", "ɝ", "eɪ", "eə", "o", "oʊ", "əʊ")},
             **{v: "low" for v in ("a", "æ", "ɐ", "ɑ", "ɑː", "ɒ", "ɔ", "ɔː", "aɪ", "aʊ", "ɔɪ")}}
ROUNDED = {"u", "uː", "ʊ", "ʊə", "o", "oʊ", "əʊ", "ɔ", "ɔː", "ɒ", "ɔɪ", "aʊ", "ɚ", "ɝ", "w", "ɹ", "r", "ʃ", "ʒ", "tʃ", "dʒ"}
LABIAL = {"p", "b", "m", "f", "v", "w"}
STOPS = {"p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "m", "n", "ŋ", "ɾ", "ʔ"}
FRICATIVES = {"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"}
APPROXIMANTS = {"ɹ", "r", "l", "w", "j"}
# Consonants made without bringing the jaw up: approximants and the glottals.
JAW_OPEN = APPROXIMANTS | {"h", "ʔ"}
NASAL = {"m", "n", "ŋ"}
VOICELESS = {"p", "t", "k", "f", "θ", "s", "ʃ", "tʃ", "h", "ʔ"}
PLACE = {**{p: "labial" for p in ("p", "b", "m", "f", "v", "w")},
         **{p: "coronal" for p in ("θ", "ð", "t", "d", "n", "s", "z", "l", "ɹ", "r", "ɾ", "ʃ", "ʒ", "tʃ", "dʒ", "j")},
         **{p: "dorsal" for p in ("k", "ɡ", "ŋ")}, **{p: "glottal" for p in ("h", "ʔ")}}

OBSERVABLE = ("jaw", "round", "labial", "closure")
CONTROLS = ("voicing", "nasal", "place")


def features(phone):
    vowel = phone in VOWEL_JAW
    if not vowel and phone not in STOPS | FRICATIVES | APPROXIMANTS:
        raise ValueError(f"No articulatory features for {phone!r}")
    return dict(jaw=VOWEL_JAW.get(phone, "open" if phone in JAW_OPEN else "closed"),
                round=int(phone in ROUNDED), labial=int(phone in LABIAL),
                closure="vowel" if vowel else "stop" if phone in STOPS else "fricative" if phone in FRICATIVES else "approximant",
                voicing=int(phone not in VOICELESS), nasal=int(phone in NASAL), place=PLACE.get(phone, "vowel"))


def coverage_check(language):
    missing = [p["ipa"] for p in language["inventory"] if p["ipa"] not in VOWEL_JAW and p["ipa"] not in STOPS | FRICATIVES | APPROXIMANTS]
    if missing:
        raise ValueError(f"Articulatory table lacks inventory phones: {', '.join(missing)}")


# A conventional place-based viseme set, included to show what asking for tongue place costs.
VISEME_STD = [{"p", "b", "m"}, {"f", "v"}, {"θ", "ð"}, {"t", "d", "n", "s", "z", "l", "ɾ"}, {"ʃ", "ʒ", "tʃ", "dʒ"},
              {"k", "ɡ", "ŋ", "h", "ʔ"}, {"ɹ", "r", "ɝ", "ɚ"}, {"w"}, {"j"},
              {"a", "æ", "ɐ", "ɑ", "ɑː", "ʌ", "aɪ", "aʊ"}, {"e", "ɛ", "ɛː", "eɪ", "eə", "ə", "ɜ", "ɜː"},
              {"i", "iː", "ɪ", "ɪə"}, {"o", "ɔ", "ɔː", "ɒ", "oʊ", "əʊ", "ɔɪ"}, {"u", "uː", "ʊ", "ʊə"}]

# Fixed before any result was seen; the coarse auxiliary head picks among these by a lexicon rule only.
ALPHABETS = {
    "phone": lambda p: p,
    "jaw4": lambda p: features(p)["jaw"],
    "jaw_round_labial": lambda p: tuple(features(p)[k] for k in ("jaw", "round", "labial")),
    "observable": lambda p: tuple(features(p)[k] for k in OBSERVABLE),
    "viseme_std": lambda p: next(i for i, group in enumerate(VISEME_STD) if p in group),
}


def unit_string(phones, alphabet):
    return tuple(ALPHABETS[alphabet](p) for p in phones)


def cohort_entropy(language, alphabet, weights=None):
    """How many lexicon words share each unit string: words the decoder can't tell apart
    however good the motion model becomes. Uniform word weights unless `weights` given."""
    cohorts = defaultdict(set)
    for word, variants in language["entries"].items():
        for variant in variants:
            if variant["ipa"]:
                cohorts[unit_string(variant["ipa"], alphabet)].add(word)
    membership = defaultdict(set)
    for members in cohorts.values():
        for word in members:
            membership[word] |= members
    words = sorted(membership)
    weight = {w: (weights or {}).get(w, 0 if weights else 1) for w in words}
    total = sum(weight.values()) or 1
    # H(word | units) for a uniform choice among each word's cohort mates.
    entropy = sum(weight[w] * math.log2(len(membership[w])) for w in words) / total
    return dict(alphabet=alphabet, words=len(words), unit_strings=len(cohorts),
                unique_word_share=sum(len(membership[w]) == 1 for w in words) / max(1, len(words)),
                mean_cohort_size=sum(len(membership[w]) for w in words) / max(1, len(words)),
                conditional_entropy_bits=entropy)


def transmitted_information(pairs):
    """Miller–Nicely transmitted information T(x;y) in bits from (reference, prediction) pairs,
    with the Miller–Madow correction because counts are small."""
    n = len(pairs)
    if not n:
        return dict(bits=None, relative=None, reference_entropy=None, n=0)
    joint, xs, ys = Counter(pairs), Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    def entropy(counts):
        return -sum(c / n * math.log2(c / n) for c in counts.values()) + (len(counts) - 1) / (2 * n * math.log(2))
    hx, hy, hxy = entropy(xs), entropy(ys), entropy(joint)
    bits = hx + hy - hxy
    return dict(bits=bits, relative=bits / hx if hx > 0 else None, reference_entropy=hx, n=n)
