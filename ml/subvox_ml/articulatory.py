"""Articulatory features per phone and the coarse alphabets built from them.

Jaw height, lip rounding, labial closure and closure type are what a jaw-coupled earbud can
plausibly observe. Voicing, nasality and tongue place are kept only as negative controls: a
measured transfer for them means an alignment or leakage bug, not a capable sensor."""
from __future__ import annotations
import math
import unicodedata
from collections import Counter, defaultdict
from . import chimera_tables

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
# Added 2026-11-10 after the alphabets above were fixed; its bins are CHIMERA_JAW_BINS, fixed the
# same day before any result used them. The coarse head never selects it.
ALPHABETS["jaw_chimera"] = lambda p: chimera_jaw(p)


# Chimera's articulator values (chimera_tables.py, male body) are keyed by Chimera's own symbols.
# These rules were fixed on 2026-11-10, before any result used them.
# Diphthongs and centring vowels take their onset, as VOWEL_JAW does.
CHIMERA_ONSET = {"eɪ": "e", "aɪ": "a", "aʊ": "a", "ɔɪ": "ɔ", "oʊ": "o", "əʊ": "ə", "ɪə": "ɪ", "eə": "e", "ʊə": "ʊ"}
# ɚ is ɝ in Chimera's planner; ɜ has no Chimera target and takes ɝ as the nearest calibrated vowel.
CHIMERA_SUBSTITUTES = {"ɡ": "g", "ɚ": "ɝ", "ɜ": "ɝ"}
# Length, nasalisation, aspiration and syllabicity change no stored jaw or lip target in Chimera.
CHIMERA_IGNORED_MARKS = ("ː", "\u0303", "ʰ", "\u0329")


def chimera_key(phone):
    """Chimera's symbol for a Subvox phone. Raises for a phone Chimera has no target for."""
    key = unicodedata.normalize("NFC", phone)
    for mark in CHIMERA_IGNORED_MARKS:
        key = key.replace(mark, "")
    key = CHIMERA_ONSET.get(key, key)
    key = CHIMERA_SUBSTITUTES.get(key, key)
    if key not in chimera_tables.VOWELS and key not in chimera_tables.CONSONANTS:
        raise ValueError(f"No Chimera articulator target for {phone!r}")
    return key


def chimera_values(phone):
    """Chimera's jaw and lip values for a phone, in its template cm. None where Chimera leaves the
    articulator to coarticulation; these are fitted model targets, never measured positions."""
    key = chimera_key(phone)
    values = chimera_tables.VOWELS.get(key) or chimera_tables.CONSONANTS[key]
    return dict(zip(chimera_tables.KEYS, values))


# Vowel jaw bins on Chimera's male incisor opening (cm), fixed on 2026-11-10 before any result.
CHIMERA_JAW_BINS = ((0.6, "high"), (1.1, "mid"), (math.inf, "low"))


def chimera_jaw(phone):
    """Vowels binned by Chimera's jaw opening; consonants keep the jaw4 class so every phone has a unit."""
    if phone not in VOWEL_JAW:
        return features(phone)["jaw"]
    jaw = chimera_values(phone)["jaw"]
    return next(label for limit, label in CHIMERA_JAW_BINS if jaw < limit)


def chimera_jaw_comparison():
    """Each VOWEL_JAW vowel's hand-coded bucket beside Chimera's jaw opening and bin."""
    rows = []
    for phone in sorted(VOWEL_JAW):
        key = chimera_key(phone)
        rows.append(dict(phone=phone, chimera_key=key, bucket=VOWEL_JAW[phone],
                         chimera_jaw=chimera_tables.VOWELS[key][0], chimera_bin=chimera_jaw(phone)))
    ranges = {}
    for bucket in ("high", "mid", "low"):
        jaws = [r["chimera_jaw"] for r in rows if r["bucket"] == bucket]
        ranges[bucket] = dict(min=min(jaws), max=max(jaws), vowels=len(jaws))
    return dict(chimera_rev=chimera_tables.CHIMERA_REV, body=chimera_tables.BODY, bins=[list(b) for b in CHIMERA_JAW_BINS[:-1]],
                vowels=rows, bucket_ranges=ranges, disagreements=[r["phone"] for r in rows if r["bucket"] != r["chimera_bin"]])


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
