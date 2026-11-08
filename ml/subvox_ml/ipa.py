"""One IPA phone set shared by English, French, Spanish and Mandarin (docs/next-phase-plan.md, item 1).

- English, French and Spanish go through espeak-ng (via `phonemizer`). Whole phrases are converted at
  once, so French liaison ("vous êtes" -> v u z ɛ t) comes out of the phrase, not word by word.
- Mandarin goes through `pypinyin` with phrase context (银行 hang2, 行走 xing2), then a pinyin-to-IPA
  table. Labels keep the tone of each syllable; the visual model is trained on toneless syllables
  and `restore_tones` (a phrase-dictionary language model) picks the tones back.
- The 39-unit ARPAbet inventory of the existing English data maps into the same set
  (`arpabet_to_ipa`), so old labels and checkpoints keep working.

Every phone that leaves this module is in `INVENTORY`; anything else raises, so a new espeak
symbol shows up as a test failure instead of a silent new class.
"""
from __future__ import annotations
import re
import shutil
from functools import lru_cache

LANGUAGES = {"en": "en-us", "fr": "fr-fr", "es": "es", "zh": None}
SPEAKER_STATUS = ("native", "learning", "unknown")

ARPABET = {
    "P": "p", "B": "b", "M": "m", "F": "f", "V": "v", "TH": "θ", "DH": "ð", "T": "t", "D": "d", "S": "s", "Z": "z",
    "N": "n", "L": "l", "SH": "ʃ", "ZH": "ʒ", "CH": "tʃ", "JH": "dʒ", "K": "k", "G": "ɡ", "NG": "ŋ", "R": "ɹ", "W": "w",
    "Y": "j", "HH": "h", "IY": "i", "IH": "ɪ", "EH": "ɛ", "AE": "æ", "AA": "ɑ", "AO": "ɔ", "AH": "ʌ", "UH": "ʊ",
    "UW": "u", "ER": "ɝ", "EY": "eɪ", "AY": "aɪ", "OY": "ɔɪ", "AW": "aʊ", "OW": "oʊ",
}
# Unstressed AH and ER are the reduced vowels, as in the app's lexicon.
ARPABET_UNSTRESSED = {"AH": "ə", "ER": "ɚ"}

# phone -> languages that use it. Diphthongs and affricates are single units, as in the ARPAbet set.
_BY_LANGUAGE = {
    "en": set(ARPABET.values()) | {"ə", "ɚ"},
    "fr": {"p", "b", "m", "f", "v", "t", "d", "n", "s", "z", "ʃ", "ʒ", "k", "ɡ", "ɲ", "ŋ", "l", "ʁ", "j", "w", "ɥ",
           "i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə", "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"},
    "es": {"p", "b", "β", "m", "f", "t", "d", "ð", "n", "s", "z", "θ", "x", "ɡ", "ɣ", "k", "l", "ʎ", "ɲ", "ŋ", "r", "ɾ",
           "tʃ", "ʝ", "j", "w", "a", "e", "i", "o", "u"},
    "zh": {"p", "pʰ", "m", "f", "t", "tʰ", "n", "l", "k", "kʰ", "x", "tɕ", "tɕʰ", "ɕ", "ʈʂ", "ʈʂʰ", "ʂ", "ʐ", "ts", "tsʰ",
           "s", "j", "w", "ɥ", "a", "o", "ɤ", "e", "ɛ", "ə", "i", "u", "y", "ʊ", "ŋ", "ɚ", "ɹ̩", "ɻ̩"},
}
INVENTORY = {p: tuple(sorted(l for l, s in _BY_LANGUAGE.items() if p in s)) for p in sorted(set().union(*_BY_LANGUAGE.values()))}

# espeak-ng symbols folded onto the shared set. Length is not contrastive in any of the four
# languages' labels here, and the lips cannot show it.
_ESPEAK_FOLD = {
    "en": {"ɐ": "ə", "ɜ": "ɝ", "ɒ": "ɑ", "ᵻ": "ɪ", "ɾ": "t", "ʔ": "t", "əl": "əl", "ɔː": "ɔ", "e": "ɛ", "o": "oʊ",
           "a": "æ", "ɪɹ": "ɪɹ", "ʊɹ": "ʊɹ", "ɛɹ": "ɛɹ", "ɑɹ": "ɑɹ", "ɔɹ": "ɔɹ", "aɪɚ": "aɪɚ", "iə": "iə", "r": "ɹ", "n̩": "n"},
    "fr": {"r": "ʁ", "ɜ": "ə", "ɐ": "a"},
    "es": {"ɛ": "e", "ɔ": "o", "ʒ": "ʝ", "ʃ": "tʃ", "v": "β", "ɟ": "ʝ"},
}


def _split(token, language):
    """An espeak token -> inventory phones (some espeak tokens are clusters such as 'ɑɹ' or 'əl')."""
    token = token.replace("ː", "").replace("g", "ɡ")
    token = _ESPEAK_FOLD.get(language, {}).get(token, token)
    if token in _BY_LANGUAGE[language]:
        return [token]
    phones, rest = [], token
    while rest:  # longest match first
        match = next((rest[:n] for n in (3, 2, 1) if rest[:n] in _BY_LANGUAGE[language]), None)
        if match is None:
            raise ValueError(f"espeak phone {token!r} is not in the {language} inventory")
        phones.append(match)
        rest = rest[len(match):]
    return phones


def espeak_available():
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


@lru_cache(maxsize=None)
def _backend(language):
    from phonemizer.backend import EspeakBackend
    return EspeakBackend(LANGUAGES[language], preserve_punctuation=False, with_stress=False, language_switch="remove-flags")


def phonemize(text, language):
    """Text -> list of words, each a list of IPA phones. Mandarin words are syllables."""
    if language == "zh":
        return [s["phones"] for s in mandarin_syllables(text)]
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}")
    from phonemizer.separator import Separator
    raw = _backend(language).phonemize([text], separator=Separator(phone=" ", word=" | "), strip=True)[0]
    words = [[p for token in word.split() for p in _split(token, language)] for word in raw.split("|") if word.strip()]
    if language == "fr":
        words = [_french_glides(w) for w in words]
    return words


def _french_glides(word):
    # espeak writes "suis" as s y i; before a vowel, /y/ is the glide /ɥ/ (and /u/ before /i/ is /w/ in "oui").
    vowels = {"i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə", "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"}
    out = list(word)
    for k in range(len(out) - 1):
        if out[k] == "y" and out[k + 1] in vowels and (k == 0 or out[k - 1] not in vowels):
            out[k] = "ɥ"
    return out


def phones(text, language):
    return [p for word in phonemize(text, language) for p in word]


def arpabet_to_ipa(sequence):
    """'HH AH0 L OW1' or ['HH', 'AH0', ...] -> IPA; the compatibility path for existing English labels."""
    tokens = sequence.split() if isinstance(sequence, str) else sequence
    out = []
    for token in tokens:
        base, stress = re.fullmatch(r"([A-Z]+)([0-2]?)", token).groups()
        if base not in ARPABET:
            raise ValueError(f"Unknown ARPAbet phone {token!r}")
        out.append(ARPABET_UNSTRESSED[base] if stress == "0" and base in ARPABET_UNSTRESSED else ARPABET[base])
    return out


# --- Mandarin ---------------------------------------------------------------------------------

PINYIN_INITIALS = {"b": "p", "p": "pʰ", "m": "m", "f": "f", "d": "t", "t": "tʰ", "n": "n", "l": "l", "g": "k", "k": "kʰ",
                   "h": "x", "j": "tɕ", "q": "tɕʰ", "x": "ɕ", "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ʐ", "z": "ts",
                   "c": "tsʰ", "s": "s"}
# pypinyin "strict" finals: y/w spellings are already undone and ü is written v.
PINYIN_FINALS = {
    "a": ["a"], "o": ["o"], "e": ["ɤ"], "i": ["i"], "u": ["u"], "v": ["y"], "ê": ["ɛ"],
    "ai": ["a", "i"], "ei": ["e", "i"], "ao": ["a", "u"], "ou": ["o", "u"],
    "an": ["a", "n"], "en": ["ə", "n"], "ang": ["a", "ŋ"], "eng": ["ə", "ŋ"], "ong": ["ʊ", "ŋ"], "er": ["ɚ"],
    "ia": ["j", "a"], "ie": ["j", "e"], "iao": ["j", "a", "u"], "iou": ["j", "o", "u"], "ian": ["j", "ɛ", "n"],
    "in": ["i", "n"], "iang": ["j", "a", "ŋ"], "ing": ["i", "ŋ"], "iong": ["j", "ʊ", "ŋ"],
    "ua": ["w", "a"], "uo": ["w", "o"], "uai": ["w", "a", "i"], "uei": ["w", "e", "i"], "uan": ["w", "a", "n"],
    "uen": ["w", "ə", "n"], "uang": ["w", "a", "ŋ"], "ueng": ["w", "ə", "ŋ"],
    "ve": ["ɥ", "e"], "van": ["ɥ", "ɛ", "n"], "vn": ["y", "n"],
}
TONES = {1: "˥", 2: "˧˥", 3: "˨˩˦", 4: "˥˩", 5: ""}


def _syllable_ipa(initial, final):
    if final == "i" and initial in ("z", "c", "s"):
        rhyme = ["ɹ̩"]  # apical vowel of 字 zi
    elif final == "i" and initial in ("zh", "ch", "sh", "r"):
        rhyme = ["ɻ̩"]  # apical vowel of 是 shi
    else:
        rhyme = PINYIN_FINALS[final]
    return ([PINYIN_INITIALS[initial]] if initial else []) + rhyme


def mandarin_syllables(text):
    """Characters -> [{'pinyin': 'hang2', 'toneless': 'hang', 'tone': 2, 'phones': [...]}]; non-Han text is skipped."""
    from pypinyin import Style, pinyin
    hanzi = "".join(ch for ch in text if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿")
    if not hanzi:
        return []
    tone3 = [p[0] for p in pinyin(hanzi, style=Style.TONE3, neutral_tone_with_five=True)]
    initials = [p[0] for p in pinyin(hanzi, style=Style.INITIALS, strict=True)]
    finals = [re.sub(r"\d", "", p[0]) for p in pinyin(hanzi, style=Style.FINALS_TONE3, strict=True, neutral_tone_with_five=True)]
    out = []
    for syllable, initial, final in zip(tone3, initials, finals):
        tone = int(syllable[-1]) if syllable[-1].isdigit() else 5
        out.append(dict(pinyin=syllable, toneless=re.sub(r"\d", "", syllable), tone=tone, phones=_syllable_ipa(initial, final)))
    return out


def strip_tones(syllables):
    return [re.sub(r"\d", "", s) for s in syllables]


@lru_cache(maxsize=1)
def _tone_model():
    """Toneless phrase -> tone patterns with counts, plus per-syllable tone counts, from pypinyin's phrase dictionary."""
    from collections import Counter, defaultdict
    from pypinyin import Style
    from pypinyin.constants import PHRASES_DICT
    from pypinyin.style import convert
    phrases, single = defaultdict(Counter), defaultdict(Counter)
    for readings in PHRASES_DICT.values():
        marked = [convert(r[0], Style.TONE3, strict=False, neutral_tone_with_five=True) for r in readings]
        tones = tuple(int(m[-1]) if m[-1].isdigit() else 5 for m in marked)
        bare = tuple(strip_tones(marked))
        phrases[bare][tones] += 1
        for b, t in zip(bare, tones):
            single[b][t] += 1
    return phrases, single


def restore_tones(toneless, max_phrase=4):
    """Picks tones for a toneless syllable sequence: longest dictionary phrase first, then the
    syllable's most frequent tone. A placeholder for a proper Mandarin language model; the
    interface (toneless in, toned pinyin out) is what the decoder depends on."""
    phrases, single = _tone_model()
    out, k = [], 0
    while k < len(toneless):
        for n in range(min(max_phrase, len(toneless) - k), 1, -1):
            pattern = phrases.get(tuple(toneless[k:k + n]))
            if pattern:
                out += [f"{s}{t}" for s, t in zip(toneless[k:k + n], pattern.most_common(1)[0][0])]
                k += n
                break
        else:
            counts = single.get(toneless[k])
            out.append(f"{toneless[k]}{counts.most_common(1)[0][0] if counts else 5}")
            k += 1
    return out


def validate_language_tag(tag):
    """'en', 'fr', 'es', 'zh' (optionally with a region, e.g. 'fr-CA'); returns the base language."""
    match = re.fullmatch(r"([a-z]{2})(?:-[A-Z]{2})?", str(tag))
    if not match or match.group(1) not in LANGUAGES:
        raise ValueError(f"Unsupported language tag {tag!r}")
    return match.group(1)
