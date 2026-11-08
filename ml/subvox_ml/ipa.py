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
import os
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
    "es": {"ɛ": "e", "ɔ": "o", "ʒ": "ʝ", "ʃ": "tʃ", "v": "β", "ɟ": "ʝ", "dʒ": "ʝ"},
}

# Character-level folds inside clusters. espeak writes Spanish falling diphthongs as oɪ, aɪ, eɪ,
# aʊ, eʊ; the second half is a glide (soy /soj/, causa /kawsa/), never a full vowel. It only
# writes ɪ and ʊ inside these clusters for Spanish; syllabic i and u come out as i and u.
_CHAR_FOLD = {"es": {"ɪ": "j", "ʊ": "w"}, "fr": {}, "en": {}}


def _split(token, language):
    """An espeak token -> inventory phones (some espeak tokens are clusters such as 'ɑɹ' or 'əl')."""
    token = token.replace("ː", "").replace("g", "ɡ")
    token = _ESPEAK_FOLD.get(language, {}).get(token, token)
    for a, b in _CHAR_FOLD.get(language, {}).items():
        token = token.replace(a, b)
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


for _library in ("/opt/homebrew/lib/libespeak-ng.dylib", "/usr/local/lib/libespeak-ng.dylib"):
    # Apps launched from Finder don't have Homebrew on PATH; point phonemizer at the library directly.
    if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ and os.path.exists(_library):
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _library


def espeak_available():
    return bool(shutil.which("espeak-ng") or shutil.which("espeak") or os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"))


# Regional accents: Latin American Spanish has seseo (z, ce, ci are /s/) and yeísmo (ll is /ʝ/).
ACCENTS = {("es", "MX"): "es-419", ("es", "419"): "es-419"}


def _backend(language, accent=None):
    from phonemizer.backend import EspeakBackend
    voice = ACCENTS.get((language, accent), LANGUAGES[language])
    return _cached_backend(voice)


@lru_cache(maxsize=None)
def _cached_backend(voice):
    from phonemizer.backend import EspeakBackend
    return EspeakBackend(voice, preserve_punctuation=False, with_stress=False, language_switch="remove-flags")


def phonemize(text, language, accent=None):
    """Text -> list of words, each a list of IPA phones. Mandarin words are syllables.
    `accent` picks a regional pronunciation where it changes the phones (es + "MX": seseo)."""
    if language == "zh":
        return [s["phones"] for s in mandarin_syllables(text)]
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}")
    from phonemizer.separator import Separator
    raw = _backend(language, accent).phonemize([text], separator=Separator(phone=" ", word=" | "), strip=True)[0]
    words = [[p for token in word.split() for p in _split(token, language)] for word in raw.split("|") if word.strip()]
    if language == "es":
        # espeak's Latin American voice writes intervocalic ll as j + j (calle /kajje/); it is /ʝ/.
        words = [_collapse(w, ("j", "j"), "ʝ") for w in words]
    if language == "fr":
        words = [_french_glides(w) for w in words]
        spelled = re.findall(r"[\w'’-]+", text.lower())
        if len(spelled) == len(words):  # only when espeak's words line up with the spelling
            words = [_french_gn(s, w) for s, w in zip(spelled, words)]
    return words


def _french_glides(word):
    # espeak writes "suis" as s y i; before a vowel, /y/ is the glide /ɥ/ (and /u/ before /i/ is /w/ in "oui").
    vowels = {"i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə", "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"}
    out = list(word)
    for k in range(len(out) - 1):
        if out[k] == "y" and out[k + 1] in vowels and (k == 0 or out[k - 1] not in vowels):
            out[k] = "ɥ"
    return out


def _collapse(word, pair, into):
    out, k = [], 0
    while k < len(word):
        if tuple(word[k:k + 2]) == pair:
            out.append(into); k += 2
        else:
            out.append(word[k]); k += 1
    return out


def _french_gn(spelling, word):
    # espeak writes "gn" before a vowel as n + j (agneau /anjo/); it is /ɲ/ (agneau /aɲo/).
    out = list(word)
    for _ in re.finditer(r"gn(?=[aeiouyéèêàâôîû])", spelling):
        k = next((i for i in range(len(out) - 1) if out[i] == "n" and out[i + 1] == "j"), None)
        if k is None:
            break
        out[k:k + 2] = ["ɲ"]
    return out


def phones(text, language, accent=None):
    return [p for word in phonemize(text, language, accent) for p in word]


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


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC
    return OpenCC("t2s")


def _to_simplified(text):
    return _converter().convert(text) if text else text


NUMERALS = set("〇零一二三四五六七八九十百千万两几第")
PRONOUNS_BEFORE_DEI = set("我你他她它们咱还也就都先")
DEI_VERBS = set("走去来回做学买吃喝跑看上下写说找交给办")


def _corrected(hanzi, tone3, sandhi):
    """Fixes the readings pypinyin's dictionary gets wrong in everyday text.

    Labels and pinyin use citation tones (一 yī, 不 bù) unless `sandhi` asks for the spoken tone
    changes, which only the tone-change drills do: 一/不 become 2nd tone before a 4th tone, and
    一 becomes 4th tone before tones 1-3 (but stays yī in numbers such as 十一 and 第一)."""
    out = list(tone3)
    for i, ch in enumerate(hanzi):
        prev, nxt = (hanzi[i - 1] if i else ""), (hanzi[i + 1] if i + 1 < len(hanzi) else "")
        if ch == "只" and prev in NUMERALS | set("这那哪"):
            out[i] = "zhi1"  # measure word, not "only"
        elif ch == "得" and i:
            out[i] = "dei3" if prev in PRONOUNS_BEFORE_DEI and nxt in DEI_VERBS else "de5"
        elif ch == "着" and not (nxt == "急" or prev in "睡找" or nxt == "火"):
            out[i] = "zhe5"
        elif ch == "谁":
            out[i] = "shei2"
        elif ch in "一不":
            out[i] = "yi1" if ch == "一" else "bu4"
    if sandhi:
        for i, ch in enumerate(hanzi):
            nxt_tone = int(out[i + 1][-1]) if i + 1 < len(out) and out[i + 1][-1].isdigit() else None
            in_number = (i and hanzi[i - 1] in NUMERALS) or (i + 1 < len(hanzi) and hanzi[i + 1] in NUMERALS - {"两", "几"})
            if ch == "一" and nxt_tone and not in_number:
                out[i] = "yi2" if nxt_tone == 4 else "yi4"
            elif ch == "不" and nxt_tone == 4:
                out[i] = "bu2"
    return out


def mandarin_syllables(text, sandhi=False):
    """Characters -> [{'pinyin': 'hang2', 'toneless': 'hang', 'tone': 2, 'phones': [...]}]; non-Han text is skipped."""
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
    hanzi = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf")
    # pypinyin's phrase dictionary is keyed on Simplified characters, so Traditional text is
    # converted first; otherwise 銀行 loses its context and reads yín xíng.
    hanzi = _to_simplified(hanzi)
    if not hanzi:
        return []
    tone3 = _corrected(hanzi, [p[0] for p in pinyin(hanzi, style=Style.TONE3, neutral_tone_with_five=True)], sandhi)
    out = []
    for syllable in tone3:
        tone = int(syllable[-1]) if syllable[-1].isdigit() else 5
        bare = re.sub(r"\d", "", syllable)
        spelled = bare.replace("v", "ü")
        initial, final = get_initials(spelled, strict=True), get_finals(spelled, strict=True).replace("ü", "v")
        out.append(dict(pinyin=syllable, toneless=bare, tone=tone, phones=_syllable_ipa(initial, final)))
    return out


def pinyin_marks(text, sandhi=False):
    """Tone-marked pinyin for display ("yín háng"), from the same corrected readings as the labels."""
    from pypinyin.contrib.tone_convert import to_tone
    return " ".join(to_tone(s["pinyin"]).replace("v", "ü") for s in mandarin_syllables(text, sandhi))


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


def accent_of(tag):
    """'es-MX' -> 'MX' (a regional accent the converter knows), else None."""
    region = str(tag).split("-")[-1] if "-" in str(tag) else None
    return region if (str(tag)[:2], region) in ACCENTS else None


def validate_language_tag(tag):
    """'en', 'fr', 'es', 'zh', optionally with a region ('fr-CA') or, for Chinese, a script
    ('zh-Hans', 'zh-Hant'); returns the base language."""
    match = re.fullmatch(r"([a-z]{2})(?:-[A-Z]{2}|-Han[st])?", str(tag))
    if not match or match.group(1) not in LANGUAGES:
        raise ValueError(f"Unsupported language tag {tag!r}")
    return match.group(1)
