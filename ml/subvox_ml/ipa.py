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
           "s", "j", "w", "ɥ", "a", "o", "ɤ", "e", "ɛ", "ə", "i", "u", "y", "ʊ", "ŋ", "ɚ", "ɹ̩", "ɻ̩", "ɻ"},
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


def _use_packaged_espeak():
    """Point phonemizer at the eSpeak NG library and data shipped in the `espeakng-loader` wheel, so the
    app's own Python runtime needs nothing installed on the Mac. A library already named in
    PHONEMIZER_ESPEAK_LIBRARY wins; its data path is then left to that installation."""
    if "PHONEMIZER_ESPEAK_LIBRARY" in os.environ:
        return
    try:
        import espeakng_loader
        library, data = espeakng_loader.get_library_path(), espeakng_loader.get_data_path()
    except (ImportError, RuntimeError):
        return
    if os.path.exists(library):
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = library
        os.environ["PHONEMIZER_ESPEAK_DATA_PATH"] = data


_use_packaged_espeak()


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
    spoken = _french_spelling(text) if language == "fr" else text
    raw = _backend(language, accent).phonemize([spoken], separator=Separator(phone=" ", word=" | "), strip=True)[0]
    words = [[p for token in word.split() for p in _split(token, language)] for word in raw.split("|") if word.strip()]
    if language == "es":
        # espeak's Latin American voice writes intervocalic ll as j + j (calle /kajje/); it is /ʝ/.
        words = [_collapse(w, ("j", "j"), "ʝ") for w in words]
    if language == "fr":
        # Word by word, so espeak's over-eager liaison never leaks in; linking consonants come
        # back only where French makes liaison (see _french_word).
        spelled = re.findall(r"[\w'’-]+", _french_spelling(text).lower())
        if len(spelled) == len(words):
            # In-sentence readings (they settle homographs like habitions, est), minus wrong liaisons.
            words = [_french_glides(_french_gn(w, _french_in_context(spelled, i, words[i]))) for i, w in enumerate(spelled)]
        else:
            words = [_french_glides(_french_gn(w, _french_word(spelled, i))) for i, w in enumerate(spelled)]
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


# espeak makes liaison after almost any word before a vowel ("tu vas‿au travail"). Liaison is
# only made after these words (determiners, pronouns, numbers, short prepositions and adverbs,
# est, and a few adjectives before a noun); everywhere else the linking consonant is dropped.
LIAISON_AFTER = set("""les des ces mes tes ses nos vos leurs aux un mon ton son en on nous vous ils elles dont quand chez dans
sans sous très plus moins tout deux trois six dix vingt quels quelles quel aucun plusieurs certains est c'est petit petits
grand grands bon bons gros mauvais premier dernier aux leur cent cents quelques certain ancien moyen plein prochain""".split())
DENASALISE = {"bon": ("ɔ̃", "ɔ"), "certain": ("ɛ̃", "ɛ"), "ancien": ("ɛ̃", "ɛ"), "moyen": ("ɛ̃", "ɛ"), "plein": ("ɛ̃", "ɛ"), "prochain": ("ɛ̃", "ɛ")}
FRENCH_WORDS = {"rouen": ["ʁ", "w", "ɑ̃"], "amiens": ["a", "m", "j", "ɛ̃"], "dreyfus": ["d", "ʁ", "ɛ", "f", "y", "s"]}
# Old spellings (Ronsard) read as their modern pronunciation.
ARCHAIC = {"avoit": "avait", "desclose": "déclose", "ceste": "cette", "vostre": "votre", "vesprée": "vêprée", "estoit": "était",
           "point": "point", "ayant": "ayant"}


def _french_spelling(text):
    """Spelling tweaks before espeak: '-y' is the pronoun y (vas-y), not the letter; archaic forms."""
    # vas-y, allons-y: obligatory liaison, and espeak would read a lone "y" as the letter "i grec".
    text = re.sub(r"s-y\b", "-zi", text)
    text = re.sub(r"-y\b", "-i", text)
    return re.sub(r"\b(" + "|".join(ARCHAIC) + r")\b", lambda m: ARCHAIC[m.group(1)], text)


@lru_cache(maxsize=8192)
def _french_alone(word):
    return tuple(p for w in _french_tokens(word) for p in w)


def _french_tokens(text):
    from phonemizer.separator import Separator
    raw = _backend("fr").phonemize([text], separator=Separator(phone=" ", word=" | "), strip=True)[0]
    return [[p for token in word.split() for p in _split(token, "fr")] for word in raw.split("|") if word.strip()]


VOWEL_START = re.compile(r"^[aeiouyéèêëàâîïôûùœæh]")


def _french_in_context(spelled, i, word):
    """espeak's reading of word i in its sentence, with a final linking consonant removed unless
    French makes liaison there. A final consonant counts as linking when the word said alone
    doesn't end in it and the next word starts with a vowel or h."""
    word = list(word)
    spelling = spelled[i]
    if spelling in FRENCH_WORDS:
        return list(FRENCH_WORDS[spelling])
    nxt = spelled[i + 1] if i + 1 < len(spelled) else ""
    if spelling == "comment" and nxt.startswith(("a", "e", "i", "o", "u", "é")) and word[-1:] != ["t"]:
        return word + ["t"]
    if spelling in DENASALISE and nxt and VOWEL_START.match(nxt) and nxt != "et" and word and word[-1] == DENASALISE[spelling][0]:
        return word[:-1] + [DENASALISE[spelling][1], "n"]  # ancien‿ami /ɑ̃sjɛn ami/
    if len(word) >= 2 and word[-1] == word[-2] and word[-1] in ("z", "t", "n", "p", "ʁ"):
        word = word[:-1]  # espeak's liaison on top of a final consonant (roses‿envolées /ʁozz/)
        if spelling not in LIAISON_AFTER:
            return word
    if not word or not nxt or not VOWEL_START.match(nxt) or word[-1] not in ("z", "t", "n", "p", "ʁ"):
        return word
    alone = list(_french_alone(spelling))
    if alone and alone[-1] == word[-1]:
        return word  # a consonant the word always has (sept, avec), not a liaison
    trigger = re.split(r"['’]", spelling)[-1]
    if spelling == "vingt" and nxt == "et":
        return word  # vingt et un: the t is said
    if nxt != "et" and (spelling in LIAISON_AFTER or trigger in LIAISON_AFTER):
        if spelling in DENASALISE:
            nasal, oral = DENASALISE[spelling]
            return [oral if p == nasal else p for p in word]
        return word
    return word[:-1]


def _french_word(spelled, i):
    """Phones of word i: said alone, plus the linking consonant when liaison applies (the
    previous-word rule: after liaison words, before a vowel or mute h, never before et)."""
    word = spelled[i]
    if word in FRENCH_WORDS:
        return list(FRENCH_WORDS[word])
    alone = list(_french_alone(word))
    nxt = spelled[i + 1] if i + 1 < len(spelled) else ""
    trigger = re.split(r"['’]", word)[-1]  # j'en -> en, c'est -> est (and the whole word)
    if not nxt or nxt == "et" or not VOWEL_START.match(nxt):
        return alone
    if word == "comment" and nxt.startswith(("a", "e", "i", "o", "u", "é")):
        return alone + ["t"]
    if word not in LIAISON_AFTER and trigger not in LIAISON_AFTER:
        return alone
    pair = _french_tokens(f"{word} {nxt}")
    linked = pair[0] if len(pair) == 2 else alone  # espeak decides h aspiré (les haricots)
    if len(linked) == len(alone) + 1 and linked[:-1] == alone and linked[-1] in ("z", "t", "n", "p", "ʁ"):
        if word in DENASALISE:
            nasal, oral = DENASALISE[word]
            linked = [oral if p == nasal else p for p in linked]
        return linked
    return alone


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
DURATIVE_VERBS = set("拿看坐站躺穿戴开关等听唱笑哭写想带放挂住拉抱牵背跟陪守开着关")


# Readings that depend on meaning in a specific context (classical poems above all):
# (context, character, reading). 朝辞白帝 zhāo "morning", 一日还 huán "return", 万重山 chóng "layers".
CONTEXT_READINGS = [("朝辞", "朝", "zhao1"), ("日还", "还", "huan2"), ("重山", "重", "chong2"), ("不胜簪", "胜", "sheng1"),
                    ("返景", "景", "ying3")]
# 儿化: these words are one rhotacised syllable (哪儿 nǎr), not two.
ERHUA = set("哪儿 这儿 那儿 点儿 会儿 块儿 玩儿 孩儿 边儿 空儿 事儿 片儿 味儿 劲儿 本儿 号儿 儿".split()) - {"儿"}

NEUTRAL_SECOND = set("""师傅 爸爸 妈妈 哥哥 姐姐 弟弟 妹妹 爷爷 奶奶 姥姥 叔叔 舅舅 宝宝 谢谢 看看 说说 试试 想想 听听 走走
东西 时候 什么 怎么 喜欢 衣服 朋友 学生 多少 麻烦 漂亮 休息 知道 明白 先生 地方 事情 认识 意思 舒服 客气 告诉 名字
眼睛 耳朵 头发 早上 晚上 屋子 房子 桌子 椅子 孩子 儿子 杯子 包子 饺子 筷子 本子 日子 鼻子 肚子 我们 你们 他们 她们
咱们 人们 这么 那么 多么 关系 便宜 认得 记得 觉得 晓得 舍得 部分 窗户 豆腐 萝卜 葡萄 玻璃 姑娘 丈夫 媳妇 打扮 打听
聪明 糊涂 热闹 凉快 暖和 清楚 干净 结实 商量 收拾 招呼 照顾 消息 生意 力气 价钱 规矩 道理 钥匙 太阳 月亮 尾巴""".split())


def _corrected(hanzi, tone3, sandhi, neutral=True):
    """Fixes the readings pypinyin's dictionary gets wrong in everyday text.

    Labels and pinyin use citation tones (一 yī, 不 bù) unless `sandhi` asks for the spoken tone
    changes, which only the tone-change drills do: 一/不 become 2nd tone before a 4th tone, and
    一 becomes 4th tone before tones 1-3 (but stays yī in numbers such as 十一 and 第一)."""
    out = list(tone3)
    for i, ch in enumerate(hanzi):
        prev, nxt = (hanzi[i - 1] if i else ""), (hanzi[i + 1] if i + 1 < len(hanzi) else "")
        if ch == "只" and prev in NUMERALS | set("这那哪"):
            out[i] = "zhi1"  # measure word, not "only"
        elif ch == "得" and i and out[i] in ("de2", "de5") and nxt not in "到了意分知出" and prev not in "获取赢难不晓懂":
            out[i] = "dei3" if prev in PRONOUNS_BEFORE_DEI and nxt in DEI_VERBS else "de5"
        elif ch == "着" and prev in DURATIVE_VERBS and nxt not in "急火了":
            out[i] = "zhe5"  # durative 着 after a verb (拿着, 看着); other readings left to the dictionary
        elif ch == "谁":
            out[i] = "shei2"
        elif ch in "一不":
            out[i] = "yi1" if ch == "一" else "bu4"
    # Words whose second syllable is lexically neutral (not tone sandhi, so shown in every drill).
    for context, ch, reading in CONTEXT_READINGS:
        at = hanzi.find(context)
        while at >= 0:
            k = at + context.index(ch)
            out[k] = reading
            at = hanzi.find(context, at + 1)
    for k in range(len(hanzi) - 1) if neutral else ():
        if hanzi[k:k + 2] in NEUTRAL_SECOND and hanzi[k:k + 4] != "东西南北":
            out[k + 1] = re.sub(r"\d$", "5", out[k + 1])
    if sandhi:
        for i, ch in enumerate(hanzi):
            nxt_tone = int(out[i + 1][-1]) if i + 1 < len(out) and out[i + 1][-1].isdigit() else None
            prev = hanzi[i - 1] if i else ""
            nxt = hanzi[i + 1] if i + 1 < len(hanzi) else ""
            in_number = prev in NUMERALS or nxt in set("〇零一二三四五六七八九十") or nxt in set("月号楼")
            between = prev and nxt and prev == nxt  # 看一看, 去不去: neutral
            if between:
                out[i] = ("yi5" if ch == "一" else "bu5")
            elif ch == "一" and nxt_tone and not in_number:
                out[i] = "yi2" if nxt_tone == 4 else "yi4"
            elif ch == "不" and nxt_tone == 4:
                out[i] = "bu2"
    return out


def mandarin_syllables(text, sandhi=False, neutral=True):
    """Characters -> [{'pinyin': 'hang2', 'toneless': 'hang', 'tone': 2, 'phones': [...]}]; non-Han text is skipped."""
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
    hanzi = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf")
    # pypinyin's phrase dictionary is keyed on Simplified characters, so Traditional text is
    # converted first; otherwise 銀行 loses its context and reads yín xíng.
    hanzi = _to_simplified(hanzi)
    if not hanzi:
        return []
    tone3 = _corrected(hanzi, [p[0] for p in pinyin(hanzi, style=Style.TONE3, neutral_tone_with_five=True)], sandhi, neutral)
    out = []
    for k, syllable in enumerate(tone3):
        if hanzi[k] == "儿" and k and hanzi[k - 1:k + 1] in ERHUA and out:
            # 儿化: one rhotacised syllable (哪儿 nǎr /naɻ/); drop a final n, as in 点儿 diǎnr /tjaɻ/.
            prev = out[-1]
            phones = prev["phones"]
            if phones[-1:] == ["n"]:
                phones = ["a" if p == "ɛ" else p for p in phones[:-1]]  # 点儿 /tjɛn/ -> /tjaɻ/
            prev.update(pinyin=prev["toneless"] + "r" + str(prev["tone"]), toneless=prev["toneless"] + "r", phones=phones + ["ɻ"])
            continue
        tone = int(syllable[-1]) if syllable[-1].isdigit() else 5
        bare = re.sub(r"\d", "", syllable)
        spelled = bare.replace("v", "ü")
        initial, final = get_initials(spelled, strict=True), get_finals(spelled, strict=True).replace("ü", "v")
        phones = _syllable_ipa(initial, final)
        if tone == 5:
            phones = ["ə" if p == "ɤ" else p for p in phones]  # neutral 的 了 么 个 着: /tə lə mə/
        out.append(dict(pinyin=syllable, toneless=bare, tone=tone, phones=phones))
    return out


def pinyin_marks(text, sandhi=False, neutral=True):
    """Tone-marked pinyin for display ("yín háng"), from the same corrected readings as the labels."""
    from pypinyin.contrib.tone_convert import to_tone
    return " ".join(to_tone(s["pinyin"]).replace("v", "ü") for s in mandarin_syllables(text, sandhi, neutral))


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
