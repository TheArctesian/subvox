"""Builds Corpus/practice/practice-v1.json for the Train tabs (docs/next-phase-plan.md, item 3).

Every text is public domain in both the UK (author died more than 70 years ago) and the US
(published before 1930). No Churchill or King, and no US federal works beyond 19th-century ones
that are public domain by age everywhere. IPA comes from ml/subvox_ml/ipa.py so the app shows the
same phones the trainer uses.

    cd ml && uv run python ../Tools/corpus/practice.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from subvox_ml import ipa  # noqa: E402

# (id, language, kind, title, author, year, died, lines)
PIECES = [
    ("gettysburg", "en", "speech", "Gettysburg Address", "Abraham Lincoln", 1863, 1865, [
        "Four score and seven years ago our fathers brought forth on this continent a new nation,",
        "conceived in Liberty, and dedicated to the proposition that all men are created equal.",
        "Now we are engaged in a great civil war,",
        "testing whether that nation, or any nation so conceived and so dedicated, can long endure."]),
    ("tilbury", "en", "speech", "Speech to the Troops at Tilbury", "Elizabeth I", 1588, 1603, [
        "I know I have the body of a weak and feeble woman;",
        "but I have the heart and stomach of a king, and of a king of England too."]),
    ("aint-i-a-woman", "en", "speech", "Ain't I a Woman?", "Sojourner Truth", 1851, 1883, [
        "That man over there says that women need to be helped into carriages,",
        "and lifted over ditches, and to have the best place everywhere."]),
    ("sonnet-18", "en", "poem", "Sonnet 18", "William Shakespeare", 1609, 1616, [
        "Shall I compare thee to a summer's day?", "Thou art more lovely and more temperate:",
        "Rough winds do shake the darling buds of May,", "And summer's lease hath all too short a date."]),
    ("tyger", "en", "poem", "The Tyger", "William Blake", 1794, 1827, [
        "Tyger Tyger, burning bright,", "In the forests of the night;",
        "What immortal hand or eye,", "Could frame thy fearful symmetry?"]),
    ("corbeau", "fr", "poem", "Le Corbeau et le Renard", "Jean de La Fontaine", 1668, 1695, [
        "Maître Corbeau, sur un arbre perché,", "Tenait en son bec un fromage.",
        "Maître Renard, par l'odeur alléché,", "Lui tint à peu près ce langage :"]),
    ("demain-des-l-aube", "fr", "poem", "Demain, dès l'aube", "Victor Hugo", 1856, 1885, [
        "Demain, dès l'aube, à l'heure où blanchit la campagne,", "Je partirai. Vois-tu, je sais que tu m'attends.",
        "J'irai par la forêt, j'irai par la montagne.", "Je ne puis demeurer loin de toi plus longtemps."]),
    ("rima-xxi", "es", "poem", "Rima XXI", "Gustavo Adolfo Bécquer", 1871, 1870, [
        "¿Qué es poesía?, dices mientras clavas", "en mi pupila tu pupila azul.",
        "¡Qué es poesía! ¿Y tú me lo preguntas?", "Poesía... eres tú."]),
    ("caminante", "es", "poem", "Proverbios y cantares XXIX", "Antonio Machado", 1912, 1939, [
        "Caminante, son tus huellas el camino y nada más;", "caminante, no hay camino, se hace camino al andar."]),
    ("jing-ye-si", "zh", "poem", ("静夜思", "靜夜思"), ("李白", "李白"), 750, 762, [
        ("床前明月光，", "床前明月光，"), ("疑是地上霜。", "疑是地上霜。"), ("举头望明月，", "舉頭望明月，"), ("低头思故乡。", "低頭思故鄉。")]),
    ("chun-xiao", "zh", "poem", ("春晓", "春曉"), ("孟浩然", "孟浩然"), 730, 740, [
        ("春眠不觉晓，", "春眠不覺曉，"), ("处处闻啼鸟。", "處處聞啼鳥。"), ("夜来风雨声，", "夜來風雨聲，"), ("花落知多少。", "花落知多少。")]),
]

# Grammar drills per language (Tools/corpus/grammar/*.json): A1–B1 (CEFR) and HSK 1–4 topics,
# 5–8 short phrases each, with a level, topic and one-line note. Chinese phrases carry both
# simplified ("hans") and traditional ("hant") forms.
GRAMMAR = [ROOT / "Tools" / "corpus" / "grammar" / name for name in ("grammar-fr-es.json", "grammar-zh-en.json", "everyday.json", "extra.json")]
# Valencian is not Spanish: the Spanish converter would give it wrong IPA, so it waits for a Catalan one.
SKIP_DRILLS = {"es-valencia"}


def load_drills():
    drills = []
    for path in GRAMMAR:
        for language, items in json.loads(path.read_text()).items():
            for d in items:
                if f"{language}-{d['id']}" in SKIP_DRILLS:
                    continue
                phrases = [p if isinstance(p, str) else dict(p) for p in d["phrases"]]
                drills.append((f"{language}-{d['id']}", language, d["title"], phrases,
                               dict(level=d.get("level"), topic=d.get("topic"), note=d.get("note"), source=d.get("source"),
                                    region=d.get("region"))))
    return drills


# Pairs aimed at sounds that look alike on the lips (or look alike but differ inside the mouth).
MINIMAL_PAIRS = [
    ("en", "p b m", [["pat", "bat", "mat"], ["pie", "buy", "my"], ["pea", "bee", "me"]]),
    ("en", "f v", [["fan", "van"], ["fine", "vine"], ["safe", "save"]]),
    ("en", "θ s", [["think", "sink"], ["thigh", "sigh"]]),
    ("en", "i ɪ", [["sheep", "ship"], ["feet", "fit"], ["leave", "live"]]),
    ("en", "b v", [["ban", "van"], ["berry", "very"], ["boat", "vote"]]),
    ("en", "ʃ tʃ", [["ship", "chip"], ["share", "chair"], ["wash", "watch"]]),
    ("en", "s z", [["bus", "buzz"], ["sip", "zip"], ["rice", "rise"]]),
    ("en", "h", [["hat", "at"], ["hold", "old"], ["heat", "eat"]], "h or no h"),
    ("en", "ɪ ɛ", [["bit", "bet"], ["sit", "set"], ["pin", "pen"]]),
    ("en", "p b", [["cap", "cab"], ["rope", "robe"], ["tap", "tab"]], "word endings p b"),
    ("en", "t d", [["bat", "bad"], ["seat", "seed"], ["cart", "card"]], "word endings t d"),
    ("en", "s ʃ", [["sea", "she"], ["sip", "ship"], ["sell", "shell"]]),
    ("en", "j dʒ", [["yes", "Jess"], ["yellow", "jello"], ["year", "jeer"]]),
    ("en", "æ ʌ ɑ", [["cat", "cut", "cot"], ["cap", "cup", "cop"], ["hat", "hut", "hot"]]),
    ("fr", "y u", [["rue", "roue"], ["tu", "tout"], ["vu", "vous"], ["dessus", "dessous"]]),
    ("fr", "p b m", [["pain", "bain", "main"], ["pas", "bas", "mas"]]),
    ("fr", "f v", [["fin", "vin"], ["faux", "veau"]]),
    ("fr", "ɔ̃ ɑ̃", [["bon", "banc"], ["long", "lent"]]),
    ("es", "ɾ r", [["pero", "perro"], ["caro", "carro"], ["coro", "corro"]]),
    ("es", "p b m", [["peso", "beso"], ["pala", "bala", "mala"]]),
    ("zh", "p pʰ m", [[("八", "八"), ("趴", "趴"), ("妈", "媽")], [("包", "包"), ("抛", "拋"), ("猫", "貓")]]),
    ("zh", "y i u", [[("女", "女"), ("你", "你"), ("怒", "怒")], [("鱼", "魚"), ("移", "移"), ("无", "無")]]),
]


# When each speech was given; poems show their publication year instead.
SPOKEN = {"gettysburg": "19 November 1863", "tilbury": "9 August 1588", "aint-i-a-woman": "29 May 1851"}


# Vosotros forms are Spain-only; Español (México) leaves those phrases out.
import re as _re
VOSOTROS = _re.compile(r"\b(vosotr[oa]s|os|sois|tenéis|estáis|vais|habéis|sabéis|podéis|queréis|hacéis|decís|venís|divertís|"
                       r"coméis|habláis|vivís|seáis|tengáis|perdonad|sabéis|tenéis|levantaos|sentaos)\b", _re.IGNORECASE)

# Tone-change drills show the spoken tones of 一 and 不; everything else shows dictionary tones.
SANDHI_DRILLS = {"zh-yi-tone-change", "zh-bu-tone-change"}


def entry(text, language, sandhi=False, accent=None):
    """A line or phrase. Mandarin text is a (simplified, traditional) pair; IPA and pinyin come
    from the simplified form, which pypinyin reads with the most reliable phrase context."""
    hant = None
    if isinstance(text, tuple):
        text, hant = text
    item = dict(text=text, ipa=["".join(word) for word in ipa.phonemize(text, language, accent)])
    if language == "es" and accent is None:
        # Shared Spanish lines carry Mexican phones too (seseo, yeísmo); the app shows them in Español (México).
        mx = ["".join(word) for word in ipa.phonemize(text, language, "MX")]
        if mx != item["ipa"]:
            item["ipa_mx"] = mx
        if VOSOTROS.search(text):
            item["region"] = "ES"
    if language == "zh":
        item["hant"] = hant or text
        item["pinyin"] = ipa.pinyin_marks(text, sandhi)
        from pypinyin.style.bopomofo import BopomofoConverter
        item["zhuyin"] = " ".join(BopomofoConverter().to_bopomofo(p) for p in item["pinyin"].split())
    return item


def traditional(text):
    """Simplified -> Traditional with Mainland vocabulary kept (OpenCC s2t, not s2tw)."""
    from opencc import OpenCC
    return OpenCC("s2t").convert(text)


def traditional_title(title, meta):
    # OpenCC can't tell the measure word 隻 from the adverb 只 out of context, and keeps the
    # aspect particle as 着 where Traditional text (and these drills' phrases) write 著.
    hant = traditional(title).replace("着", "著")
    return hant.replace("只", "隻") if meta.get("topic") == "Measure words" else hant


# Meanings for the minimal-pair words, so a learner can tell "caro" from "carro".
PAIR_GLOSS = {
    "fr": dict(rue="street", roue="wheel", tu="you", tout="all", vu="seen", vous="you (formal)", dessus="on top", dessous="underneath",
               pain="bread", bain="bath", main="hand", pas="step; not", bas="low", mas="farmhouse", fin="end", vin="wine",
               faux="false", veau="calf", bon="good", banc="bench", long="long", lent="slow"),
    "es": dict(pero="but", perro="dog", caro="expensive", carro="cart; car", coro="choir", corro="I run", peso="weight", beso="kiss",
               pala="shovel", bala="bullet", mala="bad"),
    "zh": {"八": "eight", "趴": "lie face down", "妈": "mum", "包": "bag", "抛": "throw", "猫": "cat", "女": "woman", "你": "you",
           "怒": "anger", "鱼": "fish", "移": "move", "无": "without"},
}


def glossed(word, language):
    item = entry(word, language)
    gloss = PAIR_GLOSS.get(language, {}).get(item["text"])
    if gloss:
        item["gloss"] = gloss
    return item


def phrase(p, language, sandhi=False, accent=None):
    """A drill phrase: a string, or {"hans", "hant"} / {"text"} with an optional English gloss."""
    if isinstance(p, str):
        return entry(p, language, sandhi, accent)
    item = entry((p["hans"], p["hant"]) if "hans" in p else p["text"], language, sandhi, accent)
    if p.get("gloss"):
        item["gloss"] = p["gloss"]
    return item


def both(value):
    return value if isinstance(value, tuple) else (value, value)


# More pieces (Tools/corpus/pieces/*.json): each with source URL and a public-domain portrait.
PIECE_FILES = sorted((ROOT / "Tools" / "corpus" / "pieces").glob("*.json"))
SOUNDS = ROOT / "Tools" / "corpus" / "sounds.json"
# Guides for learners whose own language isn't English: {speaker: {practised language: {ipa: {name, guide, tip}}}}.
LOCAL_SOUNDS = sorted((ROOT / "Tools" / "corpus" / "sounds-local").glob("*.json"))


def readable(line, limit=18):
    """Splits a long prose sentence at commas, semicolons and colons into chunks of at most about
    `limit` words, so each take is one breath. The words are unchanged, only re-broken."""
    import re as _re
    if len(line.split()) <= limit + 4:
        return [line]
    parts = _re.split(r"(?<=[,;:])\s+", line)
    chunks, current = [], ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if current and len(candidate.split()) > limit:
            chunks.append(current)
            current = part
        else:
            current = candidate
    return chunks + ([current] if current else [])


def load_pieces():
    extra = []
    for path in PIECE_FILES:
        for p in json.loads(path.read_text())["pieces"]:
            lines = [(l["hans"], l["hant"]) if isinstance(l, dict) else chunk for l in p["lines"]
                     for chunk in ([l] if isinstance(l, dict) else readable(l))]
            title = (p["title"], p.get("title_hant") or p["title"]) if p["language"] == "zh" else p["title"]
            author = (p["author"], p.get("author_hant") or p["author"]) if p["language"] == "zh" else p["author"]
            extra.append((p["id"], p["language"], p["kind"], title, author, p["year"], p["died"], lines, p.get("date")))
    return extra


def load_sounds():
    """Per language: each sound with a guide, a mouth tip, an IPA string to synthesise, and
    example words whose phones are checked to contain the sound."""
    if not SOUNDS.exists():
        return {}, []
    out, problems = {}, []
    local = {}
    for path in LOCAL_SOUNDS:
        for speaker, targets in json.loads(path.read_text()).items():
            for target, guides in targets.items():
                for symbol, guide in guides.items():
                    local.setdefault((target, symbol), {})[speaker] = guide
    for language, items in json.loads(SOUNDS.read_text()).items():
        for item in items:
            examples = []
            for e in item["examples"]:
                try:
                    examples.append(entry((e["hans"], e["hant"]) if isinstance(e, dict) else e, language))
                except ValueError as error:
                    problems.append(f"{language} {item['ipa']}: example {e!r} dropped ({error})")
            if item["group"] != "Tones":
                if item["ipa"] not in ipa.INVENTORY or language not in ipa.INVENTORY[item["ipa"]]:
                    problems.append(f"{language} {item['ipa']}: not in the shared inventory")
                for ex in examples:
                    if item["ipa"] not in [ph for word in ipa.phonemize(ex["text"], language) for ph in word]:
                        problems.append(f"{language} {item['ipa']}: example {ex['text']!r} has /{' '.join(ex['ipa'])}/")
            out.setdefault(language, []).append(dict(ipa=item["ipa"], group=item["group"], name=item["name"], guide=item["guide"],
                                                     tip=item.get("tip"), say=item["say"], examples=examples,
                                                     local=local.get((language, item["ipa"]))))
    return out, problems


def build(year=2026):
    pieces = []
    for pid, language, kind, title, author, published, died, lines, *rest in PIECES + load_pieces():
        date = rest[0] if rest else SPOKEN.get(pid)
        if not (died + 70 < year and published < 1930):
            raise ValueError(f"{title} is not public domain in both the UK and US")
        (title, title_hant), (author, author_hant) = both(title), both(author)
        pieces.append(dict(id=pid, language=language, kind=kind, title=title, title_hant=title_hant, author=author,
                           author_hant=author_hant, year=published, date=date or (f"c. {published}" if published < 1000 else None),
                           lines=[entry(line, language) for line in lines]))
    drills = [dict(id=did, language=language, title=title, title_hant=traditional_title(title, meta) if language == "zh" else None,
                   note_hant=traditional_title(meta.get("note") or "", meta) if language == "zh" and meta.get("note") else None,
                   phrases=[phrase(p, language, did in SANDHI_DRILLS, meta.get("region")) for p in phrases], **meta)
              for did, language, title, phrases, meta in load_drills()]
    pairs = [dict(language=language, contrast=contrast, label=(rest[0] if rest else None),
                  sets=[[glossed(w, language) for w in group] for group in groups])
             for language, contrast, groups, *rest in MINIMAL_PAIRS]
    sounds, problems = load_sounds()
    return dict(version=1, pieces=pieces, drills=drills, minimal_pairs=pairs, sounds=sounds), problems


if __name__ == "__main__":
    out = ROOT / "Corpus" / "practice" / "practice-v1.json"
    out.parent.mkdir(exist_ok=True)
    corpus, problems = build()
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=1) + "\n")
    print(out)
    for problem in problems:
        print("check:", problem)
