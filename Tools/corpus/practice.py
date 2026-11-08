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
    ("jing-ye-si", "zh", "poem", "静夜思", "李白", 750, 762, ["床前明月光，", "疑是地上霜。", "举头望明月，", "低头思故乡。"]),
    ("chun-xiao", "zh", "poem", "春晓", "孟浩然", 730, 740, ["春眠不觉晓，", "处处闻啼鸟。", "夜来风雨声，", "花落知多少。"]),
]

DRILLS = [
    ("etre", "fr", "être (to be), present", ["je suis", "tu es", "il est", "elle est", "nous sommes", "vous êtes", "ils sont", "elles sont"]),
    ("avoir", "fr", "avoir (to have), present", ["j'ai", "tu as", "il a", "nous avons", "vous avez", "ils ont"]),
    ("ser", "es", "ser (to be), present", ["yo soy", "tú eres", "él es", "nosotros somos", "vosotros sois", "ellos son"]),
    ("estar", "es", "estar (to be), present", ["yo estoy", "tú estás", "él está", "nosotros estamos", "vosotros estáis", "ellos están"]),
    ("to-be", "en", "to be, present", ["I am", "you are", "she is", "we are", "they are"]),
    ("shi", "zh", "是 (to be)", ["我是学生", "你是老师", "他是医生", "我们是朋友"]),
    ("greetings-zh", "zh", "Greetings", ["你好", "谢谢", "再见", "对不起", "没关系"]),
]

# Pairs aimed at sounds that look alike on the lips (or look alike but differ inside the mouth).
MINIMAL_PAIRS = [
    ("en", "p b m", [["pat", "bat", "mat"], ["pie", "buy", "my"], ["pea", "bee", "me"]]),
    ("en", "f v", [["fan", "van"], ["fine", "vine"], ["safe", "save"]]),
    ("en", "θ s", [["think", "sink"], ["thigh", "sigh"]]),
    ("en", "i ɪ", [["sheep", "ship"], ["feet", "fit"]]),
    ("fr", "y u", [["rue", "roue"], ["tu", "tout"], ["vu", "vous"], ["dessus", "dessous"]]),
    ("fr", "p b m", [["pain", "bain", "main"], ["pas", "bas", "mas"]]),
    ("fr", "f v", [["fin", "vin"], ["faux", "veau"]]),
    ("fr", "ɔ̃ ɑ̃", [["bon", "banc"], ["long", "lent"]]),
    ("es", "ɾ r", [["pero", "perro"], ["caro", "carro"], ["coro", "corro"]]),
    ("es", "p b m", [["peso", "beso"], ["pala", "bala", "mala"]]),
    ("zh", "p pʰ m", [["八", "趴", "妈"], ["包", "抛", "猫"]]),
    ("zh", "y i u", [["女", "你", "怒"], ["鱼", "移", "无"]]),
]


def entry(text, language):
    item = dict(text=text, ipa=[" ".join(word) for word in ipa.phonemize(text, language)])
    if language == "zh":
        item["pinyin"] = " ".join(s["pinyin"] for s in ipa.mandarin_syllables(text))
    return item


def build(year=2026):
    pieces = []
    for pid, language, kind, title, author, published, died, lines in PIECES:
        if not (died + 70 < year and published < 1930):
            raise ValueError(f"{title} is not public domain in both the UK and US")
        pieces.append(dict(id=pid, language=language, kind=kind, title=title, author=author, year=published,
                           lines=[entry(line, language) for line in lines]))
    drills = [dict(id=did, language=language, title=title, phrases=[entry(p, language) for p in phrases])
              for did, language, title, phrases in DRILLS]
    pairs = [dict(language=language, contrast=contrast, sets=[[entry(w, language) for w in group] for group in groups])
             for language, contrast, groups in MINIMAL_PAIRS]
    return dict(version=1, pieces=pieces, drills=drills, minimal_pairs=pairs)


if __name__ == "__main__":
    out = ROOT / "Corpus" / "practice" / "practice-v1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n")
    print(out)
