#!/usr/bin/env python3
"""Community reading corpus: public-domain sources -> easy-to-say prompts grouped in passages.

Pipeline (docs/community-dataset-design.md, section 3):
  1. `sources.tsv` lists every text with provenance and a UK + US public-domain basis. The build
     refuses any source that fails the rule for the current year.
  2. Each text is downloaded from Project Gutenberg once, cached, and checked against its title.
  3. Sentences are filtered so anyone can read them: 4-16 words, Flesch-Kincaid grade <= 6, every
     word in the app's pronunciation lexicon, no numerals/abbreviations/dialogue fragments/names,
     no archaic forms unless the source allows them, British spellings normalised to US.
  4. Consecutive accepted sentences form passages (recorded in order, shown with title/author).
  5. A fixed calibration set (40 short sentences, greedy diphone cover) is recorded by everyone.
  6. 10% of passages, chosen by hash, are held out for evaluation and never assigned.
  7. `assign` picks passages that fill diphone gaps given current counts.

Usage:
  Tools/corpus/community.py build [--version 1] [--out Corpus/community]
  Tools/corpus/community.py assign --counts counts.json [--n 5] [--recorded ids.txt]
"""

import argparse
import collections
import csv
import datetime
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "Corpus"
CACHE = Path.home() / "Library/Caches/Subvox/community"

MIN_WORDS, MAX_WORDS = 4, 16
MAX_GRADE = 6.0
PASSAGE_GAP = 6        # accepted sentences more than this many source sentences apart start a new passage
PASSAGE_MAX = 6        # prompts per passage
CALIBRATION_SIZE = 40
CALIBRATION_WORDS = (4, 10)
CALIBRATION_PER_SOURCE = 3
CALIBRATION_MAX_RANK = 5000  # calibration words must be among the 5,000 most common in Corpus/words.tsv
HELDOUT_MODULUS = 10   # passage hash % 10 == 0 -> held out (about 10%)

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
ARCHAIC = {"thee", "thou", "thy", "thine", "hath", "doth", "dost", "hast", "art", "wilt", "shalt", "ye", "ere", "oft",
           "nay", "whence", "thither", "hither", "wherefore", "methinks", "o'er", "e'er", "ne'er", "'tis", "'twas",
           "tis", "twas", "unto", "betwixt", "anon", "forsooth", "prithee", "wouldst", "couldst", "shouldst", "didst",
           "canst", "mayst", "saith", "spake", "aught", "naught", "whilst", "amongst", "thereof", "wherein", "fain",
           "hence", "yonder", "alas", "lo", "doth", "maketh", "goeth", "cometh"}
BLOCKED = {"nigger", "niggers", "nigga", "negro", "negroes", "darky", "darkies", "coon", "coons", "injun", "injuns",
           "savage", "savages", "squaw", "gypsy", "gypsies", "lynch", "lynched", "lynching",
           "whip", "whipped", "whipping", "kill", "killed", "dead", "blood", "dat", "dem", "dis", "ain't"}
ABBREVIATION = re.compile(r"\b(Mr|Mrs|Dr|St|Mt|Jr|Sr|Co|vs|etc|viz|i\.e|e\.g)\b\.?", re.I)
ALLOWED_CHARS = re.compile(r"^[A-Za-z ,;:'!?.\-]+$")
US_SPELLING = {
    "colour": "color", "colours": "colors", "coloured": "colored", "honour": "honor", "honours": "honors",
    "honoured": "honored", "favour": "favor", "favourite": "favorite", "favoured": "favored", "labour": "labor",
    "neighbour": "neighbor", "neighbours": "neighbors", "neighbourhood": "neighborhood", "behaviour": "behavior",
    "humour": "humor", "rumour": "rumor", "harbour": "harbor", "parlour": "parlor", "vapour": "vapor",
    "splendour": "splendor", "odour": "odor", "valour": "valor", "armour": "armor", "endeavour": "endeavor",
    "grey": "gray", "centre": "center", "theatre": "theater", "metre": "meter", "travelled": "traveled",
    "travelling": "traveling", "jewellery": "jewelry", "plough": "plow", "realise": "realize",
    "realised": "realized", "recognise": "recognize", "recognised": "recognized", "apologise": "apologize",
    "civilised": "civilized", "organised": "organized", "connexion": "connection", "shew": "show", "shewed": "showed",
    "to-day": "today", "to-morrow": "tomorrow", "to-night": "tonight",
}


# ---------- sources and provenance ----------

def load_sources(path=CORPUS / "community/sources.tsv"):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def public_domain_problems(source, year=None):
    """Reasons a source is not public domain in both the UK (death + 70) and US (published 95+ years ago)."""
    year = year or datetime.date.today().year
    problems = []
    death = int(source["author_death_year"])
    if death > year - 71:
        problems.append(f"author died {death}; UK protection runs to the end of {death + 70}")
    if int(source["first_published"]) > year - 96:
        problems.append(f"first published {source['first_published']}; not yet US public domain")
    if source.get("translator"):
        translator_death = source.get("translator_death_year") or ""
        if not translator_death.isdigit() or int(translator_death) > year - 71:
            problems.append("translation without a translator death year old enough for UK public domain")
    return problems


def fetch(source, cache=CACHE):
    """Downloads once, then checks the Gutenberg header names the expected title."""
    cache.mkdir(parents=True, exist_ok=True)
    gid = source["gutenberg_id"]
    path = cache / f"pg{gid}.txt"
    if not path.exists():
        print(f"Downloading {source['title']}", file=sys.stderr)
        urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", path)
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    header = re.search(r"^Title:\s*(.+)$", text, re.M)
    expected = re.sub(r"\W+", " ", source["title"]).lower().split()[:2]
    if not header or not all(word in re.sub(r"\W+", " ", header.group(1)).lower() for word in expected):
        raise ValueError(f"Gutenberg #{gid} is not {source['title']!r}")
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    return text[start.end() if start else 0:end.start() if end else len(text)]


# ---------- lexicon and phonetics ----------

def load_lexicon(corpus=CORPUS):
    """The app's own pronunciations (Corpus/lexicon.tsv, then Corpus/words.tsv), stress removed."""
    lexicon = {}
    for name in ("lexicon.tsv", "words.tsv"):
        for line in (corpus / name).read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] not in lexicon:
                lexicon[parts[0].lower()] = [p.rstrip("012") for p in parts[1].split()]
    return lexicon


def load_ranks(corpus=CORPUS):
    """Frequency rank (0 = most common) from Corpus/words.tsv, which is sorted by pool-book count."""
    return {line.split("\t")[0].lower(): i for i, line in enumerate((corpus / "words.tsv").read_text().splitlines())}


def words_of(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def prompt_id(text):
    """Same identity as SentencePrompt.identity in the app and group_id in subvox_ml."""
    return hashlib.sha256(" ".join(words_of(text)).encode()).hexdigest()


def phones_of(words, lexicon):
    return [lexicon[w] for w in words]


def diphones(phone_words):
    flat = [p for word in phone_words for p in word]
    return [a + " " + b for a, b in zip(flat, flat[1:])]


def grade(words, phone_words):
    """Flesch-Kincaid grade for one sentence; syllables are vowel nuclei in the lexicon."""
    syllables = sum(max(1, sum(p in VOWELS for p in word)) for word in phone_words)
    return 0.39 * len(words) + 11.8 * syllables / len(words) - 15.59


# ---------- sentences ----------

def normalise_spelling(sentence):
    def swap(match):
        word = match.group(0)
        replacement = US_SPELLING.get(word.lower())
        if replacement is None:
            return word
        return replacement.capitalize() if word[0].isupper() else replacement
    return re.sub(r"[A-Za-z]+(?:-[a-z]+)?", swap, sentence)


def clean_text(text):
    text = re.sub(r"\[[^\]]*\]", " ", text)                  # [Illustration], footnote marks
    text = re.sub(r"[_*]", "", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"--+|—|–", ", ", text)
    lines = [line for line in text.split("\n") if not re.match(r"^\s*(CHAPTER|BOOK|PART|STAVE|SONNET|[IVXLC]+\.?)\b[^a-z]*$", line)]
    return re.sub(r"\s+", " ", " ".join(lines))


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", clean_text(text)) if s.strip()]


def reject_reason(sentence, lexicon, allow_archaic=False):
    """None when a sentence is easy to read aloud; otherwise why not."""
    if '"' in sentence:
        return "dialogue"
    if re.search(r"\d", sentence):
        return "numeral"
    if ABBREVIATION.search(sentence):
        return "abbreviation"
    if not ALLOWED_CHARS.match(sentence) or not sentence[:1].isupper() or sentence[-1] not in ".!?":
        return "punctuation"
    words = words_of(sentence)
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return "length"
    if sentence.isupper():
        return "punctuation"
    if any(w[:1].isupper() and w not in ("I", "I'm", "I'll", "I've", "I'd", "O") for w in re.findall(r"[A-Za-z']+", sentence)[1:]):
        return "name"
    if re.search(r"(^|\s)'|'(\s|[,;:!?.]|$)", sentence):
        return "dialect"
    if BLOCKED & set(words):
        return "blocked word"
    if not allow_archaic and ARCHAIC & set(words):
        return "archaic"
    if any(w not in lexicon or (len(w) == 1 and w not in "ai") for w in words):
        return "not in lexicon"
    if grade(words, phones_of(words, lexicon)) > MAX_GRADE:
        return "reading level"
    return None


def accepted_sentences(text, lexicon, allow_archaic, rejections, missing):
    """(position in source, sentence) for every accepted sentence."""
    out = []
    for position, raw in enumerate(split_sentences(text)):
        sentence = normalise_spelling(re.sub(r"\s+([,;:!?.])", r"\1", raw))
        reason = reject_reason(sentence, lexicon, allow_archaic)
        if reason is None:
            out.append((position, sentence))
        else:
            rejections[reason] += 1
            if reason == "not in lexicon":
                missing.update(w for w in words_of(sentence) if w not in lexicon)
    return out


def passages_of(source, accepted, gap=PASSAGE_GAP, limit=PASSAGE_MAX):
    """Runs of nearby accepted sentences, in source order, at most `limit` prompts each."""
    groups, current, last = [], [], None
    for position, sentence in accepted:
        if current and (position - last > gap or len(current) >= limit):
            groups.append(current)
            current = []
        current.append(sentence)
        last = position
    if current:
        groups.append(current)
    return [dict(passage_id=f"{source['source_id']}-{i:04d}", source_id=source["source_id"], title=source["title"],
                 author=source["author"], kind=source["kind"], sentences=group) for i, group in enumerate(groups)]


# ---------- calibration, held-out, assignment ----------

def heldout(passage_id, modulus=HELDOUT_MODULUS):
    return int(hashlib.sha256(passage_id.encode()).hexdigest(), 16) % modulus == 0


def greedy_cover(candidates, size, key_words, per_source=CALIBRATION_PER_SOURCE):
    """Deterministic greedy set cover over diphones; ties by prompt id. `candidates` are dicts
    with prompt_id, source_id, diphones and words."""
    covered, chosen, used = collections.Counter(), [], collections.Counter()
    pool = sorted(candidates, key=lambda c: c["prompt_id"])
    while len(chosen) < size:
        best, best_gain = None, 0.0
        for c in pool:
            if used[c["source_id"]] >= per_source or c in chosen:
                continue
            gain = sum(1 / (1 + covered[d]) for d in set(c["diphones"])) / len(c[key_words])
            if gain > best_gain + 1e-12:
                best, best_gain = c, gain
        if best is None:
            break
        chosen.append(best)
        used[best["source_id"]] += 1
        covered.update(set(best["diphones"]))
    return chosen


def assign(passages, counts, n=5, recorded=()):
    """The `n` passages that best fill diphone gaps given current dataset counts, never held-out
    or already-recorded ones. Gain per word, so long passages don't win by size alone; counts are
    updated as passages are chosen so the picks complement each other."""
    counts = collections.Counter(counts)
    recorded = set(recorded)
    pool = [p for p in passages if not p.get("heldout") and p["passage_id"] not in recorded]
    chosen = []
    while len(chosen) < n and pool:
        def gain(p):
            pairs = [d for prompt in p["prompts"] for d in prompt["diphones"]]
            words = sum(prompt["words"] for prompt in p["prompts"])
            return sum(1 / (1 + counts[d]) for d in pairs) / max(1, words)
        best = min(pool, key=lambda p: (-gain(p), p["passage_id"]))
        chosen.append(best)
        pool.remove(best)
        counts.update(d for prompt in best["prompts"] for d in prompt["diphones"])
    return chosen


def existing_prompt_ids(corpus=CORPUS):
    """Sentences already in the app's bundled banks (including its reserved final-test pool)."""
    ids = set()
    for path in corpus.glob("*.txt"):
        for line in path.read_text().splitlines():
            if words_of(line):
                ids.add(prompt_id(line))
    return ids


def build(sources, lexicon, texts, version=1, existing=frozenset(), year=None, ranks=None):
    """Returns (prompts, calibration, heldout_prompts, report). `texts` maps source_id -> body."""
    problems = {s["source_id"]: public_domain_problems(s, year) for s in sources}
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        raise ValueError(f"Sources not public domain in both the UK and US: {problems}")
    rejections, missing, seen = collections.Counter(), collections.Counter(), set(existing)
    passages, per_source = [], collections.Counter()
    for source in sources:
        accepted = accepted_sentences(texts[source["source_id"]], lexicon, source.get("allow_archaic") == "1", rejections, missing)
        unique = []
        for position, sentence in accepted:
            pid = prompt_id(sentence)
            if pid in seen:
                rejections["duplicate or existing bank sentence"] += 1
                continue
            seen.add(pid)
            unique.append((position, sentence))
        for passage in passages_of(source, unique):
            passage["prompts"] = []
            for order, sentence in enumerate(passage.pop("sentences")):
                words = words_of(sentence)
                phone_words = phones_of(words, lexicon)
                passage["prompts"].append(dict(prompt_id=prompt_id(sentence), text=sentence, words=len(words),
                                               phones=[" ".join(w) for w in phone_words], diphones=diphones(phone_words),
                                               grade=round(grade(words, phone_words), 2)))
            passage["heldout"] = heldout(passage["passage_id"])
            passages.append(passage)
            per_source[source["source_id"]] += len(passage["prompts"])

    lo, hi = CALIBRATION_WORDS
    plain = {s["source_id"] for s in sources if s.get("allow_archaic") != "1" and s["kind"] != "poetry"}
    common = (lambda text: all(ranks.get(w, CALIBRATION_MAX_RANK) < CALIBRATION_MAX_RANK for w in words_of(text))) if ranks else (lambda text: True)
    # Everyone reads the calibration set, so it uses only plain modern prose and very common words.
    candidates = [dict(prompt, source_id=p["source_id"], passage_id=p["passage_id"], title=p["title"], author=p["author"])
                  for p in passages if not p["heldout"] and p["source_id"] in plain
                  for prompt in p["prompts"] if lo <= prompt["words"] <= hi and common(prompt["text"])]
    calibration = greedy_cover(candidates, CALIBRATION_SIZE, "diphones")
    calibration_ids = {c["prompt_id"] for c in calibration}
    for p in passages:
        p["prompts"] = [q for q in p["prompts"] if q["prompt_id"] not in calibration_ids]
    passages = [p for p in passages if p["prompts"]]

    def lines(selected):
        return [dict(prompt_id=q["prompt_id"], passage_id=p["passage_id"], order=i, text=q["text"], source_id=p["source_id"],
                     title=p["title"], author=p["author"], kind=p["kind"], words=q["words"], phones=q["phones"], grade=q["grade"],
                     version=version)
                for p in selected for i, q in enumerate(p["prompts"])]
    training = [p for p in passages if not p["heldout"]]
    held = [p for p in passages if p["heldout"]]
    calibration_lines = [dict(prompt_id=c["prompt_id"], order=i, text=c["text"], source_id=c["source_id"], title=c["title"],
                              author=c["author"], words=c["words"], phones=c["phones"], grade=c["grade"], version=version)
                         for i, c in enumerate(calibration)]
    all_pairs = collections.Counter(d for p in training for q in p["prompts"] for d in q["diphones"])
    calibration_pairs = {d for c in calibration for d in c["diphones"]}
    report = dict(version=version, sources=len(sources), passages=len(training), heldout_passages=len(held),
                  prompts=sum(len(p["prompts"]) for p in training), heldout_prompts=sum(len(p["prompts"]) for p in held),
                  calibration=len(calibration), prompts_per_source=dict(sorted(per_source.items())),
                  rejections=dict(rejections.most_common()), distinct_diphones=len(all_pairs),
                  calibration_distinct_diphones=len(calibration_pairs),
                  phones_never_seen=sorted({p for word in lexicon.values() for p in word} - {p for p in all_pairs for p in p.split()}),
                  missing_words_for_review=[w for w, _ in missing.most_common(200)],
                  rules=dict(words=[MIN_WORDS, MAX_WORDS], max_flesch_kincaid_grade=MAX_GRADE, spelling="en-US",
                             heldout=f"sha256(passage_id) % {HELDOUT_MODULUS} == 0", calibration=f"{CALIBRATION_SIZE} greedy diphone cover, {lo}-{hi} words"))
    return lines(training), calibration_lines, lines(held), report


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--version", type=int, default=1)
    b.add_argument("--out", type=Path, default=CORPUS / "community")
    a = sub.add_parser("assign")
    a.add_argument("--counts", type=Path, required=True, help="JSON object of diphone -> count")
    a.add_argument("--n", type=int, default=5)
    a.add_argument("--recorded", type=Path, help="passage ids already recorded, one per line")
    a.add_argument("--prompts", type=Path, default=CORPUS / "community/prompts-v1.jsonl")
    args = parser.parse_args()
    lexicon = load_lexicon()
    if args.command == "build":
        sources = load_sources(args.out / "sources.tsv")
        texts = {s["source_id"]: fetch(s) for s in sources}
        prompts, calibration, held, report = build(sources, lexicon, texts, args.version, existing_prompt_ids(), ranks=load_ranks())
        v = args.version
        write_jsonl(args.out / f"prompts-v{v}.jsonl", prompts)
        write_jsonl(args.out / f"calibration-v{v}.jsonl", calibration)
        write_jsonl(args.out / f"heldout-v{v}.jsonl", held)
        (args.out / f"build-report-v{v}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({k: report[k] for k in ("passages", "prompts", "heldout_passages", "calibration", "distinct_diphones")}))
    else:
        rows = [json.loads(line) for line in args.prompts.read_text().splitlines()]
        passages = collections.OrderedDict()
        for r in rows:
            p = passages.setdefault(r["passage_id"], dict(passage_id=r["passage_id"], title=r["title"], prompts=[]))
            phone_words = [w.split() for w in r["phones"]]
            p["prompts"].append(dict(diphones=diphones(phone_words), words=r["words"], text=r["text"]))
        recorded = args.recorded.read_text().split() if args.recorded else ()
        for p in assign(list(passages.values()), json.loads(args.counts.read_text()), args.n, recorded):
            print(p["passage_id"], "|", p["title"], "|", " ".join(q["text"] for q in p["prompts"]))


if __name__ == "__main__":
    main()
