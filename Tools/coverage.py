#!/usr/bin/env python3
"""Phonetic coverage report for a reading passage.

Looks every word up in the CMU Pronouncing Dictionary (General American, ARPAbet) and
reports which speech structures the text exercises, from the general inventory down to
the jaw-specific classes Subvox can plausibly sense:

  phonemes        all 39 ARPAbet phonemes, and how often each occurs
  stressed vowels every vowel under primary stress (Wells lexical sets, GA mergers applied)
  diphones        adjacent phone pairs in connected speech, against the set making up 99% of running English
  clusters        word-initial and word-final consonant clusters
  shapes          syllable counts and stress patterns per word
  jaw             vowel height classes and the height-to-height transitions between syllables

Usage: Tools/coverage.py PASSAGE.txt [MORE.txt ...] [--dict cmudict.dict] [--missing]
"""

import argparse
import collections
import re
import sys
import urllib.request
from pathlib import Path

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
CACHE = Path.home() / "Library/Caches/Subvox/cmudict.dict"

VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
CONSONANTS = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R",
              "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]

# Stressed vowel (+ following R where it defines the set) -> Wells lexical set.
# General American merges BATH into TRAP and CLOTH into THOUGHT/LOT, so those are not separable here.
LEXICAL_SETS = {
    "IH": "KIT", "EH": "DRESS", "AE": "TRAP", "AA": "LOT/PALM", "AH": "STRUT", "UH": "FOOT",
    "IY": "FLEECE", "EY": "FACE", "AO": "THOUGHT", "OW": "GOAT", "UW": "GOOSE",
    "AY": "PRICE", "OY": "CHOICE", "AW": "MOUTH", "ER": "NURSE",
    "IH R": "NEAR", "IY R": "NEAR", "EH R": "SQUARE", "AA R": "START", "AO R": "NORTH/FORCE",
    "UH R": "CURE", "UW R": "CURE",
}
UNSTRESSED_SETS = {"AH": "commA", "ER": "lettER", "IY": "happY"}

# Jaw opening at the vowel nucleus. Diphthongs are glides, so they're tracked as start->end:
# that movement is exactly what a jaw sensor should see best.
JAW_HEIGHT = {
    "IY": "close", "IH": "close", "UW": "close", "UH": "close",
    "EY": "mid", "EH": "mid", "AH": "mid", "ER": "mid", "OW": "mid",
    "AE": "open", "AA": "open", "AO": "open",
    "AY": "open>close", "AW": "open>close", "OY": "mid>close",
}
# Consonants whose articulation drives the jaw up toward closure.
JAW_CLOSING = {"P", "B", "M", "F", "V", "S", "Z", "SH", "ZH", "CH", "JH", "T", "D", "TH", "DH"}


def load_dict(path):
    if path is None:
        path = CACHE
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading CMU dict to {path}", file=sys.stderr)
            urllib.request.urlretrieve(CMUDICT_URL, path)
    entries = {}
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        if "(" in word:  # alternate pronunciations; keep the first
            continue
        entries[word] = phones
    return entries


def base(phone):
    return phone.rstrip("012")


def stress(phone):
    return phone[-1] if phone[-1].isdigit() else None


def words_of(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def diphones(words, cmu):
    """Adjacent phone pairs in connected speech, within words and across word boundaries.
    An out-of-dictionary word breaks the chain rather than inventing a junction."""
    pairs = []
    previous = None
    for word in words:
        phones = cmu.get(word)
        if phones is None:
            previous = None
            continue
        bases = [base(p) for p in phones]
        if previous:
            pairs.append(f"{previous} {bases[0]}")
        pairs.extend(f"{a} {b}" for a, b in zip(bases, bases[1:]))
        previous = bases[-1]
    return pairs


# Public-domain books from Project Gutenberg: the reference for which diphones occur in real
# running English, and the candidate pool the sentence bank is drawn from.
POOL_BOOKS = {
    11: "Alice's Adventures in Wonderland", 16: "Peter Pan", 45: "Anne of Green Gables",
    55: "The Wonderful Wizard of Oz", 74: "The Adventures of Tom Sawyer", 113: "The Secret Garden",
    120: "Treasure Island", 215: "The Call of the Wild", 236: "The Jungle Book",
    271: "Black Beauty", 289: "The Wind in the Willows", 514: "Little Women",
    1342: "Pride and Prejudice", 1661: "The Adventures of Sherlock Holmes",
}
POOL_DIR = CACHE.parent / "pool"


def load_pool():
    """Returns {book title: body text}, downloading and caching books on first use."""
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    books = {}
    for number, title in POOL_BOOKS.items():
        path = POOL_DIR / f"pg{number}.txt"
        if not path.exists():
            print(f"Downloading {title}", file=sys.stderr)
            urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt", path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
        end = re.search(r"\*\*\* ?END OF", text)
        books[title] = text[start.end() if start else 0:end.start() if end else len(text)]
    return books


def frequent_diphones(cmu, share=0.99):
    """The smallest set of diphones that accounts for `share` of all diphone tokens in the pool."""
    counts = collections.Counter()
    for text in load_pool().values():
        counts.update(diphones(words_of(text), cmu))
    total = sum(counts.values())
    target, running = set(), 0
    for pair, n in counts.most_common():
        if running >= share * total:
            break
        target.add(pair)
        running += n
    return target


def analyse(words, cmu):
    report = collections.defaultdict(collections.Counter)
    missing = []
    jaw_sequence = []
    for word in words:
        phones = cmu.get(word)
        if phones is None:
            missing.append(word)
            continue
        bases = [base(p) for p in phones]
        report["phonemes"].update(bases)

        vowel_idx = [i for i, p in enumerate(phones) if stress(p) is not None]
        report["syllables"][len(vowel_idx)] += 1
        report["stress"]["".join(stress(phones[i]) for i in vowel_idx)] += 1

        for i in vowel_idx:
            v = bases[i]
            with_r = f"{v} R" if i + 1 < len(bases) and bases[i + 1] == "R" else None
            if stress(phones[i]) == "1":
                report["lexical"][LEXICAL_SETS.get(with_r) or LEXICAL_SETS[v]] += 1
            elif stress(phones[i]) == "0" and v in UNSTRESSED_SETS:
                report["lexical"][UNSTRESSED_SETS[v]] += 1
            jaw_sequence.append(JAW_HEIGHT[v])

        first_v = vowel_idx[0] if vowel_idx else len(bases)
        last_v = vowel_idx[-1] if vowel_idx else -1
        onset, coda = bases[:first_v], bases[last_v + 1:]
        if len(onset) >= 2:
            report["onset clusters"][" ".join(onset)] += 1
        if len(coda) >= 2:
            report["coda clusters"][" ".join(coda)] += 1

    report["diphones"].update(diphones(words, cmu))
    for a, b in zip(jaw_sequence, jaw_sequence[1:]):
        report["jaw transitions"][f"{a} -> {b}"] += 1
    report["jaw heights"].update(jaw_sequence)
    return report, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("passages", nargs="+", type=Path)
    parser.add_argument("--dict", type=Path, help="cmudict.dict path (downloaded and cached if omitted)")
    parser.add_argument("--missing", action="store_true", help="list everything not yet covered")
    args = parser.parse_args()

    cmu = load_dict(args.dict)
    targets = {
        "diphones": sorted(frequent_diphones(cmu)),
        "phonemes": VOWELS + CONSONANTS,
        "lexical": sorted(set(LEXICAL_SETS.values()) | set(UNSTRESSED_SETS.values())),
        "jaw heights": sorted(set(JAW_HEIGHT.values())),
        "jaw transitions": [f"{a} -> {b}" for a in sorted(set(JAW_HEIGHT.values()))
                            for b in sorted(set(JAW_HEIGHT.values()))],
    }

    for path in args.passages:
        words = words_of(path.read_text())
        report, missing = analyse(words, cmu)
        print(f"\n== {path.name}: {len(words)} words, {sum(report['syllables'].values()) and sum(k * v for k, v in report['syllables'].items())} syllables")
        if missing:
            print(f"   not in dictionary (ignored): {', '.join(sorted(set(missing)))}")
        for name, target in targets.items():
            have = [t for t in target if report[name][t]]
            print(f"   {name:<16} {len(have):>3}/{len(target):<4}", end="")
            if name == "phonemes":
                rare = sorted((n, p) for p, n in report[name].items())[:5]
                print(f"  rarest: {', '.join(f'{p}×{n}' for n, p in rare)}", end="")
            print()
            if args.missing and len(have) < len(target):
                print(f"      missing: {', '.join(t for t in target if not report[name][t])}")
        print(f"   {'onset clusters':<16} {len(report['onset clusters']):>3}      {', '.join(sorted(report['onset clusters']))}")
        print(f"   {'coda clusters':<16} {len(report['coda clusters']):>3}      {', '.join(sorted(report['coda clusters']))}")
        print(f"   {'syllables/word':<16}          {dict(sorted(report['syllables'].items()))}")
        print(f"   {'stress patterns':<16} {len(report['stress']):>3}      {', '.join(p for p, _ in report['stress'].most_common())}")


if __name__ == "__main__":
    main()
