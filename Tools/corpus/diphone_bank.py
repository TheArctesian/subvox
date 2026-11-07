#!/usr/bin/env python3
"""Builds a phonetically compact sentence bank that tops a reading passage up to full diphone coverage.

The method is TIMIT's SX design: greedy set cover. Starting from what the seed passage already
covers, repeatedly take the candidate sentence that adds the most still-needed diphones per word,
until every diphone in the target (the set making up 99% of running English) has been heard
`--reps` times or no candidate helps. Candidates are natural sentences from the public-domain
pool books, so the bank reads as ordinary English rather than tongue-twisters.

Usage: Tools/corpus/diphone_bank.py [--seed Corpus/harbor.txt] [--reps 2] [--out Corpus/diphones.txt]
"""

import argparse
import collections
import re
from pathlib import Path

from coverage import diphones, frequent_diphones, load_dict, load_pool, words_of

MIN_WORDS, MAX_WORDS = 5, 14
ALLOWED = re.compile(r"^[A-Za-z ,;:'!?.]+$")
# Period language in the pool books that nobody should be asked to read aloud.
BLOCKED = {"nigger", "niggers", "nigga", "negro", "negroes", "darky", "darkies", "blacks", "coon", "coons",
           "injun", "injuns", "savage", "savages", "squaw", "half", "breed", "gypsy", "gypsies",
           "lynch", "lynched", "lynching", "dat", "dem", "dis", "ruther", "dam", "ain't"}


def candidates(cmu):
    """Clean, readable sentences from the pool whose every word has a pronunciation."""
    seen = set()
    books = load_pool()
    # A word counts as ordinary (not a name) if the books also use it in lower case.
    ordinary = set(re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", " ".join(books.values())))
    for title, text in books.items():
        text = re.sub(r"[_*]", "", text)  # Gutenberg italics/emphasis markers
        text = re.sub(r"[‘’]", "'", re.sub(r"[“”\"]", "", text))
        for raw in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
            sentence = raw.strip()
            if not ALLOWED.match(sentence) or not sentence[:1].isupper() or sentence[-1] not in ".!?":
                continue
            words = words_of(sentence)
            if not MIN_WORDS <= len(words) <= MAX_WORDS or sentence.isupper():
                continue
            # Capitalised words past the first are names; they buy rare diphones with unreadable prompts.
            if any(w[:1].isupper() and w not in ("I", "I'm", "I'll", "I've", "I'd")
                   for w in sentence.split()[1:]):
                continue
            if words[0] not in ordinary and words[0] != "i" and not words[0].startswith("i'"):
                continue
            # Dialect spellings ('em, nothin', th') are hard to read and the CMU entries for them are guesses.
            if re.search(r"(^|\s)'|'(\s|[,;:!?.]|$)", sentence):
                continue
            if BLOCKED & set(words):
                continue
            if any(w not in cmu or (len(w) == 1 and w not in "ai") for w in words):
                continue
            key = " ".join(words)
            if key in seen:
                continue
            seen.add(key)
            yield sentence, title, collections.Counter(diphones(words, cmu))


def build(seed_pairs, pool, target, reps):
    need = collections.Counter({d: reps for d in target})
    need.subtract(p for p in seed_pairs if p in target)
    need = +need  # drop satisfied entries

    def gain(pairs):
        return sum(min(n, need[d]) for d, n in pairs.items() if d in need)

    chosen = []
    remaining = [c for c in pool if gain(c[2])]
    while need and remaining:
        scored = [(gain(c[2]) / len(words_of(c[0])), c) for c in remaining]
        score, best = max(scored, key=lambda s: s[0])
        if score == 0:
            break
        chosen.append(best)
        need.subtract({d: min(n, need[d]) for d, n in best[2].items() if d in need})
        need = +need
        remaining = [c for s, c in scored if s > 0 and c is not best]
    return chosen, need


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--seed", type=Path, default=root / "Corpus/harbor.txt")
    parser.add_argument("--reps", type=int, default=2, help="times each diphone should be heard (default 2)")
    parser.add_argument("--out", type=Path, default=root / "Corpus/diphones.txt")
    parser.add_argument("--dict", type=Path)
    args = parser.parse_args()

    cmu = load_dict(args.dict)
    target = frequent_diphones(cmu)
    seed = diphones(words_of(args.seed.read_text()), cmu)
    pool = list(candidates(cmu))
    chosen, unmet = build(seed, pool, target, args.reps)

    attainable = set().union(*(c[2] for c in pool)) | set(seed)
    covered = (set(seed) | set().union(*(c[2] for c in chosen))) & target
    words = sum(len(words_of(c[0])) for c in chosen)
    args.out.write_text("".join(f"{c[0]}\n" for c in chosen))

    print(f"pool: {len(pool)} candidate sentences from {len({c[1] for c in pool})} books")
    print(f"target: {len(target)} diphones (99% of running English); seed covers {len(set(seed) & target)}")
    print(f"bank: {len(chosen)} sentences, {words} words (~{words / 130:.0f} min read aloud) -> {args.out}")
    print(f"covered: {len(covered)}/{len(target)} at least once;"
          f" {len(target) - len(unmet)}/{len(target)} at least {args.reps}x")
    absent = sorted(target - attainable)
    if absent:
        print(f"not in any candidate sentence: {', '.join(absent)}")


if __name__ == "__main__":
    main()
