#!/usr/bin/env python3
"""Writes the pronunciation files the app bundles instead of the full 135k-word CMU dictionary.

  Corpus/lexicon.tsv  every word in the Corpus reading texts:  word <TAB> phones
  Corpus/words.tsv    the decoding vocabulary: every pool-book word with a pronunciation, most
                      frequent first: word <TAB> phones <TAB> occurrences in the pool books.
                      The Train checklist also picks its prompts from the common end of it.

Rerun after editing a corpus text.

Usage: Tools/lexicon.py [--dict cmudict.dict]
"""

import argparse
import collections
import re
from pathlib import Path

from coverage import load_dict, load_pool, words_of
from diphone_bank import BLOCKED


def vocabulary(cmu):
    """Pool words used as ordinary words (seen in lower case), with a pronunciation, not blocked."""
    text = " ".join(load_pool().values())
    lower = collections.Counter(re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text))
    return [(w, cmu[w], n) for w, n in lower.most_common()
            if w in cmu and w not in BLOCKED and (len(w) > 1 or w in ("a", "i"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dict", type=Path)
    args = parser.parse_args()

    corpus = Path(__file__).resolve().parent.parent / "Corpus"
    cmu = load_dict(args.dict)

    words = sorted({w for path in corpus.glob("*.txt") for w in words_of(path.read_text())})
    missing = [w for w in words if w not in cmu]
    (corpus / "lexicon.tsv").write_text("".join(f"{w}\t{' '.join(cmu[w])}\n" for w in words if w in cmu))
    print(f"{len(words) - len(missing)} corpus words -> {corpus / 'lexicon.tsv'}")
    if missing:
        print(f"no pronunciation (will be timed as a single unit): {', '.join(missing)}")

    words = vocabulary(cmu)
    (corpus / "words.tsv").write_text("".join(f"{w}\t{' '.join(p)}\t{n}\n" for w, p, n in words))
    print(f"{len(words)} vocabulary words -> {corpus / 'words.tsv'}")


if __name__ == "__main__":
    main()
