"""Prompt banks designed around what the earbud can observe.

grid: GRID-style slot sentences ("Place the blue bag by nine."), each recorded once, so the
model can't learn ten sentences by heart; slot words span jaw height and labial onsets and each
appears in several contexts.
frames: "Say ___ again." with targets in sets differing in one observable feature (jaw height,
rounding, labial closure, closure type). The carrier stays fixed, so differences between takes
belong to the target. frames.tsv keeps the feature and set of every frame for evaluation.

Words avoid accent-variable classes (post-vocalic r, BATH and CLOTH words) because the
speaker's accent is mixed."""
from __future__ import annotations
import itertools
import random
from pathlib import Path

SLOTS = dict(
    verb=["place", "bring", "move", "put", "lay", "set", "keep", "pick"],
    colour=["blue", "red", "green", "white", "black", "pink", "gold", "brown"],
    noun=["bag", "box", "cup", "book", "key", "coat", "bowl", "map", "pen", "sock"],
    preposition=["by", "in", "on", "at", "with", "beside"],
    number=["one", "two", "six", "eight", "nine", "ten", "three", "five"],
)
TEMPLATE = "{verb} the {colour} {noun} {preposition} {number}."

FRAME_SETS = [
    ("jaw", ["beat", "bit", "bet", "bat"]),
    ("jaw", ["leak", "lick", "neck", "lack"]),
    ("round", ["beet", "boot"]), ("round", ["seat", "suit"]), ("round", ["tea", "too"]), ("round", ["feed", "food"]),
    ("labial", ["bay", "day"]), ("labial", ["pea", "tea"]), ("labial", ["me", "knee"]), ("labial", ["my", "tie"]),
    ("closure", ["tip", "sip"]), ("closure", ["toe", "so"]), ("closure", ["pea", "fee"]), ("closure", ["dime", "lime"]),
]
CARRIER = "Say {} again."


def generate_grid(count=120, seed=7, minimum_contexts=4):
    """Unique slot sentences, drawn so every slot word is used about equally often."""
    rng = random.Random(seed)
    combos = list(itertools.product(*SLOTS.values()))
    rng.shuffle(combos)
    used, sentences = {w: 0 for words in SLOTS.values() for w in words}, []
    # Greedy balance: prefer combinations whose words have been used least so far.
    while len(sentences) < count and combos:
        combos.sort(key=lambda c: sum(used[w] for w in c))
        choice = combos.pop(0)
        sentences.append(TEMPLATE.format(**dict(zip(SLOTS, choice))).capitalize())
        for w in choice:
            used[w] += 1
    if min(used.values()) < minimum_contexts:
        raise ValueError("Too few sentences for every slot word to appear in enough contexts")
    return sentences


def generate_frames():
    rows = []
    for index, (feature, targets) in enumerate(FRAME_SETS):
        for target in targets:
            rows.append((CARRIER.format(target), feature, f"{feature}-{index}", target, ",".join(targets)))
    return rows


def write_banks(corpus=Path(__file__).resolve().parents[2] / "Corpus", lexicon=None):
    """Writes grid-sentences.txt, frames.txt and frames.tsv; every word must have a pronunciation."""
    grid = generate_grid()
    frames = generate_frames()
    if lexicon is not None:
        from .data import words
        missing = sorted({w for text in grid + [f[0] for f in frames] for w in words(text) if w not in lexicon})
        if missing:
            raise ValueError(f"No pronunciation for: {', '.join(missing)}")
    (corpus / "grid-sentences.txt").write_text("\n".join(grid) + "\n")
    # A target in two sets (pea, tea) is one prompt; frames.tsv keeps both memberships.
    (corpus / "frames.txt").write_text("\n".join(dict.fromkeys(f[0] for f in frames)) + "\n")
    (corpus / "frames.tsv").write_text("text\tfeature\tset\ttarget\tmembers\n" + "".join("\t".join(f) + "\n" for f in frames))
    return len(grid), len(frames)
