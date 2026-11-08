"""Phone check from audio (docs/next-phase-plan.md, item 2).

Inputs for one spoken take:
- the text that was shown (what was meant),
- the local Whisper transcript (what words were said),
- optionally, phone intervals from local MFA alignment or a phone recogniser, already mapped to IPA
  (what phones were produced, with timing).

Output: expected and produced phones side by side, an alignment between them, and whether the take
may train. Produced phones come from the phone intervals when present (the real check). Without
them they come from the Whisper words converted to IPA, which catches wrong, missing and extra
words but not a mispronounced sound inside a correctly recognised word; `source` says which.
"""
from __future__ import annotations
import re
from . import ipa

TEXT_AGREEMENT = 0.85  # the existing Whisper agreement rule for free text


def words(text):
    return re.findall(r"[\w'’-]+", text.lower())


def align(expected, produced):
    """Levenshtein alignment -> list of (op, expected_phone, produced_phone), op in match/sub/del/ins."""
    n, m = len(expected), len(produced)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (expected[i - 1] != produced[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (expected[i - 1] != produced[j - 1]):
            ops.append(("match" if expected[i - 1] == produced[j - 1] else "sub", expected[i - 1], produced[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", expected[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, produced[j - 1]))
            j -= 1
    return ops[::-1]


def word_agreement(expected_text, heard_text):
    e, h = words(expected_text), words(heard_text)
    if not e:
        return 0.0
    ops = align(e, h)
    return sum(op == "match" for op, _, _ in ops) / max(len(e), len(h))


def _characters(text):
    return [ch for ch in text if "一" <= ch <= "鿿"]


def check(text, language, whisper_text, phone_intervals=None):
    """phone_intervals: [{'phone': IPA, 'start': s, 'end': s}] from MFA or a phone recogniser, or None."""
    accent = ipa.accent_of(language)
    language = ipa.validate_language_tag(language)
    expected = ipa.phones(text, language, accent)
    if phone_intervals:
        unknown = sorted({p["phone"] for p in phone_intervals} - set(ipa.INVENTORY))
        if unknown:
            raise ValueError(f"Phone intervals use symbols outside the shared inventory: {unknown}")
        produced, source = [p["phone"] for p in phone_intervals], "phone-intervals"
    else:
        produced, source = ipa.phones(whisper_text, language, accent), "whisper-words"
    if language == "zh":
        e, h = _characters(text), _characters(whisper_text)
        agreement = sum(op == "match" for op, _, _ in align(e, h)) / max(len(e), len(h), 1)
    else:
        agreement = word_agreement(text, whisper_text)
    ops = align(expected, produced)
    phone_errors = sum(op != "match" for op, _, _ in ops)
    return dict(language=language, expected_phones=expected, produced_phones=produced, source=source,
                alignment=[dict(op=op, expected=e, produced=p) for op, e, p in ops],
                word_agreement=round(agreement, 4), phone_error_rate=round(phone_errors / max(len(expected), 1), 4),
                trainable=agreement >= TEXT_AGREEMENT,
                timing=phone_intervals if phone_intervals else None)


def mispronounced(result):
    """(expected, produced) pairs a learner should see, most frequent first."""
    from collections import Counter
    counts = Counter((a["expected"], a["produced"]) for a in result["alignment"] if a["op"] != "match")
    return [pair for pair, _ in counts.most_common()]
