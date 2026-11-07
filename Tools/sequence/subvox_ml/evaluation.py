"""Measurements that show motion information while phoneme and word error are pinned near 1.

Closed-set forced choice asks a sharper question than free decoding: among the mouthed sentence
and same-length distractors the model never trained on, does its CTC likelihood prefer the one
actually mouthed? Training sentences are never candidates: a model memorises their label
sequences, and they would win on familiarity rather than motion. A label-randomisation null over
whole sentence groups and a duration-only baseline say whether a preference beats chance and
beats ranking by length alone."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from .data import pronunciations, group_id, words, RATE
from .decoding import SEPARATOR
from .articulatory import features, OBSERVABLE, CONTROLS


def candidate_paths(texts, language, index):
    """Token-index pronunciation paths per candidate text; texts with unknown words are skipped."""
    result = {}
    for text in texts:
        try:
            paths, _ = pronunciations(text, language)
        except ValueError:
            continue
        mapped = [[index[p] for p in path if p != SEPARATOR] for path in paths]
        if all(mapped):
            result[text] = mapped
    return result


DISTRACTOR_BANK = Path(__file__).resolve().parents[3]/"Corpus"/"diphones.txt"
DISTRACTORS = 9


def distractor_candidates(truth, language, index, excluded_groups, count=DISTRACTORS, bank=DISTRACTOR_BANK):
    """The truth plus `count` bank sentences of the nearest phone count, chosen by a hash of the
    truth's sentence group so every repetition of a sentence faces the same candidates.
    Matching phone count matters: a length-normalised CTC score from a blank-heavy model favours
    longer phone strings, which would rank by length rather than motion.
    `excluded_groups` holds the fold's training groups and the reserved final-test groups."""
    target = candidate_paths([truth], language, index)
    if not target:
        return {}
    length = min(map(len, target[truth]))
    pool = [line.strip() for line in Path(bank).read_text().splitlines() if line.strip()]
    pool = [t for t in pool if group_id(t) not in excluded_groups and group_id(t) != group_id(truth)]
    paths = candidate_paths(pool, language, index)
    key = lambda t: (abs(min(map(len, paths[t])) - length), hashlib.sha256((group_id(truth) + t).encode()).hexdigest())
    chosen = sorted(paths, key=key)[:count]
    return {**{t: paths[t] for t in chosen}, **target}


def motion_nulls(raw, rng, block=.2, rate=RATE, fade=3):
    """Same-length copies of a clip that keep its energy and value distribution but break its
    temporal order: time reversal (primary: no discontinuities, and conservative because jaw
    cycles are nearly symmetric), a 200 ms block shuffle and a half-length circular shift.
    Shuffle and shift joins are crossfaded over `fade` frames: a hard cut is a step the
    derivative features read as a burst of motion, which made the null look *more* speech-like
    than the real clip."""
    raw = np.asarray(raw, dtype=np.float64)
    size = max(1, int(block * rate))
    blocks = [raw[i:i + size] for i in range(0, len(raw), size)]
    def join(pieces):
        out = pieces[0].copy()
        for piece in pieces[1:]:
            n = min(fade, len(out), len(piece))
            weight = np.linspace(0, 1, n + 2)[1:-1, None]
            out[-n:] = out[-n:] * (1 - weight) + piece[:n] * weight
            out = np.concatenate([out, piece[n:]])
        return out
    def pad(x):
        return np.concatenate([x, np.repeat(x[-1:], len(raw) - len(x), axis=0)]) if len(x) < len(raw) else x[:len(raw)]
    order = rng.permutation(len(blocks))
    half = len(raw) // 2
    return dict(reverse=raw[::-1].copy(), shuffle=pad(join([blocks[i] for i in order])), shift=pad(join([raw[half:], raw[:half]])))


def order_candidates(path, rng, count=9):
    """The truth's own phones in other orders: same multiset, same length, so only the order of
    motion evidence can separate them. Keyed '<order-i>' with a single path each."""
    from .network import fits
    seen, out = {tuple(path)}, {}
    trials = [list(reversed(path))] + [list(rng.permutation(path)) for _ in range(count * 20)]
    for candidate in trials:
        key = tuple(int(t) for t in candidate)
        if key in seen:
            continue
        seen.add(key)
        out[f"<order-{len(out)}>"] = [list(key)]
        if len(out) == count:
            break
    return out


def forced_choice(log_probs, candidates, truth, log_prior=None, null_log_probs=None):
    """Length-normalised CTC log-likelihood of every candidate; `truth` is the mouthed text.
    With `log_prior` (the model's training label prior), each score is a likelihood ratio
    against that prior held constant over the same frames: a blank-heavy model otherwise ranks
    candidates by how common their phones are, not by the motion.
    With `null_log_probs` (emissions of order-broken copies of this same clip), each score is a
    likelihood ratio against them: only evidence carried by the order of motion survives.
    Candidates no path of which fits the frame count are listed, never scored as wrong."""
    from .network import ctc_variant_loss, fits
    lp = torch.as_tensor(np.asarray(log_probs), dtype=torch.float32)
    background = None
    if log_prior is not None:
        prior = np.asarray(log_prior, dtype=np.float64)
        background = torch.as_tensor(np.tile(prior - np.logaddexp.reduce(prior), (len(lp), 1)), dtype=torch.float32)
    scores, unfit = {}, []
    for text, paths in candidates.items():
        if not any(fits(p, len(lp)) for p in paths):
            unfit.append(text)
            continue
        scores[text] = -float(ctc_variant_loss(lp, paths))
        if background is not None:
            scores[text] += float(ctc_variant_loss(background, paths))
        if null_log_probs:
            nulls = [-float(ctc_variant_loss(torch.as_tensor(n, dtype=torch.float32), paths)) for n in null_log_probs]
            scores[text] -= float(np.logaddexp.reduce(nulls) - np.log(len(nulls)))
    if truth not in scores:
        return dict(fc_scores=scores, fc_unfit=unfit, fc_rank=None, fc_reciprocal_rank=0.0, fc_margin=None, fc_candidates=len(scores))
    ordered = sorted(scores, key=lambda t: (-scores[t], t))
    best_wrong = max((s for t, s in scores.items() if t != truth), default=None)
    rank = ordered.index(truth) + 1
    return dict(fc_scores=scores, fc_unfit=unfit, fc_rank=rank, fc_reciprocal_rank=1 / rank,
                fc_margin=None if best_wrong is None else scores[truth] - best_wrong, fc_candidates=len(scores))


def duration_rank(frames, candidates, truth, frames_per_phone):
    """Rank by how well each candidate's phone count fits the clip length: the floor any
    forced-choice result has to beat, since CTC fit is length-sensitive."""
    if truth not in candidates:
        return None
    gap = {t: min(abs(frames - len(p) * frames_per_phone) for p in paths) for t, paths in candidates.items()}
    return sorted(gap, key=lambda t: (gap[t], t)).index(truth) + 1


def frames_per_phone(rows):
    pairs = [(len(r["raw"]), min(len([p for p in t if p != SEPARATOR]) for t in r["targets"])) for r in rows if r["activity"]]
    return sum(f for f, _ in pairs) / max(1, sum(n for _, n in pairs))


def permutation_null(rows, draws=2000, seed=42):
    """One-sided p-value for mean top-1 under label randomisation: each sentence group gets a
    random candidate from its own set as the pseudo-truth, shared by all its repetitions, so
    repetitions never count as independent evidence."""
    scored = [r for r in rows if r.get("fc_rank") is not None]
    if not scored:
        return None
    clusters = sorted({tuple(r["groups"]) for r in scored})
    members = {c: [r for r in scored if tuple(r["groups"]) == c] for c in clusters}
    winners = {id(r): max(r["fc_scores"], key=lambda t: (r["fc_scores"][t], t)) for r in scored}
    observed = float(np.mean([winners[id(r)] == r["reference_text"] for r in scored]))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(draws):
        hits = []
        for c in clusters:
            options = sorted(members[c][0]["fc_scores"])
            pseudo = options[int(rng.integers(0, len(options)))]
            hits += [winners[id(r)] == pseudo for r in members[c]]
        null.append(float(np.mean(hits)))
    return dict(observed_top1=observed, null_mean_top1=float(np.mean(null)), clusters=len(clusters),
                p_value=(1 + sum(v >= observed for v in null)) / (draws + 1))


def proportional_positions(frames, path, vowels):
    """Phone position per frame from proportional timing (vowels weighted 2), independent of
    the model. Aligning with the model's own posteriors would make its guesses agree with the
    reference by construction."""
    weights = np.array([2.0 if p in vowels else 1.0 for p in path])
    edges = np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * frames
    return np.searchsorted(edges, np.arange(frames) + .5, side="right") - 1


def frame_alignment(log_probs, reference_paths, vowels=()):
    """(frame, reference phone index, predicted phone index) for every frame, the reference
    from proportional timing of the first pronunciation variant, the prediction the most likely
    non-blank token there."""
    lp = np.asarray(log_probs)
    if not reference_paths or not reference_paths[0]:
        return []
    path = reference_paths[0]
    positions = proportional_positions(len(lp), path, vowels)
    guesses = lp[:, 1:].argmax(1) + 1
    return [(t, path[i], int(guesses[t])) for t, i in enumerate(positions)]


def mutual_information(x, y):
    """Miller–Madow corrected T(x;y) in bits for integer-coded label arrays."""
    n = len(x)
    if not n:
        return 0.0, 0.0
    def entropy(codes):
        counts = np.bincount(codes)
        counts = counts[counts > 0]
        probability = counts / n
        return float(-(probability * np.log2(probability)).sum() + (len(counts) - 1) / (2 * n * np.log(2)))
    hx, hy = entropy(x), entropy(y)
    return hx + hy - entropy(x * (y.max() + 1) + y), hx


def feature_transfer(rows, tokens, draws=200, seed=42):
    """Transmitted bits per articulatory feature, and the excess over a circular-shift null that
    keeps each row's label statistics but breaks its timing."""
    sequences = [r["frame_pairs"] for r in rows if r.get("frame_pairs")]
    if not sequences:
        return None
    rng = np.random.default_rng(seed)
    shifts = [[int(rng.integers(1, max(2, len(s)))) for s in sequences] for _ in range(draws)]
    result = {}
    for name in OBSERVABLE + CONTROLS + ("phone",):
        values = {}
        code = lambda i: values.setdefault(tokens[i] if name == "phone" else features(tokens[i])[name], len(values))
        refs = [np.array([code(a) for a, _ in seq]) for seq in sequences]
        guesses = [np.array([code(b) for _, b in seq]) for seq in sequences]
        reference = np.concatenate(refs)
        bits, hx = mutual_information(reference, np.concatenate(guesses))
        null = [mutual_information(reference, np.concatenate([np.roll(g, -k) for g, k in zip(guesses, draw)]))[0] for draw in shifts]
        result[name] = dict(bits=bits, relative=bits / hx if hx > 0 else None, reference_entropy=hx, n=len(reference),
                            null_bits=float(np.mean(null)), excess_bits=bits - float(np.mean(null)),
                            null_95=float(np.quantile(null, .95)), control=name in CONTROLS)
    return result


def cluster_interval(rows, statistic, draws=500, seed=42):
    """95% bootstrap interval resampling whole sentence groups, not repetitions."""
    groups = sorted({tuple(r["groups"]) for r in rows})
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    by_group = {g: [r for r in rows if tuple(r["groups"]) == g] for g in groups}
    values = [statistic([r for i in rng.integers(0, len(groups), len(groups)) for r in by_group[groups[i]]]) for _ in range(draws)]
    return np.quantile(values, [.025, .975]).tolist()


def input_dependence(log_probs, nulls, paths):
    """Per-phone log-likelihood of the reference on the real clip minus on each order-broken
    copy: about 0 when the model ignores motion order, positive when it reads it."""
    from .network import ctc_variant_loss
    real = -float(ctc_variant_loss(torch.as_tensor(np.asarray(log_probs), dtype=torch.float32), paths))
    return {f"indep_{name}": real + float(ctc_variant_loss(torch.as_tensor(np.asarray(lp), dtype=torch.float32), paths)) for name, lp in nulls.items()}


def ranking_summary(rows, tokens=None, prefix="fc"):
    """Forced-choice top-1, MRR and margin with group intervals, chance, the duration floor and
    the permutation p-value; plus per-feature transmitted bits and top-k frame recall."""
    scored = [r for r in rows if r.get("activity") and r.get(f"{prefix}_rank") is not None]
    if not scored:
        return {}
    top1 = lambda rs: float(np.mean([r[f"{prefix}_rank"] == 1 for r in rs]))
    mrr = lambda rs: float(np.mean([r[f"{prefix}_reciprocal_rank"] for r in rs]))
    margins = [r[f"{prefix}_margin"] for r in scored if r[f"{prefix}_margin"] is not None]
    result = {f"{prefix}_utterances": len(scored), f"{prefix}_top1": top1(scored), f"{prefix}_top1_95": cluster_interval(scored, top1),
              f"{prefix}_mrr": mrr(scored), f"{prefix}_mrr_95": cluster_interval(scored, mrr),
              f"{prefix}_margin_mean": float(np.mean(margins)) if margins else None,
              f"{prefix}_chance_top1": float(np.mean([1 / r[f"{prefix}_candidates"] for r in scored]))}
    null = permutation_null([dict(r, fc_scores=r[f"{prefix}_scores"], fc_rank=r[f"{prefix}_rank"]) for r in scored])
    result[f"{prefix}_permutation"] = null
    if prefix == "fc":
        durations = [r["duration_rank"] for r in scored if r.get("duration_rank") is not None]
        result["duration_baseline_top1"] = float(np.mean([d == 1 for d in durations])) if durations else None
        hits = [r["frame_topk"] for r in scored if r.get("frame_topk")]
        if hits:
            total = sum(h["frames"] for h in hits)
            result["frame_topk_recall"] = {k: sum(h[k] for h in hits) / max(1, total) for k in ("1", "3", "5")}
        if tokens:
            result["feature_transfer"] = feature_transfer(scored, tokens)
    return result


def dependence_summary(rows):
    result = {}
    for name in ("shuffle", "shift", "reverse"):
        key = f"indep_{name}"
        values = [r for r in rows if r.get("activity") and r.get(key) is not None]
        if values:
            mean = lambda rs: float(np.mean([r[key] for r in rs]))
            result[key] = mean(values)
            result[key + "_95"] = cluster_interval(values, mean)
    return result
