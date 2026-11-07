"""Jaw-cycle counting: the syllabic open–close rhythm is the most robust thing a jaw-coupled
earbud can see. Counted cycles enter decoding only after a gate shows they track the reference
vowel count; otherwise they are reported and left out."""
from __future__ import annotations
import math
import numpy as np
from .data import RATE

BAND = (1.5, 8.0)        # Hz: syllable rate sits around 3–6; below 1.5 is head posture
MIN_SEPARATION = .12     # s between cycle peaks
PROMINENCE = 3.0         # × MAD of the rest window's projection; set on synthetic bursts, before any real result
GATE_RHO = .3            # lower 95% bound of Spearman ρ that lets the count prior into decoding


def biquad(kind, frequency, rate=RATE, q=1 / math.sqrt(2)):
    """Second-order Butterworth section (RBJ cookbook), as (b, a)."""
    w = 2 * math.pi * frequency / rate
    alpha, cos = math.sin(w) / (2 * q), math.cos(w)
    if kind == "low":
        b = [(1 - cos) / 2, 1 - cos, (1 - cos) / 2]
    else:
        b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]
    a = [1 + alpha, -2 * cos, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def lfilter(b, a, x):
    y = np.zeros_like(x, dtype=np.float64)
    for n in range(len(x)):
        y[n] = b[0] * x[n] + (b[1] * x[n-1] - a[1] * y[n-1] if n > 0 else 0) + (b[2] * x[n-2] - a[2] * y[n-2] if n > 1 else 0)
    return y


def bandpass(x, low=BAND[0], high=BAND[1], rate=RATE):
    """Zero-phase band-pass (forward and backward), per column."""
    x = np.asarray(x, dtype=np.float64)
    columns = x.reshape(len(x), -1)
    out = np.empty_like(columns)
    for k in range(columns.shape[1]):
        y = columns[:, k] - columns[:, k].mean()
        for kind, f in (("high", low), ("low", high)):
            b, a = biquad(kind, f, rate)
            y = lfilter(b, a, lfilter(b, a, y)[::-1])[::-1]
        out[:, k] = y
    return out.reshape(x.shape)


def channels(raw):
    """Rotation rate and gravity (head pitch/roll), the channels jaw cycles move."""
    raw = np.asarray(raw)
    return bandpass(raw[:, 3:9])


def fit_projection(raws):
    """First principal axis of the band-passed channels over training clips."""
    stacked = np.concatenate([channels(r) for r in raws])
    scale = stacked.std(0) + 1e-9
    _, _, vt = np.linalg.svd(stacked / scale, full_matrices=False)
    return dict(axis=vt[0].tolist(), scale=scale.tolist())


def project(raw, projection):
    return (channels(raw) / np.asarray(projection["scale"])) @ np.asarray(projection["axis"])


def count_cycles(signal, floor, rate=RATE, min_separation=MIN_SEPARATION, prominence=PROMINENCE):
    """Positive peaks (one per open–close cycle) inside active stretches, where a 0.25 s moving
    RMS exceeds `prominence`×`floor`, each rising that far above the deeper of the troughs
    separating it from its neighbours. Resting noise outside activity is never counted."""
    x = np.asarray(signal, dtype=np.float64)
    width = max(1, int(.25 * rate))
    envelope = np.sqrt(np.convolve(x * x, np.ones(width) / width, mode="same"))
    active = envelope > prominence * floor
    peaks = [i for i in range(1, len(x) - 1) if active[i] and x[i] >= x[i-1] and x[i] > x[i+1] and x[i] > 0]
    kept = []
    for i in sorted(peaks, key=lambda i: -x[i]):
        if any(abs(i - j) < min_separation * rate for j in kept):
            continue
        left = x[max([j for j in kept if j < i], default=0):i + 1].min()
        right = x[i:min([j for j in kept if j > i], default=len(x) - 1) + 1].min()
        if x[i] - max(left, right) >= prominence * floor:
            kept.append(i)
    return len(kept)


def mad(x):
    x = np.asarray(x)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) + 1e-9


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def gate(rows, vowel_counts, draws=1000, seed=42):
    """Spearman ρ between detected cycles and reference vowel counts, with a sentence-group
    bootstrap interval and a group-level permutation p-value. `rows` carry `cycles` and `groups`."""
    cycles = [r["cycles"] for r in rows]
    rho = spearman(cycles, vowel_counts)
    groups = sorted({tuple(r["groups"]) for r in rows})
    by_group = {g: [i for i, r in enumerate(rows) if tuple(r["groups"]) == g] for g in groups}
    rng = np.random.default_rng(seed)
    boot, null = [], []
    for _ in range(draws):
        sample = [i for g in rng.integers(0, len(groups), len(groups)) for i in by_group[groups[g]]]
        boot.append(spearman([cycles[i] for i in sample], [vowel_counts[i] for i in sample]))
        shuffled = rng.permutation(len(groups))
        counts = {g: vowel_counts[by_group[groups[j]][0]] for g, j in zip(groups, shuffled)}
        null.append(spearman(cycles, [counts[tuple(r["groups"])] for r in rows]))
    low, high = np.quantile(boot, [.025, .975]).tolist() if len(groups) >= 2 else (None, None)
    return dict(rho=rho, rho_95=[low, high], groups=len(groups), utterances=len(rows),
                p_value=(1 + sum(v >= rho for v in null)) / (draws + 1),
                passes=low is not None and low > GATE_RHO)


def rhythm_report(examples, language):
    """Gate per condition over every activity row: the projection is unsupervised (no labels),
    and nothing here is selected or fitted on reference text, so using all rows is safe."""
    from .articulatory import VOWEL_JAW
    from .data import pronunciations
    rest = {r["id"]: r["raw"] for r in examples if not r["activity"]}
    report = {}
    for condition in sorted({r["condition"] for r in examples}):
        active = [r for r in examples if r["condition"] == condition and r["activity"]]
        if len(active) < 4:
            continue
        projection = fit_projection([r["raw"] for r in active])
        rows, vowels = [], []
        for r in active:
            try:
                paths, _ = pronunciations(r["text"], language)
            except ValueError:
                continue
            signal = project(r["raw"], projection)
            reference = rest.get(r["id"] + "/rest")
            floor = mad(project(reference, projection)) if reference is not None else mad(signal)
            rows.append(dict(cycles=count_cycles(signal, floor), groups=r["groups"]))
            vowels.append(min(sum(p in VOWEL_JAW for p in path) for path in paths))
        report[condition] = gate(rows, vowels)
    return report


def band_energy(x, low, high, rate=RATE):
    return float(np.mean(bandpass(np.asarray(x)[:, None], low, high, rate) ** 2))


def signal_quality(raw, rest=None, rate=RATE):
    """Report-only take flags. jaw_snr: 1.5–8 Hz rotation-rate energy in the take over the same
    in its rest window. contamination: slow (0.2–1.5 Hz) head rotation over jaw-band rotation,
    i.e. nodding or turning mixed into the take. Neither excludes anything."""
    # Per-axis energies summed: a magnitude would rectify, smearing a nod into the jaw band.
    def energy(r, low, high):
        r = np.asarray(r)[:, 3:6]
        return sum(band_energy(r[:, k], low, high, rate) for k in range(3))
    jaw = energy(raw, *BAND)
    result = dict(contamination=energy(raw, .2, BAND[0]) / max(jaw, 1e-12))
    if rest is not None and len(rest) >= 25:
        result["jaw_snr"] = jaw / max(energy(rest, *BAND), 1e-12)
    return result
