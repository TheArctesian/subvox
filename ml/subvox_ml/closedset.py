"""Closed-set sentence identification by template matching, per input modality.

Unlike CTC, a template cannot compose an unseen sentence; it asks the smaller question a few
recordings can answer: which already-recorded sentence is this? Two protocols:

- ``enrol_spoken``: each sentence's spoken take is its template; identify every mouthed take
  among all enrolled sentences ("say it aloud once, then mouth it").
- ``cross_session``: identify each take of a sentence recorded in several sessions among every
  sentence with a template from a *different* session of the same condition.
- ``commands_*``: the fixed command set only (candidates are commands): mouthed takes against other
  blocks of the same session, against other sessions, and against the spoken enrolment takes.

Each result carries two controls: a duration-only matcher (sentence length alone can give the
answer away) and a label-permutation null on the same distances.
"""
import numpy as np
from collections import defaultdict
from . import camera as camera_input
from .data import features, group_id

STEP = 2  # 50 Hz -> 25 Hz for matching
MODALITIES = ("imu", "camera", "fused")
CAMERA_COLUMNS = camera_input.VALUE_WIDTH + len(camera_input.SCALARS)


def sequence(row, modality):
    """Per-clip centred frames (posture and seating offsets removed), or None if unavailable."""
    parts = []
    if modality in ("imu", "fused"):
        parts.append(("imu", features(np.asarray(row["raw"]))))
    if modality in ("camera", "fused"):
        if not row.get("camera") or camera_input.coverage(row["camera"]) < .8:
            return None
        values = camera_input.features(np.asarray(row["camera"]))
        parts.append(("camera", values[:, :CAMERA_COLUMNS]))
    out = []
    for name, x in parts:
        x = x[::STEP].astype(np.float64)
        out.append((name, x - x.mean(0)))
    return out


def scales(sequences):
    """Pooled per-channel scale for each modality block, so blocks weigh equally."""
    blocks = defaultdict(list)
    for parts in sequences:
        for name, x in parts:
            blocks[name].append(x)
    return {name: np.concatenate(xs).std(0).clip(1e-4) for name, xs in blocks.items()}


def flatten(parts, scale):
    return np.concatenate([x / scale[name] / np.sqrt(x.shape[1]) for name, x in parts], axis=1)


def dtw(a, b):
    """Length-normalised DTW distance with Euclidean frame cost."""
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    n, m = cost.shape
    previous = np.full(m + 1, np.inf)
    previous[0] = 0
    for i in range(n):
        current = np.full(m + 1, np.inf)
        row = cost[i]
        for j in range(1, m + 1):
            current[j] = row[j - 1] + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return float(previous[-1] / (n + m))


def score(queries, templates, distance, rng, permutations=1000):
    """Top-1/top-3 over sentence classes (best template per class), a duration-only matcher on
    the same candidates, and a permutation p-value that shuffles template labels."""
    labels = sorted({t["label"] for t in templates})
    index = {label: i for i, label in enumerate(labels)}
    template_labels = np.array([index[t["label"]] for t in templates])
    rows = []
    for q in queries:
        allowed = np.array([q["allow"](t) for t in templates])
        if not allowed.any() or not any(allowed[template_labels == index.get(q["label"], -1)]):
            continue
        d = np.array([distance(q, t) if ok else np.inf for t, ok in zip(templates, allowed)])
        durations = np.array([abs(np.log(q["frames"] / t["frames"])) if ok else np.inf for t, ok in zip(templates, allowed)])
        rows.append((index[q["label"]], d, durations, allowed, int((allowed & (template_labels == index[q["label"]])).sum())))
    if not rows:
        return None

    def ranks(labels_of_templates, use_durations=False):
        result = []
        for truth, d, durations, allowed, _ in rows:
            values = durations if use_durations else d
            best = np.full(len(labels), np.inf)
            np.minimum.at(best, labels_of_templates[allowed], values[allowed])
            candidates = np.isfinite(best)
            order = np.argsort(best[candidates], kind="stable")
            names = np.flatnonzero(candidates)[order]
            result.append((int(np.flatnonzero(names == truth)[0]) + 1 if truth in names else len(names) + 1, int(candidates.sum())))
        return result

    observed = ranks(template_labels)
    single = np.array([count == 1 for *_, count in rows])
    hits = np.array([r == 1 for r, _ in observed])
    top1 = float(hits.mean())
    exceed = 0
    null_hits = np.zeros(len(rows))
    for _ in range(permutations):
        shuffled = np.array([r == 1 for r, _ in ranks(rng.permutation(template_labels))])
        null_hits += shuffled
        if float(shuffled.mean()) >= top1:
            exceed += 1
    null_hits /= max(1, permutations)
    duration = np.array([r == 1 for r, _ in ranks(template_labels, use_durations=True)])
    result = dict(queries=len(observed), top1=top1, top3=float(np.mean([r <= 3 for r, _ in observed])),
                  chance_top1=float(np.mean([1 / n for _, n in observed])),
                  # Shuffled labels keep each sentence's template count, so this is the honest chance level
                  # when some sentences have many templates and others one.
                  null_mean_top1=float(null_hits.mean()),
                  mean_candidates=float(np.mean([n for _, n in observed])),
                  duration_top1=float(duration.mean()),
                  p_value=(exceed + 1) / (permutations + 1), permutations=permutations)
    if single.any():
        result["one_template"] = dict(queries=int(single.sum()), top1=float(hits[single].mean()), duration_top1=float(duration[single].mean()),
                                      null_mean_top1=float(null_hits[single].mean()))
    return result


def closed_set(examples, seed=7, permutations=1000):
    rng = np.random.default_rng(seed)
    active = [r for r in examples if r["activity"]]
    # One shared cohort: every modality is scored on exactly the clips that have camera points.
    cohort = [r for r in active if sequence(r, "camera") is not None]
    report = dict(protocols={}, cohort=dict(utterances=len(cohort), excluded_without_camera=len(active) - len(cohort)),
                  method="25 Hz per-clip-centred features, DTW, best template per sentence; durations and shuffled labels as controls")
    for modality in MODALITIES:
        items = []
        for r in cohort:
            parts = sequence(r, modality)
            items.append(dict(id=r["id"], label=group_id(r["text"]), session=r["session"], condition=r["condition"],
                              pair=r["provenance"].get("pair"), kind=r["provenance"].get("kind"), parts=parts, frames=len(r["raw"])))
        scale = scales([i["parts"] for i in items])
        for i in items:
            i["x"] = flatten(i["parts"], scale)
        cache = {}

        def distance(q, t):
            key = (q["id"], t["id"])
            if key not in cache:
                cache[key] = dtw(q["x"], t["x"])
            return cache[key]

        spoken = [i for i in items if i["condition"] == "Normal speech"]
        mouthed = [dict(i, allow=lambda t: True) for i in items if i["condition"] == "Silent mouthing"]
        enrol = score(mouthed, spoken, distance, rng, permutations)
        report["protocols"].setdefault("enrol_spoken", {})[modality] = enrol
        for condition in ("Normal speech", "Silent mouthing"):
            pool = [i for i in items if i["condition"] == condition]
            sessions = defaultdict(set)
            for i in pool:
                sessions[i["label"]].add(i["session"])
            queries = [dict(i, allow=(lambda s: lambda t: t["session"] != s)(i["session"])) for i in pool if len(sessions[i["label"]]) >= 2]
            result = score(queries, pool, distance, rng, permutations)
            if result:
                result["repeated_sentences"] = sum(len(s) >= 2 for s in sessions.values())
            report["protocols"].setdefault("cross_session", {}).setdefault(condition, {})[modality] = result
        commands = [i for i in items if i["kind"] == "command set"]
        if commands:
            mouthed = [i for i in commands if i["condition"] == "Silent mouthing"]
            spoken = [i for i in commands if i["condition"] == "Normal speech"]
            # Same session, other blocks: repeatability before any reseating. Never the query itself.
            within = [dict(i, allow=(lambda q: lambda t: t["session"] == q["session"] and t["id"] != q["id"])(i)) for i in mouthed]
            cross = [dict(i, allow=(lambda q: lambda t: t["session"] != q["session"])(i)) for i in mouthed]
            enrol_queries = [dict(i, allow=lambda t: True) for i in mouthed]
            for key, queries, templates in (("commands_within_session", within, mouthed), ("commands_cross_session", cross, mouthed),
                                            ("commands_enrol_spoken", enrol_queries, spoken)):
                result = score(queries, templates, distance, rng, permutations) if templates else None
                report["protocols"].setdefault(key, {})[modality] = result
    report["interpretation"] = ("Closed-set identification of already-recorded sentences. It does not show open-sentence recognition; "
                                "a top-1 near duration_top1 means length, not articulation, carried the answer.")
    return report
