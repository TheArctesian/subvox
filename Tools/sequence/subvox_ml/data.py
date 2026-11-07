"""Immutable-source dataset preparation. Split sentence groups AND sessions before features."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np

SCHEMA = 3
RATE = 50
VECTOR_KEYS = ("ax", "ay", "az", "rx", "ry", "rz", "gx", "gy", "gz")
QUAT_KEYS = ("qw", "qx", "qy", "qz")
BASE_NAMES = list(VECTOR_KEYS) + ["acc_vertical", "rot_vertical", "acc_horizontal", "rot_horizontal", "acc_norm", "rot_norm", "acc_dot_rot",
                                  "att_x", "att_y", "att_z", "att_norm"]
FEATURE_NAMES = BASE_NAMES + ["d_" + k for k in BASE_NAMES]
FEATURE_WIDTH = len(FEATURE_NAMES)
# Channels whose sign or axis depends on which ear the sensor sits in; masked when the side is unknown.
DIRECTIONAL = [i for i, n in enumerate(BASE_NAMES) if n in VECTOR_KEYS or n in ("att_x", "att_y", "att_z")]
DIRECTIONAL += [i + len(BASE_NAMES) for i in DIRECTIONAL]
ATTITUDE_HIGHPASS = .5  # Hz: below this, attitude change is posture or sensor-fusion drift, not jaw


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def group_id(text):
    return hashlib.sha256(" ".join(words(text)).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


# Unreviewed pairs enter when local Whisper heard at least this share of the prompt words.
# CTC needs only the transcript, not word timing, so alignment failures don't exclude a pair.
MIN_TRANSCRIPT_MATCH = .85


def discover(root, language, corrections=None, min_transcript_match=MIN_TRANSCRIPT_MATCH):
    """Never loads reserved final recordings into the training material.
    `min_transcript_match=None` admits only pairs the user has accepted."""
    root = Path(root)
    rows, sources, seen = [], {}, {}
    corrections = corrections or {}
    sentence_data = root / "sentence-training.json"
    reserved = set()
    if sentence_data.exists():
        data = json.loads(sentence_data.read_text())
        reserved = {k for k, v in data["roles"].items() if v == "Final test"}
    else:
        data = {"pairs": [], "roles": {}}

    def load(path):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def add(identity, text, groups, session, recording, provenance, original_role="flexible", prior_test=False, reference=None, confidence=None):
        # Empty targets describe non-speech; keep its explicit movement group,
        # rather than connecting every negative through the empty-text hash.
        groups = sorted(set(groups + ([group_id(text)] if words(text) else [])))
        if reserved.intersection(groups) or original_role == "Final test":
            return
        correction = corrections.get(identity)
        if correction is not None:
            text = correction
            groups = sorted(set(groups + [group_id(text)]))
        if reserved.intersection(groups):
            raise ValueError("A correction duplicates a reserved final sentence")
        row = dict(id=identity, text=text, groups=groups, session=session, condition=recording["condition"],
                   recording=recording, original_role=original_role,
                   provenance=dict(**provenance, reference="user correction" if correction is not None else reference or "recorded prompt; not acoustically verified",
                                   confidence=confidence, estimated_phone_boundaries=False, former_test=prior_test))
        fingerprint = digest(row)
        if identity in seen:
            if seen[identity] != fingerprint:
                raise ValueError(f"Conflicting recording identity {identity}")
            return
        seen[identity] = fingerprint
        rows.append(row)

    for path in sorted((root / "ControlledSignalTests").glob("*.json")):
        comparison = load(path)
        if comparison.get("endedAt") is None:
            continue
        for run in comparison["blocks"]:
            if run.get("endedAt") is None or not run.get("session"):
                continue
            for take in run["takes"]:
                if take.get("discarded") or take.get("endReason") not in ("done", "scheduled"):
                    continue
                text = language["diagnosticSentences"][take["prompt"]["sentence"]]
                add(take["id"], text, [], run["session"]["id"], take["recording"],
                    dict(source=str(path), kind="controlled diagnostic", comparison=comparison["id"]),
                    prior_test=run.get("control", {}).get("block") == "reseated")
    for path in sorted((root / "RepresentationStudies").glob("*/study.json")):
        study = load(path)
        training = {p["text"] for p in study["training"]}
        for take in study["takes"]:
            if take.get("discardedReason"):
                continue
            is_test = take["text"] not in training
            # A may train another model; an unfinished prospective B cannot be consumed.
            if is_test and not study["complete"]:
                continue
            add(take["id"], take["text"], [take["sentenceID"]], take["sessionID"], take["recording"],
                dict(source=str(path), kind="representation pilot", comparison=study["id"]),
                original_role="flexible" if is_test else "Training", prior_test=is_test)
    for path in sorted((root / "SequenceLearning" / "Collections").glob("*/session.json")):
        visit = load(path)
        # Prospective tests/inference NEVER flow back into preparation implicitly.
        if not visit["complete"] or visit["role"] not in ("training", "development", "negativeTraining", "negativeDevelopment"):
            continue
        for take in visit["takes"]:
            if take.get("discardedReason"):
                continue
            negative=visit["role"].startswith("negative")
            if negative and words(take["text"]):
                raise ValueError("Non-speech recordings must have empty text labels; instructions are not transcripts")
            add(take["id"], take["text"], [take["sentenceID"]], visit["session"]["id"], take["recording"],
                dict(source=str(path),kind="non-speech motion" if negative else "motion-only neural collection",visit=visit["id"]),
                original_role="Training" if visit["role"] in ("training","negativeTraining") else "Validation")
    if sentence_data.exists():
        paired = load(sentence_data)
        for pair in paired["pairs"]:
            groups = pair.get("promptIDs", [])
            if pair["role"] == "Final test" or reserved.intersection(groups) or pair["review"] == "Excluded":
                continue
            match = pair.get("transcriptMatchRate")
            if pair["review"] == "Accepted":
                reference = "recorded prompt; pair accepted by user"
            elif min_transcript_match is not None and match is not None and match >= min_transcript_match:
                reference = f"recorded prompt; local Whisper matched {match:.0%} of prompt words in the spoken take; user review pending"
            else:
                continue
            for key in ("spoken", "silent"):
                if pair.get(key) and pair.get("recordingSessionID"):
                    add(pair[key]["id"], pair["text"], groups, pair["recordingSessionID"], pair[key],
                        dict(source=str(sentence_data), kind="reviewed sentence pair", pair=pair["id"], review=pair["review"]),
                        original_role=pair["role"], reference=reference, confidence=match)
    return rows, sources, reserved


def assign_folds(rows, reserved, folds=4):
    """Block cross-validation over whole sessions AND connected sentence groups.

    Sessions and sentence groups are each dealt into K folds. In fold f a row validates when
    both its session and its group are in f, trains when neither is, and is quarantined
    otherwise, so no fold shares a session or a sentence between training and validation.
    Recording is crossed (every session repeats the same sentences), so only about 1/K of the
    rows are ever validated; that is the price of keeping both disjoint.
    Rows marked Training are never validated; rows marked Validation never train a fold."""
    if not rows:
        raise ValueError("No real labelled recordings found")
    # Paragraph constituents are connected: any co-occurrence makes one atomic group.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    for row in rows:
        if reserved.intersection(row["groups"]):
            raise ValueError("Reserved final-test contamination")
        for g in row["groups"]:
            union(row["groups"][0], g)
    component = {row["id"]: find(row["groups"][0]) for row in rows}
    eligible = [r for r in rows if r["original_role"] != "Training"]

    def deal(key):
        # Largest first onto the lightest fold; ties by id so preparation is reproducible.
        sizes = Counter(key(r) for r in eligible)
        count = min(folds, len(sizes))
        bins, assigned = [0] * max(count, 1), {}
        for value, n in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
            target = min(range(len(bins)), key=lambda b: (bins[b], b))
            assigned[value] = target
            bins[target] += n
        return assigned, count
    group_fold, group_count = deal(lambda r: component[r["id"]])
    sessions = Counter(r["session"] for r in eligible)
    k = min(group_count, len(sessions))
    if k < 2:
        raise ValueError("Cross-validation needs at least two sentence groups and two sessions that aren't training-only")
    # Fold numbers above k fold back in, so groups and sessions share the same k bins.
    group_fold = {g: f % k for g, f in group_fold.items()}
    # A row validates only where its session and group share a fold, so put each session in
    # the fold holding most of its eligible rows' groups; ties go to the lightest fold.
    session_fold, load = {}, [0] * k
    for session, n in sorted(sessions.items(), key=lambda kv: (-kv[1], kv[0])):
        overlap = Counter(group_fold[component[r["id"]]] for r in eligible if r["session"] == session)
        target = max(range(k), key=lambda f: (overlap[f], -load[f], -f))
        session_fold[session] = target
        load[target] += n

    roles = {row["id"]: [] for row in rows}
    summaries = []
    for f in range(k):
        for row in rows:
            g, s = group_fold.get(component[row["id"]], -1), session_fold.get(row["session"], -1)
            if g == f and s == f and row["original_role"] != "Training":
                role = "validation"
            elif g != f and s != f and row["original_role"] != "Validation":
                role = "training"
            else:
                role = "quarantine"
            roles[row["id"]].append(role)
        train = [r for r in rows if roles[r["id"]][f] == "training"]
        val = [r for r in rows if roles[r["id"]][f] == "validation"]
        assert not ({r["session"] for r in train} & {r["session"] for r in val})
        assert not ({component[r["id"]] for r in train} & {component[r["id"]] for r in val})
        summaries.append(dict(fold=f, training=len(train), validation=len(val),
                              quarantine=sum(roles[r["id"]][f] == "quarantine" for r in rows),
                              validation_sessions=sorted({r["session"] for r in val}),
                              validation_groups=sorted({g for r in val for g in r["groups"]})))
    if sum(1 for f in summaries if f["training"] and f["validation"]) < 2:
        raise ValueError("Fewer than two folds have both training and validation recordings")
    return roles, dict(folds=summaries, k=k,
                       protocol="block cross-validation; whole sessions and connected sentence groups disjoint within each fold; crossed cells quarantined per fold")


def resample(recording, baseline=False):
    samples = recording["samples"]
    if len(samples) < 12:
        raise ValueError("too few motion samples")
    t = np.asarray([s["t"] for s in samples], dtype=np.float64)
    vectors = np.asarray([[s[k] for k in VECTOR_KEYS] for s in samples], dtype=np.float64)
    dt = np.diff(t)
    if not np.isfinite(t).all() or not np.isfinite(vectors).all() or (dt <= 0).any():
        raise ValueError("nonfinite values or unordered timestamps")
    start, end = recording["cueStart"], recording["cueEnd"]
    if not math.isfinite(start) or not math.isfinite(end) or not .5 <= end - start <= 48:
        raise ValueError("invalid cue duration")
    if baseline:
        start, end = start - 2.5, start - .5
    lo, hi = max(start, t[0]), min(end, t[-1])
    if lo - start > .1 or end - hi > .1 or hi - lo < .9 * (end - start):
        raise ValueError("motion does not cover cue (100 ms / 90% rule)")
    selected = (t >= lo - .1) & (t <= hi + .1)
    locations = {s.get("sensorLocation", -1) for s, use in zip(samples, selected) if use}
    locations = {-1 if s in (None, 0, -1) else s for s in locations}
    if len(locations) != 1:
        raise ValueError("sensor source switched inside utterance/context")
    if np.diff(t[selected]).max(initial=0) > .12:
        raise ValueError("internal motion gap exceeds 120 ms")
    intervals=dt[selected[:-1] & selected[1:]]
    rate = 1 / float(np.median(intervals)) if len(intervals) else 0
    if not math.isfinite(rate) or rate < 40:
        raise ValueError("motion rate below 40 Hz; no invented high-rate information")
    grid = lo + np.arange(int(math.floor((hi - lo) * RATE)) + 1) / RATE
    raw = np.stack([np.interp(grid, t, vectors[:, k]) for k in range(9)], axis=1).astype(np.float32)
    if all(k in samples[0] for k in QUAT_KEYS):
        quaternions = hemisphere(np.asarray([[s[k] for k in QUAT_KEYS] for s in samples], dtype=np.float64))
        attitude = np.stack([np.interp(grid, t, quaternions[:, k]) for k in range(4)], axis=1)
        attitude /= np.linalg.norm(attitude, axis=1, keepdims=True).clip(1e-9)
        attitude_state = "identity" if np.abs(attitude - [1, 0, 0, 0]).max() < 1e-6 else "ok"
    else:
        attitude = np.tile([1., 0, 0, 0], (len(grid), 1))
        attitude_state = "missing"
    raw = np.concatenate([raw, attitude.astype(np.float32)], axis=1)
    if len(raw) < 25 or (np.linalg.norm(raw[:, 6:9], axis=1) < .1).any():
        raise ValueError("short sequence or invalid gravity vector")
    flags = []
    if np.max(np.linalg.norm(raw[:, :6], axis=1)) < .005:
        flags.append("low motion energy retained; not an exclusion")
    side = next(iter(locations))
    if side == -1:
        flags.append("sensor side unknown; directional channels masked")
    if attitude_state != "ok":
        flags.append(f"attitude {attitude_state}; jaw-angle channels are zero")
    return raw, dict(rate=rate, source=side, frames=len(raw), start=lo, end_gap=end-hi, flags=flags, attitude=attitude_state)


def hemisphere(q):
    """q and -q are the same rotation; flip signs so consecutive quaternions agree."""
    q = np.array(q, dtype=np.float64)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i-1]) < 0:
            q[i] = -q[i]
    return q


def quaternion_multiply(a, b):
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1)


def attitude_channels(quaternions):
    """Jaw-angle proxy: rotation vector of each attitude relative to the clip's mean attitude,
    high-passed at 0.5 Hz so posture and drift drop out, plus its norm, which doesn't depend on
    the ear. Rotation rate alone gives velocity; this keeps the opening angle itself."""
    from .rhythm import biquad, lfilter
    q = hemisphere(np.asarray(quaternions, dtype=np.float64))
    if np.abs(q - [1, 0, 0, 0]).max() < 1e-6:
        return np.zeros((len(q), 4), dtype=np.float32)
    mean = q.mean(0)
    mean /= np.linalg.norm(mean)
    relative = quaternion_multiply(mean * [1, -1, -1, -1], q)
    relative *= np.sign(relative[:, :1] + 1e-12)
    angle = 2 * np.arccos(np.clip(relative[:, 0], -1, 1))
    axis = relative[:, 1:] / np.maximum(np.sin(angle / 2), 1e-9)[:, None]
    vector = axis * angle[:, None]
    b, a = biquad("high", ATTITUDE_HIGHPASS)
    for k in range(3):
        y = vector[:, k] - vector[:, k].mean()
        vector[:, k] = lfilter(b, a, lfilter(b, a, y)[::-1])[::-1]
    return np.concatenate([vector, np.linalg.norm(vector, axis=1, keepdims=True)], axis=1).astype(np.float32)


def features(raw, mask_direction=False):
    """Per-frame features from resampled raw motion (9 vector columns, then optionally the
    attitude quaternion): raw axes, gravity-frame invariants, jaw-angle channels, derivatives."""
    raw = np.asarray(raw, dtype=np.float32)
    attitude = attitude_channels(raw[:, 9:13]) if raw.shape[1] >= 13 else np.zeros((len(raw), 4), np.float32)
    raw = raw[:, :9]
    a, r, g = raw[:, :3], raw[:, 3:6], raw[:, 6:9]
    unit = g / np.linalg.norm(g, axis=1, keepdims=True).clip(.1)
    av, rv = (a * unit).sum(1), (r * unit).sum(1)
    aa, rr = (a*a).sum(1), (r*r).sum(1)
    inv = np.stack([av, rv, np.sqrt(np.maximum(0, aa-av*av)), np.sqrt(np.maximum(0, rr-rv*rv)),
                    np.sqrt(aa), np.sqrt(rr), (a*r).sum(1)], axis=1)
    base = np.concatenate([raw, inv, attitude], axis=1)
    derivative = np.zeros_like(base)
    derivative[1:] = (base[1:] - base[:-1]) * RATE
    result = np.concatenate([base, derivative], axis=1)
    if mask_direction:
        result[:, DIRECTIONAL] = 0
    return result.astype(np.float32)


def pronunciations(text, language, max_paths=32):
    paths = [[]]
    unconfirmed = []
    for index, word in enumerate(words(text)):
        variants = language["entries"].get(word, [])
        if not variants:
            raise ValueError(f"unknown pronunciation: {word}")
        if not all(v["speakerConfirmed"] for v in variants):
            unconfirmed.append(word)
        alternatives = sorted({tuple(v["ipa"]) for v in variants})
        if len(paths) * len(alternatives) > max_paths:
            raise ValueError("too many pronunciation combinations; review variants (no silent truncation)")
        paths = [p + (["|"] if index else []) + list(v) for p, v in itertools.product(paths, alternatives)]
    return paths, sorted(set(unconfirmed))


def prepare(root, language_path, output, corrections=None, folds=4, min_transcript_match=MIN_TRANSCRIPT_MATCH):
    output = Path(output)
    if output.exists():
        raise ValueError("Dataset output already exists; create a new immutable snapshot")
    language = json.loads(Path(language_path).read_text())
    rows, sources, reserved = discover(root, language, corrections, min_transcript_match)
    roles, split = assign_folds(rows, reserved, folds)
    examples, excluded = [], []
    # Folds above are fixed before resampling, any estimated alignment, or augmentation.
    # Every row is kept: each fold reads its own roles, and the final refit trains on all.
    for row in rows:
        row = dict(row, fold_roles=roles[row["id"]])
        try:
            raw, quality = resample(row["recording"])
            targets, proposals = pronunciations(row["text"], language)
            if not targets or min(len(p) + sum(a == b for a, b in zip(p, p[1:])) for p in targets) > len(raw):
                raise ValueError("insufficient CTC frames for transcript/repeated phones")
            item = {k:v for k,v in row.items() if k != "recording"}
            item.update(raw=raw.tolist(), targets=targets, quality=quality, unconfirmed_pronunciations=proposals, activity=int(bool(words(row["text"]))),
                        alignment_provenance="CTC will marginalise alignment; no phone boundaries supplied")
            examples.append(item)
            try:
                if not item["activity"]:continue
                rest, q = resample(row["recording"], baseline=True)
                negative = dict(item, id=row["id"]+"/rest", text="", raw=rest.tolist(), targets=[[]], quality=q, activity=0,
                                alignment_provenance="instructed relaxed-mouth baseline; not acoustic VAD ground truth")
                examples.append(negative)
            except ValueError:
                pass
        except (ValueError, KeyError) as error:
            excluded.append(dict(id=row["id"], reason=str(error), text=row["text"], session=row["session"], groups=row["groups"],
                                 condition=row["condition"], fold_roles=row["fold_roles"]))
    from .rhythm import signal_quality
    rest = {e["id"]: e["raw"] for e in examples if not e["activity"]}
    for e in examples:
        if e["activity"]:
            e["quality"]["signal"] = signal_quality(e["raw"], rest.get(e["id"] + "/rest"))
    # Fill in the sensor side of older takes only when a leave-one-session-out check passes.
    from .side import infer_sides
    rest = {e["id"]: e["raw"] for e in examples if not e["activity"]}
    inferred, side_report = infer_sides([dict(e, baseline_raw=rest.get(e["id"] + "/rest")) for e in examples if e["activity"]])
    for e in examples:
        side = inferred.get(e["id"].removesuffix("/rest"))
        if side is not None and e["quality"]["source"] not in (1, 2):
            e["quality"]["source"] = side
            e["quality"]["source_inferred"] = dict(side=side, method="rest gravity direction; leave-one-session-out gated")
            e["quality"]["flags"] = [f.replace("sensor side unknown; directional channels masked", "sensor side inferred; directional channels used")
                                     for f in e["quality"]["flags"]]
    for condition in ("Normal speech", "Silent mouthing"):
        if sum(e["condition"] == condition and e["activity"] for e in examples) < 4:
            raise ValueError(f"Need at least four valid training utterances for {condition}")
    manifest = dict(schema=SCHEMA, split=split, sources=sources, language_hash=hashlib.sha256(Path(language_path).read_bytes()).hexdigest(), canonical_language_hash=digest(language),
                    reserved_final_groups=sorted(reserved), excluded=excluded,
                    promoted_former_tests=[e["id"] for e in examples if e["provenance"]["former_test"] and e["activity"]],
                    admission=dict(min_transcript_match=min_transcript_match,
                                   unreviewed_pairs=sorted({e["provenance"]["pair"] for e in examples
                                                            if e["provenance"].get("review") not in (None, "Accepted")})),
                    warning="Former diagnostic tests used in training are NOT independent tests of this model. Historical models/reports are unchanged.",
                    features=dict(version="timestamp-50hz-raw-invariant-derivative-v1", rate=RATE, names=FEATURE_NAMES, low_energy="retain"),
                    side_inference=side_report, examples_hash=digest(examples))
    snapshots={};historical=[];material={}
    for name,sha in sources.items():
        raw=Path(name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError("Source changed during snapshot; retry preparation")
        snapshots[name]="source_material/"+sha+".json";material[sha]=raw
        original=json.loads(raw)
        if original.get("teacherScores"):
            historical.append(dict(comparison=original["id"],scores=original["teacherScores"],
                interpretation="Historical three-phrase camera/raw benchmark, different protocol/cohort; not evidence of improvement in CTC word decoding"))
    manifest["source_snapshots"]=snapshots;manifest["historical_baselines"]=historical
    output.mkdir(parents=True)
    (output/"source_material").mkdir()
    for sha,raw in material.items():(output/"source_material"/(sha+".json")).write_bytes(raw)
    atomic_json(output/"language.json", language)
    atomic_json(output/"examples.json", examples)
    atomic_json(output/"manifest.json", manifest)
    return manifest


def load_dataset(path):
    path = Path(path)
    manifest = json.loads((path/"manifest.json").read_text())
    # Schema 2 snapshots lack the attitude quaternion; their jaw-angle channels are simply zero.
    if manifest.get("schema", 1) < 2:
        raise ValueError("This dataset predates cross-validation folds; prepare a new dataset snapshot")
    examples = json.loads((path/"examples.json").read_text())
    language = json.loads((path/"language.json").read_text())
    if digest(examples) != manifest["examples_hash"] or hashlib.sha256((path/"language.json").read_bytes()).hexdigest() != manifest["language_hash"]:
        # Language serialization is canonical in snapshots; compare canonical source hash too.
        if digest(examples) != manifest["examples_hash"] or digest(language) != manifest.get("canonical_language_hash", manifest["language_hash"]):
            raise ValueError("Dataset/language fingerprint mismatch")
    return examples, language, manifest
