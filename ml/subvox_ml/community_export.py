"""Local recordings -> community dataset layout (docs/community-dataset-design.md, section 4).

    data/<contributor>/<session-id>/
      session.json     schema version, app version, device models (null until recorded), sensor side,
                       sample rates, consent version, month
      takes.jsonl      take id, prompt id, passage id, mode, cue start/end, eligibility flags, redo count
      motion.parquet   take id, t, acceleration, rotation, gravity, attitude, side
      camera.parquet   take id, t, eye-frame landmark channels and masks (50 Hz grid)

Only community, calibration, and separately opted-in library pilot prompts are exported.
Times are seconds from the session start;
the only date is the month. Session and take IDs are one-way hashes of local IDs, so re-exporting
is idempotent and nothing local (paths, device IDs, wall-clock times) leaves the Mac.
`validate_session` is the same check the validator Space runs before merging a PR.
"""
import datetime
import hashlib
import json
import math
import re
from pathlib import Path
from uuid import UUID
import numpy as np
from . import camera as camera_input
from . import ipa
from .data import RATE, resample

SCHEMA_VERSION = 2  # v2: language tags, speaker status, speaking style, expected/produced phones per take
SUPPORTED_SCHEMAS = (1, 2)
LIBRARY_CATALOG = "library-pilot-v1.jsonl"
LIBRARY_PROMPTS_VERSION = 2
LIBRARY_CONSENT = "consent-v3-library-pilot"
LIBRARY_PIECES = {"invictus": ("poetry", 4), "gettysburg": ("speech", 3)}
SESSION_FILES = ("camera.parquet", "motion.parquet", "session.json", "takes.jsonl")
SWIFT_EPOCH = datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc)
MOTION_COLUMNS = ["take_id", "t", "ax", "ay", "az", "rx", "ry", "rz", "gx", "gy", "gz", "qw", "qx", "qy", "qz", "side"]
CAMERA_COLUMNS = ["take_id", "t"] + camera_input.RAW_NAMES
SESSION_KEYS = {"schema_version", "app_version", "contributor", "session_id", "airpods_model", "airpods_vendor_id", "airpods_product_id",
                "airpods_firmware", "motion_source", "mac_model", "camera_model", "sensor_side", "sample_rates", "consent_version",
                "month", "prompts_version", "takes"}
TAKE_KEYS = {"take_id", "prompt_id", "passage_id", "mode", "cue_start", "cue_end", "motion_hz", "eligible", "ineligible_reason",
             "camera_coverage", "camera_eligible", "redo_count"}
TIMING_KEYS = {"word_timing_source", "word_timing_model_revision"}
TAKE_KEYS_V2 = TAKE_KEYS | {"language", "speaker_status", "first_language", "speaking_style", "expected_phones", "produced_phones"}
SPEAKING_STYLES = ("natural", "exaggerated")
HEX_ID = re.compile(r"^0x[0-9A-F]{4}$")
CONTRIBUTOR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")
MAX_SESSION_BYTES = 200 * 1024 * 1024


def opaque(*parts):
    return hashlib.sha256(":".join(["subvox-community", *map(str, parts)]).encode()).hexdigest()[:32]


def find_prompts_dir(package=Path(__file__).resolve().parent):
    candidates = [package.parent / "Corpus" / "community", package.parents[1] / "Corpus" / "community"]
    return next((p for p in candidates if p.exists()), candidates[0])


def load_prompt_index(prompts_dir):
    """prompt_id -> (passage_id or None for calibration), plus the held-out IDs that must never appear."""
    prompts_dir = Path(prompts_dir)
    index, heldout = {}, set()
    for path in sorted(prompts_dir.glob("prompts-v*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            index[row["prompt_id"]] = row["passage_id"]
    for path in sorted(prompts_dir.glob("calibration-v*.jsonl")):
        for line in path.read_text().splitlines():
            index[json.loads(line)["prompt_id"]] = None
    for path in sorted(prompts_dir.glob("heldout-v*.jsonl")):
        heldout.update(json.loads(line)["prompt_id"] for line in path.read_text().splitlines())
    return index, heldout


def library_prompt_identity(language, piece_id, text):
    exact = "\n".join(("library-pilot-v1", language, piece_id, text))
    return hashlib.sha256(exact.encode("utf-8")).hexdigest()


def load_library_pilot(prompts_dir):
    path = Path(prompts_dir) / LIBRARY_CATALOG
    if not path.exists():
        return {}
    sections, positions = {}, set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        piece_id = row.get("piece_id")
        kind, section_count = LIBRARY_PIECES.get(piece_id, (None, 0))
        lines = row.get("lines")
        if (kind is None or kind != row.get("kind") or row.get("language") != "en"
                or not isinstance(lines, list) or not lines or any(not isinstance(s, str) or not s.strip() for s in lines)
                or row.get("text") != " ".join(lines) or row.get("passage_id") != f"library-pilot-v1:{piece_id}"
                or type(row.get("section")) is not int or not 1 <= row["section"] <= section_count):
            raise ValueError("Invalid library pilot catalog section")
        prompt_id = library_prompt_identity(row["language"], piece_id, row["text"])
        position = (piece_id, row["section"])
        if row.get("prompt_id") != prompt_id or prompt_id in sections or position in positions:
            raise ValueError("Invalid or duplicate library pilot prompt ID")
        sections[prompt_id] = row
        positions.add(position)
    if positions != {(piece_id, section) for piece_id, (_, count) in LIBRARY_PIECES.items()
                     for section in range(1, count + 1)}:
        raise ValueError("Incomplete library pilot catalog")
    return sections


def library_pair_prompt(pair, visit, grant, catalog, roles):
    """Return the approved prompt ID only for a verified physical spoken/mouthed pair."""
    if (visit.get("libraryContributionID") != grant or visit.get("endedAt") is None
            or visit.get("recordingSessionID") != pair.get("recordingSessionID")
            or pair.get("id") not in visit.get("pairIDs", []) or pair.get("review") == "Excluded"
            or pair.get("role") not in ("Training", "Validation") or pair.get("error")):
        return None
    piece_id, language, text = pair.get("literaturePieceID"), pair.get("language"), pair.get("text")
    if piece_id not in LIBRARY_PIECES or language != "en" or not isinstance(text, str):
        return None
    prompt_id = library_prompt_identity(language, piece_id, text)
    row = catalog.get(prompt_id)
    if row is None or row["kind"] != pair.get("practiceMode"):
        return None
    # Pilot lines are English and use the existing legacy identity in the local store.
    ids = [prompt_identity(line) for line in row["lines"]]
    if ids != pair.get("promptIDs") or any(roles.get(id) == "Final test" for id in ids):
        return None
    timing = pair.get("timedWordCoverage") or 0
    if not (pair.get("alignment") or {}).get("words") or not isinstance(timing, (float, int)) or not .8 <= timing <= 1:
        return None
    source = pair.get("wordTimingSource")
    if source == "forcedAlignment":
        if not re.fullmatch(r"[0-9a-f]{40}", pair.get("wordTimingModelRevision") or ""):
            return None
    elif source == "whisperAttention":
        agreement = pair.get("transcriptMatchRate") or 0
        if not isinstance(agreement, (float, int)) or not .8 <= agreement <= 1:
            return None
    else:
        return None
    start, spoken, silent = pair.get("sessionStartUptime"), pair.get("spoken"), pair.get("silent")
    if (not isinstance(start, (float, int)) or not math.isfinite(start) or start <= 0
            or not spoken or not silent or spoken.get("condition") != "Normal speech"
            or silent.get("condition") != "Silent mouthing"
            or not (spoken.get("id") and silent.get("id")) or spoken["id"] == silent["id"]):
        return None
    try:
        for recording in (spoken, silent):
            resample(recording)
            if start > recording["samples"][0]["t"] or start > recording["cueStart"]:
                return None
        if silent["cueEnd"] > spoken["cueStart"]:
            return None
    except (ValueError, KeyError, TypeError):
        return None
    return prompt_id


def native_rate(samples):
    t = np.asarray([s["t"] for s in samples], dtype=float)
    d = np.diff(t)
    d = d[d > 0]
    return round(float(1 / np.median(d)), 2) if len(d) else None


def month_of(swift_seconds):
    return (SWIFT_EPOCH + datetime.timedelta(seconds=float(swift_seconds))).strftime("%Y-%m")


def side_name(samples):
    sides = {s.get("sensorLocation") for s in samples}
    return {frozenset({1}): "left", frozenset({2}): "right"}.get(frozenset(sides), "unknown")


def take_rows(recording, origin, take_id, readings):
    samples = recording["samples"]
    motion = {c: [] for c in MOTION_COLUMNS}
    for s in samples:
        motion["take_id"].append(take_id)
        motion["t"].append(s["t"] - origin)
        for key in MOTION_COLUMNS[2:-1]:
            motion[key].append(float(s.get(key, 1.0 if key == "qw" else 0.0)))
        motion["side"].append(int(s.get("sensorLocation") or 0))
    camera_table = {c: [] for c in CAMERA_COLUMNS}
    coverage = 0.0
    if readings and len(samples) >= 2:
        start = samples[0]["t"]
        frames = int(np.floor((samples[-1]["t"] - start) * RATE)) + 1
        grid = camera_input.grid(readings, start, frames)
        times = start + np.arange(frames) / RATE
        window = (times >= recording["cueStart"]) & (times <= recording["cueEnd"])
        coverage = float(grid[window, camera_input.MOUTH_MASK].mean()) if window.any() else 0.0
        if grid[:, camera_input.MOUTH_MASK].any() or grid[:, camera_input.FACE_MASK].any():
            camera_table["take_id"] = [take_id] * frames
            camera_table["t"] = (times - origin).tolist()
            for k, name in enumerate(camera_input.RAW_NAMES):
                camera_table[name] = grid[:, k].astype(float).tolist()
    try:
        resample(recording)
        reason = None
    except (ValueError, KeyError) as error:
        reason = str(error)
    return motion, camera_table, coverage, reason


def export_sessions(root, output, contributor, consent_version, prompts_dir=None, app_version="0.1.0", prompts_version=1,
                    skip=(), automatic_grant=None, library_grant=None):
    """Writes one folder per recording session that has community takes; returns their paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not CONTRIBUTOR.match(contributor):
        raise ValueError("Contributor ID must be 2-64 letters, digits, '.', '_' or '-'")
    prompts_dir = prompts_dir or find_prompts_dir()
    index, heldout = load_prompt_index(prompts_dir)
    library = {}
    if library_grant is not None:
        UUID(library_grant)  # The app records the exact grant used to start a physical visit.
        library = load_library_pilot(prompts_dir)
        if not library:
            raise ValueError("The library pilot catalog is missing")
        if set(library) & (set(index) | heldout):
            raise ValueError("Library pilot prompt IDs overlap the reading corpus")
    data = json.loads((Path(root) / "sentence-training.json").read_text())
    visits = {v["id"]: v for v in data.get("visits", [])}
    devices = {key: v.get("motionDevice") for key, v in visits.items()}
    kinds = {p_id: p.get("kind", "sentence") for p in data.get("prompts", []) for p_id in [prompt_identity(p["text"])]}
    speaker = data.get("speaker") or {}
    roles = data.get("roles") or {}
    pending_visits = {pair.get("visitID") for pair in data.get("pairs", [])
                      if pair.get("needsBackgroundCheck") is True} if library else set()
    sessions = {}
    for pair in data.get("pairs", []):
        visit = visits.get(pair.get("visitID"), {})
        if pair.get("literaturePieceID") or visit.get("libraryContributionID") or pair.get("practiceMode") in ("poetry", "speech"):
            if library_grant is None or not pair.get("recordingSessionID") or pair.get("visitID") in pending_visits:
                continue
            prompt_id = library_pair_prompt(pair, visit, library_grant, library, roles)
            if prompt_id is not None:
                sessions.setdefault(pair["recordingSessionID"], []).append((pair, prompt_id))
            continue
        if pair.get("practiceMode") is not None:
            continue  # A practice take can reuse the text of a community prompt.
        if automatic_grant is None and library_grant is not None:
            continue  # A library-only export is not permission to include any other visits.
        if automatic_grant is not None:
            if (visit.get("automaticContributionID") != automatic_grant or visit.get("endedAt") is None
                    or visit.get("recordingSessionID") != pair.get("recordingSessionID")
                    or pair.get("review") == "Excluded"):
                continue
        ids = pair.get("promptIDs") or []
        if len(ids) != 1 or not pair.get("recordingSessionID") or pair.get("role") == "Final test":
            continue
        if kinds.get(ids[0]) not in ("community", "calibration") or ids[0] not in index or ids[0] in heldout:
            continue
        if not any((pair.get(key) or {}).get("samples") for key in ("spoken", "silent")):
            continue
        sessions.setdefault(pair["recordingSessionID"], []).append((pair, ids[0]))
    written = []
    for local_session, candidates in sorted(sessions.items()):
        session_id = opaque("session", local_session)
        if session_id in skip:
            continue  # already shared; never re-exported or offered again
        pairs = [pair for pair, _ in candidates]
        is_library = any(prompt_id in library for _, prompt_id in candidates)
        if is_library and (not all(prompt_id in library for _, prompt_id in candidates)
                           or len({p["visitID"] for p in pairs}) != 1
                           or len({p["sessionStartUptime"] for p in pairs}) != 1
                           or len({p["literaturePieceID"] for p in pairs}) != 1):
            continue  # One export must have a single recording clock, scope and piece.
        origin = min(p.get("sessionStartUptime") or min(r["samples"][0]["t"] for r in (p.get("spoken"), p.get("silent")) if r and r["samples"])
                     for p in pairs)
        takes, motion, cameras, all_samples, readings_all = [], [], [], [], []
        included_library_prompts = set()
        for pair, prompt_id in sorted(candidates, key=lambda row: row[0]["createdAt"]):
            if is_library:
                if prompt_id in included_library_prompts:
                    continue
                included_library_prompts.add(prompt_id)
            passage_id = library[prompt_id]["passage_id"] if is_library else index[prompt_id]
            readings = pair.get("cameraReadings") or []
            check = pair.get("phoneCheck") or {}
            expected = check.get("expected_phones") or expected_phones(pair.get("text", ""), pair.get("language") or "en")
            intervals = (pair.get("spokenPhones") or {}).get("phones") or []
            # A partial sequence is useful as timed annotations locally, but
            # would be an incorrect whole-take CTC target in the dataset.
            labels = [phone["ipa"] for phone in intervals]
            produced = labels if expected and len(labels) == len(expected) and all(p in ipa.INVENTORY for p in labels) else None
            if produced is None and pair.get("wordTimingSource") != "forcedAlignment" and check.get("source") == "phone-intervals":
                produced = check.get("produced_phones")
            rejected_alignment = pair.get("wordTimingSource") == "forcedAlignment" and bool(pair.get("error"))
            if rejected_alignment:
                produced = None
            timing = {}
            if pair.get("alignment") and not pair.get("error") and pair.get("wordTimingSource"):
                timing["word_timing_source"] = pair["wordTimingSource"]
                if pair.get("wordTimingModelRevision"):
                    timing["word_timing_model_revision"] = pair["wordTimingModelRevision"]
            readings_all += readings
            for mode, key in (("spoken", "spoken"), ("mouthed", "silent")):
                recording = pair.get(key)
                if not recording or not recording.get("samples"):
                    continue
                take_id = opaque("take", recording["id"])
                m, c, coverage, reason = take_rows(recording, origin, take_id, readings)
                if rejected_alignment and reason is None:
                    reason = "Word alignment did not verify the reading"
                motion.append(m); cameras.append(c); all_samples += recording["samples"]
                takes.append(dict(take_id=take_id, prompt_id=prompt_id, passage_id=passage_id, mode=mode,
                                  cue_start=recording["cueStart"] - origin, cue_end=recording["cueEnd"] - origin,
                                  motion_hz=native_rate(recording["samples"]),
                                  eligible=reason is None, ineligible_reason=reason, camera_coverage=round(coverage, 4),
                                  camera_eligible=coverage >= .8, redo_count=None,
                                  language=pair.get("language") or "en", speaker_status=pair.get("speakerStatus") or speaker.get("status") or "unknown",
                                  first_language=pair.get("firstLanguage") or speaker.get("firstLanguage"),
                                  speaking_style=pair.get("speakingStyle") or "natural",
                                  expected_phones=expected,
                                  produced_phones=produced if mode == "spoken" else None,
                                  **timing))
        if not takes:
            continue
        times = sorted(r["t"] for r in readings_all)
        camera_hz = float(1 / np.median(np.diff(times))) if len(times) > 2 and np.median(np.diff(times)) > 0 else None
        motion_t = sorted(s["t"] for s in all_samples)
        motion_hz = float(1 / np.median(np.diff(motion_t))) if len(motion_t) > 2 else None
        device = next((devices.get(p.get("visitID")) for p in pairs if devices.get(p.get("visitID"))), None) or {}
        session = dict(schema_version=SCHEMA_VERSION, app_version=app_version, contributor=contributor, session_id=session_id,
                       airpods_model=device.get("model"), airpods_vendor_id=device.get("vendorID"), airpods_product_id=device.get("productID"),
                       airpods_firmware=device.get("firmware"), motion_source="CMHeadphoneMotionManager",
                       mac_model=None, camera_model=None, sensor_side=side_name(all_samples),
                       sample_rates=dict(motion_hz=round(motion_hz, 2) if motion_hz else None, camera_hz=round(camera_hz, 2) if camera_hz else None),
                       consent_version=LIBRARY_CONSENT if is_library else consent_version, month=month_of(min(p["createdAt"] for p in pairs)),
                       prompts_version=LIBRARY_PROMPTS_VERSION if is_library else prompts_version, takes=len(takes))
        folder = Path(output) / "data" / contributor / session_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "session.json").write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")
        (folder / "takes.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in takes))
        merged_motion = {c: [v for m in motion for v in m[c]] for c in MOTION_COLUMNS}
        pq.write_table(pa.table(merged_motion), folder / "motion.parquet")
        merged_camera = {c: [v for m in cameras for v in m[c]] for c in CAMERA_COLUMNS}
        schema = pa.schema([("take_id", pa.string())] + [(c, pa.float64()) for c in CAMERA_COLUMNS[1:]])
        pq.write_table(pa.table(merged_camera, schema=schema), folder / "camera.parquet")
        problems = validate_session(folder, index, heldout, library=library)
        if problems:
            raise ValueError(f"Export failed its own schema check: {problems}")
        written.append(folder)
    return written


def prompt_identity(text):
    return hashlib.sha256(" ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())).encode()).hexdigest()


def validate_session(folder, index, heldout, library=None):
    """Schema and privacy checks for one contributed session; an empty list means valid."""
    import pyarrow.parquet as pq
    folder = Path(folder)
    problems = []
    for name in SESSION_FILES:
        if not (folder / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems
    session = json.loads((folder / "session.json").read_text())
    is_library = session.get("consent_version") == LIBRARY_CONSENT
    allowed = SESSION_FILES if is_library else SESSION_FILES + ("alignment.jsonl", "audio")
    extra = sorted(p.name for p in folder.iterdir() if p.name not in allowed)
    if extra:
        problems.append(f"unexpected files {extra}")
    if sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) > MAX_SESSION_BYTES:
        problems.append("session larger than 200 MB")
    if set(session) != SESSION_KEYS:
        problems.append(f"session.json keys differ: extra {sorted(set(session) - SESSION_KEYS)}, missing {sorted(SESSION_KEYS - set(session))}")
    version = session.get("schema_version")
    if version not in SUPPORTED_SCHEMAS:
        problems.append("unsupported schema_version")
    if is_library and (version != SCHEMA_VERSION or session.get("prompts_version") != LIBRARY_PROMPTS_VERSION):
        problems.append("unsupported library pilot schema or catalog version")
    if not re.fullmatch(r"\d{4}-\d{2}", str(session.get("month"))):
        problems.append("month must be YYYY-MM")
    if not CONTRIBUTOR.match(str(session.get("contributor"))) or folder.parent.name != session.get("contributor") or folder.name != session.get("session_id"):
        problems.append("contributor/session folder mismatch")
    for key in ("airpods_vendor_id", "airpods_product_id"):
        if session.get(key) is not None and not HEX_ID.match(str(session[key])):
            problems.append(f"{key} must be a 0xNNNN hex ID")
    if session.get("sensor_side") not in ("left", "right", "unknown"):
        problems.append("sensor_side must be left, right or unknown")
    for key, value in session.items():
        if isinstance(value, str) and ("/" in value or "\\" in value):
            problems.append(f"{key} looks like a path")
    takes = [json.loads(line) for line in (folder / "takes.jsonl").read_text().splitlines()]
    ids = set()
    library_sections = {}
    library_pieces = set()
    for take in takes:
        expected_keys = TAKE_KEYS_V2 if version == 2 else TAKE_KEYS
        keys = set(take) - (TIMING_KEYS if version == 2 else set())
        if keys != expected_keys:
            problems.append(f"take keys differ: {sorted(keys ^ expected_keys)}")
            continue
        if version == 2:
            problems += language_problems(take)
        ids.add(take["take_id"])
        if take["mode"] not in ("spoken", "mouthed"):
            problems.append(f"bad mode {take['mode']}")
        if take["prompt_id"] in heldout:
            problems.append("held-out prompt recorded")
        elif is_library:
            row = (library or {}).get(take["prompt_id"])
            if row is None or row["passage_id"] != take["passage_id"] or row["language"] != take.get("language"):
                problems.append("unknown library pilot prompt, passage or language")
            else:
                library_pieces.add(row["piece_id"])
                library_sections.setdefault(take["prompt_id"], []).append(take)
            if (take["eligible"] is not True or take["ineligible_reason"] is not None
                    or take.get("word_timing_source") not in ("forcedAlignment", "whisperAttention")):
                problems.append("library pilot takes must have eligible motion and verified word timing")
        elif take["prompt_id"] not in index or index[take["prompt_id"]] != take["passage_id"]:
            problems.append("unknown prompt or passage")
        if not take["cue_end"] > take["cue_start"]:
            problems.append("cue end before start")
    if is_library:
        if len(library_pieces) != 1:
            problems.append("library pilot session must contain one approved piece")
        for section in library_sections.values():
            by_mode = {take["mode"]: take for take in section}
            if len(section) != 2 or set(by_mode) != {"mouthed", "spoken"}:
                problems.append("library pilot sections must contain one spoken and one mouthed take")
            elif by_mode["mouthed"]["cue_end"] > by_mode["spoken"]["cue_start"]:
                problems.append("library pilot sections must be mouthed before they are spoken")
    if len(ids) != len(takes) or session.get("takes") != len(takes):
        problems.append("take count or duplicate take IDs")
    motion = pq.read_table(folder / "motion.parquet")
    if motion.column_names != MOTION_COLUMNS:
        problems.append("motion.parquet columns differ")
    else:
        take_col = motion.column("take_id").to_pylist()
        t = np.asarray(motion.column("t").to_pylist(), dtype=float)
        if set(take_col) - ids:
            problems.append("motion rows for unknown takes")
        for take_id in ids:
            mask = np.asarray([x == take_id for x in take_col])
            if mask.sum() < 12 or (np.diff(t[mask]) <= 0).any():
                problems.append("motion missing or not increasing in time")
                break
        numeric = np.asarray([motion.column(c).to_pylist() for c in MOTION_COLUMNS[1:-1]], dtype=float)
        if not np.isfinite(numeric).all():
            problems.append("non-finite motion values")
        if is_library:
            samples_by_take = {}
            for row in motion.to_pylist():
                samples_by_take.setdefault(row.pop("take_id"), []).append(dict(row, sensorLocation=row["side"]))
            for take in takes:
                samples = samples_by_take.get(take.get("take_id"), [])
                try:
                    if samples and samples[0]["t"] < 0:
                        raise ValueError("negative sample time")
                    resample(dict(samples=samples, cueStart=take["cue_start"], cueEnd=take["cue_end"]))
                    if take["cue_start"] < 0:
                        raise ValueError("negative cue time")
                except (ValueError, KeyError, TypeError):
                    problems.append("library pilot motion failed the recording eligibility checks")
                    break
    camera_table = pq.read_table(folder / "camera.parquet")
    if camera_table.column_names != CAMERA_COLUMNS:
        problems.append("camera.parquet columns differ")
    elif set(camera_table.column("take_id").to_pylist()) - ids:
        problems.append("camera rows for unknown takes")
    return problems


def language_problems(take):
    problems = []
    for key in ("language", "first_language"):
        if take[key] is None and key == "first_language":
            continue
        try:
            ipa.validate_language_tag(take[key])
        except ValueError:
            if key == "language" or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Z]{2})?", str(take[key])):
                problems.append(f"bad {key} tag {take[key]!r}")
    if take["speaker_status"] not in ipa.SPEAKER_STATUS:
        problems.append(f"bad speaker_status {take['speaker_status']!r}")
    if take["speaking_style"] not in SPEAKING_STYLES:
        problems.append(f"bad speaking_style {take['speaking_style']!r}")
    for key in ("expected_phones", "produced_phones"):
        value = take[key]
        if value is not None and (not isinstance(value, list) or any(p not in ipa.INVENTORY for p in value)):
            problems.append(f"{key} must be a list of shared-inventory IPA phones")
    source = take.get("word_timing_source")
    revision = take.get("word_timing_model_revision")
    if source not in (None, "forcedAlignment", "whisperAttention"):
        problems.append("unknown word timing source")
    if source == "forcedAlignment":
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            problems.append("forced alignment requires a model revision")
    elif revision is not None:
        problems.append("unexpected word timing model revision")
    return problems


def training_phones(take, produced_source=None):
    """The label a take trains on (data rule 3): the phones actually produced when an audio phone
    recogniser or aligner ran; the textbook phones only for a native speaker reading known text;
    otherwise None (practice only). Phones guessed from the words Whisper heard are not 'produced':
    a mispronounced sound inside a recognised word would become a correct-looking label."""
    if take.get("eligible") is False:
        return None
    if take.get("produced_phones") and produced_source in (None, "phone-intervals"):
        return take["produced_phones"]
    if take.get("speaker_status") == "native" and take.get("speaking_style") == "natural":
        return take.get("expected_phones")
    return None


def expected_phones(text, language):
    try:
        base = ipa.validate_language_tag(language)
        return ipa.phones(text, base, ipa.accent_of(language)) if base == "zh" or ipa.espeak_available() else None
    except (ValueError, RuntimeError):
        return None


def without_takes(folder, drop, destination):
    """A copy of a session with the user's dropped takes removed from every file."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    drop = set(drop)
    takes = [json.loads(line) for line in (folder / "takes.jsonl").read_text().splitlines()]
    kept = [t for t in takes if t["take_id"] not in drop]
    if not kept:
        raise ValueError("Every take was dropped; nothing to upload")
    (destination / "takes.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in kept))
    session = json.loads((folder / "session.json").read_text())
    session["takes"] = len(kept)
    (destination / "session.json").write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")
    keep = pa.array(sorted({t["take_id"] for t in kept}))
    for name in ("motion.parquet", "camera.parquet"):
        table = pq.read_table(folder / name)
        pq.write_table(table.filter(pc.is_in(table.column("take_id"), value_set=keep)), destination / name)
    return destination


def library_receipt_takes(path):
    ids = [json.loads(line)["take_id"] for line in Path(path).read_text().splitlines()]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[0-9a-f]{32}", take_id) for take_id in ids):
        raise ValueError("Invalid library pilot receipt")
    return dict(takes=len(ids), take_ids=ids)


def upload_session(folder, repo, token, drop=(), pr=None, prompts_dir=None, api=None, automatic=False, download=None):
    """One commit per session, opened as a pull request on `repo` (a dataset). Re-uploading with
    `pr` adds to that PR's `refs/pr/N` branch instead of opening a duplicate. The files are checked
    against the schema again right before sending."""
    import tempfile
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    download = download or hf_hub_download
    folder = Path(folder)
    prompts_dir = Path(prompts_dir or find_prompts_dir())
    index, heldout = load_prompt_index(prompts_dir)
    original = json.loads((folder / "session.json").read_text())
    is_library = original.get("consent_version") == LIBRARY_CONSENT
    library = load_library_pilot(prompts_dir) if is_library else None
    if is_library and drop:
        # Dropping a pair cannot make an unvalidated source eligible for the pilot.
        problems = validate_session(folder, index, heldout, library=library)
        if problems:
            raise ValueError(f"Session failed validation; not uploaded: {problems}")
    with tempfile.TemporaryDirectory() as tmp:
        source = without_takes(folder, drop, Path(tmp) / folder.parent.name / folder.name) if drop else folder
        problems = validate_session(source, index, heldout, library=library)
        if problems:
            raise ValueError(f"Session failed validation; not uploaded: {problems}")
        session = json.loads((source / "session.json").read_text())
        prefix = f"data/{session['contributor']}/{session['session_id']}"
        operations = [CommitOperationAdd(path_in_repo=f"{prefix}/{name}", path_or_fileobj=str(source / name))
                      for name in SESSION_FILES]
        api = api or HfApi(token=token)
        message = f"Add Subvox session {session['session_id'][:8]} ({session['takes']} takes)"
        description = (f"Contributed with Subvox {session['app_version']}. Schema v{session['schema_version']}, "
                       f"{session['airpods_model'] or 'unknown headphones'}, consent {session['consent_version']}.")
        if automatic and pr is None:
            # Recover the PR if the previous request reached the Hub but the app never got its receipt.
            message = f"Add Subvox session {session['session_id']} ({session['takes']} takes)"
            marker = f"Add Subvox session {session['session_id']} ("
            for discussion in api.get_repo_discussions(repo, repo_type="dataset", author=session["contributor"],
                                                       discussion_type="pull_request"):
                if discussion.title.startswith(marker):
                    counts = dict(takes=session["takes"])
                    if is_library:
                        try:
                            remote_takes = download(repo_id=repo, filename=f"{prefix}/takes.jsonl", repo_type="dataset",
                                                    revision=f"refs/pr/{discussion.num}", token=token, force_download=True)
                            counts = library_receipt_takes(remote_takes)
                        except (OSError, KeyError, TypeError, ValueError) as error:
                            raise ValueError("Could not read the takes from the existing library pilot pull request.") from error
                    return dict(session_id=session["session_id"], pr_url=discussion.url,
                                pr_num=discussion.num, commit=None, **counts)
        counts = library_receipt_takes(source / "takes.jsonl") if is_library else dict(takes=session["takes"])
        if is_library:
            try:
                remote = download(repo_id=repo, filename=f"prompts/{LIBRARY_CATALOG}",
                                  repo_type="dataset", token=token, force_download=True)
                matches = Path(remote).read_bytes() == (prompts_dir / LIBRARY_CATALOG).read_bytes()
            except (OSError, ValueError):
                matches = False
            if not matches:
                raise ValueError("Library pilot is awaiting setup on this dataset.")
        if pr is not None:
            info = api.create_commit(repo_id=repo, repo_type="dataset", operations=operations, commit_message=message,
                                     revision=f"refs/pr/{int(pr)}")
        else:
            info = api.create_commit(repo_id=repo, repo_type="dataset", operations=operations, commit_message=message,
                                     commit_description=description, create_pr=True)
        return dict(session_id=session["session_id"], pr_url=getattr(info, "pr_url", None),
                    pr_num=getattr(info, "pr_num", pr), commit=getattr(info, "oid", None), **counts)
