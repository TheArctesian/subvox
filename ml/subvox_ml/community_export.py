"""Local recordings -> community dataset layout (docs/community-dataset-design.md, section 4).

    data/<contributor>/<session-id>/
      session.json     schema version, app version, device models (null until recorded), sensor side,
                       sample rates, consent version, month
      takes.jsonl      take id, prompt id, passage id, mode, cue start/end, eligibility flags, redo count
      motion.parquet   take id, t, acceleration, rotation, gravity, attitude, side
      camera.parquet   take id, t, eye-frame landmark channels and masks (50 Hz grid)

Only community and calibration prompts are exported. Times are seconds from the session start;
the only date is the month. Session and take IDs are one-way hashes of local IDs, so re-exporting
is idempotent and nothing local (paths, device IDs, wall-clock times) leaves the Mac.
`validate_session` is the same check the validator Space runs before merging a PR.
"""
import datetime
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from . import camera as camera_input
from .data import RATE, resample

SCHEMA_VERSION = 1
SWIFT_EPOCH = datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc)
MOTION_COLUMNS = ["take_id", "t", "ax", "ay", "az", "rx", "ry", "rz", "gx", "gy", "gz", "qw", "qx", "qy", "qz", "side"]
CAMERA_COLUMNS = ["take_id", "t"] + camera_input.RAW_NAMES
SESSION_KEYS = {"schema_version", "app_version", "contributor", "session_id", "airpods_model", "mac_model", "camera_model",
                "sensor_side", "sample_rates", "consent_version", "month", "prompts_version", "takes"}
TAKE_KEYS = {"take_id", "prompt_id", "passage_id", "mode", "cue_start", "cue_end", "eligible", "ineligible_reason",
             "camera_coverage", "camera_eligible", "redo_count"}
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


def export_sessions(root, output, contributor, consent_version, prompts_dir=None, app_version="0.1.0", prompts_version=1):
    """Writes one folder per recording session that has community takes; returns their paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not CONTRIBUTOR.match(contributor):
        raise ValueError("Contributor ID must be 2-64 letters, digits, '.', '_' or '-'")
    index, heldout = load_prompt_index(prompts_dir or find_prompts_dir())
    data = json.loads((Path(root) / "sentence-training.json").read_text())
    kinds = {p_id: p.get("kind", "sentence") for p in data.get("prompts", []) for p_id in [prompt_identity(p["text"])]}
    sessions = {}
    for pair in data.get("pairs", []):
        ids = pair.get("promptIDs") or []
        if len(ids) != 1 or not pair.get("recordingSessionID") or pair.get("role") == "Final test":
            continue
        if kinds.get(ids[0]) not in ("community", "calibration") or ids[0] not in index or ids[0] in heldout:
            continue
        sessions.setdefault(pair["recordingSessionID"], []).append(pair)
    written = []
    for local_session, pairs in sorted(sessions.items()):
        session_id = opaque("session", local_session)
        origin = min(p.get("sessionStartUptime") or min(r["samples"][0]["t"] for r in (p.get("spoken"), p.get("silent")) if r and r["samples"])
                     for p in pairs)
        takes, motion, cameras, all_samples, readings_all = [], [], [], [], []
        for pair in sorted(pairs, key=lambda p: p["createdAt"]):
            readings = pair.get("cameraReadings") or []
            readings_all += readings
            for mode, key in (("spoken", "spoken"), ("mouthed", "silent")):
                recording = pair.get(key)
                if not recording or not recording.get("samples"):
                    continue
                take_id = opaque("take", recording["id"])
                m, c, coverage, reason = take_rows(recording, origin, take_id, readings)
                motion.append(m); cameras.append(c); all_samples += recording["samples"]
                takes.append(dict(take_id=take_id, prompt_id=pair["promptIDs"][0], passage_id=index[pair["promptIDs"][0]], mode=mode,
                                  cue_start=recording["cueStart"] - origin, cue_end=recording["cueEnd"] - origin,
                                  eligible=reason is None, ineligible_reason=reason, camera_coverage=round(coverage, 4),
                                  camera_eligible=coverage >= .8, redo_count=None))
        if not takes:
            continue
        times = sorted(r["t"] for r in readings_all)
        camera_hz = float(1 / np.median(np.diff(times))) if len(times) > 2 and np.median(np.diff(times)) > 0 else None
        motion_t = sorted(s["t"] for s in all_samples)
        motion_hz = float(1 / np.median(np.diff(motion_t))) if len(motion_t) > 2 else None
        session = dict(schema_version=SCHEMA_VERSION, app_version=app_version, contributor=contributor, session_id=session_id,
                       airpods_model=None, mac_model=None, camera_model=None, sensor_side=side_name(all_samples),
                       sample_rates=dict(motion_hz=round(motion_hz, 2) if motion_hz else None, camera_hz=round(camera_hz, 2) if camera_hz else None),
                       consent_version=consent_version, month=month_of(min(p["createdAt"] for p in pairs)),
                       prompts_version=prompts_version, takes=len(takes))
        folder = Path(output) / "data" / contributor / session_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "session.json").write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")
        (folder / "takes.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in takes))
        merged_motion = {c: [v for m in motion for v in m[c]] for c in MOTION_COLUMNS}
        pq.write_table(pa.table(merged_motion), folder / "motion.parquet")
        merged_camera = {c: [v for m in cameras for v in m[c]] for c in CAMERA_COLUMNS}
        schema = pa.schema([("take_id", pa.string())] + [(c, pa.float64()) for c in CAMERA_COLUMNS[1:]])
        pq.write_table(pa.table(merged_camera, schema=schema), folder / "camera.parquet")
        problems = validate_session(folder, index, heldout)
        if problems:
            raise ValueError(f"Export failed its own schema check: {problems}")
        written.append(folder)
    return written


def prompt_identity(text):
    return hashlib.sha256(" ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())).encode()).hexdigest()


def validate_session(folder, index, heldout):
    """Schema and privacy checks for one contributed session; an empty list means valid."""
    import pyarrow.parquet as pq
    folder = Path(folder)
    problems = []
    required = ["session.json", "takes.jsonl", "motion.parquet", "camera.parquet"]
    for name in required:
        if not (folder / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems
    extra = sorted(p.name for p in folder.iterdir() if p.name not in required and p.name not in ("alignment.jsonl", "audio"))
    if extra:
        problems.append(f"unexpected files {extra}")
    if sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) > MAX_SESSION_BYTES:
        problems.append("session larger than 200 MB")
    session = json.loads((folder / "session.json").read_text())
    if set(session) != SESSION_KEYS:
        problems.append(f"session.json keys differ: extra {sorted(set(session) - SESSION_KEYS)}, missing {sorted(SESSION_KEYS - set(session))}")
    if session.get("schema_version") != SCHEMA_VERSION:
        problems.append("unsupported schema_version")
    if not re.fullmatch(r"\d{4}-\d{2}", str(session.get("month"))):
        problems.append("month must be YYYY-MM")
    if not CONTRIBUTOR.match(str(session.get("contributor"))) or folder.parent.name != session.get("contributor") or folder.name != session.get("session_id"):
        problems.append("contributor/session folder mismatch")
    if session.get("sensor_side") not in ("left", "right", "unknown"):
        problems.append("sensor_side must be left, right or unknown")
    for key, value in session.items():
        if isinstance(value, str) and ("/" in value or "\\" in value):
            problems.append(f"{key} looks like a path")
    takes = [json.loads(line) for line in (folder / "takes.jsonl").read_text().splitlines()]
    ids = set()
    for take in takes:
        if set(take) != TAKE_KEYS:
            problems.append(f"take keys differ: {sorted(set(take) ^ TAKE_KEYS)}")
            continue
        ids.add(take["take_id"])
        if take["mode"] not in ("spoken", "mouthed"):
            problems.append(f"bad mode {take['mode']}")
        if take["prompt_id"] in heldout:
            problems.append("held-out prompt recorded")
        elif take["prompt_id"] not in index or index[take["prompt_id"]] != take["passage_id"]:
            problems.append("unknown prompt or passage")
        if not take["cue_end"] > take["cue_start"]:
            problems.append("cue end before start")
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
    camera_table = pq.read_table(folder / "camera.parquet")
    if camera_table.column_names != CAMERA_COLUMNS:
        problems.append("camera.parquet columns differ")
    elif set(camera_table.column("take_id").to_pylist()) - ids:
        problems.append("camera rows for unknown takes")
    return problems
