"""Audit paired AirPods captures locally and freeze speaker/prompt-family splits."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata
from uuid import UUID

from .community_export import find_prompts_dir
from .data import MIN_TRANSCRIPT_MATCH, atomic_json, digest, resample

PHASES = ("first", "repeat", "reseated")
PARTITIONS = ("train", "development", "test")
CONDITIONS = {"spoken": "Normal speech", "silent": "Silent mouthing"}


def corpus_families(directory):
    families = {}
    for path in sorted(Path(directory).glob("prompts-v*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            prompt = json.loads(line)
            families[prompt["prompt_id"]] = prompt["source_id"]
    return families


def normalize_id(value):
    return str(UUID(value))


def audit_motion(recording, expected):
    if not recording:
        return {"issue": "missing take"}
    if recording.get("condition") != expected:
        return {"issue": "condition mismatch"}
    try:
        _, quality = resample(recording)
        return {
            "issue": None, "hz": round(quality["rate"], 2), "side": quality["source"],
            "seconds": round(recording["cueEnd"] - recording["cueStart"], 4),
            "flags": quality["flags"],
        }
    except ValueError as error:
        return {"issue": str(error)}
    except (KeyError, TypeError, IndexError):
        return {"issue": "malformed motion data"}


def audit_pair(pair, visit, reserved, families, speaker):
    issues = []
    prompt_ids = pair.get("promptIDs", [])
    motion = {kind: audit_motion(pair.get(kind), expected) for kind, expected in CONDITIONS.items()}
    if pair.get("role") == "Final test" or reserved.intersection(prompt_ids):
        issues.append("reserved final test")
    if pair.get("review") == "Excluded":
        issues.append("excluded locally")
    pair_session = pair.get("recordingSessionID")
    if not pair_session or normalize_id(visit["recordingSessionID"]) != normalize_id(pair_session):
        issues.append("missing or inconsistent recording session")
    if pair.get("needsBackgroundCheck"):
        issues.append("spoken check still processing")
    match = pair.get("transcriptMatchRate")
    coverage = pair.get("timedWordCoverage")
    reliable = isinstance(match, (int, float)) and MIN_TRANSCRIPT_MATCH <= match <= 1
    if pair.get("wordTimingSource") == "forcedAlignment" and pair.get("practiceMode") != "freeText":
        reliable = (not pair.get("error") and isinstance(coverage, (int, float)) and .8 <= coverage <= 1
                    and bool((pair.get("alignment") or {}).get("words")))
    if not reliable and pair.get("review") != "Accepted":
        issues.append("spoken text needs confirmation")
    if all(take["issue"] is None for take in motion.values()):
        spoken, silent = pair["spoken"], pair["silent"]
        if spoken.get("id") == silent.get("id") or not (spoken.get("id") and silent.get("id")):
            issues.append("take identities missing or reused")
        if max(spoken["cueStart"], silent["cueStart"]) < min(spoken["cueEnd"], silent["cueEnd"]):
            issues.append("spoken and silent cues overlap")
    for kind, take in motion.items():
        if take["issue"] is not None:
            issues.append(f'{kind}: {take["issue"]}')

    groups = {families.get(key) for key in prompt_ids}
    if pair.get("literaturePieceID") and pair.get("practiceMode") in ("speech", "poetry"):
        family = f'literature:{pair["literaturePieceID"]}'
    else:
        family = next(iter(groups)) if len(groups) == 1 and None not in groups else None
    folded = unicodedata.normalize("NFKC", pair.get("text") or "").casefold()
    text_key = digest("".join(character for character in folded if character.isalnum()))
    return {
        "id": digest(["airpods-audit-v1", speaker, normalize_id(pair["id"])]),
        "family": family, "prompt_keys": {*prompt_ids, text_key}, "motion": motion, "issues": issues,
        "text_key": text_key, "usable_silent_seconds": motion["silent"].get("seconds", 0) if not issues else 0,
    }


def report_session(rows):
    counts = Counter(issue for row in rows for issue in row["issues"])
    rates = [take["hz"] for row in rows for take in row["motion"].values() if take["issue"] is None]
    sides = sorted({take["side"] for row in rows for take in row["motion"].values() if take["issue"] is None})
    return {
        "pairs": len(rows), "usable_pairs": sum(not row["issues"] for row in rows),
        "usable_silent_seconds": round(sum(row["usable_silent_seconds"] for row in rows), 4),
        "median_motion_hz": round(statistics.median(rates), 2) if rates else None,
        "sensor_sides": sides, "issues": dict(sorted(counts.items())),
    }


def split_rows(rows, specification):
    assignments = {"speakers": {}, "families": {}}
    for partition, spec in specification.items():
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown split: {partition}")
        for field, groups in assignments.items():
            for group in spec.get(field, []):
                if group in groups:
                    raise ValueError(f"Split {field} must be disjoint")
                groups[group] = partition

    candidates = {partition: [] for partition in PARTITIONS}
    quarantine = Counter()
    prompt_partitions = {}
    for speaker, row in rows:
        if row["issues"]:
            quarantine["failed paired quality check"] += 1
            continue
        by_speaker = assignments["speakers"].get(speaker)
        by_family = assignments["families"].get(row["family"])
        if not by_speaker or not by_family:
            quarantine["speaker or prompt family unassigned"] += 1
        elif by_speaker != by_family:
            quarantine["speaker and family assigned to different splits"] += 1
        else:
            candidates[by_speaker].append(row)
            for key in row["prompt_keys"]:
                prompt_partitions.setdefault(key, set()).add(by_speaker)

    collisions = {key for key, partitions in prompt_partitions.items() if len(partitions) > 1}
    splits = {}
    for name, candidates_in_split in candidates.items():
        splits[name] = []
        for row in candidates_in_split:
            if collisions.intersection(row["prompt_keys"]):
                quarantine["shared prompt across splits"] += 1
            else:
                splits[name].append(row["id"])
        splits[name].sort()
    return {"ready": all(splits.values()), "pair_ids": splits, "quarantine": dict(sorted(quarantine.items()))}


def audit(manifest_path, prompts_dir, early_count=3):
    manifest_path = Path(manifest_path)
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not config.get("sessions"):
        raise ValueError("Provide a version 1 manifest with at least one session")
    if early_count < 1:
        raise ValueError("Early sample count must be positive")
    families = corpus_families(prompts_dir)
    for key, family in config.get("prompt_families", {}).items():
        if not isinstance(family, str) or not family or (key in families and families[key] != family):
            raise ValueError("Custom prompt families must be named and cannot override corpus families")
        families[key] = family
    inputs, people, all_rows = {}, {}, []
    seen_sessions, seen_pairs, seen_takes = set(), set(), set()
    for entry in config["sessions"]:
        speaker, phase = entry["speaker"], entry["phase"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", speaker) or phase not in PHASES:
            raise ValueError("Use an anonymous speaker code and a phase of first, repeat or reseated")
        person = people.setdefault(speaker, {})
        if phase in person:
            raise ValueError("Each speaker may have one visit per phase")
        path = (manifest_path.parent / entry["store"]).resolve()
        if path not in inputs:
            data = json.loads(path.read_text(encoding="utf-8"))
            visits = {normalize_id(visit["id"]): visit for visit in data["visits"]}
            pairs = {}
            for pair in data["pairs"]:
                pairs.setdefault(normalize_id(pair["visitID"]), []).append(pair)
            reserved = {key for key, value in data.get("roles", {}).items() if value == "Final test"}
            inputs[path] = visits, pairs, reserved
        visits, pairs, reserved = inputs[path]
        visit_id = normalize_id(entry["visit_id"])
        visit = visits.get(visit_id)
        if (not visit or visit.get("endedAt") is None or not visit.get("recordingSessionID")
                or not math.isfinite(visit["startedAt"]) or visit["endedAt"] < visit["startedAt"]):
            raise ValueError("Each visit must exist, have ended, and have a physical session ID")
        session_id = normalize_id(visit["recordingSessionID"])
        if session_id in seen_sessions:
            raise ValueError("A physical recording session may only appear once")
        seen_sessions.add(session_id)
        selected = sorted(pairs.get(visit_id, []), key=lambda pair: (pair["createdAt"], pair["id"]))
        rows = []
        for pair in selected:
            pair_id = normalize_id(pair["id"])
            if pair_id in seen_pairs:
                raise ValueError("A recorded pair may only appear once")
            seen_pairs.add(pair_id)
            take_ids = {normalize_id(take["id"]) for kind in CONDITIONS if (take := pair.get(kind)) and take.get("id")}
            if seen_takes.intersection(take_ids):
                raise ValueError("A recorded take may only appear in one pair")
            seen_takes.update(take_ids)
            rows.append(audit_pair(pair, visit, reserved, families, speaker))
        device = visit.get("motionDevice") or {}
        hardware = (device.get("vendorID"), device.get("productID"))
        person[phase] = (rows, hardware if all(hardware) else None, visit["startedAt"])
        all_rows.extend((speaker, row) for row in rows)

    reports = {}
    for speaker, phases in sorted(people.items()):
        if "first" not in phases:
            raise ValueError("Repeat and reseated visits require a first visit for the same speaker")
        starts = [phases[phase][2] for phase in PHASES if phase in phases]
        if any(earlier >= later for earlier, later in zip(starts, starts[1:])):
            raise ValueError("Visits must occur in order: first, repeat, reseated")
        first, first_device, _ = phases["first"]
        baseline_keys = {row["text_key"] for row in first}
        comparisons = {}
        for phase in ("repeat", "reseated"):
            if phase not in phases:
                continue
            rows, hardware, _ = phases[phase]
            common = baseline_keys & {row["text_key"] for row in rows}
            comparisons[phase] = {
                "matched_prompts": len(common),
                "same_headphone_model": first_device == hardware if first_device and hardware else None,
                "first_matched": report_session([row for row in first if row["text_key"] in common]),
                "followup_matched": report_session([row for row in rows if row["text_key"] in common]),
            }
        reports[speaker] = {
            "sessions": {phase: report_session(rows) for phase, (rows, _, _) in sorted(phases.items())},
            "first_pairs": [{"id": row["id"], "issues": row["issues"], "motion": row["motion"]} for row in first[:early_count]],
            "matched_comparisons": comparisons,
        }
    return {"schema_version": 1, "speakers": reports, "splits": split_rows(all_rows, config.get("splits", {}))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompts-dir", type=Path, default=find_prompts_dir())
    parser.add_argument("--early-count", type=int, default=3)
    args = parser.parse_args()
    try:
        report = audit(args.manifest, args.prompts_dir, args.early_count)
        source_paths = [(args.manifest.parent / entry["store"]).resolve()
                        for entry in json.loads(args.manifest.read_text())["sessions"]]
        if args.output.resolve() in (args.manifest.resolve(), *source_paths):
            raise ValueError("Choose a new report path outside the input files")
        atomic_json(args.output, report)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
