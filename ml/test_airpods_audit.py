import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

from subvox_ml.airpods_audit import audit, audit_motion


def recording(condition, start, *, hz=50, side=1):
    samples = [
        dict(t=start + i / hz, ax=0, ay=0, az=0, rx=0, ry=0, rz=0,
             gx=0, gy=0, gz=-1, sensorLocation=side)
        for i in range(3 * hz)
    ]
    return dict(id=str(uuid4()), condition=condition, cueStart=start + .5, cueEnd=start + 2.5, samples=samples)


def make_visit(store, rows, *, speaker, phase, started, product="0x2027"):
    visit, session = str(uuid4()), str(uuid4())
    store["visits"].append(dict(id=visit, recordingSessionID=session, startedAt=started, endedAt=started + 100,
                                motionDevice=dict(vendorID="0x004C", productID=product)))
    for i, (prompt, text, changes) in enumerate(rows):
        spoken = recording("Normal speech", 10 * i + 5)
        silent = recording("Silent mouthing", 10 * i + 1)
        store["pairs"].append(dict(id=str(uuid4()), visitID=visit, recordingSessionID=session, promptIDs=[prompt],
                                   text=text, createdAt=started + i, spoken=spoken, silent=silent, role="Training",
                                   transcriptMatchRate=.95) | changes)
    return dict(store="capture.json", visit_id=visit, speaker=speaker, phase=phase)


def cohort(tmp_path):
    store = dict(visits=[], pairs=[], roles={"reserved": "Final test"})
    sessions = [
        make_visit(store, [("p1", "First practice line", {}), ("reserved", "Reserved line", {})],
                   speaker="A", phase="first", started=10),
        make_visit(store, [("p1", "First practice line", {})], speaker="A", phase="repeat", started=200),
        make_visit(store, [("p1", "First practice line", {"silent": recording("Silent mouthing", 1, hz=10)})],
                   speaker="A", phase="reseated", started=400),
        make_visit(store, [("p2", "Different line", {}), ("p1", "First practice line", {})],
                   speaker="B", phase="first", started=10),
        make_visit(store, [("p3", "Third practice line", {})], speaker="C", phase="first", started=10),
    ]
    manifest = dict(schema_version=1, sessions=sessions, prompt_families={"p1": "book-A", "p2": "book-B", "p3": "book-C"},
                    splits={
                        "train": {"speakers": ["A"], "families": ["book-A"]},
                        "development": {"speakers": ["B"], "families": ["book-B"]},
                        "test": {"speakers": ["C"], "families": ["book-C"]},
                    })
    (tmp_path / "capture.json").write_text(json.dumps(store))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, store


def test_motion_uses_training_thresholds_for_gaps_source_switches_and_low_energy():
    data = recording("Silent mouthing", 1)
    valid = audit_motion(data, "Silent mouthing")
    assert valid["issue"] is None
    assert valid["hz"] == 50
    assert any("low motion energy" in flag for flag in valid["flags"])
    switched = dict(data, samples=[sample | {"sensorLocation": 2 if i > 50 else 1}
                                   for i, sample in enumerate(data["samples"])])
    assert "source switched" in audit_motion(switched, "Silent mouthing")["issue"]
    gap = dict(data, samples=data["samples"][:50] + data["samples"][60:])
    assert "motion gap" in audit_motion(gap, "Silent mouthing")["issue"]
    assert "motion rate below" in audit_motion(recording("Silent mouthing", 1, hz=10), "Silent mouthing")["issue"]


def test_cli_reports_first_pairs_matched_repeats_and_disjoint_splits(tmp_path):
    path, _, _ = cohort(tmp_path)
    output = tmp_path / "report.json"
    subprocess.run([sys.executable, "-m", "subvox_ml.airpods_audit", str(path), str(output), "--prompts-dir",
                    str(tmp_path)], check=True)
    report = json.loads(output.read_text())
    assert report == audit(path, tmp_path)
    speaker = report["speakers"]["A"]
    assert speaker["sessions"]["first"]["usable_silent_seconds"] == 2
    assert speaker["first_pairs"][1]["issues"] == ["reserved final test"]
    assert speaker["matched_comparisons"]["repeat"]["followup_matched"]["usable_pairs"] == 1
    assert speaker["matched_comparisons"]["reseated"]["matched_prompts"] == 1
    assert speaker["matched_comparisons"]["reseated"]["followup_matched"]["usable_pairs"] == 0
    assert report["splits"]["ready"]
    assert {name: len(ids) for name, ids in report["splits"]["pair_ids"].items()} == {"train": 2, "development": 1, "test": 1}
    assert report["splits"]["quarantine"]["speaker and family assigned to different splits"] == 1
    assert "First practice line" not in output.read_text()


def test_text_collision_cannot_leak_across_differently_named_families(tmp_path):
    path, _, store = cohort(tmp_path)
    store["pairs"][-1]["text"] = "FIRST practice line!"
    (tmp_path / "capture.json").write_text(json.dumps(store))
    splits = audit(path, tmp_path)["splits"]
    assert not splits["ready"]
    assert splits["pair_ids"]["test"] == []
    assert splits["pair_ids"]["train"] == []
    assert splits["quarantine"]["shared prompt across splits"] == 3


def test_audit_rejects_reused_sessions_and_overlapping_speakers(tmp_path):
    path, config, _ = cohort(tmp_path)
    config["sessions"].append(config["sessions"][0] | {"speaker": "D"})
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="physical recording session"):
        audit(path, tmp_path)
    config["sessions"].pop()
    config["splits"]["test"]["speakers"].append("A")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="speakers must be disjoint"):
        audit(path, tmp_path)
