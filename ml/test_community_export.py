import json
import tempfile
import unittest
import uuid
from pathlib import Path
import pyarrow.parquet as pq
from subvox_ml import camera
from subvox_ml.community_export import export_sessions, validate_session, load_prompt_index, prompt_identity, MOTION_COLUMNS, CAMERA_COLUMNS
from test_camera import reading

COMMUNITY = "The cat sat on the mat."
CALIBRATION = "Go and ask her."
HELDOUT = "The dog ran far away."
PERSONAL = "Pack my blue bag."


def recording(condition, start):
    samples = [dict(t=start + i / 50, ax=0.01 * (i % 7), ay=0, az=0, rx=0, ry=0, rz=0, gx=0, gy=0, gz=-1, pitch=0, roll=0, yaw=0,
                    qw=1, qx=0, qy=0, qz=0, sensorLocation=2) for i in range(300)]
    return dict(id=str(uuid.uuid4()).upper(), condition=condition, cueStart=start + 3, cueEnd=start + 5.9, samples=samples)


def pair(text, session, start, role="Training", camera_on=True):
    readings = [reading(start + k / 30, opening=(k % 20) / 20) for k in range(600)] if camera_on else []  # both halves
    return dict(id=str(uuid.uuid4()).upper(), visitID=str(uuid.uuid4()).upper(), promptIDs=[prompt_identity(text)], text=text, role=role,
                createdAt=815709470.3, spoken=recording("Normal speech", start), silent=recording("Silent mouthing", start + 10),
                directory="/Users/someone/Documents/Subvox/Sessions/x", sessionStartUptime=start - 1, recordingSessionID=session,
                review="Needs review", cameraReadings=readings)


class CommunityExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root, self.prompts, self.out = root / "app", root / "prompts", root / "out"
        self.root.mkdir(); self.prompts.mkdir()
        (self.prompts / "prompts-v1.jsonl").write_text(json.dumps(dict(prompt_id=prompt_identity(COMMUNITY), passage_id="book-0001", text=COMMUNITY)) + "\n")
        (self.prompts / "calibration-v1.jsonl").write_text(json.dumps(dict(prompt_id=prompt_identity(CALIBRATION), text=CALIBRATION)) + "\n")
        (self.prompts / "heldout-v1.jsonl").write_text(json.dumps(dict(prompt_id=prompt_identity(HELDOUT), passage_id="book-0002", text=HELDOUT)) + "\n")
        self.session = str(uuid.uuid4()).upper()
        visit = str(uuid.uuid4()).upper()
        data = dict(visits=[dict(id=visit, motionDevice=dict(vendorID="0x004C", productID="0x2027", model="AirPods Pro 3", firmware="9A348"))],
                    prompts=[dict(text=COMMUNITY, kind="community"), dict(text=CALIBRATION, kind="calibration"),
                             dict(text=HELDOUT, kind="community"), dict(text=PERSONAL, kind="sentence")],
                    pairs=[dict(p, visitID=visit) for p in (pair(COMMUNITY, self.session, 100), pair(CALIBRATION, self.session, 130, camera_on=False),
                                                           pair(HELDOUT, self.session, 160), pair(PERSONAL, self.session, 190))])
        (self.root / "sentence-training.json").write_text(json.dumps(data))

    def tearDown(self):
        self.tmp.cleanup()

    def export(self):
        return export_sessions(self.root, self.out, "anon-7f3a", "consent-v1", self.prompts)

    def test_exports_only_community_and_calibration_takes_in_the_dataset_layout(self):
        [folder] = self.export()
        self.assertEqual(folder.parent.name, "anon-7f3a")
        takes = [json.loads(l) for l in (folder / "takes.jsonl").read_text().splitlines()]
        self.assertEqual(len(takes), 4, "Two prompts x spoken+mouthed; personal and held-out excluded")
        self.assertEqual({t["prompt_id"] for t in takes}, {prompt_identity(COMMUNITY), prompt_identity(CALIBRATION)})
        self.assertEqual({t["passage_id"] for t in takes}, {"book-0001", None})
        self.assertEqual({t["mode"] for t in takes}, {"spoken", "mouthed"})
        community = [t for t in takes if t["passage_id"] == "book-0001"]
        self.assertTrue(all(t["camera_eligible"] and t["eligible"] for t in community))
        self.assertTrue(all(not t["camera_eligible"] for t in takes if t["passage_id"] is None))
        motion = pq.read_table(folder / "motion.parquet")
        self.assertEqual(motion.column_names, MOTION_COLUMNS)
        self.assertEqual(motion.num_rows, 4 * 300)
        self.assertGreaterEqual(min(motion.column("t").to_pylist()), 0, "Times are relative to the session start")
        cam = pq.read_table(folder / "camera.parquet")
        self.assertEqual(cam.column_names, CAMERA_COLUMNS)
        self.assertEqual(len(set(cam.column("take_id").to_pylist())), 2, "Only takes with camera points have camera rows")
        self.assertEqual(validate_session(folder, *load_prompt_index(self.prompts)), [])

    def test_nothing_local_leaves_and_reexport_is_idempotent(self):
        [folder] = self.export()
        text = "".join(p.read_text() for p in folder.glob("*.json*"))
        for local in (self.session, "/Users/someone", "Documents", "front", "815709470"):
            self.assertNotIn(local, text)
        session = json.loads((folder / "session.json").read_text())
        self.assertEqual(session["month"], "2026-11")
        self.assertEqual(session["sensor_side"], "right")
        self.assertEqual((session["airpods_model"], session["airpods_product_id"], session["airpods_firmware"]), ("AirPods Pro 3", "0x2027", "9A348"))
        takes = [json.loads(l) for l in (folder / "takes.jsonl").read_text().splitlines()]
        self.assertTrue(all(abs(t["motion_hz"] - 50) < 1 for t in takes), "Native rate is kept per take")
        first = (folder / "takes.jsonl").read_text()
        [again] = self.export()
        self.assertEqual(again, folder)
        self.assertEqual((again / "takes.jsonl").read_text(), first)

    def test_validator_rejects_extra_fields_heldout_prompts_and_bad_contributors(self):
        [folder] = self.export()
        index, heldout = load_prompt_index(self.prompts)
        session = json.loads((folder / "session.json").read_text())
        (folder / "session.json").write_text(json.dumps(dict(session, device_name="Steve's MacBook")))
        self.assertTrue(any("keys differ" in p for p in validate_session(folder, index, heldout)))
        (folder / "session.json").write_text(json.dumps(session))
        takes = [json.loads(l) for l in (folder / "takes.jsonl").read_text().splitlines()]
        takes[0]["prompt_id"] = prompt_identity(HELDOUT)
        (folder / "takes.jsonl").write_text("".join(json.dumps(t) + "\n" for t in takes))
        self.assertIn("held-out prompt recorded", validate_session(folder, index, heldout))
        with self.assertRaisesRegex(ValueError, "Contributor ID"):
            export_sessions(self.root, self.out, "../escape", "consent-v1", self.prompts)


if __name__ == "__main__":
    unittest.main()
