import json
import tempfile
import unittest
import uuid
from pathlib import Path
import pyarrow.parquet as pq
from subvox_ml import camera
from subvox_ml.community_export import training_phones, export_sessions, validate_session, load_prompt_index, prompt_identity, MOTION_COLUMNS, CAMERA_COLUMNS
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


class Fixture(unittest.TestCase):
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


class CommunityExportTests(Fixture):

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

    def test_already_shared_sessions_are_skipped(self):
        [folder] = self.export()
        import shutil
        shutil.rmtree(self.out)
        self.assertEqual(export_sessions(self.root, self.out, "anon-7f3a", "consent-v1", self.prompts, skip={folder.name}), [])
        self.assertFalse((self.out / "data" / "anon-7f3a" / folder.name).exists())

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


class FakeHub:
    def __init__(self):
        self.calls = []

    def create_commit(self, **kwargs):
        self.calls.append(dict(kwargs, files={op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in kwargs["operations"]}))
        class Info:
            pr_url, pr_num, oid = "https://huggingface.co/datasets/x/y/discussions/7", 7, "abc"
        return Info()


class UploadTests(Fixture):
    def test_opens_one_pr_per_session_with_only_validated_files_under_the_contributor(self):
        from subvox_ml.community_export import upload_session
        [folder] = self.export()
        (folder / "alignment.jsonl").write_text("unvalidated alignment payload")
        (folder / "audio").write_bytes(b"unvalidated audio payload")
        hub = FakeHub()
        result = upload_session(folder, "stephenokita/subvox-community-test", "token", prompts_dir=self.prompts, api=hub)
        [call] = hub.calls
        self.assertTrue(call["create_pr"])
        self.assertEqual(call["repo_type"], "dataset")
        self.assertEqual(sorted(call["files"]), sorted(f"data/anon-7f3a/{folder.name}/{n}" for n in ("camera.parquet", "motion.parquet", "session.json", "takes.jsonl")))
        self.assertEqual((result["pr_num"], result["takes"]), (7, 4))

    def test_resume_adds_to_the_existing_pr_instead_of_opening_another(self):
        from subvox_ml.community_export import upload_session
        [folder] = self.export()
        hub = FakeHub()
        upload_session(folder, "stephenokita/subvox-community-test", "token", pr=7, prompts_dir=self.prompts, api=hub)
        self.assertEqual(hub.calls[0]["revision"], "refs/pr/7")
        self.assertNotIn("create_pr", hub.calls[0])

    def test_dropped_takes_vanish_from_every_file_and_invalid_sessions_never_upload(self):
        import io
        from subvox_ml.community_export import upload_session
        [folder] = self.export()
        takes = [json.loads(l) for l in (folder / "takes.jsonl").read_text().splitlines()]
        dropped = takes[0]["take_id"]
        hub = FakeHub()
        result = upload_session(folder, "r/d", "token", drop=[dropped], prompts_dir=self.prompts, api=hub)
        self.assertEqual(result["takes"], 3)
        files = hub.calls[0]["files"]
        prefix = f"data/anon-7f3a/{folder.name}/"
        self.assertNotIn(dropped, files[prefix + "takes.jsonl"].decode())
        self.assertNotIn(dropped, pq.read_table(io.BytesIO(files[prefix + "motion.parquet"])).column("take_id").to_pylist())
        self.assertEqual(json.loads(files[prefix + "session.json"])["takes"], 3)
        self.assertEqual(len(takes), 4, "The local export itself is unchanged")
        session = json.loads((folder / "session.json").read_text())
        (folder / "session.json").write_text(json.dumps(dict(session, extra="x")))
        with self.assertRaisesRegex(ValueError, "failed validation"):
            upload_session(folder, "r/d", "token", prompts_dir=self.prompts, api=FakeHub())


if __name__ == "__main__":
    unittest.main()


class LanguageTagTests(Fixture):
    def takes(self, folder):
        return [json.loads(l) for l in (folder / "takes.jsonl").read_text().splitlines()]

    def test_takes_carry_language_speaker_and_expected_phones(self):
        [folder] = self.export()
        self.assertEqual(json.loads((folder / "session.json").read_text())["schema_version"], 2)
        for take in self.takes(folder):
            self.assertEqual((take["language"], take["speaker_status"], take["speaking_style"]), ("en", "unknown", "natural"))
        community = next(t for t in self.takes(folder) if t["passage_id"] == "book-0001")
        if community["expected_phones"] is not None:  # needs espeak-ng
            self.assertEqual(community["expected_phones"][:3], ["ð", "ə", "k"])

    def test_exports_verified_timing_and_complete_phone_labels(self):
        path = self.root / "sentence-training.json"
        data = json.loads(path.read_text())
        labels = ["ʒ", "ə"]
        revision = "a" * 40
        data["pairs"][0].update(language="fr", wordTimingSource="forcedAlignment",
                                wordTimingModelRevision=revision, alignment={"words": [{}]},
                                phoneCheck={"expected_phones": labels},
                                spokenPhones={"phones": [{"ipa": phone} for phone in labels]})
        data["pairs"][1].update(wordTimingSource="forcedAlignment", error="Reading did not match")
        path.write_text(json.dumps(data))

        [folder] = self.export()
        takes = self.takes(folder)
        spoken = next(t for t in takes if t["mode"] == "spoken" and t["passage_id"] == "book-0001")
        self.assertEqual(spoken["produced_phones"], labels)
        self.assertEqual(spoken["word_timing_source"], "forcedAlignment")
        self.assertEqual(spoken["word_timing_model_revision"], revision)
        self.assertTrue(all(not t["eligible"] for t in takes if t["passage_id"] is None))
        self.assertEqual(validate_session(folder, *load_prompt_index(self.prompts)), [])

    def test_validator_rejects_bad_tags_and_off_inventory_phones(self):
        [folder] = self.export()
        takes = self.takes(folder)
        takes[0].update(language="de", produced_phones=["ʘ"], speaker_status="fluent")
        (folder / "takes.jsonl").write_text("".join(json.dumps(t) + "\n" for t in takes))
        problems = validate_session(folder, *load_prompt_index(self.prompts))
        self.assertTrue(any("language tag" in p for p in problems))
        self.assertTrue(any("produced_phones" in p for p in problems))
        self.assertTrue(any("speaker_status" in p for p in problems))

    def test_schema_v1_sessions_still_validate(self):
        [folder] = self.export()
        session = json.loads((folder / "session.json").read_text())
        (folder / "session.json").write_text(json.dumps(dict(session, schema_version=1)))
        v2_only = ("language", "speaker_status", "first_language", "speaking_style", "expected_phones", "produced_phones")
        takes = [{k: v for k, v in t.items() if k not in v2_only} for t in self.takes(folder)]
        (folder / "takes.jsonl").write_text("".join(json.dumps(t) + "\n" for t in takes))
        self.assertEqual(validate_session(folder, *load_prompt_index(self.prompts)), [])

    def test_learners_train_only_on_produced_phones(self):
        base = dict(expected_phones=["s", "ɥ", "i"], speaking_style="natural")
        self.assertIsNone(training_phones(dict(base, speaker_status="learning", produced_phones=None)))
        self.assertEqual(training_phones(dict(base, speaker_status="learning", produced_phones=["s", "y", "i"])), ["s", "y", "i"])
        self.assertEqual(training_phones(dict(base, speaker_status="native", produced_phones=None)), ["s", "ɥ", "i"])
