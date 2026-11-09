import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from subvox_ml import camera
from subvox_ml.data import resample

W, H = 1280, 720


def reading(t, opening=0.0, shift=(0, 0), roll=0.0, scale=1.0, face=True, outer_points=14, camera_id="front", confidence=.9):
    c, s = math.cos(roll), math.sin(roll)

    def place(x, y):
        x, y = (x - 640) * scale, (y - 400) * scale
        return dict(x=(640 + c * x - s * y + shift[0]) / W, y=(400 + s * x + c * y + shift[1]) / H)

    def ellipse(cx, cy, rx, ry, n):
        return [place(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]

    mouth = dict(outerLips=ellipse(640, 300, 60, 20 + 30 * opening, outer_points), innerLips=ellipse(640, 300, 40, 5 + 25 * opening, 6),
                 leftEye=place(540, 400), rightEye=place(740, 400), opening=.1 + opening, width=.6, innerArea=.02 + opening * .05,
                 outerAspect=.3 + opening, centreX=0.0, centreY=-.5 - .1 * opening, confidence=confidence)
    value = dict(t=t, cameraID=camera_id, estimator="apple-vision-face-landmarks-v1", imageWidth=W, imageHeight=H, mouth=mouth)
    if face:
        value["face"] = dict(faceContour=[place(520 + 15 * k, 260 - 40 * opening - 30 * math.sin(math.pi * k / 16)) for k in range(17)],
                             noseCrest=[place(640, 380 - 15 * k) for k in range(4)])
    return value


def recording():
    samples = [dict(t=100 + i / 50, ax=math.sin(i / 7) * .01, ay=0, az=0, rx=0, ry=0, rz=0, gx=0, gy=0, gz=-1, sensorLocation=1) for i in range(300)]
    return dict(cueStart=103, cueEnd=105.9, samples=samples, condition="Silent mouthing")


class MotionRateTests(unittest.TestCase):
    def at(self, hz):
        samples = [dict(t=100 + i / hz, ax=math.sin(i / 7) * .01, ay=0, az=0, rx=0, ry=0, rz=0, gx=0, gy=0, gz=-1, sensorLocation=1)
                   for i in range(int(6 * hz))]
        return dict(cueStart=103, cueEnd=105.5, samples=samples, condition="Silent mouthing")

    def test_slower_airpods_are_interpolated_onto_the_50_hz_grid_and_flagged(self):
        raw, quality = resample(self.at(25))
        self.assertAlmostEqual(quality["rate"], 25, delta=.5)
        self.assertEqual(len(raw), int(2.5 * 50) + 1)
        self.assertTrue(any("native motion rate 25 Hz" in f for f in quality["flags"]))
        self.assertFalse(any("native motion rate" in f for f in resample(self.at(50))[1]["flags"]))

    def test_rates_below_twenty_hertz_are_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "below 20 Hz"):
            resample(self.at(15))


class CameraInputTests(unittest.TestCase):
    def test_eye_frame_ignores_head_position_roll_and_distance(self):
        base = camera.frame_values(reading(0, opening=.4))
        moved = camera.frame_values(reading(0, opening=.4, shift=(80, -30), roll=.2, scale=1.3))
        np.testing.assert_allclose(base[0], moved[0], atol=1e-6)
        np.testing.assert_allclose(base[1], moved[1], atol=1e-6)
        opened = camera.frame_values(reading(0, opening=.9))[1]
        self.assertGreater(np.abs(opened - base[1]).max(), .05, "Jaw drop moves the face contour relative to the eyes")

    def test_contours_map_to_fixed_channels_regardless_of_point_count(self):
        a = camera.frame_values(reading(0, outer_points=14))[0]
        b = camera.frame_values(reading(0, outer_points=40))[0]
        self.assertEqual(len(a), len(camera.MOUTH_NAMES))
        np.testing.assert_allclose(a, b, atol=.02)

    def test_grid_interpolates_masks_gaps_and_keeps_one_camera(self):
        readings = [reading(100 + k / 30, opening=k / 60) for k in range(61) if not 101.0 < 100 + k / 30 < 101.3]
        readings[5] = dict(readings[5], face=None)
        readings.append(reading(100.5, opening=5, camera_id="other"))
        readings.append(reading(100.6, opening=5, confidence=.2))
        values = camera.grid(readings, 100.2, 80)
        mask = values[:, camera.MOUTH_MASK]
        times = 100.2 + np.arange(80) / 50
        self.assertTrue((mask[times < 100.95] == 1).all())
        self.assertTrue((mask[(times > 101.05) & (times < 101.25)] == 0).all(), "No interpolation across a 300 ms camera gap")
        valid = values[mask == 1, camera.SCALARS.index("opening")]
        self.assertTrue((np.diff(valid[:30]) >= -1e-6).all(), "Other camera and low-confidence frames are ignored")
        self.assertLess(valid.max(), 1.2)
        self.assertEqual(camera.coverage(np.zeros((0, camera.RAW_WIDTH))), 0)

    def test_features_zero_invalid_frames_and_skip_derivatives_across_them(self):
        values = camera.grid([reading(100 + k / 30, opening=k / 30) for k in range(31)], 100, 60)
        values[20:25, camera.MOUTH_MASK] = 0
        values[:, camera.FACE_MASK] = 0
        f = camera.features(values)
        self.assertEqual(f.shape[1], camera.FEATURE_WIDTH)
        self.assertTrue((f[20:25, :len(camera.MOUTH_NAMES)] == 0).all())
        self.assertTrue((f[:, camera.FACE] == 0).all())
        d = camera.VALUE_WIDTH + camera.SCALARS.index("opening")
        self.assertTrue((f[20:26, d] == 0).all())
        self.assertGreater(f[10, d], 0)

    def test_prepare_puts_camera_on_the_motion_grid_for_clips_and_rest_windows(self):
        from subvox_ml.data import prepare, atomic_json, load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            texts = ("pa", "ba", "pa ba", "ba pa")
            for v, text in enumerate(texts):
                takes = []
                for c, condition in enumerate(("Normal speech", "Silent mouthing")):
                    i = 2 * v + c
                    cam = [reading(100 + k / 30, opening=(k % 20) / 20) for k in range(180)] if c else []
                    takes.append(dict(id=f"take{i}", text=text, sentenceID=text, repetition=0,
                                      recording=dict(recording(), condition=condition, id=f"take{i}"), camera=cam))
                atomic_json(root / f"SequenceLearning/Collections/v{v}/session.json",
                            dict(id=f"v{v}", complete=True, role="training" if v < 2 else "development", session=dict(id=f"s{v}"), takes=takes))
            lang = dict(inventory=[dict(ipa=p) for p in ("p", "b", "a")], entries={"pa": [dict(ipa=["p", "a"], speakerConfirmed=True)], "ba": [dict(ipa=["b", "a"], speakerConfirmed=True)]},
                        diagnosticSentences=[])
            atomic_json(root / "language.json", lang)
            manifest = prepare(root, root / "language.json", root / "dataset")
            examples, _, _ = load_dataset(root / "dataset")
            with_camera = [e for e in examples if "camera" in e]
            self.assertEqual(manifest["camera"]["rows_with_camera"], len(with_camera))
            self.assertTrue(any(e["id"].endswith("/rest") for e in with_camera), "Rest windows get their own camera slice")
            for e in with_camera:
                self.assertEqual(len(e["camera"]), len(e["raw"]))
                self.assertGreater(camera.coverage(e["camera"]), .95)
            self.assertTrue(all("camera_readings" not in e for e in examples))

    def test_evaluation_scores_only_the_camera_experiment_cohort(self):
        from subvox_ml.cli import EXPERIMENTS, FUSION_EXPERIMENTS, evaluate_model

        condition = "Silent mouthing"
        camera_grid = np.zeros((5, camera.RAW_WIDTH))
        camera_grid[:4, camera.MOUTH_MASK] = 1
        partial_camera_grid = camera_grid.copy()
        partial_camera_grid[3, camera.MOUTH_MASK] = 0
        training = dict(id="training", condition=condition, fold_roles=["training"],
                        activity=1, raw=[[0]], targets=[["a"]], camera=camera_grid.tolist())
        covered = dict(training, id="covered", session="development", groups=["held-out"],
                       text="a", fold_roles=["validation"], camera=camera_grid.tolist())
        low_coverage = dict(covered, id="low-coverage", camera=partial_camera_grid.tolist())
        no_camera = dict(covered, id="no-camera", camera=[])
        excluded = dict(id="excluded", reason="Invalid motion recording", text=covered["text"],
                        session=covered["session"], groups=covered["groups"], condition=condition,
                        fold_roles=covered["fold_roles"])
        examples = [training, covered, low_coverage, no_camera]
        language = dict(entries={"a": [dict(ipa=["a"], speakerConfirmed=True)]})
        manifest = dict(training_groups=["training"], training_sessions=["training"], training_ids=["training"])
        prediction = dict(phonemes=["a"], text="a", text_with_lm="a", accepted=True)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("subvox_ml.cli.load_dataset", return_value=(examples, language, {"excluded": [excluded]})),
            patch("subvox_ml.cli.prediction", return_value=prediction),
            patch("subvox_ml.cli.ranking_fields", return_value={}),
        ):
            for experiment in (*FUSION_EXPERIMENTS, "plain"):
                with self.subTest(experiment=experiment):
                    checkpoint = dict(tokens=["a"], coverage=["a"], lm_training_text=[["a"]],
                                      config=dict(EXPERIMENTS[experiment], condition=condition, name=experiment))
                    output = Path(tmp) / f"{experiment}.json"
                    with patch("subvox_ml.network.load_model", return_value=(None, checkpoint, manifest, language)):
                        report = evaluate_model(Path(tmp) / "model", Path(tmp) / "dataset", 0, output=output)
                    expected = ["covered", "low-coverage", "no-camera", "excluded"] if experiment == "plain" else ["covered"]
                    self.assertEqual([r["id"] for r in report["rows"]], expected)
                    self.assertEqual(report["dataset_sizes"]["development"], len(expected))
                    self.assertEqual([r["id"] for r in report["quality_exclusions"]], ["excluded"])

    def test_fused_and_camera_models_fit_load_and_run_without_a_face(self):
        from subvox_ml.network import fit, load_model, emission
        from subvox_ml.cli import camera_cohort
        raw, q = resample(recording())
        lang = dict(inventory=[dict(ipa=p) for p in ("p", "b", "a")], entries={"pa": [dict(ipa=["p", "a"])]})

        def row(i, split, session, with_camera=True):
            item = dict(id=f"{i}", text="pa", targets=[["p", "a"]], condition="Silent mouthing", activity=1, raw=raw.tolist(), quality=q, session=session, groups=[split])
            if with_camera:
                item["camera"] = camera.grid([reading(q["start"] + k / 30, opening=.5 + .5 * math.sin(k / 4 + i)) for k in range(int(len(raw) / 50 * 30) + 2)],
                                             q["start"], len(raw)).tolist()
            return item

        training = [row(i, "t", "A") for i in range(4)] + [row(7, "t", "A", with_camera=False)]
        validation = [row(9, "v", "B")]
        self.assertEqual(len(camera_cohort(training)), 4)
        with tempfile.TemporaryDirectory() as tmp:
            for modalities in ("fused", "camera"):
                config = dict(name=modalities, condition="Silent mouthing", epochs=2, seed=1, transitions="none", augmentation=True,
                              activity_aux=False, teacher_aux=False, modalities=modalities)
                manifest = fit(training, validation, lang, Path(tmp) / modalities, config)
                self.assertEqual(manifest["modalities"], modalities)
                self.assertEqual(manifest["camera_feature_names"], camera.FEATURE_NAMES)
                model, checkpoint, _, _ = load_model(Path(tmp) / modalities)
                expected = camera.FEATURE_WIDTH + (len(manifest["feature_names"]) if modalities == "fused" else 0)
                self.assertEqual(model.convolution[0].in_channels, expected)
                lp, *_ = emission(model, checkpoint, validation[0])
                self.assertEqual(len(lp), len(raw))
                faceless = {k: v for k, v in validation[0].items() if k != "camera"}
                self.assertEqual(len(emission(model, checkpoint, faceless)[0]), len(raw), "Missing camera reads as invalid, not a crash")
            with self.assertRaisesRegex(ValueError, "camera landmarks"):
                fit([row(i, "t", "A", with_camera=False) for i in range(3)], [], lang, Path(tmp) / "none",
                    dict(config, modalities="fused"))


if __name__ == "__main__":
    unittest.main()
