import math
import unittest
import numpy as np
from subvox_ml import camera
from subvox_ml.closedset import closed_set, dtw

SENTENCES = ["blue bag", "red cup", "green hat", "big dog", "small cat"]


def clip(text, condition, session, signal, rng, pair):
    """150 frames for every sentence, so duration cannot identify it. With `signal`, each sentence
    has its own rotation-rate and mouth-opening trajectory; otherwise only noise."""
    k = SENTENCES.index(text)
    t = np.arange(150) / 50
    shape = np.sin(2 * np.pi * (k + 1) * t / 3) if signal else np.zeros_like(t)
    raw = np.zeros((150, 13), np.float32)
    raw[:, 8] = -1
    raw[:, 9] = 1
    raw[:, 3] = .05 * shape + rng.normal(0, .01, 150)
    raw[:, 0] = rng.normal(0, .01, 150)
    cam = np.zeros((150, camera.RAW_WIDTH), np.float32)
    cam[:, camera.SCALARS.index("opening")] = .3 + .1 * shape + rng.normal(0, .01, 150)
    cam[:, 6:camera.VALUE_WIDTH] = rng.normal(0, .01, (150, camera.VALUE_WIDTH - 6))
    cam[:, 6:30] += .05 * shape[:, None]  # lip contour points move with the mouth opening
    cam[:, camera.MOUTH_MASK] = 1
    return dict(id=f"{condition}-{session}-{text}-{rng.integers(1e9)}", text=text, activity=1, condition=condition, session=session,
                raw=raw.tolist(), camera=cam.tolist(), provenance=dict(pair=pair))


def dataset(signal, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for text in SENTENCES:
        rows += [clip(text, c, "s0", signal, rng, f"p-{text}") for c in ("Normal speech", "Silent mouthing")]
    for session in ("s1", "s2", "s3"):
        for text in SENTENCES[:2]:
            rows += [clip(text, c, session, signal, rng, f"p-{text}-{session}") for c in ("Normal speech", "Silent mouthing")]
    return rows


class ClosedSetTests(unittest.TestCase):
    def test_dtw_is_zero_on_identical_and_tolerates_time_stretch(self):
        a = np.sin(np.linspace(0, 6, 40))[:, None]
        b = np.sin(np.linspace(0, 6, 60))[:, None]
        self.assertEqual(dtw(a, a), 0)
        self.assertLess(dtw(a, b), dtw(a, -b))

    def test_separable_sentences_beat_duration_and_the_permutation_null(self):
        report = closed_set(dataset(signal=True), permutations=199)
        for modality in ("imu", "camera", "fused"):
            enrol = report["protocols"]["enrol_spoken"][modality]
            self.assertEqual(enrol["queries"], 11)
            self.assertGreater(enrol["top1"], .8, modality)
            self.assertLess(enrol["p_value"], .01)
            self.assertLessEqual(enrol["duration_top1"], .5, "Equal lengths: duration must not identify sentences")
            cross = report["protocols"]["cross_session"]["Silent mouthing"][modality]
            self.assertEqual(cross["queries"], 8, "Only sentences recorded in several sessions are queried")
            self.assertEqual(cross["repeated_sentences"], 2)
            self.assertGreater(cross["top1"], .8)

    def test_noise_only_is_rarely_significant(self):
        p = [closed_set(dataset(signal=False, seed=seed), permutations=99)["protocols"]["enrol_spoken"]["fused"]["p_value"] for seed in range(2, 8)]
        self.assertLessEqual(sum(v < .05 for v in p), 1, p)

    def test_clips_without_camera_are_left_out_of_every_modality(self):
        rows = dataset(signal=True)
        rows[0] = {k: v for k, v in rows[0].items() if k != "camera"}
        report = closed_set(rows, permutations=19)
        self.assertEqual(report["cohort"]["excluded_without_camera"], 1)
        self.assertEqual(report["protocols"]["enrol_spoken"]["imu"]["queries"], report["protocols"]["enrol_spoken"]["camera"]["queries"])


class CommandProtocolTests(unittest.TestCase):
    def commands(self, sessions, blocks):
        rng = np.random.default_rng(3)
        rows = []
        for session in sessions:
            for text in SENTENCES:
                rows.append(dict(clip(text, "Normal speech", session, True, rng, None), provenance=dict(kind="command set")))
                for _ in range(blocks):
                    rows.append(dict(clip(text, "Silent mouthing", session, True, rng, None), provenance=dict(kind="command set")))
        # Paired sentences never become command candidates.
        rows.append(dict(clip("blue bag", "Silent mouthing", "other", True, rng, "p"), provenance=dict(kind="reviewed sentence pair")))
        return rows

    def test_one_session_scores_within_session_only(self):
        report = closed_set(self.commands(["s1"], 2), permutations=99)
        within = report["protocols"]["commands_within_session"]["fused"]
        self.assertEqual(within["queries"], 10)
        self.assertEqual(within["mean_candidates"], 5, "Candidates are the commands, not every recorded sentence")
        self.assertGreater(within["top1"], .8)
        self.assertIsNone(report["protocols"]["commands_cross_session"]["fused"], "One session has no other-session templates")
        self.assertEqual(report["protocols"]["commands_enrol_spoken"]["camera"]["queries"], 10)

    def test_two_sessions_add_cross_session_scores(self):
        report = closed_set(self.commands(["s1", "s2"], 1), permutations=99)
        cross = report["protocols"]["commands_cross_session"]["imu"]
        self.assertEqual(cross["queries"], 10)
        self.assertIsNone(report["protocols"]["commands_within_session"]["imu"], "One block per session leaves no same-session template")

    def test_discover_reads_command_visits_as_training_material(self):
        import tempfile
        from pathlib import Path
        from subvox_ml.data import discover, atomic_json
        from test_sequence import recording, language
        with tempfile.TemporaryDirectory() as tmp:
            takes = [dict(id="c1", text="pa", sentenceID="pa", recording=dict(recording(), id="c1"), camera=[])]
            atomic_json(Path(tmp) / "SequenceLearning/Collections/c/session.json",
                        dict(id="c", complete=True, role="commands", session=dict(id="s"), takes=takes))
            rows, _, _ = discover(tmp, language())
            self.assertEqual([r["provenance"]["kind"] for r in rows], ["command set"])
            self.assertEqual(rows[0]["original_role"], "Training")


if __name__ == "__main__":
    unittest.main()
