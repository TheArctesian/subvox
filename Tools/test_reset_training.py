import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reset_training import archive_training


class ResetTrainingTests(unittest.TestCase):
    def test_archive_retains_all_data_and_clears_active_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            support, sessions, archives = (root / name for name in ["support", "sessions", "archives"])
            support.mkdir(); sessions.mkdir()
            (support / "dataset.json").write_text('{"templates":[1,2]}')
            (support / "sentence-training.json").write_text('{"pairs":[3]}')
            (sessions / "visit").mkdir()
            (sessions / "visit/audio.caf").write_bytes(b"recorded audio")
            target = archive_training(support, sessions, archives)
            self.assertEqual(list(support.iterdir()), [])
            self.assertEqual(list(sessions.iterdir()), [])
            self.assertEqual((target / "sessions/visit/audio.caf").read_bytes(), b"recorded audio")
            self.assertEqual((target / "application-support/dataset.json").read_text(), '{"templates":[1,2]}')
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["sources"][0]["files"]), 2)

    def test_partial_move_failure_restores_original_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            support, sessions, archives = (root / name for name in ["support", "sessions", "archives"])
            support.mkdir(); sessions.mkdir()
            (support / "dataset.json").write_text("keep this")
            (sessions / "audio.caf").write_bytes(b"keep audio")
            import shutil
            move = shutil.move
            def fail_second(source, destination):
                if Path(source) == sessions:
                    raise OSError("fixture move failure")
                return move(source, destination)
            with patch("reset_training.shutil.move", side_effect=fail_second):
                with self.assertRaises(OSError):
                    archive_training(support, sessions, archives)
            self.assertEqual((support / "dataset.json").read_text(), "keep this")
            self.assertEqual((sessions / "audio.caf").read_bytes(), b"keep audio")

    def test_rejects_archive_inside_active_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                archive_training(root / "support", root / "sessions", root / "support/archive")


if __name__ == "__main__":
    unittest.main()
