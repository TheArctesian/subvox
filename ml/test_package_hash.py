import hashlib
import tempfile
import unittest
from pathlib import Path
from subvox_ml.cli import package_hash
from subvox_ml.data import digest


class PackageHashTests(unittest.TestCase):
    def test_python_only_package_hashes_as_the_earlier_py_only_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.py").write_text("x = 1\n")
            (root / "b.py").write_text("y = 2\n")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
            earlier = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")})
            self.assertEqual(package_hash(root), earlier)

    def test_changing_bundled_data_changes_the_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.py").write_text("x = 1\n")
            (root / "data").mkdir()
            table = root / "data" / "table.json"
            table.write_text("{}")
            before = package_hash(root)
            table.write_text('{"jaw": 0.6}')
            self.assertNotEqual(package_hash(root), before)


if __name__ == "__main__":
    unittest.main()
