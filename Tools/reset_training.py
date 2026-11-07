#!/usr/bin/env python3
"""Archive Subvox datasets and recordings, then recreate empty active folders.

Close Subvox before running. Model files and local Whisper preferences are kept.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid


def inventory(root):
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob("*")) if p.is_file()}


def archive_training(application_support, sessions, archive_root):
    sources = [("application-support", Path(application_support)), ("sessions", Path(sessions))]
    archive_root = Path(archive_root)
    paths = [p.resolve() for _, p in sources]
    if any(p.is_symlink() for _, p in sources):
        raise ValueError("Training directories must not be symbolic links")
    for i, path in enumerate(paths):
        for other in paths[i + 1:] + [archive_root.resolve()]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError("Archive and active training directories must be separate")
    snapshots = [(name, path, inventory(path)) for name, path in sources if path.exists()]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    target = archive_root / (stamp + "-" + uuid.uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "archiving", "createdAt": datetime.now(timezone.utc).isoformat(),
                "sources": [{"original": str(path), "archive": name, "files": files}
                            for name, path, files in snapshots]}
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    moved = []
    try:
        for name, path, files in snapshots:
            destination = target / name
            shutil.move(str(path), str(destination))
            moved.append((path, destination))
            if inventory(destination) != files:
                raise IOError("Archive verification failed for " + str(path))
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except Exception:
        for path, destination in reversed(moved):
            shutil.move(str(destination), str(path))
        manifest["status"] = "rolled back"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    for _, path in sources:
        path.mkdir(parents=True, exist_ok=True)
    return target


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Move current data into a verified recovery archive")
    args = parser.parse_args()
    support = home / "Library/Application Support/Subvox"
    sessions = home / "Documents/Subvox/Sessions"
    archive_root = home / "Documents/Subvox/Archives"
    if not args.execute:
        print("Will archive:", support, sessions, "to", archive_root, sep="\n")
        print("Run with --execute after closing Subvox.")
        return
    if subprocess.run(["pgrep", "-x", "Subvox"], capture_output=True).returncode == 0:
        raise SystemExit("Close Subvox before resetting training data.")
    target = archive_training(support, sessions, archive_root)
    # This preference belonged to the removed legacy source selector.
    subprocess.run(["defaults", "delete", "com.joeblau.subvox", "useReviewedSentenceModel"], capture_output=True)
    print(target)


if __name__ == "__main__":
    main()
