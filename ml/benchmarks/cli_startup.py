"""Measure fresh-process pronunciation command latency with synthetic prompts."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time


def measure(source, repeats):
    cases = {
        "phones-en": ["phones", "--language", "en", "--text", "Have a nice day."],
        "phones-zh": ["phones", "--language", "zh-Hans", "--text", "你好，我今天很忙。"],
        "phone-check-en": [
            "phone-check", "--language", "en", "--text", "Have a nice day.",
            "--whisper-text", "Have a nice day.",
        ],
    }
    launcher = (
        "import runpy,sys; sys.path.insert(0, sys.argv.pop(1)); "
        "runpy.run_module('subvox_ml.cli', run_name='__main__')"
    )
    timings = {name: [] for name in cases}
    memory = {name: [] for name in cases}
    outputs = {}
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "result.json"
        for _ in range(repeats):
            for name, args in cases.items():
                output.unlink(missing_ok=True)
                command = [sys.executable, "-c", launcher, str(source), *args, "--output", str(output)]
                if sys.platform == "darwin":
                    command = ["/usr/bin/time", "-l", *command]
                start = time.perf_counter()
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                timings[name].append(time.perf_counter() - start)
                canonical = json.dumps(json.loads(output.read_bytes()), sort_keys=True)
                digest = hashlib.sha256(canonical.encode()).hexdigest()
                if outputs.setdefault(name, digest) != digest:
                    raise ValueError(f"Non-deterministic JSON output for {name}")
                rss = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
                if rss:
                    memory[name].append(int(rss[1]))
    return {
        name: {
            "median_ms": round(statistics.median(samples) * 1000, 1),
            "max_ms": round(max(samples) * 1000, 1),
            "median_rss_mib": round(statistics.median(memory[name]) / 2**20, 1) if memory[name] else None,
            "json_sha256": outputs[name],
        }
        for name, samples in timings.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps(measure(args.source.resolve(), args.repeats), indent=2))
