"""Builds Subvox/Localizable.xcstrings from translation JSON files.

    python3 Tools/l10n/build_catalog.py Tools/l10n/fr-es.json Tools/l10n/zh.json

Each input is {"<language>": {"<English key>": "<translation>"}}. A translation is rejected
(and reported) when its format specifiers (%@, %lld, %1$@ …) differ from the key's, because a
mismatch crashes or garbles the string at runtime. English is the source language.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "Subvox" / "Localizable.xcstrings"
SPEC = re.compile(r"%(?:\d+\$)?(?:lld|ld|d|@|f|\.\d+f|%)")


def specifiers(text):
    return sorted(re.sub(r"\d+\$", "", s) for s in SPEC.findall(text))


def main(paths):
    strings, problems = {}, []
    for path in paths:
        for language, table in json.loads(Path(path).read_text()).items():
            for key, value in table.items():
                if not isinstance(value, str) or not value.strip():
                    continue
                if specifiers(key) != specifiers(value):
                    problems.append(f"{language}: {key!r} -> {value!r}")
                    continue
                entry = strings.setdefault(key, {"localizations": {}})
                entry["localizations"][language] = {"stringUnit": {"state": "translated", "value": value}}
    catalog = {"sourceLanguage": "en", "strings": dict(sorted(strings.items())), "version": "1.0"}
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(strings)} strings -> {CATALOG}")
    for problem in problems:
        print("rejected (format specifiers differ):", problem)


if __name__ == "__main__":
    main(sys.argv[1:])
