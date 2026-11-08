"""Every string the app shows in the Traditional view must contain no Simplified-only characters."""
import json
import unittest
from pathlib import Path
from opencc import OpenCC

CORPUS = Path(__file__).resolve().parents[1] / "Corpus" / "practice" / "practice-v1.json"
# Characters OpenCC's s2t maps to a variant even though they are standard Traditional too.
SHARED = set("吃着里干后周表台面发松游床")


def simplified_only(text, s2t=OpenCC("s2t")):
    return sorted({c for c in text if c not in SHARED and s2t.convert(c) != c})


class TraditionalView(unittest.TestCase):
    def test_no_simplified_characters_in_traditional_strings(self):
        d = json.loads(CORPUS.read_text())
        bad = []
        for x in d["drills"]:
            if x["language"] == "zh":
                for value in [x.get("title_hant"), x.get("note_hant")] + [p["hant"] for p in x["phrases"]]:
                    if value and simplified_only(value):
                        bad.append((x["id"], value, simplified_only(value)))
        for p in d["pieces"]:
            if p["language"] == "zh":
                for value in [p.get("title_hant"), p.get("author_hant")] + [l["hant"] for l in p["lines"]]:
                    if value and simplified_only(value):
                        bad.append((p["id"], value, simplified_only(value)))
        for s in d.get("sounds", {}).get("zh", []):
            for e in s["examples"]:
                if simplified_only(e["hant"]):
                    bad.append((s["ipa"], e["hant"], simplified_only(e["hant"])))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
