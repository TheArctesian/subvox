"""Fetches a public-domain portrait for each practice piece from Wikimedia Commons.

For every piece in Tools/corpus/pieces/*.json without Corpus/practice/portraits/<id>.jpg, this
asks the Commons API for the file's licence, refuses anything not marked public domain (or CC0),
downloads a 400 px thumbnail into --downloads (a scratch folder), crops a square from the top
(where the face usually is), saves a 160 px JPEG and adds a row to SOURCES.tsv. Crops are a
starting point: look at them and adjust with --crop id=size,top,left if a face is cut.

    python3 Tools/corpus/portraits.py --downloads /tmp/portraits
"""
import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Corpus" / "practice" / "portraits"
UA = {"User-Agent": "SubvoxPortraitFetch/1.0 (Subvox practice corpus build)"}
ALLOWED = ("Public domain", "CC0")


def commons(title):
    query = dict(action="query", format="json", titles=title, prop="imageinfo", iiprop="url|extmetadata", iiurlwidth=400)
    request = urllib.request.Request("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(query), headers=UA)
    with urllib.request.urlopen(request, timeout=30) as response:
        page = next(iter(json.load(response)["query"]["pages"].values()))
    info = page["imageinfo"][0]
    return info, info["extmetadata"].get("LicenseShortName", {}).get("value", "")


def sips(*args):
    subprocess.run(["sips", *map(str, args)], check=True, capture_output=True)


def size(path):
    out = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)], check=True, capture_output=True, text=True).stdout
    values = [int(line.split()[-1]) for line in out.splitlines() if "pixel" in line]
    return values[0], values[1]


def crop(source, target, square=None, top=None, left=None):
    work = target.with_suffix(".work.jpg")
    sips("-s", "format", "jpeg", "--resampleWidth", 256, source, "--out", work)
    width, height = size(work)
    square = square or min(width, height, 200)
    top = int(height * 0.04) if top is None else top
    left = (width - square) // 2 if left is None else left
    sips("--cropOffset", top, left, "--cropToHeightWidth", square, square, work, "--out", work)
    sips("-z", 160, 160, "-s", "formatOptions", 85, work, "--out", target)
    work.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", required=True, type=Path)
    parser.add_argument("--crop", action="append", default=[], help="id=size,top,left to redo one crop")
    args = parser.parse_args()
    args.downloads.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sources = OUT / "SOURCES.tsv"
    rows = sources.read_text().splitlines() if sources.exists() else ["piece_id\tcreator_and_date\tcommons_file\tcommons_page\tlicence"]
    known = {row.split("\t")[0] for row in rows[1:]}
    overrides = {k: [int(v) for v in rest.split(",")] for k, rest in (c.split("=") for c in args.crop)}
    for path in sorted((ROOT / "Tools" / "corpus" / "pieces").glob("*.json")):
        for piece in json.loads(path.read_text())["pieces"]:
            pid, target = piece["id"], OUT / f"{piece['id']}.jpg"
            raw = args.downloads / f"{pid}.img"
            if pid in overrides and raw.exists():
                crop(raw, target, *overrides[pid])
                print("recropped", pid)
                continue
            if target.exists():
                continue
            info, licence = commons(piece["portrait"])
            if not licence.startswith(ALLOWED):
                print("skipped", pid, "licence:", licence)
                continue
            time.sleep(3)
            with urllib.request.urlopen(urllib.request.Request(info["thumburl"], headers=UA), timeout=60) as response:
                raw.write_bytes(response.read())
            crop(raw, target)
            if pid not in known:
                rows.append("\t".join([pid, f"{piece['author']}, portrait", piece["portrait"], info["descriptionurl"], licence]))
            print("saved", pid)
            time.sleep(3)
    sources.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
