import hashlib
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from subvox_ml import forced_align as fa, ipa


def test_viterbi_with_repeated_phones():
    ids = [0, 1, 1, 0, 1, 2, 2, 0]
    probs = np.full((len(ids), 3), -12.0)
    probs[np.arange(len(ids)), ids] = 0
    assert fa.viterbi(probs, [1, 1, 2], 0) == [(1, 3), (4, 5), (5, 7)]
    assert fa.viterbi(probs[:2], [1, 1], 0) is None


def test_unspoken_middle_word_is_untimed():
    vocab = {"<pad>": 0, "a": 1, "x": 2, "n": 3}
    ids = [0, 1, 1, 0, 0, 3, 3, 0]
    probs = np.full((len(ids), len(vocab)), -15.0)
    probs[np.arange(len(ids)), ids] = 0
    result = fa.score_alignment(probs, [("a", ["a"]), ("extra", ["x"]), ("n", ["n"])],
                                "es", vocab, np.full(len(ids), .01))
    assert [word["timed"] for word in result["words"]] == [True, False, True]
    assert result["words"][1]["start"] is None
    assert result["words"][1]["phones"] == []
    assert not result["accepted"]


@pytest.fixture(scope="module")
def vocab():
    # Exact upstream vocabulary from fa.REVISION, under Apache-2.0.
    raw = (Path(__file__).parent / "test_data/forced_align_vocab.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "d732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0"
    return json.loads(raw)


@pytest.mark.parametrize("tag", ["en", "fr", "es", "es-MX", "zh-Hans", "zh-Hant"])
def test_entire_inventory_maps_to_pinned_vocabulary(vocab, tag):
    language = ipa.validate_language_tag(tag)
    lookup = fa.vocabulary_ids(vocab, language)
    for phone, languages in ipa.INVENTORY.items():
        if language in languages:
            [(label, ids)] = fa.acoustic_units([phone], language, lookup)
            assert label == phone
            assert ids and not set(ids).intersection(range(4))


def test_mandarin_keeps_written_script_and_does_not_split_erhua():
    _, targets = fa.text_targets("哪兒有書？", "zh-Hant")
    assert [word for word, _ in targets] == ["哪兒", "有", "書"]


CASES = [
    ("en", "Samantha (English (US))", "Shall I compare thee to a summer’s day? Thou art more lovely and more temperate."),
    ("fr", "Thomas", "Bonjour, nous parlons avec nos amis."),
    ("es", "Mónica", "La casa tiene una puerta roja cerca de la plaza."),
    ("es-MX", "Paulina", "La casa tiene una puerta roja cerca de la plaza."),
    ("zh-Hans", "Tingting", "你好，我今天很忙。"),
    ("zh-Hant", "Meijia", "你好，我今天讀書。"),
]


@pytest.fixture(scope="module")
def model(vocab):
    if sys.platform != "darwin" or not shutil.which("ffmpeg"):
        pytest.skip("the audio integration tests require macOS say and ffmpeg")
    if not (fa.MODEL_DIRECTORY / "pytorch_model.bin").is_file():
        pytest.skip("install the pinned forced alignment weights to run audio integration tests")
    return fa.load_model(fa.MODEL_DIRECTORY)


@pytest.fixture(scope="module")
def recordings(model, tmp_path_factory):
    output = tmp_path_factory.mktemp("align-speech")
    return model, output, {}


def synthesize(recordings, tag, voice, text):
    model, output, cache = recordings
    if tag not in cache:
        available = subprocess.check_output(["say", "-v", "?"], text=True)
        if not any(line.startswith(voice + " ") for line in available.splitlines()):
            pytest.skip(f"install the macOS {voice} voice")
        aiff, wav = output / f"{tag}.aiff", output / f"{tag}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(aiff), "-af", "adelay=1000",
                        "-ar", "16000", "-ac", "1", str(wav)], check=True)
        with wave.open(str(wav), "rb") as file:
            duration = file.getnframes() / fa.SAMPLE_RATE
        audio, _ = fa.read_audio(wav, 0, duration)
        energy = (audio[:len(audio) // fa.HOP * fa.HOP].reshape(-1, fa.HOP) ** 2).mean(axis=1)
        cache[tag] = fa.posteriors(audio, *model), energy
    return cache[tag]


@pytest.mark.parametrize("tag,voice,text", CASES, ids=[case[0] for case in CASES])
def test_real_audio_all_words_timed(recordings, vocab, tag, voice, text):
    probs, energy = synthesize(recordings, tag, voice, text)
    language, targets = fa.text_targets(text, tag)
    result = fa.score_alignment(probs, targets, language, vocab, energy)
    words = result["words"]
    assert result["accepted"]
    assert all(word["timed"] and word["phones"] for word in words)
    assert words[0]["start"] >= 1
    assert all(a["end"] <= b["start"] for a, b in zip(words, words[1:]))
    active = np.flatnonzero(energy > energy.max() * .01)
    assert abs(words[-1]["end"] - (active[-1] + 1) * fa.FRAME_SECONDS) < .35
    assert abs(words[0]["start"] - active[0] * fa.FRAME_SECONDS) < .35


def test_appended_or_wrong_passage_fails(recordings, vocab):
    tag, voice, text = CASES[0]
    probs, energy = synthesize(recordings, tag, voice, text)
    extra = "Zebra hunters gathered outside the silent railway station."
    _, actual = fa.text_targets(text, tag)
    for candidate in (text + " " + extra, extra + " " + text, extra):
        language, targets = fa.text_targets(candidate, tag)
        result = fa.score_alignment(probs, targets, language, vocab, energy)
        assert result["coverage"] < .8
        assert not result["accepted"]
        if candidate.startswith(text):
            assert not any(word["timed"] for word in result["words"][len(actual):])
        if candidate.endswith(text):
            assert not any(word["timed"] for word in result["words"][:-len(actual)])


def test_silence_and_noise_fail(recordings, vocab):
    model, _, _ = recordings
    _, targets = fa.text_targets("The camera caught a bright yellow flower.", "en")
    for audio in (np.zeros(32000, np.float32), np.random.default_rng(71).normal(0, .05, 32000).astype(np.float32)):
        energy = (audio.reshape(-1, fa.HOP) ** 2).mean(axis=1)
        result = fa.score_alignment(fa.posteriors(audio, *model), targets, "en", vocab, energy)
        assert not result["accepted"]
