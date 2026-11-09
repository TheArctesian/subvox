"""Offline IPA CTC alignment with an acoustic evidence gate.

The coverage floor is 80% of expected words. Timed spans must account for at
least half the speech-range energy, and silence below -55 dBFS RMS is rejected.
The -4.8 mean log posterior and 4.7-nat greedy-path gap preserve the six
macOS voices in the integration tests and the two saved English speech takes
(24 words and 9 words). The 50% energy gate includes a one-frame collar when
measuring support because CTC emits narrow spikes within sustained sounds.
Deleting a word costs 4.8 nats per acoustic unit, so skipped words can
leave the path without pulling neighboring boundaries into silence. The
35% active-frame floor guards against spans crammed into energy peaks.
Appended and unrelated passages from the same audio were used as negatives.
These are local acceptance thresholds, not calibrated probabilities.
"""
from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import numpy as np

from . import ipa

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
REVISION = "2c733782da5604684829819a5eb744c193fe9398"
MODEL_DIRECTORY = Path.home() / "Library/Application Support/Subvox/ForcedAligner" / REVISION
SAMPLE_RATE = 16000
HOP = 320
FRAME_SECONDS = HOP / SAMPLE_RATE
THRESHOLDS = dict(word_score=-4.8, free_decode_gap=4.7, coverage=.8, energy=.5, rms=.0018, active_frames=.35, skip_per_unit=4.8)

# Literal symbols from vocab.json at REVISION. Each row lists the identity
# mappings; aliases below are explicit acoustic alternatives for folded IPA.
EXACT = frozenset("""
p b m f v θ ð t d s z n l ʃ ʒ tʃ dʒ k ɡ ŋ ɹ w j h i ɪ ɛ æ ɑ ɔ ʌ ʊ u
eɪ aɪ ɔɪ aʊ oʊ ə ɚ ɲ ʁ e a o y ø œ ɛ̃ ɑ̃ ɔ̃ œ̃ β x ɣ ʎ r ɾ ʝ
pʰ tʰ kʰ tɕ ɕ ʂ ʐ ts ɻ
""".split())
ALIASES = {
    "en": {
        "i": ("iː",), "u": ("uː",), "ɑ": ("ɑː", "ɒ"),
        "ɔ": ("ɔː", "oː"), "ɝ": ("ɜː", "ɚ"),  # rhotic vowel, absent as ɝ
        "ə": ("ɐ",), "ʊ": ("uː",), "ɪ": ("ᵻ",), "t": ("ɾ", "ʔ"),  # folded tap and glottal stop
    },
    "fr": {
        "ɥ": ("j", "w"),  # absent rounded palatal glide, palatal/rounded alternatives
        "a": ("aː",), "e": ("eː",), "o": ("oː",), "y": ("yː",),
    },
    "es": {
        "ʝ": ("j", "ɟ", "ʒ"),  # yeísmo and its reinforced/regionally fricated variants
        "β": ("b",), "ð": ("d",), "ɣ": ("ɡ",),  # stop/continuant allophones
    },
    "zh": {
        "pʰ": ("ph",), "tʰ": ("th",), "kʰ": ("kh",),
        "tɕʰ": ("tɕh",), "tsʰ": ("tsh",), "ʈʂ": ("ts.",), "ʈʂʰ": ("ts.h",),
        "ʂ": ("s.",),  # checkpoint espeak's dotted retroflex spelling
        "ɤ": ("ə",),  # absent unrounded back vowel
        "ɹ̩": ("i̪", "ɨ"), "ɻ̩": ("i.", "ɨ"),  # espeak apical-vowel spellings
        "ɥ": ("j", "w"),  # absent rounded palatal glide
        "ɚ": ("ər",), "ʊ": ("o",), "o": ("uo", "u"),  # rounded vowel in /wo/
    },
}


def symbols(phone, language):
    exact = (phone,) if phone in EXACT else ()
    variants = exact + ALIASES[language].get(phone, ())
    if not variants:
        raise ValueError(f"No {language} CTC mapping for {phone!r}")
    return variants


# Mandarin espeak emits some diphthongs and medial+vowel sequences as one
# acoustic token. Keep their observed span for the word, but don't split it
# into independently timed IPA phones.
MANDARIN_CLUSTERS = {
    ("j", "o", "u"): ("iou",), ("w", "e", "i"): ("uei",), ("w", "a", "i"): ("uai",),
    ("a", "i"): ("ai",), ("e", "i"): ("ei",), ("a", "u"): ("ɑu", "au"), ("o", "u"): ("ou",),
    ("j", "a"): ("iɑ", "ja"), ("j", "e"): ("ie",), ("j", "ɛ"): ("iɛ",),
    ("w", "a"): ("ua",), ("ɥ", "e"): ("yə",), ("ɥ", "ɛ"): ("yæ", "yɛ"),
    ("ʊ", "ŋ"): ("onɡ",),
}


def vocabulary_ids(vocab, language):
    lookup = {}
    for token, index in vocab.items():
        # Tone-bearing and toneless variants share the same acoustic state.
        bare = re.sub(r"[1245ɜ]$", "", token) if language == "zh" and len(token) > 1 else token
        lookup.setdefault(bare, []).append(index)
    return lookup


def acoustic_units(phones, language, lookup):
    index = 0
    while index < len(phones):
        if language == "zh":
            for size in (3, 2):
                cluster = tuple(phones[index:index + size])
                aliases = MANDARIN_CLUSTERS.get(cluster, ()) if len(cluster) == size else ()
                ids = [i for alias in aliases for i in lookup.get(alias, ())]
                if ids:
                    yield None, ids
                    index += size
                    break
            else:
                size = 0
            if size:
                continue
        phone = phones[index]
        ids = [i for alias in symbols(phone, language) for i in lookup.get(alias, ())]
        if not ids:
            raise ValueError(f"No {language} vocabulary ids for {phone!r}")
        yield phone, ids
        index += 1


def text_targets(text, tag):
    language = ipa.validate_language_tag(tag)
    pronunciation = ipa.phonemize(text, language, ipa.accent_of(tag))
    if language == "zh":
        words = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text)
        simplified = ipa._to_simplified("".join(words))
        words = [word + (words[i + 1] if simplified[i:i + 2] in ipa.ERHUA else "")
                 for i, word in enumerate(words) if i == 0 or simplified[i - 1:i + 1] not in ipa.ERHUA]
    else:
        words = re.findall(r"[^\W_]+(?:['’\-][^\W_]+)*", text, flags=re.UNICODE)
        if len(words) != len(pronunciation):
            # eSpeak may split compounds and expand numbers. Preserve their
            # original orthographic word and pool its spoken components.
            pronunciation = [ipa.phones(word, language, ipa.accent_of(tag)) for word in words]
    if not words or len(words) != len(pronunciation) or any(not phones for phones in pronunciation):
        raise ValueError("Could not match the expected text to its pronunciations")
    return language, list(zip(words, pronunciation))


def viterbi(log_probs, targets, blank, word_sizes=()):
    """CTC best path, with optional sets of equivalent symbols per target.

    Return half-open frame spans in target order, or None if no path exists.
    A shared symbol between adjacent targets requires an intervening blank.
    If word sizes are supplied, a whole word can be deleted at a blank
    boundary for a per-unit cost. Deleted units have empty spans.
    """
    target_sets = [(target,) if isinstance(target, (int, np.integer)) else tuple(target) for target in targets]
    frames = len(log_probs)
    if not frames or not target_sets:
        return None
    emissions = np.empty((frames, 2 * len(target_sets) + 1), dtype=np.float32)
    emissions[:, ::2] = log_probs[:, blank, None]
    for i, group in enumerate(target_sets):
        emissions[:, 2 * i + 1] = np.logaddexp.reduce(log_probs[:, group], axis=1)
    skip = np.zeros(emissions.shape[1], dtype=bool)
    skip[3::2] = [not set(left).intersection(right) for left, right in zip(target_sets, target_sets[1:])]
    scores = np.full(emissions.shape[1], -np.inf, dtype=np.float32)
    scores[0] = 0
    back = np.empty(emissions.shape, dtype=np.uint8)
    boundaries = np.r_[0, 2 * np.cumsum(word_sizes)].astype(int)
    skip_cost = boundaries / 2 * THRESHOLDS["skip_per_unit"]
    deletions = np.full((frames + 1, len(boundaries)), -1, dtype=np.int32)
    boundary_index = {state: index for index, state in enumerate(boundaries)}

    def close_deletions(time):
        # Prefix maximum of score minus deletion cost between blank states.
        best_score, best_index = -np.inf, -1
        for i, state in enumerate(boundaries):
            value = scores[state] + skip_cost[i]
            if best_score > value:
                scores[state] = best_score - skip_cost[i]
                deletions[time, i] = best_index
            else:
                best_score, best_index = value, i

    close_deletions(0)
    for frame, emission in enumerate(emissions):
        step = np.r_[-np.inf, scores[:-1]]
        jump = np.where(skip, np.r_[-np.inf, -np.inf, scores[:-2]], -np.inf)
        choices = np.stack((scores, step, jump))
        back[frame] = choices.argmax(axis=0)
        scores = choices.max(axis=0) + emission
        close_deletions(frame + 1)
    state = len(scores) - 1 - int(scores[-2] > scores[-1])
    if not np.isfinite(scores[state]):
        return None
    starts = np.zeros(len(target_sets), dtype=int)
    ends = np.zeros_like(starts)
    for frame in range(frames - 1, -1, -1):
        if state in boundary_index:
            previous = deletions[frame + 1, boundary_index[state]]
            if previous >= 0:
                state = boundaries[previous]
        if state % 2:
            index = state // 2
            starts[index] = frame
            if not ends[index]:
                ends[index] = frame + 1
        state -= back[frame, state]
    return list(zip(starts.tolist(), ends.tolist()))


def score_alignment(log_probs, targets, language, vocab, energy, start=0.0):
    lookup = vocabulary_ids(vocab, language)
    units = [list(acoustic_units(phones, language, lookup)) for _, phones in targets]
    groups = [ids for word in units for _, ids in word]
    path = viterbi(log_probs, groups, vocab["<pad>"], [len(word) for word in units])
    free = log_probs.max(axis=1)  # frame scores of the unconstrained greedy CTC path
    covered = np.zeros(len(energy), dtype=bool)
    words = []
    index = 0
    for (text, _), word_units in zip(targets, units):
        word = dict(text=text, start=None, end=None, score=None, timed=False, phones=[])
        if path is not None and path[index][1] > path[index][0]:
            word_lo, word_hi = path[index][0], path[index + len(word_units) - 1][1]
            boundaries, scores, gaps = [], [], []
            for phone, ids in word_units:
                lo, hi = path[index]
                values = np.logaddexp.reduce(log_probs[lo:hi, ids], axis=1)
                score = float(values.mean())
                gap = float((free[lo:hi] - values).mean())
                scores.append(score)
                gaps.append(gap)
                if phone and score >= THRESHOLDS["word_score"] and gap <= THRESHOLDS["free_decode_gap"]:
                    boundaries.append(dict(ipa=phone, start=start + lo * FRAME_SECONDS,
                                           end=start + hi * FRAME_SECONDS, score=score))
                index += 1
            word["score"] = float(np.mean(scores))
            # A known word competes with a garbage model scored by free CTC
            # decoding. A penalty on that alternative allows accent/model errors.
            if word["score"] >= THRESHOLDS["word_score"] and np.mean(gaps) <= THRESHOLDS["free_decode_gap"]:
                word.update(start=start + word_lo * FRAME_SECONDS, end=start + word_hi * FRAME_SECONDS,
                            timed=True, phones=boundaries)
                covered[max(0, word_lo - 1):word_hi + 1] = True
        else:
            index += len(word_units)
        words.append(word)
    coverage = sum(word["timed"] for word in words) / len(words)
    total = float(energy.sum())
    energy_coverage = float(energy[covered].sum() / total) if total else 0.0
    active = energy >= max(THRESHOLDS["rms"] ** 2, float(energy.max(initial=0)) * .01)
    active_coverage = np.count_nonzero(active & covered) / max(1, np.count_nonzero(active))
    return dict(words=words, coverage=coverage, energy_coverage=energy_coverage,
                accepted=bool(coverage >= THRESHOLDS["coverage"] and energy_coverage >= THRESHOLDS["energy"]
                              and active_coverage >= THRESHOLDS["active_frames"]
                              and len(energy) and np.sqrt(energy.mean()) >= THRESHOLDS["rms"]),
                model=MODEL, revision=REVISION, thresholds=THRESHOLDS)


def read_audio(path, start, end):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("Alignment needs a 16 kHz mono 16-bit PCM WAV")
        duration = wav.getnframes() / SAMPLE_RATE
        if not 0 <= start < min(end, duration) or min(end, duration) - start > 60:
            raise ValueError("Invalid alignment range (maximum 60 seconds)")
        lo, hi = round(start * SAMPLE_RATE), round(min(end, duration) * SAMPLE_RATE)
        wav.setpos(lo)
        return np.frombuffer(wav.readframes(hi - lo), dtype="<i2").astype(np.float32) / 32768, lo / SAMPLE_RATE


def load_model(directory):
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    # Pass only local files and load the pinned state dict without pickle code.
    config = Wav2Vec2Config.from_json_file(str(directory / "config.json"))
    with torch.device("meta"):
        model = Wav2Vec2ForCTC(config)
    weights = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(weights, assign=True)
    model.eval()
    return model, Wav2Vec2FeatureExtractor.from_json_file(str(directory / "preprocessor_config.json"))


def posteriors(audio, model, extractor):
    import torch

    with torch.inference_mode():
        inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        return model(inputs).logits[0].log_softmax(dim=-1).numpy()


def align(audio_path, start, end, text, tag, directory=MODEL_DIRECTORY):
    language, targets = text_targets(text, tag)
    vocab = json.loads((directory / "vocab.json").read_text())
    audio, start = read_audio(audio_path, start, end)
    energy = (audio[:len(audio) // HOP * HOP].reshape(-1, HOP) ** 2).mean(axis=1)
    if len(audio) < 400 or not len(energy) or np.sqrt(energy.mean()) < THRESHOLDS["rms"]:
        return score_alignment(np.empty((0, len(vocab))), targets, language, vocab, energy, start)
    log_probs = posteriors(audio, *load_model(directory))
    # Include tail energy in the denominator even when the convolution has
    # no final frame for it.
    return score_alignment(log_probs, targets, language, vocab, energy, start)
