# Brief: a multilingual forced aligner for Subvox training takes

Paste this whole file to a coding agent working in the Subvox repo (`main`).

## Goal

Stop discarding Reading and Speech & Poetry takes because Whisper misheard or lost track of words. We know the text that was read, so we shouldn't ask a recogniser what was said. Instead, a **forced aligner** should find where each known word, and each phone in it, falls in the audio. This has to work for English, French, Spanish (including es-MX) and Mandarin (zh-Hans and zh-Hant).

Done means:

1. A take where the person read the text is kept for training even when Whisper heard "Summerslees have all to short a date". It gets word times and phone times from the aligner.
2. Takes already rejected as "Too few words have usable audio-aligned times" or "Whisper did not match enough of the spoken text" can be re-processed from their saved audio and recovered, with no re-recording.
3. A take where the person really didn't read the text is still rejected: skipped lines, the wrong passage, silence, or mostly noise. Times are never invented.

## Why takes fail today (read these first)

- **The rejection rules:** `Subvox/Collection/SentenceTrainingWorkflow.swift`, `process(_:whisper:)`, around lines 560–630. Two rules:
  - Whisper text agreement must be at least 80%, or the take fails with `Failure.transcript`.
  - Whisper must give word times for at least 80% of the expected words, or it fails with `Failure.timing`.
  - Whisper's word times are DTW attention points (`Subvox/Transcription/AudioWordTiming.swift`). They're unreliable on long (~30-word) takes and on archaic text (hath, thee).
- **Word anchors drive the motion alignment:** `SentenceReference.anchors` in `Subvox/Collection/SentenceTrainingData.swift` produces `[MotionWordAnchor]` (`word`, `start`, `end` in session uptime seconds). These feed `SentenceMotionAlignment.align(voiced:silent:words:)`. The aligner's output has to become these anchors.
- **The existing phone aligner:** `Subvox/Linguistics/LocalPhonemeAligner.swift` is an optional MFA adapter that only runs for `en`. Its `PhoneAlignment` result goes in `pair.spokenPhones` and is transferred to the mouthed half via `CameraPhoneAlignment`. Reuse `PhoneAlignment` for the new aligner's phone output so all of that keeps working.
- **Phone labels:** `ml/subvox_ml/ipa.py` produces expected phones from text: `phonemize`, `phones`, `mandarin_syllables`, `strip_tones`, and `accent_of` for es-MX. It holds the shared IPA `INVENTORY`, and Mandarin model targets are toneless. `ml/subvox_ml/phone_check.py` and `community_export.py` (schema v2) use these labels. `community_export.training_phones` uses `produced_phones` only when they come from phone intervals, so real phone intervals from the aligner make more data usable.
- **The highlight is not timing:** the on-screen highlight is a fixed 0.65 s-per-word metronome (`Subvox/Core/WordFocusGuide.swift`) and readers don't follow it. Never use it as labels. At most it's a loose prior: the person can't be on word 25 in the first second.

## Recommended approach

Do CTC forced alignment over IPA phones with **`facebook/wav2vec2-xlsr-53-espeak-cv-ft`** (Apache-2.0). It's a multilingual wav2vec2 model whose output vocabulary is espeak IPA phones, the same family `ipa.py` already produces.

1. **Model check before writing code.** Confirm the model's licence and vocabulary on Hugging Face, and pin a revision hash. If a better permissively licensed multilingual phone CTC model exists, use it. Don't use the MMS forced-alignment models (`torchaudio.pipelines.MMS_FA`, `facebook/mms-*`): they're CC-BY-NC, which doesn't fit this AGPL project's community dataset.
2. **Map phones to the model.** Write an explicit map from our `INVENTORY` phones to the model's vocabulary, one file per language if needed. Base it on the real vocab file, with no guessed remapping (the same rule `LocalPhonemeAligner` follows).
   - A phone with no model equivalent maps to its closest listed symbol only through an entry in the map, with a comment.
   - Mandarin uses toneless targets.
   - Test that every phone `ipa.py` can emit for each language is mapped.
3. **Write the CTC Viterbi alignment ourselves** (numpy, about 60 lines). `torchaudio.functional.forced_align` is deprecated in recent torchaudio, so don't depend on it.
   - Inputs: frame log-probabilities and the target phone sequence with blanks.
   - Outputs: a frame span per phone, then word spans as the union of each word's phone spans.
   - Frames are 20 ms; convert to seconds against the audio range.
4. **Make skipped and misread words detectable.** Forced alignment always returns *an* answer, so it needs gates:
   - A per-phone score (mean log-posterior over its frames) and a per-word score (mean over its phones).
   - A per-word "garbage" alternative: also align with an optional star/garbage token, or compare each word's score against an unconstrained (free) CTC decode over the same frames. A word is *timed* only if it beats that alternative by a margin.
   - Pick thresholds from data (below), not by guessing.
   - The take is kept when at least 80% of the expected words are timed, matching today's rule. Untimed words are omitted, never filled in.
   - Also reject takes where the aligned speech covers less than about 50% of the voiced audio energy (reading the wrong text), or where most words cram into a tiny span.
5. **Add a Python command:**
   ```
   subvox-ml align --audio <wav> --start <s> --end <s> --text <text> --language <tag> --output <json>
   ```
   - The output JSON holds the words `{text, start, end, score, timed}`, the phones `{ipa, start, end, score}` per word, plus the model id, revision and thresholds used.
   - Put the code in `ml/subvox_ml/forced_align.py`. Add `transformers` (or load the weights directly with torch) to `ml/pyproject.toml` with bounded versions, and run `uv lock`.
6. **Model download:** a user-initiated download (a button, the same pattern as the Whisper setup sheet) into Application Support, at the pinned revision with a size and sha256 check. Nothing downloads during alignment, and alignment must work fully offline once installed.
7. **Swift integration:**
   - Add a `ForcedAligner` wrapper that runs `subvox-ml align` the same way `PracticeChecker` runs `subvox-ml phones` (in `Subvox/Practice/PracticeCorpus.swift`).
   - In `process(_:whisper:)`, when the forced aligner is installed, build the anchors from its timed words instead of Whisper's.
   - Keep Whisper as the transcript check, and add the expected text as Whisper's initial prompt (whisper.cpp `--prompt`). This helps archaic text.
   - Change the text-agreement rule so that a take passes when **either** Whisper agrees at least 80% **or** the forced aligner times at least 80% of the words with passing scores. Keep the 85% rule for Free Practice: the typed text there isn't known to be what was said, so the aligner alone mustn't be trusted.
   - Store the aligner's phones in `pair.spokenPhones` for every language, so the camera transfer and the community export's `produced_phones` work beyond English.
   - Record where the timing came from on the pair (for example `wordTimingSource: "forcedAlignment" | "whisperAttention"`, plus the model revision) and include it in the export.
   - Keep the MFA path working for English when it's configured, but the new aligner takes priority.
8. **Recover old takes:** add "Re-align saved takes" (on the Reading page's data area, or wherever `transcribePending` is exposed). It re-runs `process` on pairs whose saved error is `.timing` or `.transcript`, using the audio already on disk. Report how many were recovered.
9. **UI wording:** results currently say "Too few words have usable audio-aligned times…". When the aligner rescued a take, say "Kept for training (words matched to your audio)". Use plain words and add the new strings to `Tools/l10n/translations.json` for fr, es, es-419, zh-Hans and zh-Hant, then rebuild the catalog with `python3 Tools/l10n/build_catalog.py Tools/l10n/translations.json`.

## Testing (required)

- **Python unit tests (`ml/test_forced_align.py`):**
  - Viterbi on hand-built posteriors gives known spans.
  - The phone map covers every phone `ipa.py` emits for en, fr, es, es-MX and zh.
  - The garbage gate marks an unspoken word as untimed.
- **Real audio, one test per language:** synthesise the text with macOS `say` using a voice per language (`say -v '?'` lists them), with 1 s of silence padded before. Assert:
  - every word is timed;
  - the first word starts after the padding;
  - word order is monotonic;
  - the total span roughly matches the speech.
- **Negative test:** align a text with an extra sentence that wasn't spoken. Assert those words are untimed and the take fails the 80% gate.
- **Thresholds:** pick them from these tests plus any real takes in the user's data folder, and write the chosen values and why into the module docstring.
- **Swift:** a unit test that a pair with Whisper agreement at 55% but aligner coverage of at least 80% becomes `usesAlignedWords`, and one where both fail stays rejected. Put them in `SubvoxTests/Collection/PracticeTests.swift`, or a new file after `xcodegen generate`.
- **Commands:**
  - Run the Python tests with `cd ml && uv run python -m pytest`.
  - Build with `xcodegen generate && xcodebuild -project Subvox.xcodeproj -scheme Subvox -configuration Release -derivedDataPath build/dd CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM= build`.
  - Run the Swift tests with the same flags plus `test -only-testing:SubvoxTests/PracticeTests`.
  - MouthTeacherTests and CameraMotionCompensationTests already fail; leave them.

## Rules

- Never invent times: no even spreading, no highlight-schedule labels. An untimed word stays untimed.
- Work on a branch off `main`, commit as you go, push, and open a PR. Don't push directly to `main`.
- Don't kill processes broadly (`pkill Subvox`, `pkill osascript`). Quit the app with `osascript -e 'tell application "Subvox" to quit'`.
- If a Keychain password prompt appears, press Deny and never type a password. The test host already uses an in-memory token store.
- Keep the docs honest: update `docs/next-phase-plan.md` (the open item "phone-level feedback needs MFA or a phone recogniser") and NOTICE (model attribution).
