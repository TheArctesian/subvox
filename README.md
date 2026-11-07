# Subvox

A macOS prototype that tests whether AirPods Pro head-motion data (`CMHeadphoneMotionManager`) carries enough information to turn **silently mouthed words into text**.

The motion decoder builds personal word models. Local Whisper supplies spoken-reference transcripts and estimated word timings for paired training; it does not decode silent motion.

## Build & run

```sh
brew install xcodegen   # once
xcodegen generate
open Subvox.xcodeproj   # ⌘R
```

Or from the command line:

```sh
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build -allowProvisioningUpdates -configuration Release build
open build/Build/Products/Release/Subvox.app
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build test
```

Use the Release build for decoding: it matches each word against ~18,000 templates, which takes about 0.4 s optimised and over a minute in Debug.

Requires macOS 14+, AirPods Pro or another headphone that supports head tracking, connected as the Mac's audio output. On first launch, grant **Motion**. The mic and speech prompts only appear when you start the mic.

## Screens

| Screen | What it does |
|---|---|
| **Sensors** | Live plots of user acceleration, rotation rate and attitude, the measured sample rate, a mic input picker and level meter, and a recorder for experiment sessions. |
| **Train** | One paired sentence workflow: join sentences into short passages, speak then mouth each prompt, align motion using local Whisper reference timing, review both traces, fill remaining pronunciation gaps with word repairs, and build/check the existing word decoder. Large colored cues and live plots stay visible. |
| **Signal check** | Controlled spoken, mouthed and mouth-still trials for measuring motion signal. Diagnostic recordings stay out of model training. |
| **Word diagnostics** | Persistent sessions, notes, a sound-coverage checklist, per-take motion plots and quality review. Decoder validation measures unseen word groups; 25% of word groups stay reserved for the final test. |
| **Read** | A teleprompter for the reading corpus (the passage, or a block of the diphone sentences). Pick a condition (normal, whisper or silent), read each line and press Space. When you stop, Subvox writes `labels.csv` next to the session's motion and audio: every line, word, phone and vowel jaw class, timed on the motion clock. Voiced and whispered words are timed by the on-device speech recognizer; silent readings, and any word the recognizer missed, are spread over the line's active motion window and marked `estimated`. |
| **Decode** | The motion-to-text screen, open vocabulary. Hold Space, or use auto-detect after calibrating the noise floor. Each segment is matched against all ~18,000 vocabulary words, built from your trained sounds, so it can output words you never recorded. A live mic transcript runs alongside it for comparison. |

## Paired sentence workflow

**Train** keeps collection, alignment review, pronunciation coverage, word repairs and the decoder check together. The default **Short passages** mode joins up to three sentences; choose **Sentences** for individual prompts. Each prompt has an amber Relax baseline, green Speak cue, recovery, another baseline, then a blue Mouth cue. Read once naturally and wait. Comfortable cues are at least 12 seconds; Extra time uses at least 16 seconds, with longer passages allocated more time (up to 45 seconds). No per-word Space presses are needed. Live motion stays pinned below the prompt; review preserves both saved traces with selectable channels.

The authored 32-sentence corpus gets a separate persistent split: 18 fitting sentences, 6 development validation sentences, and **8 (25%) sealed final sentences**. Normalized text identifies a sentence, so repeats and punctuation/case variants keep their role. Passage groups never cross roles. Final text is excluded from acquisition plans, coverage, repairs and model fitting. The earlier word dataset and its assignments remain separate.

Each visit records local audio and raw motion, cue clocks, microphone, notes, pace, presentation and measured rate. Each completed half is saved immediately; Stop retains completed halves and processes complete pairs. A partial pair can be re-recorded. Existing OpenWhispr files power local, hint-free Whisper transcription of each spoken cue independently. Audio-file times are converted using the recorder's audio offset and monotonic start clock. At least 80% of prompted words must match in order; unmatched words receive no invented timestamps.

Full-path DTW aligns acceleration and rotation from the paired cues, with pair-local scaling and monotonic word-time transfer. Flat/invalid motion, sampling gaps, malformed anchors and excessive warps fail alignment. Estimated boundaries, cost and warp warnings are shown for review. These are annotation estimates, not verified phone boundaries or evidence of recognition. Normal and silent motion can differ substantially; accept only after review, or retry alignment/re-record. Both traces show sample rates and quality hints, and **Open raw recording** exposes the saved audio/motion for inspection.

Only accepted fitting pairs produce silent word slices, with stable IDs and sentence/pair/visit lineage. Dictionary phoneme counts measure collection coverage, with a target of four occurrences per unit; a mixed accent may differ from dictionary pronunciations. Isolated word repairs are offered only after fitting sentences have been reviewed and only for remaining coverage gaps. Validation/final examples never influence those gaps. The greedy repair planner favors broad coverage; it does not guarantee the mathematical minimum or independent examples.

**Build from reviewed sentences** uses those slices in the existing word decoder. **Check aligned validation words** fits only training sentences and scores words from accepted held-out validation sentences; it shows top-1/top-5 and how much validation text had usable estimated timing and dictionary entries. This is an assisted boundary diagnostic, potentially within the same recording visit, not continuous sentence transcription accuracy or separate-session generalization. Continuous sentence decoding and the sentence final-test runner are not implemented; the reserved 25% remain untouched for that future test.

Sentence state is separate at `~/Library/Application Support/Subvox/sentence-training.json`. Raw visits use `~/Documents/Subvox/Sessions/`; each successfully aligned pair also gets a `sentence-pair-<UUID>.json` sidecar. The original `dataset.json` remains the word experiment.

## How decoding works

1. **Segmentation** (`CaptureController`). Space-hold, or an energy trigger above a calibrated noise floor with pre-roll and hangover.
2. **Features** (`MotionFeatures`). Per frame: user acceleration (3), rotation rate (3), and attitude change since the start of the segment (3). Each channel is mean-centered, and quiet lead-in and lead-out frames are trimmed.
3. **Sound models** (`PhoneticDecoder`). Each training take's phones (from the pronunciation dictionary) are spread over its active motion, vowels weighted double. Each phoneme's model is its medoid example. Channels are z-scaled across the training set.
4. **Spelling words** (`PhoneticDecoder`). Every vocabulary word (`Corpus/words.tsv`) gets a template: its phonemes' models joined end to end. A segment is DTW-matched (Sakoe–Chiba band, early abandoning) against templates of similar length, with a small *common-word bias* to break ties between words the jaw can't tell apart. A segment is rejected if the best score is above *max distance* or the margin to the runner-up is below *min confidence*.
5. **Evaluation** (`DecoderEvaluation`). The shared `MotionWordDecoder` facade is used by both Decode and evaluation. Reports top-1, top-5, acceptance and accepted-correct rates on validation word groups. Missing sounds, out-of-vocabulary words and rejected predictions remain in the denominator. The checklist measures collection coverage, not phoneme accuracy.

## Experimental learning methods

The learning-method council (modeling, evaluation skeptic, engineering) reviewed the supplied Claude research report and agreed on a small native sequence model as an **experimental challenger**, retaining the original default. The report's results concern different sensors/tasks; this implementation is a reasoned small-data experiment, not a reproduction of MuteIt or a proven AirPods result. The strongest objections were self-reinforcing guessed alignments, unverified coarse units, sparse independent sessions and incomparable score scales.

Train offers four methods:

- **Original phone templates:** the existing per-phone medoid/DTW baseline.
- **Coarse jaw templates:** the same guessed boundaries and DTW, but pooled into five existing vowel jaw classes plus one consonant class (`jaw6-v1`). This isolates the effect of pooling. These are hypotheses, not established sound equivalences.
- **Jaw sequence, initial alignment:** tied diagonal Gaussian unit emissions with a left-to-right self-loop/advance topology. Weighted duration guesses initialize the model; no realignment iterations are applied.
- **Jaw sequence, learned alignment:** the same model with up to six transcript-constrained Viterbi realignment/refit iterations. Variance floors and pooled shrinkage reduce degenerate estimates; the best finite training-objective checkpoint is retained. Transcripts constrain training only. Decoding scores every candidate pronunciation without consulting the expected word. Inferred boundaries remain estimates.

**Compare learning methods** runs all four against fixed common-word prefixes of 100, 1,000 and the full vocabulary. All configurations use the same fitting/validation takes and zero frequency bias, with a separate frequency-only baseline. No expected test word is injected into a candidate list. Coverage and conditional within-list counts are shown alongside all-take accuracy. Unit-sequence collisions expose words the selected representation cannot distinguish. Comparison does not automatically promote a model or touch the sealed final test.

Sequence scores use their own maximum score and minimum runner-up gap controls in Decode. DTW thresholds do not calibrate these scores, and displayed margins are not probabilities. Tune rejection on development validation before freezing. Saved reports include method/configuration, candidate coverage, collisions and hashes of fitting recordings (including order and pronunciations), vocabulary/pronunciations/counts, and settings. Final-test resume checks that manifest before scoring. Old settings/reports decode as the original method with the full vocabulary, including frozen experiments.

The offline benchmark avoids launching the GUI and never scores final-test takes:

```sh
Tools/benchmark_learning.sh /path/to/dataset.json /tmp/subvox-validation.json
```

Neural CTC/self-supervised pretraining and forced phoneme alignment remain future experiments requiring more independently recorded data. The paired workflow below provides experimental word-timing transfer, without assuming it works reliably for AirPods. Transfer from voiced to silent signals is demonstrated for facial EMG in [Gaddy & Klein (2020)](https://aclanthology.org/2020.emnlp-main.445/); it does not establish that AirPods can do the same.

## Experiment protocol

Use **Record Session** in the toolbar or on the Sensors screen. Recordings go to `~/Documents/Subvox/Sessions/<timestamp>-<condition>/`:

- `motion.csv`: every motion sample. `t` is seconds from session start.
- `audio.caf`: microphone audio, if the mic is running.
- `events.csv`: every Space-held segment, with its label or prediction.
- `meta.json`: condition, prompt, and the audio start offset for aligning audio with motion.
- `labels.csv` (Read sessions only): `tier,start,end,label,source`, with tiers `line`, `word`, `phone` and `jaw`, and source `speech`, `estimated` or `prompt`. Times are seconds from session start, the same clock as `motion.csv`.

For checklist training, choose a speaking condition and session notes, then hold Space for each prompt. New takes retain a session ID, condition, measured sample rate and quality review decision. A stopped session retains its end time; a session interrupted by quitting has no end time. Legacy takes retain unknown session/condition metadata.

Quality review flags recordings shorter than 0.35 s or 12 samples, longer than 6 s, sensor gaps (over 120 ms or 3× the median interval), very low motion energy, or unusually large motion. These are heuristic review hints, not calibrated jaw/head discrimination. Flagged takes are quarantined until accepted or excluded; invalid sensor values or non-increasing timestamps cannot be accepted. The live motion plot stays above the scrolling training content. Each saved take retains switchable rotation, acceleration and attitude plots.

The experiment assigns **25% of word groups** to the final test (rounded up); the other 75% is development data, with a quarter of those groups used for validation (roughly 56% training / 19% validation / 25% final overall). Assignments persist by normalized word, including skipped/deleted words and future re-recordings. Percentages apply to allocated word groups, not necessarily eligible takes after exclusions or condition filtering. Validation/final words are never prompted as fitting data. Normal, whisper and silent conditions are evaluated separately; legacy takes with unknown condition remain eligible and are explicitly disclosed.

Run **validation** to measure the same decoder used by Decode; only accepted training takes fit it during development. Validation errors can request new word contexts; they never diagnose an individual phoneme. Validation reports are invalidated by data/settings/review changes. Raw data and reports stay local.

Only press **Freeze and run final test** when finished tuning. All pending reviews must be resolved first. The experiment archives its data snapshot, locks collection/review/settings, refits on accepted development takes, and evaluates the reserved final takes once. Its report is saved in the dataset; reopening displays that report rather than rerunning the test. If processing is interrupted, the frozen test can resume with the same data/settings. Existing recordings were used before this split existed, so a historical holdout is not an independent fresh benchmark. Word grouping prevents same-word leakage, but does not measure generalization to entirely separate recording sessions. **Archive and start fresh** saves the whole previous dataset to a local experiment archive and begins with empty data; it never repartitions previously used takes into a new supposedly fresh test.

The training set lives at `~/Library/Application Support/Subvox/dataset.json`.

## Is there usable signal?

The **Signal check** screen provides a controlled acquisition visit. Six already-assigned training words are shown across three randomized instructions: speak aloud, mouth silently, or read internally without moving mouth/head. Every round contains all word/condition combinations. The comfortable pace records a 2-second blank baseline followed by a 5-second cue and 2-second recovery; extra time uses 3/7/2 seconds. The cue fills and centers in the recording area, with amber Relax, green Speak, blue Mouth and purple Think labels and icons; future words are hidden during the baseline. This removes per-trial Space presses and word-block order as obvious confounds. Stop keeps complete trials and discards the current unfinished one. Repeat the visit after reseating the AirPods to enable a separate-session test.

Diagnostic takes retain raw samples, exact monotonic cue start/end, speaking condition, instruction purpose, visit ID and measured rate. They are explicitly excluded from normal model fitting/validation/final scoring, even if accepted in quality review. They do not change existing word-role assignments. A mixed-condition recording folder retains synchronized audio, motion and events. Whisper checks spoken references only; it never produces silent labels or verified phone boundaries.

The local adapter discovers OpenWhispr's `~/.cache/openwhispr/whisper-models/ggml-base.bin` and `/Applications/OpenWhispr.app/Contents/Resources/bin/whisper-server-darwin-arm64`; another GGML Whisper model/server can be chosen in the UI. It reuses those existing files rather than bundling a copy. Each transcription starts its own server bound to `127.0.0.1`, uses a hint-free request, rejects redirects, and stops only its own child process. No models/audio are downloaded or sent to a cloud API. CAF audio is converted locally to 16 kHz mono WAV, and transcript/estimated word times plus `voice-verification.json` are saved beside the recording. ONNX Parakeet models are not accepted as Whisper models.

The offline diagnostic fits/scores **development data only**:

```sh
Tools/audit_signal.sh /path/to/dataset.json /tmp/subvox-signal-audit.json
```

It compares whole-word leave-one-out DTW without phoneme labels, acceleration/rotation/attitude ablations, cropping/trimming, pre-cue motion, recording-time and duration controls, shuffled-label nulls with feature-search correction, and compositional decoding on the known repeated-word task. Once dedicated probes exist, the primary repetition test uses their silent takes rather than mixing historical word blocks into the new experiment. Mouth-still controls and explicit cross-session results are separate. All these are exploratory diagnostics, not production validation or final-test accuracy. Time-block confounds and unknown historical sessions prevent conclusions about jaw specificity from old data alone.

## Known limits

- Headphone motion is processed and low-rate (about 50 Hz on AirPods Pro; check the toolbar badge for yours). The speech-detecting accelerometer and the inward mic are not exposed. Expect jaw and head movement, not vocal vibration.
- Templates are user- and session-specific. Re-seating the AirPods can shift the signal.
- Choosing AirPods as the mic input switches them to the Bluetooth headset (HFP) profile.

## Reading passage

`Corpus/harbor.txt` is a 200-word passage written to cover every structure the decoder might learn from: all 39 English phonemes (each at least 3 times), all 23 Wells lexical sets that General American distinguishes, every jaw-height class and all 25 transitions between them, 18 onset and 18 coda clusters, and words of 1–6 syllables. The standard passages (Rainbow, North Wind, Please Call Stella) each miss the NEAR and CURE vowel sets and 8–12 of the jaw transitions.

`Corpus/diphones.txt` tops that up to full **diphone** coverage: 235 short sentences (about 12 minutes read aloud) that, together with the passage, contain every one of the 858 phone pairs that make up 99% of running English, almost all of them at least twice. It is built the way TIMIT's phonetically compact sentences were: greedy set cover over about 12,500 natural sentences from public-domain Project Gutenberg books, filtered for names, dialect spellings and offensive period language.

```sh
Tools/coverage.py Corpus/harbor.txt --missing   # coverage report for any passage
Tools/diphone_bank.py --reps 2                  # rebuild the sentence bank
Tools/lexicon.py                                # refresh Corpus/lexicon.tsv after editing a text
```

The CMU Pronouncing Dictionary and the pool books are downloaded to `~/Library/Caches/Subvox` on first use.
