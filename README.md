# Subvox

A macOS prototype exploring whether AirPods motion can recognize silently mouthed words. Train on paired spoken and silently mouthed passages, review their motion alignment, then check the resulting personal word model.

## Build and run

Requires macOS 14+, Xcode and XcodeGen. AirPods must support head tracking and be connected as the Mac’s audio output.

```sh
xcodegen generate
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build \
  CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM= CODE_SIGN_STYLE=Manual \
  -configuration Release build
open build/Build/Products/Release/Subvox.app
```

Use Release: matching thousands of vocabulary templates is much slower in Debug. Grant Motion and Microphone access when requested. All transcription uses existing local Whisper files; no speech-recognition service is used.

## The app

- **Train:** collect paired passages, inspect spoken/mouthed traces and transcripts, accept or skip pairs, fill remaining sound gaps, and check validation words. Models rebuild automatically from accepted training pairs.
- **Try model:** mouth one word while holding Space, or calibrate automatic capture. Recognition thresholds and candidate details sit inside disclosures. It starts with a training invitation when no model exists.
- **Devices:** inspect AirPods status, measured sample rate and motion channels, and check microphone input. Recording belongs to the training workflow; there is no separate recorder to compete with it.

Large prompts use amber for Relax, green for Speak and blue for Mouth. Short passages join up to three sentences. Comfortable prompts last at least 12 seconds; Extra time uses at least 16 seconds. Longer passages receive up to 45 seconds. Speak/mouth each prompt once naturally, then wait. Live motion remains visible throughout acquisition and review.

Recording settings hold the batch size, microphone, notes and local model paths. Use the built-in microphone to avoid changing AirPods’ Bluetooth audio mode. The old word checklist, controlled signal experiment, reading teleprompter, legacy dataset and model-source switch have been removed.

## Training and validation

The authored 32-sentence corpus has a persistent split: **18 training, 6 validation, 8 final-test sentences**. Sentence identity ignores punctuation/case, so repetitions keep their assignment. Passage groups never cross splits. The final 25% stays out of collection plans, fitting, coverage, repairs and validation.

Local Whisper transcribes each spoken cue without expected-text hints. At least 80% of prompt words must match in order. No missing word receives an invented timing. Full-path DTW transfers estimated word boundaries from spoken motion to mouthed motion using pair-local scaling; it rejects flat/invalid data, sampling gaps, invalid anchors and excessive warp. Boundaries remain annotation estimates. Review both traces and the transcript before accepting.

Each completed half immediately saves its recording identity, directory and monotonic audio clock. Recorder metadata is checkpointed during collection. Stop and normal app termination close the audio file and retain completed halves. An interrupted current cue does not become a complete take. Each visit retains microphone, notes, pace, measured rate and cue times.

Only accepted training pairs supply silent word examples. Coverage counts dictionary phoneme occurrences, targeting four per unit; dictionary pronunciations may differ from a mixed accent. Single-word repairs appear only after training sentences have been reviewed and only for remaining coverage gaps. Validation/final examples do not influence repairs.

The word check fits the same decoder used in Try model and scores accepted validation word slices. It reports top-1/top-5, acceptance, correctly emitted words as a fraction of all evaluated slices, and how much validation text had usable annotation. Correctly ranked but rejected candidates are not correctly emitted words. The check uses estimated boundaries and may share a recording visit with training; it does not measure continuous sentence transcription or independent-session generalization. Continuous sentence decoding and its final-test runner remain unimplemented; the reserved sentences stay untouched.

Model settings offer motion templates and an experimental sequence model. Motion templates allocate each word’s phones across its motion and match synthetic word templates by DTW. The sequence model ties diagonal Gaussian emissions to six hypothesized motion units and learns transcript-constrained timing with up to six Viterbi refits. Both retain estimated within-word boundaries. Sequence scores, DTW distances and confidence margins are uncalibrated; tune thresholds on validation data. Neither buildable vocabulary size nor collection coverage establishes recognition accuracy.

## Local files

- Active training: `~/Library/Application Support/Subvox/sentence-training.json`.
- Raw recordings: `~/Documents/Subvox/Sessions/`, containing audio, motion, events, metadata and aligned-pair sidecars.
- Recovery archives: `~/Documents/Subvox/Archives/`. These are never loaded as active training.
- Existing OpenWhispr files: `~/.cache/openwhispr/whisper-models/ggml-base.bin` and `/Applications/OpenWhispr.app/Contents/Resources/bin/whisper-server-darwin-arm64` by default; alternate GGML models can be selected.

Whisper runs in an owned child process bound to `127.0.0.1`. Requests contain no expected-text prompt, refuse redirects, and convert CAF audio locally to 16 kHz mono WAV. Model files stay in their existing location.

To start clean again, close Subvox, inspect the reset paths, then archive active training:

```sh
python3 Tools/reset_training.py
python3 Tools/reset_training.py --execute
```

Reset archives datasets, prior snapshots and recording folders with SHA-256 verification, recreates empty active folders, and retains local Whisper model files/preferences. Failed moves roll back; no permanent deletion is performed.

## Checks

```sh
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build \
  CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM= CODE_SIGN_STYLE=Manual \
  -configuration Release ENABLE_TESTABILITY=YES test
python3 -m unittest discover -s Tools -p test_reset_training.py
```

Tests cover word matching, timing transfer, sentence-role isolation, annotation review, rejection metrics, model parity, recorder ownership, recovery metadata, settings persistence and archive rollback. Removed workflows no longer carry compatibility tests.

## Corpus tools and limits

`Corpus/paired-sentences.txt` supplies paired prompts. The pronunciation/vocabulary tables support fitting and decoding. `harbor.txt` and `diphones.txt` remain reference material for corpus development; the retired teleprompter does not run them.

```sh
python3 Tools/coverage.py Corpus/paired-sentences.txt --missing
python3 Tools/diphone_bank.py --reps 2
python3 Tools/lexicon.py
```

Corpus tools cache pronunciation and public-domain source material under `~/Library/Caches/Subvox`. Those shared resources are not personal training examples.

AirPods motion is typically about 50 Hz and includes head movement as well as articulation. The speech accelerometer and inward microphone are not exposed. Models are personal and may change when the AirPods are reseated. This remains an experiment in signal quality and word recognition.
