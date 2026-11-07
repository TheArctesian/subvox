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

## Repository layout

```
Subvox/
  App/            entry point, AppModel, ContentView (sidebar)
  Capture/        Motion (AirPods IMU), Audio (engine/devices), Camera (pose + mouth landmarks)
  Recording/      SessionRecorder: archival session folders (motion.csv, audio.caf, …)
  Transcription/  local Whisper server + word timing
  Linguistics/    IPA inventory, pronunciation graph, lexicon, phoneme table, MFA aligner
  Alignment/      LCS, spoken→mouthed motion DTW, camera phone-boundary transfer
  Collection/     paired sentence collection: data model, store, workflow, take quality
  Core/           shared enums, deterministic RNG, word-focus pacing
  Sequence/       Swift driver for the neural pipeline in ml/
  Baselines/      historical pilots: WordDecoders (DTW/jaw HMM), SignalCheck + camera teacher, Representation
  Views/          SwiftUI, grouped like the folders above (Shared/ holds reusable components)
SubvoxTests/      XCTest, mirrored by area
ml/               subvox_ml: PyTorch CTC pipeline, bundled into the app as SequenceRuntime
Corpus/           bundled sentence banks and lexicons
Tools/corpus/     offline corpus generators (coverage, diphone bank, lexicon)
Tools/data/       data-ops utilities (reset/archive training data)
```

## Sentence recording

**Train** opens ordinary paired sentence collection using the recording interface from block A: preview each sentence, choose Start, wait through Relax, speak or mouth with highlighted words, choose Done, then review END before Next. Left/Up redoes the current take; Right/Down starts, finishes or advances. The large coloured prompt, centred timer above START/END, live motion and numeric camera markers stay visible. Pacing is a visual guide saved as provenance, never measured word/phone timing. Source switches or capture gaps discard an interrupted attempt and return to Ready. There is no mandatory A/freeze/reseat/B sequence in collection.

Choose Training or Development before a visit. Each visit uses a single role and fresh sentence groups; partial or completed sentences are not scheduled again automatically. Re-record from review when needed. Existing frozen-study sentences are excluded from this queue. **Add the 235-sentence diphone bank** expands the available corpus while preserving every existing role and reserving 25% of the expanded sentence set. It does not collect or expose the reserved final prompts. The bank supplies varied sentences, not evidence of complete measured phone observability.

Sentence visits explicitly record microphone audio and motion, plus numeric camera landmarks when enabled, never video. The Mac microphone remains preferred. Every new completed spoken take goes through local Whisper; the pending-transcription action also handles saved spoken-only takes while leaving excluded/final pairs alone. Transcript agreement and estimated word timing are separate: a word-alignment failure can still leave a sentence-sequence candidate for CTC. Neither ASR agreement nor a visual word cue proves a usable motion signal.

**Record non-speech examples** collects nods/turns, empty-mouth chewing, natural swallowing and posture shifts, with the same centred prompts and next/redo controls. Animated native SF Symbols demonstrate each movement, with directional cues, neutral surfaces and restrained state accents. These are labelled examples, not camera feedback or completion detection; Reduce Motion uses static cues. Ready is cyan, Relax amber, movement purple and Saved green. The prompt scrolls within an explicitly bounded viewport; separate fixed sections keep the action bar and live plots visible. Stopped partial non-speech sessions can continue in a new recorder session with completed cells skipped; original records remain unchanged. These separate visits save motion only: microphone, camera and video are off. Each take has an empty text/phone target and an explicit movement group. Movement instructions never become transcripts. Valid hard negatives enter new training/development snapshots according to the chosen role; invalid negatives remain quality exclusions and cannot count as successful rejection. Existing models never change automatically.

**Advanced** opens neural training/results, saved frozen-model studies and motion checks. Existing A recordings and frozen fingerprints remain resumable there. Reload can recover an omitted embedded timing-policy field only if restoring that one field reproduces the complete original fingerprint from the separate frozen reference; changed weights/recordings still fail verification, and reload rewrites neither file.

## Neural motion → phonemes → words

Choose **Train → Advanced → Neural model training and results**. This is an actual trainable PyTorch convolutional/bidirectional-GRU network with a CTC phoneme head, alongside the existing template, jaw-sequence and representation pilots. It processes completed clips, not streaming partial speech. Training, inference and teacher processing run locally. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if unavailable; the app copies its versioned Python source and pinned lockfile into Application Support and prepares a runtime there. First setup downloads Python/packages; recording data is not uploaded. Xcode builds bundle source and the lockfile, never the development virtual environment.

The first implementation has five comparisons, fitted separately for spoken and silently mouthed recordings:

| Experiment | Learning objective |
| --- | --- |
| Plain | Sentence-sequence phoneme CTC |
| Diphone | CTC plus next-phone auxiliary prediction from soft CTC correspondences |
| Augmented | Diphone model plus training-only speed variation, small common vector rotations, noise and time/channel masking |
| Activity | Augmented model plus relaxed-mouth versus prompted-articulation auxiliary loss |
| Camera teacher, separate second stage | Diphone model plus confidence-weighted observed mouth-shape targets; original timelines, no teacher time warping |

Raw acceleration, rotation and gravity remain inputs, together with invariant summaries and temporal derivatives: 32 features on an observed 50 Hz timeline. The encoder provides temporal context. Training-only statistics standardise inputs; validation/fresh sessions supply no scaling, checkpoint fitting or language-model counts. Unknown/untrained sensor sides mask directional features and retain invariants; this is a fallback, not evidence of fit invariance. Source switches, long gaps, inadequate sampling and missing START/END coverage have explicit exclusions. Subtle low-energy mouthing stays eligible. No extrapolated END tail is manufactured.

The shared Swift linguistic graph exports editable speaker pronunciation alternatives, IPA, ARPAbet compatibility and articulatory attributes. CTC marginalises supported pronunciation sequences without guessed phoneme times. A soft estimated next-phone target models transitions without a quadratic diphone vocabulary. These posterior correspondences are model estimates, not measured boundaries. Phone and diphone coverage are separate; pronunciation proposals are not ground truth for a mixed-accent speaker. Broad gesture priors do not make voicing, tongue positions or every linguistic contrast sensor-observable. Existing exploratory sensor-overlap estimates remain separate from neural recognition accuracy.

Decoding exposes a free-phone prefix beam for phoneme error rate, plus a pronunciation-lexicon beam for words. A source-only word bigram model supplies a separately reported assisted result. Same-phone spellings are exposed as homophone alternatives; strict word error remains separate from homophone-merged error. Ties between different phone strings and insufficient phone evidence can reject output. Margins, activity scores and estimated temporal correspondences are uncalibrated. Expected text is used only for visual focus and scoring after inference, never motion features, emissions or search. Corrections are versioned label sidecars used by a future dataset/run; they do not silently retrain a frozen model.

### What the first real training run established

The initial snapshot trained **29 spoken and 28 mouthed utterances**, plus their relaxed-mouth baselines, for 60 epochs per comparison. Entire recording sessions and connected sentence groups were separated before preprocessing, with crossed cells quarantined. The development set contains one unseen sentence, “Pack my blue bag.”, with **2 usable spoken and 3 usable mouthed takes**. One additional spoken take failed recording-coverage checks. The initial historical report scores usable recordings and lists the exclusion separately; newly prepared snapshots also retain excluded planned takes in the error denominator.

**All ten fitted models have 100% development phoneme and word error, with and without the language model, and reject all positive development takes.** Falling training loss has not produced demonstrated unseen-sentence recognition. Zero negative false accepts here is trivial because everything is rejected. There is no defensible sentence-group bootstrap interval with only one development sentence. The separate camera stage used 16,387 valid training teacher frames and had no reviewed acoustic phone alignments or Whisper proposals in this snapshot; it showed no gain either. These are retrospective checkpoint-selection results, not final-test scores or prospective reseating evidence.

The matched raw whole-utterance DTW baseline also has 100% word error on this cohort; it cannot compose a sentence absent from its training bank. Historical word/jaw/camera baselines and their frozen reports remain available, but their different candidate sets and protocols do not support a direct improvement claim. Former diagnostic B takes included in this new training snapshot explicitly lose independent-test status for these new models only. Original recordings, references and earlier scores are preserved. No anatomical simulation, synthetic anatomy data or RLHF is used.

### Cross-validation and the handoff

New snapshots use schema 2: connected whole-sentence groups and entire sessions are kept apart within each block cross-validation fold, and crossed cells are quarantined per fold. Ordinary pairs explicitly excluded by the user stay out. Accepted pairs are eligible; pending pairs are candidates when Whisper word agreement is at least 85%, with review status, agreement and uncertain prompt-label provenance retained. Word-alignment failures do not by themselves block CTC. The CLI can select accepted-only admission. This is an annotation admission rule, not proof that the mouthed take was correct or the sensors carried speech.

Each condition/experiment fits fold-specific models and pools only their validation predictions. After choosing an epoch budget, a separate deployed model refits on all eligible recordings. Pooled scores describe the method's development folds, **not independent performance of those refitted weights**. Fresh tests must use new sentence groups and sessions and frozen weights. Final-test data never enters these folds or the refit. Original schema-1 snapshots/models are retained; prepare a new snapshot before new training rather than rewriting the old dataset.

The first completed four-fold handoff run has 15 validation utterances from four sentence groups per condition. It reports approximately 98.6–100% phoneme error, 100% word error with/without LM and homophones merged, and 0–21% false accepts on scored rest clips. Its 5–7-epoch refits came from the earlier stopping policy. It establishes no useful recognition. Its completed dataset and fold/refit models were copied into `datasets/imported-cv-20261107` and `runs/imported-cv-20261107` with unchanged weight files and explicit import provenance. The source run remains untouched.

The other agent's later decoder/optimiser changes are integrated as source: new phone heads omit the word separator, lexical search supplies boundaries and ends on complete words, next-phone transition assistance is reported separately, and CTC/posteriors are batched. Legacy separator checkpoints retain a compatibility path. Minimum-epoch/early-stopping changes and a rejection threshold fitted on training rest clips are **not measured by the first completed CV report**. The subsequent `cv-run2` comparison remains a separate background experiment; no improved score is claimed before that run finishes.

Use the main **Train** screen to collect varied paired sentences. Prepare a new snapshot under Advanced, then train. Fresh same-fit/reseated visits in Advanced remain optional evaluations and never flow into training automatically. Those tests use different available non-final groups, not a randomized causal test of reseating. Missing/invalid positive takes count as errors; recording/label exclusions remain visible. The eight original final sentences retain their roles; expanding the corpus adds a 25% reservation for the enlarged set. No final-test runner is enabled.

The optional second-stage controls route to explicit voiced/camera **training** or **development** paired collection. These record microphone audio and numeric camera points, never video. Local Whisper proposes transcripts/quality; review and correction stay in pair review. The separate optional local acoustic aligner described below retains estimated timing, confidence and provenance; Whisper word durations are never converted into measured phone boundaries. Mouthed teacher intervals use a separate camera warp and can differ in pacing. The neural camera auxiliary uses eligible training-frame landmarks only; neither audio nor camera is required at inference. Acoustic phone annotations are preserved as fallible provenance, not yet an additional neural timing loss. Self-supervised unlabelled-motion pretraining is deferred.

### Reproduce locally

Artifacts live under `~/Library/Application Support/Subvox/SequenceLearning/`: dataset snapshots, model weights, pronunciation snapshots, feature schema, training/development identities, source hashes, coverage, training curves, seeds, dependency versions, source-code snapshots and reports. `initial-dataset`, `initial-neural-run`, `initial-teacher-dataset` and `initial-teacher-run` contain the measured runs above. The app exports a fresh `language.json` from the production Swift graph when preparing a dataset; an earlier snapshot can be reused for a repeat with identical pronunciation proposals.

```sh
uv run --project ml python -m unittest discover -s ml -p test_sequence.py
uv run --project ml subvox-ml prepare --root "$HOME/Library/Application Support/Subvox" \
  --language /path/to/language.json --output /path/to/new-dataset
uv run --project ml subvox-ml train --dataset /path/to/new-dataset \
  --output /path/to/new-run --epochs 60
uv run --project ml subvox-ml infer --model /path/to/new-run/plain-mouthed \
  --input /path/to/input.json --output /path/to/prediction.json
uv run --project ml subvox-ml evaluate-fresh --session /path/to/closed-test/session.json
uv run --project ml subvox-ml prepare-teachers --dataset /path/to/new-dataset \
  --base-run /path/to/new-run --output /path/to/new-teacher-dataset
uv run --project ml subvox-ml train --dataset /path/to/new-teacher-dataset \
  --output /path/to/new-teacher-run --experiments teacher --epochs 60
```

Inference input is `{"recording": <SentenceRecording JSON>, "expectedText": "optional reference"}`. Preparation accepts `--corrections`, `--folds` and `--min-transcript-match` (negative for accepted-only); output directories must be new. Re-evaluation requires `--fold` and a new `--output` path so frozen reports are preserved. Reproducible source snapshots accompany each model run. Algorithm/gradient tests establish implementation behaviour, not physical sensor accuracy. The architecture uses [PyTorch CTC loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html) and [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html); [Sequence Modeling with CTC](https://distill.pub/2017/ctc/) explains alignment marginalisation and prefix search.

## Raw + camera + linguistic representation pilot

Choose **Train → Advanced → Saved frozen-model studies** for the separate guided representation workflow. This is a separate, versioned experiment; the earlier three-phrase camera-teacher comparisons, recordings, frozen models and ordinary word decoder remain available. The latest completed three-phrase study tied in its primary mouthed linear comparison (raw 5/9, camera-taught 5/9). Raw temporal matching reached 8/9 versus 2/9 for the compressed student, and the reported AirPod source changed during B. Those development observations motivate retaining raw channels; they do not establish a new model's accuracy.

The small pilot collects four **training** sentences twice, spoken and mouthed (16 takes in A), then four different **validation** sentences twice in a fresh recording session after reseating (16 takes in B). Sentence identities are normalised hashes, paragraphs are not split into independent examples, and the entire B session is excluded from fitting. The existing eight reserved final-test sentences (25%) cannot enter collection, candidate banks, training, coverage repairs or evaluation. B is a development transfer test, not permission to tune on the reserved final test. Existing recordings are never relabelled or deleted.

Prompts stay large and colour coded: orange baseline, green speech, blue mouthing. Word focus advances every 0.65 seconds after a 0.4-second lead-in and holds the last word until Done; it resets on each spoken/mouthed take and redo. The timer sits centred directly above START/END, and long sentences wrap without moving words as the highlight changes. Focus settings are saved as visual-guide provenance only, never measured word/phoneme boundaries or decoder inputs. Move your head comfortably with your face visible. Choose **Done / next** as soon as you finish; **Redo take** repeats mistakes. Right/down advances, left/up redoes; key repeats and text editing do not carry actions into subsequent takes. Live motion and numeric camera markers stay visible. Diagnostics explicitly stop the microphone and record **only numeric points and motion**, never audio or video.

The original directional acceleration/rotation context at −240/−120/0 ms is retained alongside gravity-invariant context: 39 channels per frame. The three learned lip-velocity channels are **appended**, not substituted. A second auxiliary branch learns articulatory gesture summaries. All arms use the same fixed-ridge decoder method and the same two-half mean/standard-deviation pooling, although augmented arms have more input dimensions. The comparisons are raw; raw plus camera teaching; raw plus camera and linguistic supervision; and raw plus a half-sequence-shifted camera teacher. All coefficients, candidate signatures, pronunciation snapshots and scaling are frozen after A. Inference receives AirPods data alone; B camera frames cannot change an AirPods prediction. Linear auxiliary predictions add an inductive bias, not new sensor information at inference. There is no hyperparameter search in this pilot.

The auxiliary camera learner retains relaxed-mouth baselines as zero mouth-motion targets and a fixed 0.5 penalty on covariance with camera head movement during those baselines. Eye centre, spacing and roll supply the head reference. The shifted teacher moves only cue targets, preserving identical baseline treatment. This nuisance penalty changes the auxiliary predictions only; it never subtracts or overwrites the raw channels. It is an experimental separation mechanism, not proof that head-motion confounding is gone.

Core Motion side changes are discontinuities. Acquisition flags and retains interrupted attempts and requires a redo after at least 1.5 seconds of stable source data. It never interpolates across a switch or gap. Full directional heads are trained separately for each reported source when enough complete examples exist. An untrained side uses a clearly reported, source-independent **gravity-invariant fallback**, with no B fitting or inferred axis mirroring. Raw axes remain recorded. Fallback use and scored/missing/ambiguous counts are shown explicitly against all planned takes. Neither invariant features nor a fresh fit are assumed to solve reseating automatically.

### Linguistic labels and observable movement

`ArticulatoryInventory` maps all 39 existing ARPAbet units into IPA and explicit place, manner, voicing, vowel height/backness/rounding and diphthong trajectories. Stress-bearing `AH0`/`ER0` map to schwa/rhotic schwa rather than stressed vowels. The editable speaker profile supports alternative space-separated IPA sequences per word, including common regional vowel, length, tap and glottal variants. It does not infer a pronunciation from the speaker's family or choose General American as ground truth. The bundled dictionary is labelled an **unconfirmed proposal**. Unknown symbols/words remain unsupported, with no invented junction across an unknown word. This first inventory is not the complete IPA.

`LinguisticGraph` connects words, pronunciation alternatives, IPA phones, within-word and cross-word diphones, and broad articulatory gesture hypotheses. Alternative coverage is fractional expected annotation count, with equal variant weighting explicitly a placeholder. Diphthongs retain internal gesture transitions. Voicing and internal tongue positions are not declared measurable camera/IMU classes; /p b m/ may share a gesture prior. With no phone aligner, linguistic supervision is an explicitly **weak sentence-level** gesture/transition summary, never invented phone timing. The decoder retrieves among the predeclared A+B sentence bank, reports ties/alternatives and uncalibrated margins, and counts ambiguous outputs as incorrect.

Reviewed phone annotations from accepted, single-sentence **training-only** pairs can enter the same representation learner when their frozen pronunciation profile matches. These enable time-local gesture supervision and exploratory per-modality sensor-overlap estimates. Overlap uses observed camera shape or invariant AirPods features, requires multiple sentence contexts and pools occurrences rather than treating frames as independent phone examples. Similar distributions retain both IPA hypotheses; different sample means do not establish reliable phoneme separation. With no reviewed annotations, the UI explicitly says that these empirical phone-level estimates are unavailable. Training coverage and actual development errors generate targeted example requests without changing the frozen model or final reservation.

### Optional local acoustic phone alignment

In normal paired sentence training, expand **Optional local phoneme alignment and camera points**. Whisper remains responsible for word-level audio estimates; it is not a phoneme aligner. The optional adapter supports the MFA 3.x legacy `align_one` interface with an **already local acoustic model file**, local executable and explicit JSON mapping of acoustic-model phone symbols to IPA, for example `{"P":"p","AE1":"æ","K":"k"}`. The dictionary exported for alignment includes all editable pronunciation alternatives; the phone map must agree with the acoustic model. Nothing installs or downloads automatically. MFA and a compatible acoustic model are not bundled or currently installed on this development machine.

The adapter aligns the cropped spoken audio locally and keeps its raw output, exact cropped-audio/model/map/profile hashes, audio clock, word association and matching pronunciation variant IDs. Acoustic boundaries remain **estimates**; absent confidence stays absent, and backend likelihoods are not rebranded as probabilities. Failure leaves the existing word alignment intact. Word-duration spreading is never accepted as acoustic phoneme timing. The adapter's parser/process integration is tested with fixtures; real acoustic alignment quality still needs verification with the chosen local MFA installation/model and this speaker's accent.

Numeric camera capture during ordinary sentence training is opt-in. When available, mouthed phone intervals are estimated separately by a monotonic camera-shape warp from spoken to mouthed video **points**, allowing different pacing and using no mouthed audio evidence. The warp records its source alignment, cost and uncalibrated confidence; it does not use IMU data to manufacture IMU labels. Inspect both estimates in pair review before marking them reviewed. These phone sidecars never overwrite existing recordings or word boundaries. Normal spoken sentence training continues recording microphone audio; diagnostic pilots never do.

This pilot establishes infrastructure and a prospective comparison, **not** continuous transcription, phoneme recognition, pure jaw sensing or isolated unseen-word accuracy. The report counts B word types absent from A, but its metric is sentence retrieval containing those words. Four held-out sentence groups, one person and A-then-B order are insufficient to isolate a causal reseating effect from time/practice or to establish general vocabulary performance. A future continuous decoder and reserved final-test runner remain separate work. No new measured accuracy is claimed until new recordings are evaluated.

The IPA descriptions follow the [International Phonetic Association chart](https://www.internationalphoneticassociation.org/content/chart). Local alignment interfaces and pronunciation alternatives follow the [MFA alignment reference](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) and [dictionary format](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html); MFA 4 changes these command signatures and is not silently assumed compatible.

Pilot files live under `~/Library/Application Support/Subvox/RepresentationStudies/<study-id>/`: an immutable `frozen-reference.json` and a separate `study.json`, with `latest.json` for navigation. Source takes have a saved content digest, and model fingerprints are checked before B, on reload and before scoring. `pronunciation-profile.json` affects future studies only; each reference owns its snapshot. Earlier `controlled-signal-test.json`, `frozen-signal-reference.json` and sentence-test reservation files are not modified by the pilot.

Completed A blocks that failed during fitting can use **Fit saved A and continue to B**. Recovery preserves an exact `before-refit-<id>.json` backup and reuses the saved recordings. The observed-tail feature policy uses only available motion grid points when capture already passes the 100 ms END-coverage check and at least 90% of the cue is supported; internal gaps, source switches and larger missing tails remain rejected. It never extends samples or moves START/END. Older frozen representation models retain their original strict-grid evaluator when the feature-policy field is absent; a recovery fit records the new policy explicitly. B remains a separate prospective session and is never used to repair or fit A.

## The app

- **Train:** collect paired passages, inspect spoken/mouthed traces and transcripts, accept or skip pairs, fill remaining sound gaps, and check validation words. Models rebuild automatically from accepted training pairs.
- **Try model:** mouth one word while holding Space, or calibrate automatic capture. Recognition thresholds and candidate details sit inside disclosures. It starts with a training invitation when no model exists.
- **Devices:** inspect AirPods status, measured sample rate and motion channels, and check microphone input. Recording belongs to the training workflow; there is no separate recorder to compete with it.

Large prompts use amber for Relax, green for Speak and blue for Mouth. The current recording screen collects individual sentences, one role per visit; the older passage acquisition implementation remains available internally. Comfortable timers allow at least 12 seconds; Extra time allows at least 16 seconds. Longer passages receive up to 45 seconds. Speak/mouth each prompt once naturally, then choose **Done / next**, press **Right Arrow** or **Down Arrow**, or reach the safety timer. Ready and END wait for explicit Next in the current recording screen. Finishing early records the actual press time and omits later motion from that take. Use **Redo take**, **Left Arrow** or **Up Arrow** after a mistake, including from the END screen. It restarts just the current spoken/mouthed half after a fresh Relax baseline and replaces any already-saved attempt, keeping the other half. Raw redo events remain auditable; discarded attempts never enter fitting. Relax/recovery pauses cannot be fast-forwarded, and held arrows do not queue future actions. Live motion remains visible throughout acquisition and review.

Recording settings hold the batch size, microphone, notes and local model paths. The app selects an available built-in Mac microphone on first device discovery, without changing the macOS system default. Explicit input choices survive refreshes for the current app session. Use it to avoid changing AirPods’ Bluetooth audio mode. The old word checklist and reading teleprompter remain removed. Historical controlled/frozen comparisons are kept under Advanced; the main recording screen does not require them.

## Training and validation

The original authored 32-sentence corpus has a persistent split: **18 training, 6 validation, 8 final-test sentences**. Sentence identity ignores punctuation/case, so repetitions keep their assignment. Passage groups never cross splits. The final 25% stays out of collection plans, fitting, coverage, repairs and validation.

Local Whisper transcribes each spoken cue in English without expected-text hints. Audio-attention DTW is enabled for standard tiny/base/small/medium/versioned-large models (including supported quantizations), with flash attention disabled. The adapter uses `t_dtw` token locations in 10 ms units, groups token fragments by the rendered transcript, estimates word boundaries between attention locations, and checks those locations against the recorded audio energy. Adjacent words sharing an attention location, unsupported/quiet locations, and unavailable timing remain untimed. Segment-only timing is never used for new labels. These are acoustic estimates, not verified forced phoneme boundaries. Expand **Check word times against audio** to inspect the waveform and play individual estimated intervals. Raw backend responses and the timing review are saved locally.

For the legacy aligned-word decoder, text agreement and usable timed-word coverage are checked separately: each must cover at least 80% of prompt words in order. The separate CTC sentence admission rule above does not require word timings. No missing word receives an invented timing. Full-path DTW transfers estimated word boundaries from spoken motion to mouthed motion using pair-local scaling; it rejects flat/invalid data, sampling gaps, invalid anchors and excessive warp. Boundaries remain annotation estimates. Review both traces and the transcript before accepting.

Each completed half immediately saves its recording identity, directory and monotonic audio clock. Recorder metadata is checkpointed during collection. Stop and normal app termination close the audio file and retain completed halves. An interrupted current cue does not become a complete take. Each visit retains microphone, notes, pace, measured rate and cue times.

Only accepted training pairs supply silent word examples. Coverage counts dictionary phoneme occurrences, targeting four per unit; dictionary pronunciations may differ from a mixed accent. Single-word repairs appear only after training sentences have been reviewed and only for remaining coverage gaps. Validation/final examples do not influence repairs.

The word check fits the same decoder used in Try model and scores accepted validation word slices. It reports top-1/top-5, acceptance, correctly emitted words as a fraction of all evaluated slices, and how much validation text had usable annotation. Correctly ranked but rejected candidates are not correctly emitted words. The check uses estimated boundaries and may share a recording visit with training; it does not measure continuous sentence transcription or independent-session generalization. The separate experimental CTC decoder above handles completed sentence clips; this legacy word check does not evaluate it. The reserved final-test runner remains disabled and reserved sentences stay untouched.

Model settings offer motion templates and an experimental sequence model. Motion templates allocate each word’s phones across its motion and match synthetic word templates by DTW. The sequence model ties diagonal Gaussian emissions to six hypothesized motion units and learns transcript-constrained timing with up to six Viterbi refits. Both retain estimated within-word boundaries. Sequence scores, DTW distances and confidence margins are uncalibrated; tune thresholds on validation data. Neither buildable vocabulary size nor collection coverage establishes recognition accuracy.

## Repeated sentence signal check

From Train, choose **Advanced → Motion signal checks**. Three independent four-word phrases are repeated three times, spoken then mouthed, with sentence order shuffled within each repetition block. They never enter model fitting or the 32-sentence split.

Preview the phrase, choose **Start take** (Right/Down), and remain still through the three-second amber baseline. Wait for the large **START** cue, say/mouth the phrase once, then choose **Done** immediately to save an explicit **END** marker. Left/Up redoes the current take; discarded attempts stay in the audit file. The END screen waits for Next, allowing review before proceeding. The 12-second safety timeout saves the take but excludes it from comparisons. Stop saves completed takes; incomplete current takes are not scored.

Results compare spoken and mouthed whole-phrase repeatability using nearest-neighbor motion DTW, holding out the entire repetition block each time. A half-clip circular time shift provides an exploratory timing control. Three-way chance is 1/3; at most nine evaluated takes per condition are insufficient to establish reliable decoding accuracy. Head movement and pacing can also produce repeatability. Quality checks report marker coverage, sensor gaps, a still-baseline comparison, and the share of rotation energy in the strongest 1% of frames. Missing/gapped baselines, missing boundary coverage, invalid sampling and timeouts are excluded. Live motion and saved START/END plots remain visible. New signal checks are points/motion only: the microphone is stopped, no audio sink is attached, and Whisper is not run. Historical checks retain their existing audio/transcripts. Paired sentence training remains a separate spoken-audio workflow.

From a completed check, choose **Freeze this session for reseat test**. The app saves an immutable reference containing all 18 usable reference takes, source-only channel scales per condition, the phrase list, matcher version and fixed DTW settings. A SHA-256 fingerprint identifies it. Existing frozen references are reused rather than overwritten. Remove and reinsert both AirPods, wait for motion to reconnect, then choose **I’ve reseated the AirPods — start second session**. This records the user's reseating confirmation and starts a separate shuffled 18-take visit with the same START/END and redo controls.

Second-session results match each new clip only against the original condition's nine reference clips. Mean centering remains fixed per-clip preprocessing; channel scales are fitted entirely on the reference session. Features, time band, step penalty and templates stay frozen. No new test take becomes a reference, and predictions are withheld until acquisition ends. Results show correct matches out of nine planned takes per condition, scored/invalid/missing counts, unmatched valid takes and the shifted-motion control. Unmatched valid takes remain in the scoring denominator. Each take also shows the predicted phrase. This is a test on three known phrases after reseating, not open-vocabulary word recognition. Incompatible or changed references fail explicitly rather than substituting a within-session check.

### Camera-assisted reseat comparison

**Camera teacher · train A and test B** is the new default. It allows natural head movement and shows both halves explicitly: 18 source takes in A, then 18 fresh takes after reseating in B. A 30-second minimum rest begins when A's recorder closes. Word pacing varies between 0.5/0.65/0.8 seconds with 0.25/0.55/0.85-second lead-ins; each phrase sees all speeds across its source repetitions, and B uses a different schedule for each matched prompt. Each take saves its actual pacing prescription. Fixed A-then-B order still leaves time, practice and fatigue as possible confounds.

The new model learns three **mouth-shape velocities** (opening, width, outer aspect), rather than assuming mouth positions are directly recoverable from an IMU. The camera teacher uses a centred 160 ms derivative. Eye-centre translation, eye spacing and eye-line roll supply a continuous head-motion reference; the model does not differentiate the coarser face-box angles. During the relaxed-mouth baseline the student target is zero. A fixed baseline-only covariance penalty (0.5) discourages retaining head-related directions; this is a learning constraint, not literal subtraction of perfectly measured head motion. Camera tracking errors and incomplete separation remain possible.

Student input is causal AirPods context at −240/−120/0 ms on a 25 Hz grid. Source validation compares directional acceleration/rotation (18 inputs) against gravity projections, perpendicular magnitudes, vector magnitudes and acceleration–rotation dot products (21 inputs). These latter quantities are invariant to a common rotation of the measured vectors; directional inputs may retain useful information that invariants discard. Standardization, target scales, ridge (0.1/1/10), feature mode, coefficients and templates are fitted before the new B block. Student inputs contain no camera measurements, expected text, labels, elapsed cue phase or word index. Spoken and mouthed students are fitted separately.

The most recent completed two-block comparison can supply two historical training sessions, alongside the new A session. Teacher A explicitly asks the user to reseat before training too, to avoid treating a second recording in the same physical fit as independent fit validation. The start button records that confirmation in `blockConfirmedAt`; B requires another reseat. This is user-confirmed, not a hardware fit detector. All source run/session IDs and canonical source hashes are saved. Former tests are explicitly training data now; they are not reported as fresh held-out results. With multiple source sessions, model selection leaves entire recording sessions out. With only one source session it leaves whole repetitions out and reports the weaker scope. A shifted-teacher control gets the same candidate search. Validation scores are used for selection and are development estimates, not unbiased final-test scores. No fitting occurs in the prospective B evaluator, and training identities/sessions are refused as new tests. The model fingerprint remains fixed through B.

The primary comparison is **camera-taught AirPods + linear decoder versus raw AirPods + the same fixed linear-decoder method**. Both use source-standardized means/standard deviations in four relative-time bins and fixed ridge=1; their scores are not probabilities. Supporting arms compare raw DTW, untrained temporal-context DTW, learned-feature DTW, camera-mouth-only DTW and camera-head-only DTW. Every arm includes a half-clip time-shift control. Camera mouth/head controls are labelled separately; deleting B's camera data cannot alter AirPods-arm predictions. Incomplete tracking affects the visual controls, while AirPods arms use motion/timer quality. Missing/unmatched takes stay visible against all nine planned takes per condition. This remains a three-phrase experiment; the normal vocabulary decoder and reserved 25% sentence test are untouched.

The approach draws on the idea of transferring supervision between paired modalities in [Cross Modal Distillation for Supervision Transfer](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gupta_Cross_Modal_Distillation_CVPR_2016_paper.pdf). That paper studies different modalities/tasks; it does not establish AirPods silent-speech performance. A small regularized model is used here because there are few independent recordings, even when there are many sensor frames.

Enable the local Mac camera and keep your face and mouth visible. There is no neutral-pose capture, fixed position, angle tolerance or hold-still requirement in new comparisons. During the three-second baseline, relax your mouth and gently move your head naturally; it supplies head-only calibration data. Vision estimates yaw, pitch, roll, centre and apparent size independently of the AirPods. Camera frames/video are never written to disk; numeric pose measurements, inner/outer lip contours, eye anchors, mouth corners/centre and timestamps are saved. Preview and processing use the same unmirrored orientation.

Controlled takes end on their fixed timer; forward arrows cannot finish them early. Left/Up redoes a mistake and preserves the discarded attempt in raw metadata. Prediction feedback stays hidden until both blocks finish (or the user ends the comparison). Pacing prescribes articulation; it does not verify actual silent word times. The tracking/timing subset requires 90% head and lip coverage plus no more than 100 ms capture-duration error. Head movement is not a quality failure. Missing/invalid takes stay visible against the nine-planned-take denominator. Incomplete comparisons and completed takes survive termination. Camera access stops when the comparison is cancelled or finished. Historical strict-pose comparisons retain their original rules and are never reinterpreted as natural-movement data.

Historical `naturalMovement` comparisons retain their original linear-subtraction method and fixed 3.4-second pacing. They map six camera velocities to the three rotation channels, fitted only on the −2.5 to −0.5-second relaxed-mouth baseline. Alternate whole baselines train/validate the correction; at least 10% held-out prediction-error reduction is required. An unhelpful fit leaves corrected scores unavailable. The final coefficients use all A baselines and remain frozen for B. The new teacher protocol does not rerun that failed subtraction method or reinterpret its old scores.

Both protocols retain camera-only context up to 250 ms after END and flush delayed camera callbacks for 500 ms before closing a block, without changing take boundaries. Missing frames, clock fallbacks, gaps over 120 ms and sensor-source transitions are not bridged to invent data. Original motion/points are preserved. New motion samples and CSV rows log Core Motion's `sensorLocation` (0 unspecified, 1 left headphone, 2 right; −1/missing for older data). A source-side change after reseating is reported explicitly; older recordings cannot retroactively identify the sensor.

The live camera preview overlays cyan lip contours/points, a pink opening marker, yellow corners, a mouth-centre marker and green eye-reference anchors. It targets up to 30 processed samples/s and displays the achieved rate and tracking confidence. Raw lip/eye points use full-image normalized coordinates with a lower-left origin; eye-aligned shape measurements use inter-eye-distance units, correcting translation, scale and roll but not yaw/pitch perspective. Width, opening, inner area, outer aspect, centre and short-interval movement/opening velocities are recorded. Missing/low-confidence detections remain absent; velocities do not bridge tracking gaps longer than 200 ms. Head-pose targets identify the camera estimator so targets from the older rectangles-only guide must be recaptured before a new comparison.

Each new signal session streams a numeric `camera-points.ndjson` sidecar continuously while its recorder is active, including ready/baseline/take intervals and missing-detection samples. Rows identify the session, relative session seconds, capture/receipt/processing uptimes, clock fallback, camera dimensions, pose and mouth points/measurements. Per-take metadata also carries numeric readings during baselines/takes. The report shows lip-tracking coverage, actual rate and receipt-time fallback counts separately from frozen-model scores. None of these visual features is fed into the existing AirPods matcher. New checks record no audio/video files and need no Whisper configuration; the metadata explicitly records this capture policy.

## Local files

- Active training: `~/Library/Application Support/Subvox/sentence-training.json`.
- Latest diagnostic: `~/Library/Application Support/Subvox/signal-check.json`; a permanent `signal-check.json` copy also stays in each raw diagnostic recording folder.
- Controlled comparison: `~/Library/Application Support/Subvox/controlled-signal-test.json`, with per-test archives in `ControlledSignalTests/` and comparison sidecars in raw block folders. Teacher comparisons save the training-source manifest, selected models, raw/learned classifiers, templates/scales, immutable fingerprint and per-arm B scores. Historical comparisons keep their original linear correction and raw/corrected scores.
- Frozen diagnostic model: `~/Library/Application Support/Subvox/frozen-signal-reference.json`. Each second-session recording folder also receives an identical reference copy; its latest-run JSON records the reference identity/fingerprint and reseating confirmation.
- Raw recordings: `~/Documents/Subvox/Sessions/`. New signal checks contain motion, numeric camera-point NDJSON, events and metadata. Paired sentence/historical recordings may also contain audio and aligned-pair sidecars.
- Recovery archives: `~/Documents/Subvox/Archives/`. These are never loaded as active training.
- Existing OpenWhispr files: `~/.cache/openwhispr/whisper-models/ggml-base.bin` and `/Applications/OpenWhispr.app/Contents/Resources/bin/whisper-server-darwin-arm64` by default; alternate GGML models can be selected.

Whisper runs in an owned child process bound to `127.0.0.1`. Requests contain no expected-text prompt, refuse redirects, and convert CAF audio locally to 16 kHz mono WAV. Model files stay in their existing location.

To start clean again, close Subvox, inspect the reset paths, then archive active training:

```sh
python3 Tools/data/reset_training.py
python3 Tools/data/reset_training.py --execute
```

Reset archives datasets, prior snapshots and recording folders with SHA-256 verification, recreates empty active folders, and retains local Whisper model files/preferences. Failed moves roll back; no permanent deletion is performed.

## Checks

```sh
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build \
  CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM= CODE_SIGN_STYLE=Manual \
  -configuration Release ENABLE_TESTABILITY=YES test
python3 -m unittest discover -s Tools -p test_reset_training.py
```

Tests cover audio-attention timing and acoustic rejection, explicit signal-check markers and redo/stop recovery, held-out repetition comparisons and controls, Mac microphone preference, word matching, timing transfer, sentence-role isolation, annotation review, rejection metrics, model parity, recorder ownership, recovery metadata, settings persistence and archive rollback. Removed workflows no longer carry compatibility tests.

## Corpus tools and limits

`Corpus/paired-sentences.txt` supplies paired prompts. The pronunciation/vocabulary tables support fitting and decoding. `harbor.txt` and `diphones.txt` remain reference material for corpus development; the retired teleprompter does not run them.

```sh
python3 Tools/corpus/coverage.py Corpus/paired-sentences.txt --missing
python3 Tools/corpus/diphone_bank.py --reps 2
python3 Tools/corpus/lexicon.py
```

Corpus tools cache pronunciation and public-domain source material under `~/Library/Caches/Subvox`. Those shared resources are not personal training examples.

AirPods motion is typically about 50 Hz and includes head movement as well as articulation. The speech accelerometer and inward microphone are not exposed. Models are personal and may change when the AirPods are reseated. This remains an experiment in signal quality and word recognition.
