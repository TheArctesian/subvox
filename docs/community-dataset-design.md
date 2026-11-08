# Community dataset and phoneme model: design

Status: build steps 1–3 are implemented (see "Build status" at the end). Steps 4–7 are not built yet.

## Goal

Free silent dictation from AirPods motion fused with camera lip reading. Free dictation needs hundreds of hours of mouthed speech from many people. The plan:

1. Subvox becomes installable by anyone. Recording, review and inference stay **local**.
2. Contributors choose to share finished sessions to one **gated Hugging Face dataset**, as pull requests.
3. Contributors read **varied public-domain literature, poetry and speeches**, assigned to cover all phonemes across contributors. Each person reads different text.
4. A shared **phoneme model** trains on the dataset, is published on the Hugging Face Hub, and is **personalised on each user's Mac** from a short calibration set.

Why this direction: the closed-set results (README, "Closed-set sentence identification") show that the sensors carry sentence-level information with repetitions. But whole-sentence templates cannot recognise text that was never recorded. Free dictation needs sub-word units (phonemes) and a lot more varied data.

## 1. Sign-in and contribution modes

The first launch shows a contribution screen with three choices. Recording works the same in all three.

| Mode | Uploads via | Credit | Identity in the dataset |
| --- | --- | --- | --- |
| **Local only** (default) | nothing | none | none |
| **Anonymous** | Subvox relay | none | random contributor ID |
| **Hugging Face** | the user's own account | PR authorship on the dataset, plus a dataset-card listing | Hugging Face username |

The mode can be changed at any time in Settings. Changing it doesn't move data that was already uploaded.

### Hugging Face mode
- OAuth with PKCE through the system browser (`ASWebAuthenticationSession`), using a registered Hugging Face OAuth app. The token is stored in the Keychain and refreshed when it expires.
- Each session is uploaded as one commit with `create_pr=true` on the dataset. Large files go through Hugging Face's large-file storage.
- **OAuth scope: still to verify with a live test.** Checked against the Hugging Face docs on 7 Nov 2026 (`docs/hub/oauth`, `repositories-pull-requests-discussions`, `huggingface_hub` upload guide, `security-tokens`):
  - `write-discussions` is the only scope documented for opening PRs ("Open discussions and Pull Requests on behalf of the user"). Opening a PR on a *private* repo also needs `read-repos`.
  - The docs do **not** say whether `write-discussions` alone allows a PR that carries commits or large (LFS/Xet) files.
  - `write-repos` covers writing only to "the user's personal repos", so it may not apply to a dataset someone else owns.
  - Nothing says whether a contributor must accept a gated dataset's terms before opening a PR.
  - **Test before building step 4:** with a token that has only `write-discussions`, call `create_commit(create_pr=True)` on a public test dataset you don't own, with one small file and one LFS-tracked file. Then repeat on a gated test dataset with an account that hasn't accepted the gate. If either fails with 403, add `write-repos` and test again.
  - Resuming an upload: re-running `create_pr=True` opens a *second* PR. Resume with `revision="refs/pr/N"`.

### Anonymous mode
- The app sends the session to a **relay** (a Hugging Face Space). The relay holds a bot token and opens the PR under the bot account.
- The app never contains a write token.
- The relay validates the schema, size and quality checks, rate-limits by install ID, and rejects anything else.
- Deletion: at first upload, the app creates a random **deletion key** and keeps it locally. Sending it to the relay removes that contributor's files.
- Abuse is the main risk. Rate limits come first. Device attestation is worth investigating if spam appears.

### Shared by both modes
- **Review queue:** a finished session is never uploaded automatically. The user sees the takes and can drop any before sending.
- **Upload queue:** a local queue that persists across restarts and retries failed uploads.
- **Validation and auto-merge:** a Hugging Face webhook calls a validator Space for every new PR. It checks schema version, file sizes, sample rates, coverage and the eligibility rules, comments on the PR, and auto-merges clean ones. It also keeps the contributor list and hours on the dataset card up to date.

## 2. Consent and privacy

Lip landmarks and head motion are biometric personal data under UK GDPR, which makes them special-category data that needs explicit consent.

- **Explicit, informed opt-in** before any upload, in plain language. It says what is collected, where it goes, who can see it, and how to withdraw. In Hugging Face mode it also says: "your contributions are linked to your Hugging Face username."
- **Adults only** (18+), confirmed at opt-in.
- **Minimise what's uploaded:**
  - Numeric camera points only, never images or video. Mouth crops are a later, separate opt-in (see 5.3).
  - Audio only for spoken takes, only with a separate opt-in, and only when it is needed for alignment (see 5.4).
  - No device names, account names, file paths or wall-clock timestamps. Times are relative to the session start, and the date is kept at month precision.
- **Gated dataset:** access requires accepting terms (research use, no attempts to re-identify contributors). Contributor PRs are visible only to approved users.
- **Withdrawal:** deleting a contributor's data removes their files from the main branch. Full removal also needs history squashing, which is run periodically and documented on the dataset card.
- **Before public launch:** get a short review from someone who knows data protection. It should cover contributors outside the UK and the lawful basis.

## 3. Reading material

### Sources
Text must be public domain in both the UK (author died more than 70 years ago) and the US (published before 1930). The dataset republishes prompt text, so this matters.

- **Literature:** for example Austen, Dickens, the Brontës, Stevenson, Wilde, Twain, Kipling's earlier works, Doyle's early Holmes stories, fairy tales and fables (Grimm and Andersen in translations old enough to be public domain).
- **Poetry:** for example Shakespeare's sonnets, Blake, Wordsworth, Keats, Dickinson, Whitman, Christina Rossetti, Stevenson's *A Child's Garden of Verses*.
- **Speeches:** for example Lincoln (Gettysburg Address, second inaugural), Frederick Douglass, Sojourner Truth, Elizabeth I at Tilbury, Patrick Henry.
- **Not allowed:** speeches still under copyright, for example Churchill (in copyright until 2035) and Martin Luther King Jr. (estate-controlled). US federal works such as JFK's inaugural address are public domain in the US but uncertain abroad, so leave them out at first.
- Every source text records its provenance (title, author, year, source URL, edition) and a public-domain justification in `Corpus/community/sources.tsv`.

### Keeping it easy to say
A corpus tool (`Tools/corpus/community.py`) splits texts into sentences, then filters them:
- 4 to 16 words.
- Readability at about reading age 11 or younger (Flesch–Kincaid grade 6 or below).
- Every word in the pronunciation lexicon, with a grapheme-to-phoneme proposal flagged for review otherwise, and within the 20,000 most common English words.
- No numerals, abbreviations, foreign phrases, dialogue fragments or archaic forms (*thee*, *'tis*, *hath*) unless a curated allowlist keeps a specific poem intact.
- Spelling normalised to one variety per prompt.

Poetry lines are joined into whole sentences or couplets so that a prompt reads naturally. The passage keeps its context: the app shows the title and author, and prompts from one passage are recorded in order, which reads better than random sentences.

### Assigning prompts
- The app downloads `prompts-vN.jsonl` (prompt ID, text, source ID, phoneme sequence) from the dataset repo.
- **Coverage-driven assignment:** the validator Space publishes the current diphone counts. The app picks passages that fill the gaps, so different people read different text and coverage stays balanced.
- **Calibration set:** 40 fixed short sentences that every contributor records once. They are used for comparisons between speakers and for personalising the model on each user's Mac (5.6).
- **Held-out text:** 10% of passages are reserved for evaluation and never assigned for training. This is separate from the existing eight reserved final-test sentences, which stay untouched.

## 4. Recording mode and data format

### Prompted reading mode
This uses the existing paired recording interface: preview, Relax, START/END and redo.
- Each prompt is recorded as a **pair: spoken, then mouthed**, the same as the current paired collection. A user can choose **mouthed only** for some or all passages, though the pairs are what make phone alignment possible.
- Existing eligibility rules apply: fresh motion, no source switch or gap, START/END coverage, and camera coverage of at least 80% for camera-eligible takes.
- Session length is capped at about 20 minutes, with breaks suggested. Takes show progress in passage pages, not raw counts.

### Dataset layout (Hugging Face)
```
data/<contributor>/<session-id>/
  session.json        schema version, app version, AirPods model, Mac/camera model,
                      sensor side, sample rates, consent version, month
  takes.jsonl         take id, prompt id, mode (spoken|mouthed), cue start/end,
                      eligibility flags, redo count
  motion.parquet      take id, t, accel xyz, rotation xyz, gravity xyz, attitude, side
  camera.parquet      take id, t, eye-frame landmark channels and masks (current 50 Hz layout)
  audio/<take>.flac   spoken takes only, only with the audio opt-in
  alignment.jsonl     optional: local MFA phone intervals + camera-warp transfers
prompts/prompts-vN.jsonl
prompts/calibration-v1.jsonl
prompts/heldout-v1.jsonl
```
- One PR per session.
- `schema_version` starts at 1 for the community format. It is separate from the local neural snapshot schema. A local exporter converts recorded sessions into this layout, and the trainer reads only this layout.
- The dataset is split by contributor in `splits/*.json`. Development and test **speakers** are disjoint from training speakers. Held-out passages give a second, text-level split.

## 5. Phoneme model changes

The current `MotionCTC` is a dilated-convolution plus bidirectional-GRU encoder with a CTC phoneme head. It concatenates 32 motion features with camera channels (README, "Camera + AirPods input fusion"). It was built for one person's recordings and keeps the strict exclusion policy. These are the changes for many speakers and free dictation.

### 5.1 Separate encoders, learned fusion
- **Motion encoder:** the existing 50 Hz features with gravity-invariant fallbacks, plus small convolutional front ends.
- **Camera encoder:** landmark channels with masks, plus their velocities.
- **Fusion:** per-frame gated fusion, where a learned gate weights each stream from its own reliability features (mask coverage, motion energy). Then a shared **Conformer** encoder.
- This replaces plain concatenation. It directly fixes the closed-set failure where equal weighting let AirPods noise drown out the camera.
- **Dropping streams in training:** drop the whole camera stream (already done at 30%) and also the whole motion stream, at about 15%. One stream is always kept.

### 5.2 Targets: phonemes first, with helper targets
- **Main output:** CTC over the existing 39-unit phone inventory (`ArticulatoryInventory`), with no word separator, as in the current heads.
- **Auxiliary heads:**
  - CTC over **viseme classes** (about 12 groups), which predicts what the lips can show;
  - frame-level **articulatory features** (lip rounding, closure, jaw height), trained when phone alignments exist;
  - the existing next-phone transition target.
- A second-pass phone head, conditioned on the viseme head, is an option if the ambiguous groups (/p b m/, /f v/) dominate the errors.

### 5.3 Camera input: landmarks now, crops later
Landmarks are private and cheap, and they transfer between people. But they lose the tongue, teeth and inner-lip shading, and pretrained lipreading models (AV-HuBERT, Auto-AVSR) expect pixel crops.
- **v1:** landmarks only, matching today's no-images policy.
- **v2 (separate opt-in):** 88×88 grey mouth crops at 25 fps, processed by a frozen, pretrained lipreading encoder whose features are fused in place of, or alongside, the landmark encoder. Uploading crops is a privacy step up and gets its own consent text.
- Decide on v2 from the scaling check in 5.7: if landmark-only error stops improving early, crops are the most likely next gain.

### 5.4 Alignment for frame-level supervision
- For spoken takes with audio, local MFA phone alignment provides phone intervals. The adapter for this already exists (README, "Optional local acoustic phone alignment").
- These intervals transfer to the paired mouthed take with the existing monotonic camera-shape warp, and are stored with their confidence.
- They are used only for auxiliary frame-level losses and evaluation. CTC still trains from text alone, so takes without alignment still count.
- Alignment can also run at the validator or trainer instead of on the user's Mac, so audio never has to be uploaded. This is the preferred option if it works well enough.

### 5.5 Decoding
1. CTC prefix beam over phones, then the pronunciation lexicon (with grapheme-to-phoneme proposals for words not in it).
2. An n-gram word language model trained on a large public-domain and permissive text corpus, not the prompts, to avoid leaking prompt text.
3. **Language-model rescoring** of the n-best list. A small local model is enough to start. This is where most of the accuracy gain is expected at first, because many lip-and-jaw ambiguities are resolved by context.

Reported separately: phoneme error rate (free-phone beam), word error rate (lexicon beam), word error rate with the n-gram model, and word error rate with rescoring.

### 5.6 Multiple speakers and personalisation
- **Inputs normalised per speaker:** statistics from each speaker's calibration and rest clips, not dataset-wide values.
- **Conditioning on device and side:** AirPods model and sensor side embeddings, since Pro, 3 and Max mount their sensors differently.
- **Personalisation on the user's Mac:** after downloading the base model, the user records the calibration set. Small adapter layers (LoRA on the Conformer, plus per-speaker input scaling) are fine-tuned locally in a few minutes. The base weights stay unchanged, and personal adapters never leave the Mac.
- **Evaluation:** always on unseen speakers, both before and after personalisation, since the gap shows how much calibration helps.

### 5.7 Scaling check and go/no-go
Train and evaluate at about 5, 20, 50 and 100 hours of eligible mouthed data, with all three input arms (`imu`, `camera`, `fused`):
- phoneme and word error on held-out speakers and held-out text;
- the fused-minus-camera gap, to check whether the AirPods keep adding information as data grows.

Decision rules:
- error still falling steeply → keep collecting;
- landmark camera flattening → move to crops (5.3);
- fused no better than camera at 20 hours or more → recording AirPods motion is optional.

### 5.8 Training and release
- Training runs off the Mac on a GPU, reading the Hugging Face dataset with `datasets`. It reuses the existing `subvox-ml` package, given a community-dataset loader.
- Released models go to a Hugging Face model repo, with versioned weights, the training dataset revision, a metrics card and a schema fingerprint.
- The app downloads them on request into its existing local runtime and verifies the fingerprint. Inference stays local.

## 6. Build order

1. **Corpus pipeline:** source list, sentence filters, coverage-driven passage selection, calibration and held-out sets. No upload needed.
2. **Prompted reading mode** in the app, using the new prompt files.
3. **Community exporter:** local sessions to the dataset layout, with a tested schema.
4. **Contribution screen:** Local / Anonymous / Hugging Face, consent flow, Keychain token, review and upload queue.
5. **Validator and auto-merge Space**, plus the relay endpoint for anonymous mode.
6. **Model v1:** separate encoders, gated fusion, Conformer, phone + viseme CTC, n-gram decoding; speaker-held-out evaluation.
7. **Personalisation** on the user's Mac, language-model rescoring, and the scaling checkpoints.

Steps 1–3 are fully local and useful for your own recordings before anyone else installs the app.

## Build status

- **Step 1 (done):**
  - `Corpus/community/sources.tsv` lists 29 sources. The build refuses any source that isn't public domain in both the UK and US for the current year.
  - `Tools/corpus/community.py` builds `prompts-v1.jsonl` (7,395 prompts in 3,289 passages), `heldout-v1.jsonl` (863 prompts in 390 passages), `calibration-v1.jsonl` (40 sentences) and `build-report-v1.json`.
  - Tests: `Tools/corpus/test_community.py`.
  - Not done: a human read-through of the calibration set and a sample of passages before launch.
- **Step 2 (done, in local paired collection):**
  - Train → **Read from: Community passages** records the calibration set first, then passages chosen by `CommunityCorpus.assign`, using the same rule as the Python pipeline. The chosen passages are checked against the Python output in `CommunityReadingTests`.
  - Pairs are spoken then mouthed, or mouthed only. Mouthed-only takes skip local Whisper, so they never enter the personal neural datasets' Whisper-based admission. They do export to the community layout.
  - Local diphone counts come from this user's own mouthed takes. The validator-published counts arrive with step 5.
- **Step 3 (done):**
  - `subvox-ml export-community --root … --output … --contributor … --consent-version …` writes the dataset layout from section 4.
  - `subvox-ml validate-community --session …` runs the schema and privacy check that the validator Space should reuse. It rejects unexpected keys, paths, held-out prompts, unknown prompts and non-increasing times.
  - Session and take IDs are one-way hashes, so re-exporting gives the same IDs.
  - Not exported yet: audio, `alignment.jsonl` and device models (left `null` until the app records them).
  - When step 4 calls the exporter from the app, it must pass `--prompts-dir` pointing at the bundled `Corpus/community`, because the runtime copy holds only the distractor bank.

## Open questions

- The minimum Hugging Face OAuth scope for PR-based contribution (step 4); docs checked, live test pending (section 1).
- Where to host the relay, and its abuse controls (step 5).
- Whether audio is uploaded or aligned elsewhere (5.4).
- When to add mouth crops (5.3), decided from the scaling check.
- Dataset licence for contributed recordings (for example CC BY 4.0 with gated access), and the consent text wording, pending data-protection review.
