# Next phase: multilingual IPA and training practice modes

Handoff from the camera-fusion / community-dataset session. Read `docs/community-dataset-design.md` first; it is the spec for everything already built.

## Where things stand

Built on `claude/camera-fusion` (not pushed):
- **Camera + AirPods fusion** in the neural model, with closed-set sentence identification (`subvox-ml closed-set`). Result: one spoken example → camera 33% top-1 of 37 (chance ~1%); repeated phrases across sessions → AirPods 60%, camera 82%, fused 90% (3 phrases, chance 17%).
- **Command-set recording mode** with closed-set scoring.
- **Community dataset:**
  - a public-domain corpus pipeline (`Tools/corpus/community.py`, `Corpus/community/`): 7,395 prompts and 40 calibration sentences;
  - a prompted reading mode;
  - `export-community` and `validate-community`;
  - Hugging Face sign-in with PR uploads;
  - the upload queue.
- **Device support:** AirPods model and firmware are recorded per session, and motion at 20 Hz and up is accepted, interpolated onto the 50 Hz grid and flagged with its native rate.
- **Licence:** AGPL-3.0-or-later for code. Recommended for data: CC BY-SA 4.0 plus gated terms (trained models must be open; no re-identification). Prompts are CC0.
- **Still open:** the Hugging Face OAuth scope needs a live test (design doc section 1), and someone needs to read through the corpus prompts.

## Data rules for every new mode

A take enters training only if:
1. **The text is known before speaking**, so the label is what was meant to be said.
2. **The language is tagged**, along with whether the speaker is native or learning it, and their accent or first language.
3. **There is spoken audio or a phone check whenever the speaker might deviate** (free reading, learners, free input). Training then uses the phones that were **actually produced** (from audio alignment or a phone recogniser), not the textbook pronunciation.
4. **The sensor rules already in place pass:** fresh motion, no gaps, camera coverage of at least 80%.
5. **Mostly mouthed takes**, plus spoken-then-mouthed pairs for alignment. Exaggerated "game" articulation is labelled as a speaking style.

## Work items

### 1. Multilingual IPA layer (English, French, Spanish, Mandarin)
- Install and pin the tools in `ml/pyproject.toml`, with `uv lock`:
  - `phonemizer` with eSpeak NG from the `espeakng-loader` wheel (no system install) for French and Spanish (and English as a fallback);
  - `pypinyin` for Mandarin (characters to pinyin, using phrase context so characters with more than one reading come out right), plus a pinyin-to-IPA table.
- **Universal phone set:** replace the 39-unit ARPAbet inventory with a shared IPA inventory, with a mapping from ARPAbet for existing English data and labels. Old checkpoints keep a compatibility path.
- **Language tag** per take, prompt and session. Add it to the community schema (bump `schema_version`) and the validator.
- **Mandarin:** keep tone in the labels, but the visual model predicts toneless syllables and a **Mandarin language model chooses the tones** (agreed). For French, nasal vowels are only weakly visible, and liaison is handled by converting whole phrases to IPA, not word by word.
- **Corpus:** add public-domain sources for French, Spanish and Mandarin (UK and US public domain, recorded in `sources.tsv`), and apply the same simplicity filters per language.
- **Tests:** a few hand-checked phrases per language (*je suis / tu es / il est* liaison, Spanish vowels, Mandarin characters with more than one reading).

### 2. Phone check from audio
- Reading and Speech & Poetry can match known text against local IPA CTC audio evidence in English, French, Spanish and Mandarin. Install the pinned checkpoint from the Reading or Speech & Poetry page. Reading can retry saved word-timing and transcript failures. Free Practice still requires 85% Whisper agreement, and configured English MFA is available when the new aligner is absent.
- The CTC aligner times supported expected phones. Some Mandarin clusters occupy one model token, so their individual boundaries are omitted. An independent phone recogniser is still needed to identify substitutions for pronunciation feedback.
- Store the expected and produced phones side by side.
- This shows learners their pronunciation, and supplies cleaner training labels.

### 3. Training menu tabs
Each mode is a separate tab under Train, reusing the paired recording interface:
1. **Reading**: the existing community passages.
2. **Speech & poetry practice / competition:** famous public-domain speeches and poems per language, read freely at the person's own pace. Whisper word timing follows the reader and highlights as they go. The score comes from audio agreement, plus fluency or completion, and leaderboard ideas are for later. **Only public-domain speeches**: no Churchill or King, and no US federal works for now.
3. **Language practice:** drills like conjugations (*être: je suis, tu es, il est…*). Shadowing order: show the phrase and play native audio, the user says it aloud, then mouths it. Never use mouthed answers from memory.
4. **IPA game:** minimal pairs aimed at sounds that look alike (/p b m/, /f v/, French /y/–/u/). Short rounds, labelled as targeted data.
5. **Free practice:** a text box for any passage the user wants to memorise. A take is used for training only when Whisper agrees with the text (the existing 85% agreement rule). Otherwise it stays as practice only.

### 4. macOS UI/UX pass
- Before building the tabs, research current macOS design guidance with subagents: Apple Human Interface Guidelines for macOS (sidebars or tabs, toolbars, settings, onboarding), patterns from well-regarded practice and learning apps, and accessibility (VoiceOver, Reduce Motion, Dynamic Type equivalents).
- Summarise the findings into a short checklist in `docs/`, and apply it to the training tabs.

## Suggested order

1. Install tools and add the IPA layer, with tests.
2. Add the language tag to the schema and validator.
3. Research the UI/UX checklist (it can run in parallel with steps 1–2).
4. Add the phone check from audio.
5. Build the tabs: Reading (move the existing mode), then Free practice, then Speech & poetry, then Language practice, then the IPA game.
6. Add multilingual corpus sources.
