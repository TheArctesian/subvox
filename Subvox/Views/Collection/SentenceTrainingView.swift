import AppKit
import SwiftUI

struct SentenceTrainingView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.colorScheme) private var colorScheme
    private var speak: Color { PhaseColor.speak(colorScheme) }
    private var mouth: Color { PhaseColor.mouth(colorScheme) }
    private var relax: Color { PhaseColor.relax(colorScheme) }
    @State private var channel = MotionPlotChannel.rotation
    @State private var promptViewportHeight: CGFloat = 0
    @State private var confirmStop = false
    @State private var showCaptureMotion = false
    @AppStorage("practice.pacingGuide") private var showPacingGuide = true
    private enum Section: String, CaseIterable { case record = "Record", review = "Review", progress = "Progress" }
    @State private var section = Section.record
    /// VoiceOver starts on the recording screen's title when a round begins and at each new take.
    @AccessibilityFocusState private var focusRecording: Bool
    @FocusState private var pacingFocused: Bool

    var body: some View {
        if model.showObservability { ObservabilityView() }
        else if model.showNonSpeechCollection {SequenceLearningView(collectionOnly:true)}
        else if model.showSequenceLearning { SequenceLearningView() }
        else if model.showRepresentation { RepresentationStudyView() }
        else if model.showSignalCheck { SignalCheckView() }
        else if model.sentenceTraining.isAcquiring {sentenceRecordingScreen}
        else if !model.researchTools { SimpleTrainingView() }
        else { trainingBody }
    }

    @ViewBuilder private var trainingBody: some View {
        let workflow = model.sentenceTraining
        AppPageLayout {
            HStack(alignment: .center) {
                AppPageHeading(title: "Reading", subtitle: "Speak, then mouth. Record at your own pace.")
                Spacer()
                if !workflow.isBusy {
                    Button("Simple view") { model.researchTools = false }
                    Menu("Advanced") {
                        Button("Sentence models") { model.showSequenceLearning = true }
                        Button("Saved frozen-model studies") { model.showRepresentation = true }
                        Button("Check motion signal") { model.showSignalCheck = true }
                    }.controlSize(.large)
                }
                if workflow.canStartRecording && !workflow.plan.isEmpty {
                    Button("Record sentences") { model.startSentenceTraining() }
                        .buttonStyle(.borderedProminent).controlSize(.large).disabled(!canStart)
                }
            }
        } content: {
            Group {
              VStack(spacing: 0) {
                Picker("Training section", selection: $section) {
                    Text("Record").tag(Section.record)
                    Text("Review" + (attentionCount > 0 ? " (\(attentionCount))" : "")).tag(Section.review)
                    Text("Progress").tag(Section.progress)
                }.pickerStyle(.segmented).labelsHidden().frame(maxWidth: 460).padding(.vertical, 12)
                    // The scroll view below can draw past its top edge on macOS 26; an opaque bar
                    // drawn above it keeps scrolled text from showing through the section picker.
                    .frame(maxWidth: .infinity).background(Color(nsColor: .windowBackgroundColor)).zIndex(1)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 24) {
                        if let error = workflow.error { Text(error).foregroundStyle(.red) }
                        switch section {
                        case .record:
                            setup.disabled(workflow.isBusy)
                            if attentionCount > 0 {
                                HStack {
                                    Label("\(attentionCount) pairs need attention; usable recordings are included automatically", systemImage: "waveform.path")
                                    Spacer()
                                    Button("Review recordings") { section = .review }
                                }.font(.callout)
                            }
                            reservation
                        case .review:
                            if workflow.data.pairs.isEmpty {
                                ContentUnavailableView("No recordings yet", systemImage: "waveform", description: Text("Record a sentence pair to compare its motion and transcript here."))
                            } else { review }
                        case .progress:
                            if !workflow.fittingWords.isEmpty { coverage; validation }
                            else { ContentUnavailableView("No aligned word examples yet", systemImage: "chart.bar", description: Text("Usable pairs build the word baseline automatically. Sentence models can use recordings even when word alignment fails.")) }
                            reservation
                        }
                    }.frame(maxWidth: 1000, alignment: .leading)
                        .padding(24).frame(maxWidth: .infinity)
                }.clipped()
              }
            }
        }
    }

    private var attentionCount: Int { model.sentenceTraining.pairUses.values.filter(\.needsAttention).count }

    private var canStart: Bool {
        model.motion.status == .streaming && (model.localWhisper.isConfigured || model.sentenceTraining.readingSource == .community) && model.sentenceTraining.storageReady
            && model.sentenceTraining.canStartRecording && !model.recorder.isRecording && !model.sequenceLearning.isBusy
    }

    private var communitySetup: some View {
        @Bindable var workflow = model.sentenceTraining
        let calibration = workflow.calibrationRemaining.count
        let plan = workflow.communityPlan
        let titles = plan.compactMap { $0.first.flatMap { workflow.community.prompt($0.id) } }
            .reduce(into: [String]()) { names, prompt in
                let name = prompt.passageID == nil ? "Calibration" : "\(prompt.title) — \(prompt.author)"
                if names.last != name { names.append(name) }
            }
        return VStack(alignment: .leading, spacing: 10) {
            Text("Short sentences from public-domain books, poems and speeches, chosen to be easy to say. Passages are read in order and picked to cover the sounds you haven't recorded yet.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if calibration > 0 {
                Label("First, \(calibration) calibration sentences that every contributor reads once.", systemImage: "scope").font(.callout)
            } else {
                Stepper("\(workflow.communityPassageCount) passages per session", value: $workflow.communityPassageCount, in: 1...10).frame(maxWidth: 260)
            }
            Toggle("Mouthed only (skip the spoken half)", isOn: $workflow.mouthedOnly)
            Text(workflow.mouthedOnly ? "Faster, but without a spoken take these can't be phone-aligned." : "Each sentence is spoken, then mouthed; the spoken take makes phone alignment possible.")
                .font(.caption).foregroundStyle(.secondary)
            if !titles.isEmpty {
                Text("Next: " + titles.prefix(4).joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Text("Recordings stay on this Mac. Sharing to the open community dataset is a separate, opt-in step.")
                    .font(.caption).foregroundStyle(.secondary)
                SettingsLink { Text("Sharing Settings…") }
            }
        }
    }

    private var setup: some View {
        @Bindable var workflow = model.sentenceTraining
        @Bindable var audio = model.audio
        @Bindable var whisper = model.localWhisper
        let data = workflow.data
        let acquired = Set(data.pairs.filter(\.complete).flatMap(\.promptIDs))
        let development = data.prompts.filter { $0.kind == .sentence && data.roles[$0.id] != .finalTest }
        return VStack(alignment: .leading, spacing: 16) {
            Text("\(development.filter { acquired.contains($0.id) }.count) of \(development.count) sentences recorded")
                .font(.title2.weight(.medium))
            ProgressView(value: Double(development.filter { acquired.contains($0.id) }.count), total: Double(max(1, development.count)))
            Text("Preview, relax, speak, then mouth the same text. Choose Done when finished, and Redo for mistakes. Each saved take waits for you to continue.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text("Sentence sessions save microphone audio and AirPods motion, with numeric camera points when enabled. Video is not saved. The separate non-speech session records motion only.")
                .font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 24) {
                Label("Relax", systemImage: "leaf").foregroundStyle(relax)
                Label("Speak aloud", systemImage: "speaker.wave.2").foregroundStyle(speak)
                Label("Mouth silently", systemImage: "mouth").foregroundStyle(mouth)
            }.font(.headline)
            if model.motion.status != .streaming {
                Label("Connect your AirPods to start.", systemImage: "headphones").foregroundStyle(.orange)
            }
            if !model.localWhisper.isConfigured && workflow.readingSource == .banks {
                Label("Choose a local Whisper model in recording settings.", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
            Picker("Read from", selection: $workflow.readingSource) {
                ForEach(SentenceTrainingWorkflow.ReadingSource.allCases) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).frame(maxWidth: 460)
            if workflow.readingSource == .community { communitySetup }
            if workflow.variedSentences && workflow.readingSource == .banks {
                Picker("This recording session",selection:Binding(get:{workflow.collectionRole ?? .training},set:{workflow.collectionRole=$0})) {
                    Text("Training sentences").tag(DataRole.training)
                    Text("Development sentences").tag(DataRole.validation)
                }.pickerStyle(.segmented)
                Text("Each visit keeps one role and fresh sentence groups. Re-record mistakes from review; completed or partial sentences are not automatically repeated in every session.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach([("235-sentence diphone bank",SentencePrompt.diphoneBank),
                         ("120 one-off slot sentences",SentencePrompt.gridBank),
                         ("30 \u{201C}Say ___ again\u{201D} minimal pairs",SentencePrompt.frameBank)],id:\.0) {name,bank in
                    if bank.allSatisfy({p in data.prompts.contains {$0.id==p.id}}) {
                        Label("\(name.prefix(1).uppercased()+name.dropFirst()) added", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.secondary)
                    } else {
                        Button("Add the \(name)") {workflow.addSentenceBank(bank)}
                    }
                }
                Text("Adding a bank keeps existing recordings and reserves 25% of the expanded corpus for the final test.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                if !workflow.variedSentences {
                    Picker("Read as", selection: $workflow.presentation) {
                        ForEach(SentenceTrainingWorkflow.Presentation.allCases) { Text($0.rawValue).tag($0) }
                    }.pickerStyle(.segmented)
                }
                Picker("Pace", selection: $workflow.pace) {
                    ForEach(TrainingPace.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented)
            }
            ForcedAlignerSetup(aligner: workflow.forcedAligner)
            if workflow.forcedAligner.isInstalled && !workflow.realignableIDs.isEmpty {
                Button("Re-align saved takes") { workflow.realignSavedTakes(whisper: model.localWhisper) }
                    .disabled(workflow.isBusy || model.recorder.isRecording)
            }
            if let report = workflow.realignmentReport {
                Text("Recovered \(report.recovered) of \(report.total) saved takes.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            if !workflow.pendingTranscriptionIDs.isEmpty {
                Button("Run local Whisper on \(workflow.pendingTranscriptionIDs.count) pending spoken takes") {
                    workflow.transcribePending(whisper:model.localWhisper)
                }.disabled(!model.localWhisper.isConfigured)
            }
            Button {
                model.showNonSpeechCollection=true
            } label: { Label("Record non-speech movements", systemImage: "figure.stand") }
                .disabled(workflow.isBusy || model.sequenceLearning.isBusy)
            Button {
                model.showObservability = true
            } label: { Label("Check whether AirPods see mouth movement", systemImage: "waveform.badge.magnifyingglass") }
                .disabled(workflow.isBusy || model.observability.isBusy)
            if !workflow.plan.isEmpty {
                Text("Next batch: \(workflow.plan.count) paired prompts, about \(max(1, Int(ceil(workflow.estimatedSeconds / 60)))) minutes.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            MotionDisclosure("Recording settings") {
                VStack(alignment: .leading, spacing: 12) {
                    Stepper("Up to \(workflow.sentenceCount) training sentences per batch", value: $workflow.sentenceCount, in: 2...8, step: 2)
                    Text(workflow.variedSentences ? "Each batch uses fresh sentences from the selected role." : "Validation sentences are added separately; passages join up to three sentences.")
                        .font(.caption).foregroundStyle(.secondary)
                    Picker("Microphone", selection: $audio.selectedDeviceID) {
                        Text("System default").tag(Optional<UInt32>.none)
                        ForEach(audio.devices) { Text($0.name).tag(Optional($0.id)) }
                    }
                    Text("Use the Mac microphone to keep AirPods in their normal audio mode.").font(.caption).foregroundStyle(.secondary)
                    TextField("Session notes: fit, posture or pronunciation", text: $workflow.notes).textFieldStyle(.roundedBorder)
                    PhoneAlignmentSettingsView(workflow: workflow)
                    MotionDisclosure("Local transcription" + (whisper.isConfigured ? " · ready" : " · setup needed")) {
                        TextField("Whisper model path", text: $whisper.modelPath).textFieldStyle(.roundedBorder)
                        TextField("Whisper server path", text: $whisper.executablePath).textFieldStyle(.roundedBorder)
                        Text("Existing local model files are reused; no download is required.").font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(.top, 8)
            }
            if !workflow.plan.isEmpty {
                MotionDisclosure("Preview next batch") {
                    ForEach(workflow.plan.indices, id: \.self) { i in
                        Text(workflow.plan[i].map(\.text).joined(separator: " ")).padding(.vertical, 4)
                    }
                }
            }
        }
    }
    private var sentenceRecordingScreen:some View {
        let workflow=model.sentenceTraining
        let resting=workflow.phase == .relax || workflow.phase == .recovery
        let color=resting ? relax:workflow.currentCondition == .normal ? speak:mouth
        let status=workflow.isSavingTake ? String(localized: "Saving…") : workflow.phase == .ready ? "Ready":resting ? "Relax":workflow.phase == .ended ? "Take saved":"Recording"
        return RecordingViewport {
            VStack(spacing:8) {
                HStack(spacing: 16) {
                    Text(recordingTitle).font(.headline).accessibilityAddTraits(.isHeader)
                        .accessibilityFocused($focusRecording)
                    Spacer()
                    Toggle("Pacing guide", isOn: $showPacingGuide).toggleStyle(.switch).controlSize(.small)
                        .focused($pacingFocused)
                    if workflow.recordsLiteratureInPasses {
                        Text(workflow.currentCondition == .silent ? "Silent pass" : "Spoken pass")
                            .font(.subheadline).foregroundStyle(.secondary)
                    }
                    if let section = workflow.literatureSectionNumber, let count = workflow.literatureSectionCount {
                        Text("Section \(section) of \(count)")
                            .font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
                    } else {
                        Text("\(min(workflow.index + 1, workflow.total)) of \(workflow.total)")
                            .font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
                ProgressView(value:Double(workflow.index),total:Double(max(1,workflow.total))).tint(color)
            }
        } prompt: {
            VStack(spacing:8) {
                if workflow.captureCameraPoints {CameraPoseGuide(camera:workflow.camera,compact:true,allowsMovement:true).frame(height:70).padding(.horizontal,12)}
                RecordingPromptPanel(color:color) {
                    VStack(spacing:18) {
                        // The countdown changes every 50 ms; VoiceOver gets the phase, not a stream of numbers.
                        RecordingCueStatus(timer: cueTimer, title: status, color: color)
                            .accessibilityElement(children: .ignore).accessibilityLabel(status)
                        if let passage = workflow.currentPassageLabel {
                            Text(passage).font(.callout.italic()).foregroundStyle(.secondary)
                        }
                        // Words and a distinct symbol for each phase, so it never depends on colour or sound.
                        Label(resting ? "REST" : workflow.currentCondition == .normal ? "SPEAK ALOUD" : "MOUTH SILENTLY",
                              systemImage: resting ? "pause.circle" : workflow.currentCondition == .normal ? "speaker.wave.2.fill" : "mouth.fill")
                            .font(.title2.weight(.bold)).foregroundStyle(color)
                            .contentTransition(.symbolEffect(.replace))
                            .motion(.state, value: resting ? 0 : workflow.currentCondition == .normal ? 1 : 2)
                        VStack(spacing: 6) {
                            WordFocusPrompt(text: workflow.currentText, activeWordIndex: showPacingGuide ? workflow.activeWordIndex : nil,
                                            color: color, pinyin: workflow.currentPinyin,
                                            verseLines: workflow.currentVerseLines, viewportHeight: promptViewportHeight)
                            if let translation = workflow.currentEnglishTranslation {
                                Text(translation).font(.title3).foregroundStyle(.secondary).multilineTextAlignment(.center)
                            }
                        }
                        if let error = workflow.error {
                            Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                        }
                        Text(resting ? "Relax your mouth. Wait for the recording cue."
                             : workflow.phase == .ready ? "Press Space when you’re ready to begin."
                             : workflow.phase == .ended ? "This take was saved on this Mac. Press Space to continue, or redo it."
                             : (!showPacingGuide || workflow.guideSecondsRemaining == nil) ? "Finished? Press Space." : "Follow the pacing guide.")
                            .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                }
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { promptViewportHeight = $0 }
            }
        } actions: {
            VStack(spacing:10) {
                HStack(spacing:14) {
                    Button {workflow.redoCurrentTake()} label: {Label("Redo",systemImage:"arrow.counterclockwise").frame(minWidth:90)}.disabled(!workflow.canRedoTake)
                    // One button carries the press, the wait and the confirmation without changing size.
                    Button {workflow.advance()} label: {
                        ZStack {
                            Label(workflow.phase == .ready ? "Start take" : workflow.phase == .ended ? workflow.continueTitle : "Finish take",systemImage:workflow.phase == .speak || workflow.phase == .mouth ? "checkmark":"chevron.right")
                                .contentTransition(.symbolEffect(.replace))
                                .opacity(workflow.isSavingTake ? 0 : 1)
                            if workflow.isSavingTake {
                                HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Saving…") }.transition(.opacity)
                            }
                        }
                        .frame(minWidth:200)
                        .motion(.state, value: workflow.isSavingTake)
                        .motion(.state, value: workflow.phase)
                    }.buttonStyle(.borderedProminent).disabled(!workflow.canAdvance)
                }.controlSize(.large).frame(maxWidth:.infinity)
                HStack(spacing:16) {
                    Text("Space or → start, finish or next  ·  ← redo  ·  Esc stop").font(.callout).foregroundStyle(.secondary)
                    Button(showCaptureMotion ? "Hide motion" : "Show motion", systemImage: "waveform") { showCaptureMotion.toggle() }
                        .help("Show or hide the live AirPods signal")
                    Button("Stop round…") {
                        model.captureShortcutsSuspended = true
                        confirmStop = true
                    }.keyboardShortcut(.cancelAction).font(.callout)
                }
            }.frame(maxWidth:.infinity)
        } motion: { LiveMotionPanel(plotHeight: 44) }
        .withMotionVisibility(showCaptureMotion)
        .confirmationDialog("Stop this round?", isPresented: $confirmStop, titleVisibility: .visible) {
            Button("Stop round", role: .destructive, action: workflow.stop)
            Button("Keep practising", role: .cancel) {}
        } message: {
            Text("Finished takes stay on this Mac. An unfinished take is discarded.")
        }
        .onAppear { focusRecording = true }
        .onChange(of: confirmStop) { model.captureShortcutsSuspended = confirmStop || pacingFocused }
        .onChange(of: pacingFocused) { model.captureShortcutsSuspended = confirmStop || pacingFocused }
        .onDisappear { model.captureShortcutsSuspended = false }
        .onChange(of: model.sentenceTraining.phase) { if model.sentenceTraining.phase == .ready { focusRecording = true } }
    }

    private var recordingTitle: String {
        switch model.sentenceTraining.currentKind {
        case .sound: TrainMode.sounds.title
        case .speech: TrainMode.speech.title
        case .poetry: TrainMode.poetry.title
        case .drill: TrainMode.language.title
        case .minimalPair: TrainMode.minimalPairs.title
        case .freeText: TrainMode.free.title
        default: TrainMode.reading.title
        }
    }

    /// The countdown follows the word highlighting; after the last word the take waits for Space,
    /// and the safety timeout only shows in its final seconds.
    private var cueTimer: String {
        let workflow = model.sentenceTraining
        if workflow.phase == .ready { return "When you’re ready" }
        if workflow.phase == .ended { return "Saved on this Mac" }
        if workflow.isSavingTake { return "Finishing this take" }
        if showPacingGuide, let left = workflow.guideSecondsRemaining { return String(format: "%.1f s", left) }
        if workflow.phase == .speak || workflow.phase == .mouth {
            return workflow.secondsRemaining < 5 ? String(format: "Stops in %.0f s", ceil(workflow.secondsRemaining)) : "Press Space when done"
        }
        return String(format: "%.1f s", workflow.secondsRemaining)
    }

    private var review: some View {
        let pairs = Array(model.sentenceTraining.data.pairs.reversed())
        let attention = pairs.filter { model.sentenceTraining.use(for: $0).needsAttention }
        let included = pairs.filter { [.included, .warnings].contains(model.sentenceTraining.use(for: $0).state) }
        let other = pairs.filter { [.excluded, .reserved].contains(model.sentenceTraining.use(for: $0).state) }
        return LazyVStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Recording decisions").font(.title2.weight(.semibold))
                Spacer()
                Picker("Motion channels", selection: $channel) {
                    ForEach(MotionPlotChannel.allCases) { Text($0.rawValue).tag($0) }
                }.labelsHidden().frame(maxWidth: 190)
            }
            Text("Usable recordings are included automatically. Word timing failures do not block sentence training; each take is checked separately. Original recordings are kept.")
                .font(.callout).foregroundStyle(.secondary)
            Text("Sentence eligibility applies to the next dataset preparation, which also checks pronunciation coverage. Existing datasets and frozen models do not change. Validation and final-test roles stay separate from training.")
                .font(.caption).foregroundStyle(.secondary)
            if !attention.isEmpty {
                Text("Needs attention (\(attention.count))").font(.headline)
                ForEach(attention) { pairCard($0) }
            }
            if !included.isEmpty {
                Text("Included automatically or by you (\(included.count))").font(.headline)
                ForEach(included) { pairCard($0) }
            }
            if !other.isEmpty {
                MotionDisclosure("Excluded or reserved recordings (\(other.count))") {
                    ForEach(other) { pairCard($0) }
                }
            }
        }
    }

    private func pairCard(_ pair: SentencePair) -> some View {
        let use = model.sentenceTraining.use(for: pair)
        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                Text(pair.text).font(.headline)
                Spacer()
                Text(pair.role.rawValue).font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Label(use.state.rawValue, systemImage: use.state == .included ? "checkmark.circle.fill" : use.state == .excluded ? "minus.circle" : "info.circle")
                    .foregroundStyle(use.state == .included ? Color.green : use.state == .warnings || use.state == .attention ? Color.orange : Color.secondary)
                Text(pair.review == .pending ? "Automatic" : "Manual override").font(.caption).foregroundStyle(.secondary)
            }.font(.headline)
            VStack(alignment: .leading, spacing: 4) {
                Text("Word model: " + (use.wordModel ? "eligible" : "not eligible for aligned word examples"))
                ForEach(use.takes.indices, id: \.self) { i in
                    let take = use.takes[i]
                    Text("Sentence model · " + take.name + ": " + (take.included ? "eligible for next dataset" : "not included — " + (take.reason ?? "unavailable")))
                }
            }.font(.caption).foregroundStyle(.secondary)
            if pair.spoken != nil || pair.silent != nil {
                MotionDisclosure("Inspect recorded motion") {
                    if let spoken = pair.spoken {
                        SignalPlot(title: "Spoken", samples: spoken.cueSamples, channels: channel.channels, minScale: channel.minScale, height: 70)
                        qualityLabel(spoken)
                    }
                    if let silent = pair.silent {
                        SignalPlot(title: "Mouthed", samples: silent.cueSamples, channels: channel.channels, minScale: channel.minScale, height: 70)
                        qualityLabel(silent)
                    }
                }
            }
            if let transcript = pair.transcript {
                Text("Heard: " + transcript.text).font(.callout)
                if pair.review != .excluded,let match=pair.transcriptMatchRate,match>=0.85 {
                    MotionDisclosure("Transcript agreement: \(Int(match*100))%") {
                        Text("Sentence training can include this recording automatically even if word alignment failed. Motion quality and pronunciation coverage are checked separately. Transcript agreement does not measure motion accuracy.")
                    }.font(.caption).foregroundStyle(.secondary)
                }
                if let directory = pair.directory, transcript.timingReview != nil {
                    MotionDisclosure("Check word times against audio") {
                        AudioTimingView(transcript: transcript, audioURL: URL(fileURLWithPath: directory).appending(path: "audio.caf"))
                    }
                }
                if let textRate = pair.transcriptMatchRate, let timingRate = pair.timedWordCoverage {
                    Text(String(format: "Text match %.0f%% · usable timed words %.0f%%", textRate * 100, timingRate * 100))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if pair.usesAlignedWords, pair.wordTimingSource == .forcedAlignment, (pair.transcriptMatchRate ?? 0) < 0.8 {
                Text("Kept for training (words matched to your audio)").font(.caption).foregroundStyle(.green)
            }
            if !pair.complete { Text("Incomplete pair. Re-record to collect both takes.").font(.caption).foregroundStyle(.orange) }
            if let error = pair.error { Text("Word alignment: " + error).font(.caption).foregroundStyle(.secondary) }
            if let note = pair.phoneAlignmentNote { Text(note).font(.caption).foregroundStyle(.secondary) }
            if pair.spokenPhones != nil {
                MotionDisclosure("Phoneme alignment estimates · review before supervision") {
                    ForEach([SpeechCondition.normal,.silent]) { condition in
                        let alignment = condition == .normal ? pair.spokenPhones : pair.mouthedPhones
                        if let alignment {
                            Text(condition.rawValue + " · " + alignment.provenance.backend).font(.headline)
                            Text(alignment.provenance.note).font(.caption)
                            ForEach(alignment.phones.indices, id: \.self) { i in
                                let phone = alignment.phones[i]
                                let origin = condition == .normal ? pair.spoken?.cueStart ?? 0 : pair.silent?.cueStart ?? 0
                                Text(String(format: "/%@/ %.3f–%.3f s from START · confidence %@", phone.ipa, phone.start - origin, phone.end - origin,
                                    phone.confidence.map { String(format: "%.2f", $0) } ?? "unavailable"))
                                    .font(.caption.monospacedDigit())
                            }
                        }
                    }
                    Button("Mark these phone estimates reviewed") { model.sentenceTraining.reviewPhoneAlignment(pair.id) }
                }
            }
            MotionDisclosure("Recording details") {
                if let transfer = pair.alignment {
                    Text(String(format: "%d aligned words · cost %.3f · duration ratio %.2f", transfer.words.count, transfer.normalizedCost, transfer.durationRatio))
                    ForEach(transfer.warnings, id: \.self) { Text($0) }
                    ForEach(transfer.words.indices, id: \.self) { i in
                        let word = transfer.words[i]
                        Text(String(format: "%@ · %.2f–%.2f s", word.word,
                            word.silentStart - (pair.silent?.cueStart ?? 0), word.silentEnd - (pair.silent?.cueStart ?? 0)))
                    }
                }
                if let visit = model.sentenceTraining.data.visits.first(where: { $0.id == pair.visitID }) {
                    Text(visit.startedAt.formatted() + " · " + (visit.audioDevice ?? "Unknown microphone"))
                    if !visit.notes.isEmpty { Text(visit.notes) }
                }
                if let directory = pair.directory {
                    Button("Open raw recording") { NSWorkspace.shared.open(URL(fileURLWithPath: directory)) }
                }
            }.font(.caption).foregroundStyle(.secondary)
            // Non-default alignment warnings are visible before accepting, not hidden in details.
            ForEach(pair.alignment?.warnings.filter { !$0.hasPrefix("Transferred word times") } ?? [], id: \.self) {
                Text($0).font(.caption).foregroundStyle(.orange)
            }
            HStack {
                Menu("Override") {
                    Button("Confirm prompt and include") { model.sentenceTraining.review(pair.id, decision: .accepted) }
                        .disabled(!use.manualInclusionAllowed || pair.review == .accepted)
                    Button("Use automatic decision") { model.sentenceTraining.review(pair.id, decision: .pending) }
                        .disabled(pair.review == .pending)
                }.disabled(use.state == .reserved)
                    .help("Confirmation approves the prompt as the label. Invalid motion and unavailable word timings are still excluded from the affected method.")
                Button(pair.review == .excluded ? "Excluded" : "Exclude pair") { model.sentenceTraining.review(pair.id, decision: .excluded) }
                    .disabled(pair.review == .excluded || use.state == .reserved)
                    .help("Exclude this pair from future training snapshots. The recording is kept.")
                Spacer()
                if pair.complete {
                    Button("Retry alignment") { model.sentenceTraining.retryProcessing(pair.id, whisper: model.localWhisper) }
                        .disabled(!model.localWhisper.isConfigured)
                }
                Button("Re-record") { model.startSentenceTraining(retry: pair.id) }.disabled(!canStart)
            }
        }.padding(16).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
    }

    private func qualityLabel(_ recording: SentenceRecording) -> some View {
        let quality = recording.quality
        return Text(String(format: "%.1f Hz", quality.sampleRate ?? 0)
            + (quality.issues.isEmpty ? "" : " · " + quality.issues.joined(separator: "; ")))
            .font(.caption).foregroundStyle(quality.issues.isEmpty ? Color.secondary : Color.orange)
    }

    private var coverage: some View {
        let workflow = model.sentenceTraining, counts = workflow.coverage
        let covered = Phoneme.all.filter { counts[$0.symbol, default: 0] >= 4 }.count
        let repairs = workflow.repairWords
        return VStack(alignment: .leading, spacing: 12) {
            Divider()
            MotionDisclosure("Sound coverage: \(covered) of \(Phoneme.all.count) units at the collection target") {
                Text("Four dictionary occurrences per sound. Coverage measures collected examples, not recognition accuracy; your pronunciation may differ.")
                    .font(.caption).foregroundStyle(.secondary)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 110))], alignment: .leading, spacing: 10) {
                    ForEach(Phoneme.all) { phone in
                        Text("/\(phone.ipa)/  \(counts[phone.symbol, default: 0])/4").font(.callout)
                    }
                }.padding(.top, 8)
            }
            if !repairs.isEmpty {
                Text("Fill remaining gaps: " + repairs.prefix(8).joined(separator: ", ")).font(.callout)
                Button("Record missing sounds") { model.startSentenceTraining(repairs: true) }.disabled(!canStart)
            }
        }
    }

    private var validation: some View {
        @Bindable var workflow = model.sentenceTraining
        let words = workflow.fittingWords
        let validationCount = workflow.data.derivedWords(role: .validation, lexicon: .bundled).count
        return VStack(alignment: .leading, spacing: 12) {
            Text("Check your model").font(.title2.weight(.semibold))
            Text("\(words.count) eligible word examples. " + (model.isBuildingDecoder ? "Building model…" : model.decoder == nil ? "More usable motion is needed." : "Ready in Try model."))
                .font(.callout).foregroundStyle(.secondary)
            if validationCount > 0 {
                Button("Check validation words") { workflow.validate() }
            } else {
                Text("Review the validation pairs to enable the word check.").font(.caption).foregroundStyle(.secondary)
            }
            if let result = workflow.data.validationReport {
                Text(String(format: "Correct output %.1f%% · accepted %.1f%%", result.acceptedCorrect * 100, result.acceptance * 100)).font(.headline)
                Text(String(format: "Top 1 %.1f%% · top 5 %.1f%% · %d word examples", result.top1 * 100, result.top5 * 100, result.expected.count))
                    .font(.callout).foregroundStyle(.secondary)
                if let total = result.expectedValidationWordCount {
                    Text("\(result.expected.count) of \(total) validation words had usable boundaries and pronunciations.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("This word baseline uses estimated boundaries. Experimental sentence decoding and its separate evaluation are under Advanced → Sentence models.")
                .font(.caption).foregroundStyle(.secondary)
            MotionDisclosure("Model settings") {
                Picker("Learning method", selection: $workflow.learningMethod) {
                    ForEach(LearningMethod.allCases) { Text($0.title).tag($0) }
                }
                Text("Sequence units and word boundaries are estimates; compare their results on validation data.")
                    .font(.caption).foregroundStyle(.secondary)
            }.disabled(workflow.settingsLocked)
        }
    }

    private var reservation: some View {
        let data = model.sentenceTraining.data
        let sentences = data.prompts.filter { $0.kind == .sentence }
        let reserved = sentences.filter { data.roles[$0.id] == .finalTest }
        return Label("\(reserved.count) of \(sentences.count) sentences reserved for the final test", systemImage: "lock.fill")
            .font(.callout).foregroundStyle(.secondary)
    }
}
