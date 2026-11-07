import AppKit
import SwiftUI

struct SentenceTrainingView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.colorScheme) private var colorScheme
    private var speak: Color { colorScheme == .dark ? Color(red: 0.34, green: 0.83, blue: 0.52) : Color(red: 0.10, green: 0.46, blue: 0.30) }
    private var mouth: Color { colorScheme == .dark ? Color(red: 0.45, green: 0.67, blue: 1) : Color(red: 0.19, green: 0.37, blue: 0.78) }
    private var relax: Color { colorScheme == .dark ? Color(red: 1, green: 0.74, blue: 0.35) : Color(red: 0.62, green: 0.39, blue: 0.08) }
    @State private var channel = MotionPlotChannel.rotation

    var body: some View {
        let workflow = model.sentenceTraining
        VStack(spacing: 0) {
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Train your model").font(.largeTitle.weight(.semibold))
                    Text("Read aloud. Then mouth the same passage silently.").foregroundStyle(.secondary)
                }
                Spacer()
                if !workflow.isBusy && !workflow.plan.isEmpty {
                    Button("Record next batch") { model.startSentenceTraining() }
                        .buttonStyle(.borderedProminent).controlSize(.large).disabled(!canStart)
                }
            }.padding(24)
            Divider()
            if workflow.isAcquiring {
                cue.padding(24)
            } else if workflow.isBusy {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(workflow.phase == .preparing ? "Starting recording…" : workflow.processingText).font(.title3)
                    Text("Processing stays on this Mac.").foregroundStyle(.secondary)
                    if workflow.phase == .preparing { Button("Cancel", action: workflow.stop) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        setup
                        if let error = workflow.error { Text(error).foregroundStyle(.red) }
                        if !workflow.data.pairs.isEmpty { review }
                        if !workflow.fittingWords.isEmpty { coverage; validation }
                        reservation
                    }.frame(maxWidth: 1000, alignment: .leading)
                        .padding(24).frame(maxWidth: .infinity)
                }
            }
            Divider()
            LiveMotionPanel().padding(16)
        }
    }

    private var canStart: Bool {
        model.motion.status == .streaming && model.localWhisper.isConfigured && model.sentenceTraining.storageReady
            && !model.sentenceTraining.isBusy && !model.recorder.isRecording
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
            Text("Read each prompt aloud, then mouth it silently. Done / next finishes a take early; Redo take repeats it after a mistake. The timer can also advance automatically.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 24) {
                Label("Relax", systemImage: "leaf").foregroundStyle(relax)
                Label("Speak aloud", systemImage: "speaker.wave.2").foregroundStyle(speak)
                Label("Mouth silently", systemImage: "mouth").foregroundStyle(mouth)
            }.font(.headline)
            if model.motion.status != .streaming {
                Label("Connect your AirPods to start.", systemImage: "headphones").foregroundStyle(.orange)
            }
            if !model.localWhisper.isConfigured {
                Label("Choose a local Whisper model in recording settings.", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
            HStack {
                Picker("Read as", selection: $workflow.presentation) {
                    ForEach(SentenceTrainingWorkflow.Presentation.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented)
                Picker("Pace", selection: $workflow.pace) {
                    ForEach(TrainingPace.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented)
            }
            if !workflow.plan.isEmpty {
                Text("Next batch: \(workflow.plan.count) paired prompts, about \(max(1, Int(ceil(workflow.estimatedSeconds / 60)))) minutes.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            DisclosureGroup("Recording settings") {
                VStack(alignment: .leading, spacing: 12) {
                    Stepper("Up to \(workflow.sentenceCount) training sentences per batch", value: $workflow.sentenceCount, in: 2...8, step: 2)
                    Text("Validation sentences are added separately; passages join up to three sentences.")
                        .font(.caption).foregroundStyle(.secondary)
                    Picker("Microphone", selection: $audio.selectedDeviceID) {
                        Text("System default").tag(Optional<UInt32>.none)
                        ForEach(audio.devices) { Text($0.name).tag(Optional($0.id)) }
                    }
                    Text("Use the Mac microphone to keep AirPods in their normal audio mode.").font(.caption).foregroundStyle(.secondary)
                    TextField("Session notes: fit, posture or pronunciation", text: $workflow.notes).textFieldStyle(.roundedBorder)
                    DisclosureGroup("Local transcription" + (whisper.isConfigured ? " · ready" : " · setup needed")) {
                        TextField("Whisper model path", text: $whisper.modelPath).textFieldStyle(.roundedBorder)
                        TextField("Whisper server path", text: $whisper.executablePath).textFieldStyle(.roundedBorder)
                        Text("Existing local model files are reused; no download is required.").font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(.top, 8)
            }
            if !workflow.plan.isEmpty {
                DisclosureGroup("Preview next batch") {
                    ForEach(workflow.plan.indices, id: \.self) { i in
                        Text(workflow.plan[i].map(\.text).joined(separator: " ")).padding(.vertical, 4)
                    }
                }
            }
        }
    }
    private var cue: some View {
        let workflow = model.sentenceTraining
        let active = workflow.phase == .speak || workflow.phase == .mouth
        let color = !active ? relax : workflow.phase == .speak ? speak : mouth
        return VStack(spacing: 18) {
            HStack {
                Text("Recording \(min(workflow.index + 1, workflow.total)) of \(workflow.total)").font(.headline)
                Spacer()
                Text(String(format: "%.1f s", workflow.secondsRemaining)).font(.title.monospacedDigit()).foregroundStyle(color)
            }
            ProgressView(value: Double(workflow.index), total: Double(max(1, workflow.total))).tint(color)
            VStack(spacing: 24) {
                Label(!active ? "Relax" : workflow.phase == .speak ? "Speak aloud" : "Mouth silently",
                      systemImage: !active ? "leaf" : workflow.phase == .speak ? "speaker.wave.2" : "mouth")
                    .font(.system(size: 34, weight: .semibold))
                Text(active ? workflow.currentText : "Breathe.")
                    .font(.system(size: Lexicon.words(in: workflow.currentText).count > 18 ? 44 : 62, weight: .medium, design: .serif))
                    .multilineTextAlignment(.center)
                    .minimumScaleFactor(0.5)
                    .padding(.horizontal, 24)
                Text(!active ? "Let your jaw rest. The text returns at the next cue."
                     : workflow.phase == .speak ? "Read once naturally. Choose Done / next when finished."
                     : "Same text, no voice. Choose Done / next when finished.")
                    .font(.title3).multilineTextAlignment(.center)
            }.foregroundStyle(color)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(24)
                .background(color.opacity(0.10), in: RoundedRectangle(cornerRadius: 18))
            HStack {
                Button("Stop and save", action: workflow.stop).keyboardShortcut(.cancelAction)
                Spacer()
                Button("Redo take", action: { workflow.redoCurrentTake() })
                    .controlSize(.large).disabled(!workflow.canRedoTake)
                    .help("Repeat this take: Left Arrow or Up Arrow")
                Button("Done / next", action: { workflow.finishCurrentTake() })
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(!workflow.canFinishTake)
                    .help("Finish this take early: Right Arrow or Down Arrow")
            }
            HStack {
                Text("Completed takes are saved. Redo replaces only this take.")
                Spacer()
                Text("← / ↑ redo    → / ↓ next")
                    .accessibilityLabel("Left or Up Arrow to redo. Right or Down Arrow for next.")
            }.font(.caption).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var review: some View {
        let pairs = Array(model.sentenceTraining.data.pairs.reversed())
        let pending = pairs.filter { $0.review == .pending }
        let reviewed = pairs.filter { $0.review != .pending }
        return VStack(alignment: .leading, spacing: 14) {
            Divider()
            HStack {
                Text(pending.isEmpty ? "Recordings reviewed" : "Review \(pending.count) recording pairs").font(.title2.weight(.semibold))
                Spacer()
                Picker("Motion channels", selection: $channel) {
                    ForEach(MotionPlotChannel.allCases) { Text($0.rawValue).tag($0) }
                }.labelsHidden().frame(maxWidth: 190)
            }
            if !pending.isEmpty {
                Text("Compare both traces and the spoken transcript before using a pair. Word timings are estimates.")
                    .font(.callout).foregroundStyle(.secondary)
                ForEach(pending) { pairCard($0) }
            }
            if !reviewed.isEmpty {
                DisclosureGroup("Reviewed recordings (\(reviewed.count))") {
                    ForEach(reviewed) { pairCard($0) }
                }
            }
        }
    }

    private func pairCard(_ pair: SentencePair) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                Text(pair.text).font(.headline)
                Spacer()
                Text(pair.role.rawValue).font(.caption).foregroundStyle(.secondary)
            }
            if let spoken = pair.spoken {
                SignalPlot(title: "Spoken", samples: spoken.cueSamples, channels: channel.channels, minScale: channel.minScale, height: 70)
                qualityLabel(spoken)
            }
            if let silent = pair.silent {
                SignalPlot(title: "Mouthed", samples: silent.cueSamples, channels: channel.channels, minScale: channel.minScale, height: 70)
                qualityLabel(silent)
            }
            if let transcript = pair.transcript { Text("Heard: " + transcript.text).font(.callout) }
            if !pair.complete { Text("Incomplete pair. Re-record to collect both takes.").font(.caption).foregroundStyle(.orange) }
            if let error = pair.error { Text(error).font(.caption).foregroundStyle(.orange) }
            DisclosureGroup("Recording details") {
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
                Button(pair.review == .accepted ? "In use" : "Use pair") { model.sentenceTraining.review(pair.id, decision: .accepted) }
                    .disabled(!pair.canAccept || pair.review == .accepted)
                Button("Skip pair") { model.sentenceTraining.review(pair.id, decision: .excluded) }
                    .disabled(pair.review == .excluded)
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
            DisclosureGroup("Sound coverage: \(covered) of \(Phoneme.all.count) units at the collection target") {
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
            Text("\(words.count) reviewed word examples. " + (model.isBuildingDecoder ? "Building model…" : model.decoder == nil ? "More usable motion is needed." : "Ready in Try model."))
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
            Text("This checks words using estimated boundaries. Continuous sentence decoding is not available yet.")
                .font(.caption).foregroundStyle(.secondary)
            DisclosureGroup("Model settings") {
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
