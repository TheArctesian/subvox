import AppKit
import SwiftUI

struct SentenceTrainingView: View {
    @Environment(AppModel.self) private var model
    private let speak = Color(red: 0.10, green: 0.46, blue: 0.30)
    private let mouth = Color(red: 0.19, green: 0.37, blue: 0.78)
    private let relax = Color(red: 0.62, green: 0.39, blue: 0.08)
    @State private var channel = MotionPlotChannel.rotation

    var body: some View {
        let workflow = model.sentenceTraining
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Sentence training").font(.largeTitle.weight(.semibold))
                    Text("Record · Align · Review · Repair · Check").foregroundStyle(.secondary)
                }
                Spacer()
                if !workflow.isBusy {
                    Button("Record next batch") { model.startSentenceTraining() }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(!canStart || workflow.plan.isEmpty)
                }
            }.padding(24)
            Divider()
            if workflow.isAcquiring {
                cue.padding(24)
            } else if workflow.isBusy {
                VStack(spacing: 18) {
                    ProgressView()
                    Text(workflow.phase == .preparing ? "Starting local recording…" : workflow.processingText)
                        .font(.title3)
                    Text("Audio and motion stay on this Mac. Completed recordings are saved before alignment.")
                        .foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        setup
                        if let error = workflow.error { Text(error).foregroundStyle(.red) }
                        if !workflow.data.pairs.isEmpty { review }
                        coverage
                        validation
                        reservation
                    }.padding(24)
                }
            }
            Divider()
            LiveMotionPanel().padding(16)
        }
    }

    private var canStart: Bool {
        model.motion.status == .streaming && model.localWhisper.isConfigured && model.sentenceTraining.storageReady
            && !model.sentenceTraining.isBusy && !model.signalCheck.isBusy && !model.isEvaluating
            && !model.recorder.isRecording && !model.reading.isReading && model.session?.isFinished != false
    }

    private var setup: some View {
        @Bindable var workflow = model.sentenceTraining
        @Bindable var audio = model.audio
        @Bindable var whisper = model.localWhisper
        let groups = workflow.plan
        return VStack(alignment: .leading, spacing: 14) {
            Text("Speak it. Then mouth it.").font(.title2.weight(.medium))
            Text("Read each sentence aloud, relax, then mouth the same text silently at a natural pace. Local Whisper checks the spoken recording. Motion alignment estimates where its words fall in the silent recording; you review those estimates before they enter training.")
                .foregroundStyle(.secondary)
            if model.motion.status != .streaming {
                Label("Connect your AirPods to start recording.", systemImage: "headphones").foregroundStyle(.orange)
            }
            HStack {
                Picker("Text", selection: $workflow.presentation) {
                    ForEach(SentenceTrainingWorkflow.Presentation.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented)
                Picker("Pace", selection: $workflow.pace) {
                    ForEach(SignalCheckPace.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented)
            }
            Stepper("Up to \(workflow.sentenceCount) training sentences + validation material per visit", value: $workflow.sentenceCount, in: 2...8, step: 2)
            Text("\(groups.count) paired prompts · about \(max(1, Int(ceil(workflow.estimatedSeconds / 60)))) minutes, plus local processing. Short passages join up to three sentences from the same split.")
                .font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 22) {
                Label("Relax", systemImage: "leaf").foregroundStyle(relax)
                Label("Speak aloud", systemImage: "speaker.wave.2").foregroundStyle(speak)
                Label("Mouth silently", systemImage: "mouth").foregroundStyle(mouth)
            }.font(.headline)
            TextField("Visit notes: fit, posture, pronunciation or changes", text: $workflow.notes).textFieldStyle(.roundedBorder)
            Picker("Microphone", selection: $audio.selectedDeviceID) {
                Text("System default").tag(Optional<UInt32>.none)
                ForEach(audio.devices) { Text($0.name).tag(Optional($0.id)) }
            }
            Text("Prefer the built-in microphone. No Space presses are needed. Say each prompt once, then wait; you do not need to stretch your speech to fill the cue.")
                .font(.caption).foregroundStyle(.secondary)
            DisclosureGroup(model.localWhisper.isConfigured ? "Local Whisper ready · \(URL(fileURLWithPath: model.localWhisper.modelPath).lastPathComponent)" : "Choose local Whisper files") {
                TextField("GGML model path", text: $whisper.modelPath).textFieldStyle(.roundedBorder)
                TextField("Local Whisper server path", text: $whisper.executablePath).textFieldStyle(.roundedBorder)
                Text("Your existing OpenWhispr files are reused. No cloud API or expected-text hints.").font(.caption).foregroundStyle(.secondary)
            }
            DisclosureGroup("Upcoming paired text") {
                ForEach(groups.indices, id: \.self) { i in
                    Text(groups[i].map(\.text).joined(separator: " ")).padding(.vertical, 4)
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
                     : workflow.phase == .speak ? "Read once, naturally. Then wait for Relax."
                     : "Same text, no voice. Your timing can differ from the spoken version.")
                    .font(.title3).multilineTextAlignment(.center)
            }.foregroundStyle(color)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(24)
                .background(color.opacity(0.10), in: RoundedRectangle(cornerRadius: 18))
            HStack {
                Text("Completed halves are saved even if you stop.").foregroundStyle(.secondary)
                Spacer()
                Button("Stop and save", action: workflow.stop).keyboardShortcut(.cancelAction)
            }
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var review: some View {
        let pairs = model.sentenceTraining.data.pairs.reversed()
        return VStack(alignment: .leading, spacing: 14) {
            Divider()
            Text("Review the pairs").font(.title2.weight(.semibold))
            Text("Compare both traces, the transcript and the transferred intervals. Low alignment cost does not establish correct timing. Acceptance makes only training-split word slices available to the word model.")
                .font(.callout).foregroundStyle(.secondary)
            Picker("Motion channels", selection: $channel) {
                ForEach(MotionPlotChannel.allCases) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented)
            ForEach(Array(pairs)) { pair in
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text(pair.text).font(.headline)
                        Spacer()
                        Text("\(pair.role.rawValue) · \(pair.review.rawValue)").font(.caption).foregroundStyle(.secondary)
                    }
                    if let spoken = pair.spoken {
                        SignalPlot(title: "Spoken reference", samples: spoken.cueSamples, channels: channel.channels,
                                   minScale: channel.minScale, height: 65)
                        qualityLabel("Spoken", recording: spoken)
                    }
                    if let silent = pair.silent {
                        SignalPlot(title: "Silently mouthed", samples: silent.cueSamples, channels: channel.channels,
                                   minScale: channel.minScale, height: 65)
                        qualityLabel("Silent", recording: silent)
                    }
                    if let transcript = pair.transcript {
                        Text("Whisper: \(transcript.text)").font(.callout)
                        Text("\(Int((pair.transcriptMatchRate ?? 0) * 100))% of expected words matched in order. Word timing is estimated; no phoneme boundaries were measured.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let transfer = pair.alignment {
                        Text(String(format: "%d transferred words · DTW cost %.3f · duration ratio %.2f", transfer.words.count, transfer.normalizedCost, transfer.durationRatio))
                            .font(.caption.monospacedDigit())
                        ForEach(transfer.warnings, id: \.self) { Text($0).font(.caption).foregroundStyle(.orange) }
                        DisclosureGroup("Estimated silent word intervals") {
                            ForEach(transfer.words.indices, id: \.self) { i in
                                let word = transfer.words[i]
                                Text(String(format: "%@ · %.2f–%.2f s", word.word,
                                    word.silentStart - (pair.silent?.cueStart ?? 0), word.silentEnd - (pair.silent?.cueStart ?? 0)))
                                    .font(.caption.monospacedDigit())
                            }
                        }
                    }
                    if !pair.complete { Text("Incomplete pair: the completed half is preserved. Re-record to collect both versions.").font(.caption).foregroundStyle(.orange) }
                    if let error = pair.error { Text(error).font(.caption).foregroundStyle(.orange) }
                    if let visit = model.sentenceTraining.data.visits.first(where: { $0.id == pair.visitID }) {
                        DisclosureGroup("Visit details") {
                            Text(visit.startedAt.formatted() + " · " + (visit.audioDevice ?? "Unknown microphone"))
                            if !visit.notes.isEmpty { Text(visit.notes) }
                            if let pace = visit.pace { Text("Pace: " + pace) }
                            if let directory = pair.directory {
                                Button("Open raw recording") { NSWorkspace.shared.open(URL(fileURLWithPath: directory)) }
                            }
                        }.font(.caption)
                    }
                    HStack {
                        Button("Accept aligned pair") { model.sentenceTraining.review(pair.id, decision: .accepted) }
                            .disabled(!pair.canAccept || pair.review == .accepted)
                        Button("Exclude") { model.sentenceTraining.review(pair.id, decision: .excluded) }
                        Button("Re-record") { model.startSentenceTraining(retry: pair.id) }.disabled(!canStart)
                        if pair.complete {
                            Button("Retry alignment") { model.sentenceTraining.retryProcessing(pair.id, whisper: model.localWhisper) }
                        }
                    }
                }
                .padding(14)
                .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private func qualityLabel(_ name: String, recording: SentenceRecording) -> some View {
        let quality = recording.quality
        return Text(String(format: "%@ · %d samples · %.1f Hz", name, recording.cueSamples.count, quality.sampleRate ?? 0)
                    + (quality.issues.isEmpty ? "" : " · " + quality.issues.joined(separator: "; ")))
            .font(.caption).foregroundStyle(quality.issues.isEmpty ? Color.secondary : Color.orange)
    }

    private var coverage: some View {
        let workflow = model.sentenceTraining
        let counts = workflow.coverage
        let missing = Phoneme.all.filter { counts[$0.symbol, default: 0] < 4 }
        let repairs = workflow.repairWords
        return VStack(alignment: .leading, spacing: 12) {
            Divider()
            Text("Coverage and targeted repairs").font(.title2.weight(.semibold))
            Text("\(39 - missing.count)/39 pronunciation units have four examples in reviewed training words. This is dictionary coverage, not recognition accuracy. Reserved and validation sentences do not contribute.")
                .font(.callout).foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 110))], alignment: .leading, spacing: 10) {
                ForEach(Phoneme.all) { phone in
                    Text("/\(phone.ipa)/  \(counts[phone.symbol, default: 0])/4")
                        .font(.callout).foregroundStyle(counts[phone.symbol, default: 0] >= 4 ? Color.secondary : Color.orange)
                }
            }
            if !repairs.isEmpty {
                Text("Word repairs: \(repairs.prefix(8).joined(separator: ", "))")
                Button("Record targeted word repairs") { model.startSentenceTraining(repairs: true) }
                    .disabled(!canStart)
            } else if !missing.isEmpty {
                Text("Finish collecting and accepting the training sentences first. Isolated-word repairs become available only for gaps that remain.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var validation: some View {
        @Bindable var model = model
        let workflow = model.sentenceTraining
        return VStack(alignment: .leading, spacing: 12) {
            Divider()
            Text("Build and check the word model").font(.title2.weight(.semibold))
            Text("\(workflow.fittingWords.count) reviewed silent word slices are available from training sentences. Motion-transferred boundaries are estimates.")
                .font(.callout).foregroundStyle(.secondary)
            Picker("Learning method", selection: $model.learningMethod) {
                ForEach(LearningMethod.allCases) { Text($0.title).tag($0) }
            }
            HStack {
                Button("Build from reviewed sentences", action: model.useReviewedSentenceModel)
                    .disabled(workflow.fittingWords.isEmpty || model.isEvaluating)
                Button("Check aligned validation words") { workflow.validate(settings: model.settings) }
                    .disabled(workflow.fittingWords.isEmpty || model.isEvaluating)
            }
            if model.usesSentenceModel {
                if model.isBuildingDecoder { Text("Building the sentence-derived word model…").font(.caption) }
                else if let decoder = model.decoder {
                    Text("Active in Decode · \(decoder.wordCount) candidate words · \(decoder.missingPhones.count) missing pronunciation units")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Not enough usable sentence-derived motion to build this method yet.")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            Text("The check uses supplied word boundaries in held-out sentences. It is an assisted segmentation diagnostic, not end-to-end sentence transcription or a final-test score.")
                .font(.caption).foregroundStyle(.secondary)
            if let result = workflow.data.validationReport {
                Text(String(format: "Assisted word check: top 1 %.1f%% · top 5 %.1f%% · %d word slices", result.top1 * 100, result.top5 * 100, result.expected.count))
                    .font(.headline)
                Text(result.settings.method.title + (result.settings == model.settings ? "" : " · settings changed; check again"))
                    .font(.caption).foregroundStyle(.secondary)
                if let total = result.expectedValidationWordCount {
                    Text("\(result.expected.count) of \(total) words in accepted validation text had usable estimated boundaries and dictionary pronunciations. Scores above apply to those slices.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var reservation: some View {
        let data = model.sentenceTraining.data
        let sentences = data.prompts.filter { $0.kind == .sentence }
        let reserved = sentences.filter { data.roles[$0.id] == .finalTest }
        return VStack(alignment: .leading, spacing: 10) {
            Divider()
            Label("\(reserved.count) of \(sentences.count) sentences reserved for the final test", systemImage: "lock.fill")
                .font(.headline)
            Text("Their paired recordings and every repetition stay in the same sentence group. They are excluded from collection plans, alignment-driven coverage, repairs, model fitting and validation. Your previous word experiment has its own unchanged split.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Continuous sentence decoding and its final evaluation are not yet implemented. This workflow creates reviewed, aligned training data and supports the assisted word check above.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
