import SwiftUI

/// Collection coverage and review for an experiment with a sealed final holdout.
struct TrainView: View {
    @Environment(AppModel.self) private var model
    @State private var inspectedSound: String?

    var body: some View {
        HSplitView {
            Checklist(selectedSound: $inspectedSound)
                .frame(minWidth: 240, idealWidth: 280, maxWidth: 340)
            VStack(spacing: 0) {
                LiveMotionPanel().padding(12)
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        ExperimentPanel()
                        if let session = model.session {
                            SessionPanel(session: session)
                        } else {
                            PlanPanel()
                        }
                        RecordedMotionPanel(sound: inspectedSound)
                    }
                    .padding(20)
                }
            }
            .frame(minWidth: 520)
        }
    }
}

// MARK: Checklist

private struct Checklist: View {
    @Binding var selectedSound: String?
    @Environment(AppModel.self) private var model

    var body: some View {
        let counts = model.phonemeCounts
        let statuses = Dictionary(uniqueKeysWithValues: Phoneme.all.map {
            ($0.symbol, model.status(of: $0.symbol, counts: counts))
        })
        List(selection: $selectedSound) {
            Section {
                Text("All recorded takes").tag("all")
                Text("Training sound coverage")
                    .font(.callout.weight(.medium))
            }
            ForEach(Phoneme.Group.allCases, id: \.self) { group in
                Section(group.rawValue) {
                    ForEach(Phoneme.all.filter { $0.group == group }) { phoneme in
                        PhonemeRow(phoneme: phoneme,
                                   count: counts[phoneme.symbol, default: 0],
                                   target: model.target(for: phoneme.symbol),
                                   status: statuses[phoneme.symbol]!)
                            .tag(phoneme.symbol)
                    }
                }
            }
        }
    }
}

private struct PhonemeRow: View {
    var phoneme: Phoneme
    var count: Int
    var target: Int
    var status: AppModel.PhonemeStatus

    var body: some View {
        HStack(spacing: 10) {
            Text("/\(phoneme.ipa)/")
                .font(.system(.body, design: .serif).weight(.medium))
                .frame(width: 44, alignment: .leading)
            VStack(alignment: .leading, spacing: 1) {
                Text(phoneme.example)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(count)/\(max(target, count))")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Image(systemName: symbol)
                .foregroundStyle(tint)
                .frame(width: 18)
        }
        .help(help)
    }

    private var detail: String { phoneme.symbol }
    private var symbol: String { status == .covered ? "checkmark.circle" : "circle.dotted" }
    private var tint: Color { status == .covered ? .blue : .secondary }
    private var help: String {
        status == .covered ? "Collection target met in accepted fitting takes. This is coverage, not recognition accuracy."
            : "Still collecting accepted training examples. Reserved takes do not count toward fitting coverage."
    }
}

// MARK: Plan

private struct PlanPanel: View {
    @Environment(AppModel.self) private var model
    @State private var confirmReset = false

    var body: some View {
        @Bindable var model = model
        let prompts = model.plannedPrompts
        let counts = model.phonemeCounts
        let short = Phoneme.all.filter { counts[$0.symbol, default: 0] < model.target(for: $0.symbol) }.count

        return GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Record as", selection: $model.sessionCondition) {
                    ForEach(SpeechCondition.allCases) { Text($0.rawValue).tag($0) }
                }
                .disabled(model.experimentLocked || model.isEvaluating)
                TextField("Session notes (fit, posture, pronunciation…)", text: $model.sessionNotes)
                    .textFieldStyle(.roundedBorder)
                if prompts.isEmpty {
                    Text("Fitting coverage targets met").font(.headline)
                    Text("Run decoder validation above to measure unseen-word recognition.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Record \(prompts.count) words").font(.headline)
                    Text("Chosen to give the \(short) unfinished sounds their examples with as few takes as possible, about \(max(1, prompts.count * 4 / 60)) min. Use your selected speaking condition while holding Space. Reserved words are kept out of fitting.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Text(prompts.sorted().joined(separator: " · "))
                        .font(.callout)
                        .lineLimit(3)
                    HStack {
                        Button("Start", systemImage: "play.fill", action: model.startSession)
                            .buttonStyle(.borderedProminent)
                            .disabled(model.motion.status != .streaming || model.experimentLocked || model.isEvaluating || !model.storageReady)
                        if model.motion.status != .streaming {
                            Label(model.motion.status.label, systemImage: "airpodspro")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }
                }
                Divider()
                HStack {
                    Text("\(model.dataset.templates.count) takes recorded")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Archive and start fresh…", role: .destructive) { confirmReset = true }
                        .buttonStyle(.borderless)
                        .font(.caption)
                        .disabled(model.dataset.templates.isEmpty || model.isEvaluating)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
        .confirmationDialog("Archive this experiment and start with an empty training set?",
                            isPresented: $confirmReset) {
            Button("Archive and start fresh", role: .destructive, action: model.resetTraining)
        }
    }
}

// MARK: Session

private struct SessionPanel: View {
    @Environment(AppModel.self) private var model
    var session: TrainingSession

    var body: some View {
        GroupBox {
            VStack(spacing: 14) {
                if let word = session.current {
                    prompt(word)
                } else {
                    finished
                }
            }
            .frame(maxWidth: .infinity)
            .padding(10)
        }
    }

    private func prompt(_ word: String) -> some View {
        VStack(spacing: 14) {
            HStack {
                Text("Word \(session.position + 1) of \(session.prompts.count)")
                    .monospacedDigit()
                Spacer()
                if let next = session.upNext {
                    Text("Next: \(next)").foregroundStyle(.secondary)
                }
            }
            .font(.callout)
            ProgressView(value: session.progress)

            VStack(spacing: 4) {
                Text(word)
                    .font(.system(size: 48, weight: .semibold, design: .rounded))
                Text(transcription(word))
                    .font(.system(.title3, design: .serif))
                    .foregroundStyle(.secondary)
            }
            .contentTransition(.opacity)
            .animation(.snappy, value: session.position)

            HoldToCaptureButton(
                title: model.sessionCondition.rawValue,
                isCapturing: model.capture.isCapturing,
                isEnabled: model.motion.status == .streaming,
                onBegin: model.capture.begin,
                onEnd: model.capture.end)
            if model.motion.status != .streaming {
                Label(model.motion.status.label, systemImage: "airpodspro")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            HStack {
                Button("Undo last", systemImage: "arrow.uturn.backward", action: model.undoSessionTake)
                    .keyboardShortcut("z")
                    .disabled(session.position == 0)
                Button("Skip", systemImage: "forward", action: model.skipSessionPrompt)
                    .keyboardShortcut(.rightArrow)
                Spacer()
                Button("End session", action: model.endSession)
                    .keyboardShortcut(.cancelAction)
            }
            Text("Relax your jaw between words. Undo with ⌘Z if you fumbled one, skip with ⌘→.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var finished: some View {
        VStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 36))
                .foregroundStyle(.green)
            Text("Session complete: \(session.recordedCount) words recorded")
                .font(.headline)
            Text("Review flagged recordings, then run decoder validation. The final test stays sealed.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button("Done", action: model.endSession)
                .keyboardShortcut(.defaultAction)
        }
    }

    private func transcription(_ word: String) -> String {
        let ipa = (model.lexicon.phones(word) ?? []).compactMap { Phoneme.bySymbol[$0]?.ipa }
        return ipa.isEmpty ? "" : "/" + ipa.joined() + "/"
    }
}

/// The original per-take motion plots, with a phoneme filter instead of a word vocabulary.
private struct RecordedMotionPanel: View {
    @Environment(AppModel.self) private var model
    var sound: String?
    @State private var channel = MotionPlotChannel.rotation
    @State private var reviewFilter = "All"

    private var takes: [Template] {
        model.dataset.templates.reversed().filter {
            if reviewFilter == "Needs review", $0.reviewDecision != .pending { return false }
            if reviewFilter == "Accepted", $0.reviewDecision != .accepted { return false }
            if reviewFilter == "Excluded", $0.reviewDecision != .excluded { return false }
            guard let sound, sound != "all" else { return true }
            return model.lexicon.phones($0.label)?.contains(sound) == true
        }
    }

    var body: some View {
        let records = takes
        GroupBox("Recorded motion · \(sound.flatMap { Phoneme.bySymbol[$0]?.ipa } ?? "all sounds") · \(records.count) takes") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Review", selection: $reviewFilter) {
                    ForEach(["All", "Needs review", "Accepted", "Excluded"], id: \.self) { Text($0).tag($0) }
                }.pickerStyle(.segmented)
                Picker("Channels", selection: $channel) {
                    ForEach(MotionPlotChannel.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                Text("Select a sound in the checklist to inspect its takes. Each plot shows the whole recorded word; sound boundaries have not been measured.")
                    .font(.caption).foregroundStyle(.secondary)
                if records.isEmpty {
                    Text("No recorded takes for this sound yet.").foregroundStyle(.secondary)
                }
                LazyVStack(spacing: 12) {
                    ForEach(records) { take in
                        VStack(alignment: .leading, spacing: 5) {
                            HStack {
                                Text(take.label).font(.headline)
                                Text(take.createdAt, style: .time).font(.caption).foregroundStyle(.secondary)
                                Spacer()
                                Text(String(format: "%.2f s · %d samples", take.duration, take.samples.count))
                                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            }
                            TakeReviewDetails(take: take)
                            if !take.quality.invalid {
                            SignalPlot(title: channel.rawValue, samples: take.samples,
                                       channels: channel.channels, minScale: channel.minScale, height: 65)
                            }
                        }
                        .padding(8)
                        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(
                            take.id == model.lastTemplateID ? Color.accentColor : .clear))
                    }
                }
            }
            .padding(6)
        }
    }
}
