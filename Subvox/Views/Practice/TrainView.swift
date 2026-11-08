import SwiftUI

/// Train: the five practice modes as tabs (docs/macos-ui-checklist.md). Recording, review overlays
/// and the shared paired recording screen take the whole page while they're active.
struct TrainView: View {
    @Environment(AppModel.self) private var model

    private var fullPage: Bool {
        model.showContribute || model.showObservability || model.showNonSpeechCollection || model.showSequenceLearning
            || model.showRepresentation || model.showSignalCheck || model.sentenceTraining.isAcquiring
    }

    var body: some View {
        @Bindable var model = model
        if fullPage { SentenceTrainingView() } else {
            TabView(selection: $model.trainMode) {
                SentenceTrainingView().tabItem { Label(TrainMode.reading.rawValue, systemImage: TrainMode.reading.symbol) }.tag(TrainMode.reading)
                SpeechPracticeView().tabItem { Label(TrainMode.speech.rawValue, systemImage: TrainMode.speech.symbol) }.tag(TrainMode.speech)
                LanguagePracticeView().tabItem { Label(TrainMode.language.rawValue, systemImage: TrainMode.language.symbol) }.tag(TrainMode.language)
                MinimalPairsView().tabItem { Label(TrainMode.minimalPairs.rawValue, systemImage: TrainMode.minimalPairs.symbol) }.tag(TrainMode.minimalPairs)
                FreePracticeView().tabItem { Label(TrainMode.free.rawValue, systemImage: TrainMode.free.symbol) }.tag(TrainMode.free)
            }
            .padding(.top, 8)
        }
    }
}

// MARK: - Shared pieces

/// Header, requirements, Start button and the last round's feedback, the same in every practice tab.
struct PracticePage<Setup: View>: View {
    @Environment(AppModel.self) private var model
    var title: String
    var subtitle: String
    var startTitle: String
    var groups: [[SentencePrompt]]
    var mode: SentencePromptKind
    @ViewBuilder var setup: () -> Setup

    private var workflow: SentenceTrainingWorkflow { model.sentenceTraining }
    private var blocker: String? {
        if model.motion.status != .streaming { return "Connect your AirPods to start." }
        if !model.localWhisper.isConfigured { return "Choose a local Whisper model in Reading → recording settings. Practice needs it to check what you said." }
        if groups.isEmpty { return "Choose something to practise." }
        return nil
    }

    var body: some View {
        AppPageLayout {
            HStack(alignment: .center) {
                AppPageHeading(title: title, subtitle: subtitle)
                Spacer()
                Button { model.startPractice(groups) } label: { Label(startTitle, systemImage: "record.circle") }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(blocker != nil || workflow.isBusy || model.recorder.isRecording)
                    .accessibilityHint(blocker ?? "Shows each phrase, then you say it aloud and mouth it.")
            }
        } content: {
            if workflow.isBusy {
                VStack(spacing: 14) {
                    ProgressView()
                    Text(workflow.processingText.isEmpty ? "Saving…" : workflow.processingText)
                    Text("Processing stays on this Mac.").foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 22) {
                        if let blocker { Label(blocker, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
                        if let error = workflow.error { Label(error, systemImage: "xmark.octagon").foregroundStyle(.red) }
                        setup()
                        PracticeFeedback(mode: mode)
                    }
                    .frame(maxWidth: 900, alignment: .leading).padding(24).frame(maxWidth: .infinity)
                }
            }
        }
    }
}

/// Feedback for the most recent round in this mode: what Whisper heard, whether the take trains,
/// and the expected vs produced phones. Never colour alone: every state has a symbol and text.
struct PracticeFeedback: View {
    @Environment(AppModel.self) private var model
    var mode: SentencePromptKind

    private var pairs: [SentencePair] {
        let ids = model.sentenceTraining.lastPracticePairIDs
        return model.sentenceTraining.data.pairs.filter { ids.contains($0.id) && $0.practiceMode == mode }
    }

    var body: some View {
        if !pairs.isEmpty {
            let matched = pairs.filter { ($0.transcriptMatchRate ?? 0) >= SentenceTrainingWorkflow.freeTextAgreement }.count
            VStack(alignment: .leading, spacing: 12) {
                Divider()
                Text("Last round: \(matched) of \(pairs.count) matched the text").font(.title3.weight(.semibold))
                ForEach(pairs) { pair in row(pair) }
            }
            .onAppear {
                AccessibilityNotification.Announcement("Last round: \(matched) of \(pairs.count) matched the text").post()
            }
        }
    }

    private func row(_ pair: SentencePair) -> some View {
        let rate = pair.transcriptMatchRate
        let ok = (rate ?? 0) >= SentenceTrainingWorkflow.freeTextAgreement && pair.error == nil
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Image(systemName: rate == nil ? "clock" : ok ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(rate == nil ? Color.secondary : ok ? Color.green : Color.orange)
                    .accessibilityHidden(true)
                Text(pair.text).font(.title3)
            }
            if let transcript = pair.transcript {
                Text("Heard: \(transcript.text.trimmingCharacters(in: .whitespaces))").foregroundStyle(.secondary)
            }
            Text(rate == nil ? "Waiting for Whisper" : ok ? "Used for training (\(Int((rate ?? 0) * 100))% of words matched)" : (pair.error ?? "Practice only"))
                .font(.callout)
            if let check = pair.phoneCheck {
                Text("Expected  /\(check.expectedPhones.joined(separator: " "))/").font(.body.monospaced())
                Text("Heard       /\(check.producedPhones.joined(separator: " "))/").font(.body.monospaced())
                if !check.mismatches.isEmpty {
                    Text("Sounds to work on: " + check.mismatches.prefix(6).map { "\($0.expected ?? "–") → \($0.produced ?? "–")" }.joined(separator: ", "))
                        .font(.callout)
                }
                Text(check.source == "phone-intervals" ? "Phones from the audio aligner." : "Phones from the words Whisper heard; sounds inside a correctly heard word aren't checked yet.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12).frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        .accessibilityElement(children: .combine)
    }
}

struct LanguagePicker: View {
    @Binding var language: PracticeLanguage
    var body: some View {
        Picker("Language", selection: $language) {
            ForEach(PracticeLanguage.allCases) { Text($0.name).tag($0) }
        }.pickerStyle(.segmented).frame(maxWidth: 420)
    }
}

/// A line of practice text with its IPA (and pinyin for Mandarin) underneath.
struct PracticeLineView: View {
    var line: PracticeCorpus.Line
    var showIPA: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(line.text).font(.title3)
            if let pinyin = line.pinyin { Text(pinyin).font(.callout).foregroundStyle(.secondary) }
            if showIPA { Text("/" + line.ipa.joined(separator: " · ") + "/").font(.callout.monospaced()).foregroundStyle(.secondary) }
        }
        .accessibilityElement(children: .combine)
    }
}

extension SentencePrompt {
    init(_ line: PracticeCorpus.Line, kind: SentencePromptKind, language: PracticeLanguage) {
        self.init(text: line.text, kind: kind, language: language.rawValue)
    }
}

// MARK: - Tabs

struct SpeechPracticeView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("practice.speech.language") private var language = PracticeLanguage.en
    @State private var selection: String?
    @AppStorage("practice.showIPA") private var showIPA = true

    private var pieces: [PracticeCorpus.Piece] { model.practice.pieces.filter { $0.language == language } }
    private var piece: PracticeCorpus.Piece? { pieces.first { $0.id == selection } ?? pieces.first }

    var body: some View {
        PracticePage(title: "Speech & Poetry", subtitle: "Read a public-domain speech or poem at your own pace. The words highlight as you go.",
                     startTitle: "Read this piece",
                     groups: piece.map { p in [p.lines.map { SentencePrompt($0, kind: .speech, language: language) }] } ?? [],
                     mode: .speech) {
            LanguagePicker(language: $language)
            Picker("Piece", selection: Binding(get: { piece?.id }, set: { selection = $0 })) {
                ForEach(pieces) { Text("\($0.title) — \($0.author) (\($0.kind))").tag(Optional($0.id)) }
            }.frame(maxWidth: 520)
            Toggle("Show IPA", isOn: $showIPA)
            if let piece {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(piece.lines, id: \.self) { PracticeLineView(line: $0, showIPA: showIPA) }
                    Text("\(piece.author), \(piece.year). Public domain in the UK and US.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("You speak the whole piece, then mouth it. Your score is how much of the text Whisper heard. Leaderboards come later.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct LanguagePracticeView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("practice.language.language") private var language = PracticeLanguage.fr
    @State private var selection: String?
    @AppStorage("practice.showIPA") private var showIPA = true

    private var drills: [PracticeCorpus.Drill] { model.practice.drills.filter { $0.language == language } }
    private var drill: PracticeCorpus.Drill? { drills.first { $0.id == selection } ?? drills.first }

    var body: some View {
        PracticePage(title: "Language", subtitle: "Shadowing: see the phrase, hear it, say it aloud, then mouth it.",
                     startTitle: "Start drill",
                     groups: drill.map { d in d.phrases.map { [SentencePrompt($0, kind: .drill, language: language)] } } ?? [],
                     mode: .drill) {
            LanguagePicker(language: $language)
            Picker("Drill", selection: Binding(get: { drill?.id }, set: { selection = $0 })) {
                ForEach(drills) { Text($0.title).tag(Optional($0.id)) }
            }.frame(maxWidth: 420)
            Toggle("Show IPA", isOn: $showIPA)
            if let drill {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(drill.phrases, id: \.self) { line in
                        HStack(alignment: .firstTextBaseline) {
                            PracticeLineView(line: line, showIPA: showIPA)
                            Spacer()
                            Button { Task { await model.referenceSpeaker.speak(line.text, language: language.rawValue) } } label: {
                                Label("Listen", systemImage: "speaker.wave.2")
                            }.accessibilityLabel("Listen to \(line.text)")
                        }
                    }
                }
            }
            Text("The phrase plays with the system voice before each spoken take. Always say it aloud before mouthing it; mouthed answers from memory aren't used.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct MinimalPairsView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("practice.pairs.language") private var language = PracticeLanguage.en
    @State private var selection: String?
    @State private var round: [PracticeCorpus.Line] = []

    private var sets: [PracticeCorpus.MinimalPairs] { model.practice.minimalPairs.filter { $0.language == language } }
    private var contrast: PracticeCorpus.MinimalPairs? { sets.first { $0.id == selection } ?? sets.first }

    var body: some View {
        PracticePage(title: "Minimal Pairs", subtitle: "Short rounds of words that look alike on the lips. Recorded as targeted data.",
                     startTitle: "Play round",
                     groups: round.map { [SentencePrompt($0, kind: .minimalPair, language: language)] },
                     mode: .minimalPair) {
            LanguagePicker(language: $language)
            Picker("Sounds", selection: Binding(get: { contrast?.id }, set: { selection = $0; shuffle() })) {
                ForEach(sets) { Text("/\($0.contrast.replacingOccurrences(of: " ", with: "/ /"))/").tag(Optional($0.id)) }
            }.frame(maxWidth: 300)
            if let contrast {
                VStack(alignment: .leading, spacing: 8) {
                    Text("This round").font(.headline)
                    ForEach(Array(round.enumerated()), id: \.offset) { _, line in PracticeLineView(line: line, showIPA: true) }
                    Button("New round", action: shuffle)
                    Text("\(contrast.sets.count) sets for this contrast; a round takes two of them.").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .onAppear { if round.isEmpty { shuffle() } }
        .onChange(of: language) { selection = nil; shuffle() }
    }

    private func shuffle() {
        guard let contrast else { round = []; return }
        var generator = SystemRandomNumberGenerator()
        round = PracticeCorpus.round(contrast, count: 2, using: &generator)
    }
}

struct FreePracticeView: View {
    @AppStorage("practice.free.language") private var language = PracticeLanguage.en
    @AppStorage("practice.free.text") private var text = ""

    /// One take per sentence, so a long passage is broken into speakable pieces.
    static func sentences(_ text: String) -> [String] {
        var out: [String] = []
        text.enumerateSubstrings(in: text.startIndex..., options: .bySentences) { s, _, _, _ in
            if let s = s?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty { out.append(s) }
        }
        return out
    }

    var body: some View {
        PracticePage(title: "Free Practice", subtitle: "Type or paste anything you want to memorise.",
                     startTitle: "Practise this text",
                     groups: Self.sentences(text).map { [SentencePrompt(text: $0, kind: .freeText, language: language.rawValue)] },
                     mode: .freeText) {
            LanguagePicker(language: $language)
            TextEditor(text: $text)
                .font(.title3).frame(minHeight: 160)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                .accessibilityLabel("Practice text")
            Text("\(Self.sentences(text).count) sentences. A take is used for training only when Whisper agrees with at least 85% of the text; otherwise it stays practice.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
