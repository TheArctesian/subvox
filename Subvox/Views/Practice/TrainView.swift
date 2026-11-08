import AppKit
import SwiftUI

/// Train: the five practice modes, chosen from the sidebar (docs/macos-ui-checklist.md). The paired
/// recording screen takes the whole page while recording; Reading's own full-page views (sharing,
/// research tools) show only while Reading is the selected mode.
struct TrainView: View {
    @Environment(AppModel.self) private var model

    private var readingOverlay: Bool {
        model.showContribute || model.showObservability || model.showNonSpeechCollection || model.showSequenceLearning
            || model.showRepresentation || model.showSignalCheck
    }

    var body: some View {
        if model.sentenceTraining.isAcquiring || (model.trainMode == .reading && readingOverlay) { SentenceTrainingView() } else {
            switch model.trainMode {
            case .reading: SentenceTrainingView()
            case .sounds: SoundsView()
            case .speech: SpeechPracticeView()
            case .language: LanguagePracticeView()
            case .minimalPairs: MinimalPairsView()
            case .free: FreePracticeView()
            }
        }
    }
}

enum PageMetrics {
    static let contentWidth: CGFloat = 820
    static let padding: CGFloat = 24
}

// MARK: - Shared page

/// The same structure on every practice page: heading and Start in the header; then any blocker,
/// the last round's results (with Practise Again), the page's controls on one leading-aligned row,
/// and the practice text.
struct PracticePage<Controls: View, Content: View>: View {
    @Environment(AppModel.self) private var model
    var title: String
    var subtitle: String
    var startTitle: String
    var groups: [[SentencePrompt]]
    var mode: SentencePromptKind
    @ViewBuilder var controls: () -> Controls
    @ViewBuilder var content: () -> Content
    @State private var showWhisperSetup = false

    private var workflow: SentenceTrainingWorkflow { model.sentenceTraining }
    private var motionReady: Bool { model.motion.status == .streaming }

    var body: some View {
        AppPageLayout(showsMotion: false) {
            HStack(alignment: .center) {
                AppPageHeading(title: title, subtitle: subtitle)
                Spacer()
                Button { model.startPractice(groups) } label: { Label(startTitle, systemImage: "record.circle") }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(!motionReady || !model.localWhisper.isConfigured || groups.isEmpty || workflow.isBusy || model.recorder.isRecording)
                    .help(groups.isEmpty ? "Nothing to practise yet" : "Shows each phrase; you say it aloud, then mouth it.")
            }
        } content: {
            if workflow.isBusy {
                VStack(spacing: 14) {
                    ProgressView()
                    Text(workflow.phase == .preparing ? "Starting…" : workflow.processingText.isEmpty ? "Saving…" : workflow.processingText)
                    Text("Processing stays on this Mac.").foregroundStyle(.secondary)
                    if workflow.phase == .preparing { Button("Cancel", action: workflow.stop) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        if !motionReady {
                            Label("Connect your AirPods to start.", systemImage: "headphones").foregroundStyle(.orange)
                        }
                        if !model.localWhisper.isConfigured {
                            HStack {
                                Label("Practice needs a local Whisper model to check what you said.", systemImage: "exclamationmark.triangle")
                                    .foregroundStyle(.orange)
                                Button("Set Up Whisper…") { showWhisperSetup = true }
                            }
                        }
                        if let error = workflow.error { Label(error, systemImage: "xmark.octagon").foregroundStyle(.red) }
                        PracticeFeedback(mode: mode, again: { model.startPractice(groups) })
                        HStack(spacing: 16) { controls() }
                        content()
                    }
                    .frame(maxWidth: PageMetrics.contentWidth, alignment: .leading)
                    .padding(PageMetrics.padding).frame(maxWidth: .infinity)
                }
            }
        }
        .sheet(isPresented: $showWhisperSetup) { WhisperSetupSheet() }
    }
}

struct WhisperSetupSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        @Bindable var whisper = model.localWhisper
        VStack(alignment: .leading, spacing: 14) {
            Text("Local Whisper").font(.title2.weight(.semibold))
            Text("Practice uses a Whisper model already on this Mac to hear what you said. Nothing is downloaded.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            TextField("Whisper model path", text: $whisper.modelPath).textFieldStyle(.roundedBorder)
            TextField("Whisper server path", text: $whisper.executablePath).textFieldStyle(.roundedBorder)
            Label(whisper.isConfigured ? "Ready" : "Not found yet", systemImage: whisper.isConfigured ? "checkmark.circle" : "questionmark.circle")
            HStack { Spacer(); Button("Done") { dismiss() }.keyboardShortcut(.defaultAction) }
        }
        .padding(24).frame(width: 520)
    }
}

/// The last round in this mode: what Whisper heard, whether each take trains, and expected vs
/// heard phones. Never colour alone: every state has a symbol and words.
struct PracticeFeedback: View {
    @Environment(AppModel.self) private var model
    var mode: SentencePromptKind
    var again: () -> Void

    private var pairs: [SentencePair] {
        let ids = model.sentenceTraining.lastPracticePairIDs
        return model.sentenceTraining.data.pairs.filter { ids.contains($0.id) && $0.practiceMode == mode }
    }
    private var summary: String {
        let matched = pairs.filter { ($0.transcriptMatchRate ?? 0) >= SentenceTrainingWorkflow.freeTextAgreement }.count
        return "\(matched) of \(pairs.count) matched the text"
    }

    var body: some View {
        if !pairs.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Last round: " + summary).font(.title3.weight(.semibold))
                    Spacer()
                    Button("Done") { model.sentenceTraining.clearPracticeResults() }
                    Button("Practise Again", action: again).buttonStyle(.borderedProminent)
                }
                ForEach(pairs) { row($0) }
            }
            .padding(16)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
            .onChange(of: model.sentenceTraining.lastPracticePairIDs) {
                AccessibilityNotification.Announcement("Last round: " + summary).post()
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
            Text(rate == nil ? "Waiting for Whisper" : ok ? "Used for training (\(Int((rate ?? 0) * 100))% of words matched)" : (pair.error ?? "Practice only"))
                .font(.callout).foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                if let transcript = pair.transcript {
                    GridRow { Text("Heard").foregroundStyle(.secondary); Text(transcript.text.trimmingCharacters(in: .whitespaces)) }
                }
                if let check = pair.phoneCheck {
                    GridRow { Text("Expected").foregroundStyle(.secondary); Text("/\(check.expectedPhones.joined())/") }
                    GridRow { Text("You said").foregroundStyle(.secondary); Text("/\(check.producedPhones.joined())/") }
                    if !check.mismatches.isEmpty {
                        GridRow {
                            Text("Work on").foregroundStyle(.secondary)
                            Text(check.mismatches.prefix(6).map { "\($0.expected ?? "–") → \($0.produced ?? "–")" }.joined(separator: ", "))
                        }
                    }
                }
            }.font(.callout)
            if let check = pair.phoneCheck, check.source != "phone-intervals" {
                Text("Phones come from the words Whisper heard, so a sound inside a correctly heard word isn't checked yet.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

/// A line of practice text, with pinyin (Chinese) and IPA underneath.
struct PracticeLineView: View {
    var line: PracticeCorpus.Line
    var language: AppLanguage
    var showIPA: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(line.text(language)).font(.title2)
            if let pinyin = line.pinyin { Text(pinyin).font(.body).foregroundStyle(.secondary) }
            if showIPA { Text("/" + line.ipa.joined(separator: " ") + "/").font(.body).foregroundStyle(.secondary).textSelection(.enabled) }
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
    }
}

extension SentencePrompt {
    init(_ line: PracticeCorpus.Line, kind: SentencePromptKind, language: AppLanguage) {
        self.init(text: line.text(language), kind: kind, language: language.tag)
    }
}

// MARK: - Speech & Poetry

/// A round portrait of the speaker or author, from Corpus/practice/portraits/<id>.jpg when one is
/// bundled (public domain or freely licensed only), otherwise their initials.
struct PortraitCircle: View {
    var piece: PracticeCorpus.Piece
    var language: AppLanguage
    var size: CGFloat = 44

    private var image: NSImage? {
        guard let url = Bundle.main.url(forResource: piece.id, withExtension: "jpg", subdirectory: "Corpus/practice/portraits") else { return nil }
        return NSImage(contentsOf: url)
    }
    private var initials: String {
        let author = piece.author(language)
        if language.isChinese { return String(author.prefix(1)) }
        return author.split(separator: " ").filter { $0.first?.isUppercase == true }.prefix(2).compactMap(\.first).map(String.init).joined()
    }

    var body: some View {
        Group {
            if let image { Image(nsImage: image).resizable().scaledToFill() } else {
                Circle().fill(.tint.opacity(0.25)).overlay(Text(initials).font(.system(size: size * 0.38, weight: .semibold)))
            }
        }
        .frame(width: size, height: size).clipShape(Circle())
        .overlay(Circle().stroke(.quaternary))
        .accessibilityHidden(true)
    }
}

struct SpeechPracticeView: View {
    @Environment(AppModel.self) private var model
    @State private var selection: String?
    @AppStorage("practice.showIPA") private var showIPA = true

    private var pieces: [PracticeCorpus.Piece] { model.practice.pieces.filter { $0.language == model.language.corpus } }
    private var piece: PracticeCorpus.Piece? { pieces.first { $0.id == selection } ?? pieces.first }

    var body: some View {
        let language = model.language
        PracticePage(title: "Speech & Poetry", subtitle: "Read a public-domain speech or poem at your own pace. The words highlight as you go.",
                     startTitle: "Read This Piece",
                     groups: piece.map { p in [p.lines.map { SentencePrompt($0, kind: .speech, language: language) }] } ?? [],
                     mode: .speech) {
            Toggle("Show IPA", isOn: $showIPA)
        } content: {
            if pieces.isEmpty {
                ContentUnavailableView("No pieces in \(language.name) yet", systemImage: "text.quote", description: Text("Choose another language from the toolbar."))
            } else {
                HStack(alignment: .top, spacing: 20) {
                    VStack(spacing: 4) {
                        ForEach(pieces) { p in
                            Button { selection = p.id } label: { pieceRow(p, selected: p.id == piece?.id) }.buttonStyle(.plain)
                                .accessibilityLabel("\(p.title(language)), \(p.author(language)), \(p.when)")
                                .accessibilityAddTraits(p.id == piece?.id ? .isSelected : [])
                        }
                    }.frame(width: 270)
                    if let piece {
                        VStack(alignment: .leading, spacing: 12) {
                            ForEach(piece.lines, id: \.self) { PracticeLineView(line: $0, language: language, showIPA: showIPA) }
                            Text("\(piece.author(language)) · \(piece.when). Public domain in the UK and US.").font(.caption).foregroundStyle(.secondary)
                            Text("Speak the whole piece, then mouth it. Your score is how much of the text Whisper heard.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
        .onChange(of: model.language) { selection = nil }
    }

    private func pieceRow(_ p: PracticeCorpus.Piece, selected: Bool) -> some View {
        HStack(spacing: 12) {
            PortraitCircle(piece: p, language: model.language)
            VStack(alignment: .leading, spacing: 2) {
                Text(p.title(model.language)).font(.headline).lineLimit(2)
                Text(p.when).font(.subheadline).foregroundStyle(.secondary)
                Text(p.author(model.language)).font(.subheadline).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(8).contentShape(Rectangle())
        .background(selected ? AnyShapeStyle(.tint.opacity(0.18)) : AnyShapeStyle(.clear), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Language drills

struct LanguagePracticeView: View {
    @Environment(AppModel.self) private var model
    @State private var selection: String?
    @AppStorage("practice.showIPA") private var showIPA = true

    private var drills: [PracticeCorpus.Drill] { model.practice.drills.filter { $0.language == model.language.corpus } }
    private var drill: PracticeCorpus.Drill? { drills.first { $0.id == selection } ?? drills.first }
    private var topics: [String] { drills.compactMap(\.topic).reduce(into: []) { if !$0.contains($1) { $0.append($1) } } }

    var body: some View {
        let language = model.language
        PracticePage(title: "Language", subtitle: "Shadowing: see the phrase, hear it, say it aloud, then mouth it.",
                     startTitle: "Start Drill",
                     groups: drill.map { d in d.phrases.map { [SentencePrompt($0, kind: .drill, language: language)] } } ?? [],
                     mode: .drill) {
            Picker("Drill", selection: Binding(get: { drill?.id }, set: { selection = $0 })) {
                if topics.isEmpty {
                    ForEach(drills) { Text($0.title(language)).tag(Optional($0.id)) }
                } else {
                    ForEach(topics, id: \.self) { topic in
                        Section(topic) {
                            ForEach(drills.filter { $0.topic == topic }) { d in
                                Text(d.level.map { "\(d.title(language)) · \($0)" } ?? d.title(language)).tag(Optional(d.id))
                            }
                        }
                    }
                }
            }.fixedSize()
            Toggle("Show IPA", isOn: $showIPA)
        } content: {
            if let drill {
                VStack(alignment: .leading, spacing: 12) {
                    if let note = drill.note { Text(note).foregroundStyle(.secondary) }
                    ForEach(drill.phrases, id: \.self) { line in
                        HStack(alignment: .firstTextBaseline) {
                            PracticeLineView(line: line, language: language, showIPA: showIPA)
                            Spacer()
                            Button { Task { await model.referenceSpeaker.speak(line.text(language), language: language.tag) } } label: {
                                Label("Listen", systemImage: "speaker.wave.2")
                            }.accessibilityLabel("Listen to \(line.text(language))")
                        }
                    }
                    Text("Each phrase plays in a system voice before you say it. Always say it aloud before mouthing it; mouthed answers from memory aren't used.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            } else {
                ContentUnavailableView("No drills in \(language.name) yet", systemImage: "character.bubble", description: Text("Choose another language from the toolbar."))
            }
        }
        .onChange(of: model.language) { selection = nil }
    }
}

// MARK: - Minimal pairs

struct MinimalPairsView: View {
    @Environment(AppModel.self) private var model
    @State private var selection: String?
    @State private var round: [PracticeCorpus.Line] = []
    @AppStorage("practice.showIPA") private var showIPA = true

    private var sets: [PracticeCorpus.MinimalPairs] { model.practice.minimalPairs.filter { $0.language == model.language.corpus } }
    private var contrast: PracticeCorpus.MinimalPairs? { sets.first { $0.id == selection } ?? sets.first }

    var body: some View {
        let language = model.language
        PracticePage(title: "Minimal Pairs", subtitle: "Short rounds of words that look alike on the lips. Recorded as targeted data.",
                     startTitle: "Play Round",
                     groups: round.map { [SentencePrompt($0, kind: .minimalPair, language: language)] },
                     mode: .minimalPair) {
            Picker("Sounds", selection: Binding(get: { contrast?.id }, set: { selection = $0; shuffle() })) {
                ForEach(sets) { Text("/\($0.contrast.replacingOccurrences(of: " ", with: "/ /"))/").tag(Optional($0.id)) }
            }.fixedSize()
            Button("Shuffle", systemImage: "shuffle", action: shuffle).disabled(contrast == nil)
            Toggle("Show IPA", isOn: $showIPA)
        } content: {
            if contrast == nil {
                ContentUnavailableView("No minimal pairs in \(language.name) yet", systemImage: "circle.grid.2x1", description: Text("Choose another language from the toolbar."))
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), alignment: .topLeading)], alignment: .leading, spacing: 12) {
                    ForEach(Array(round.enumerated()), id: \.offset) { _, line in
                        PracticeLineView(line: line, language: language, showIPA: showIPA)
                            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
        .onAppear { if round.isEmpty { shuffle() } }
        .onChange(of: model.language) { selection = nil; shuffle() }
    }

    private func shuffle() {
        guard let contrast else { round = []; return }
        var generator = SystemRandomNumberGenerator()
        round = PracticeCorpus.round(contrast, count: 2, using: &generator)
    }
}

// MARK: - Free practice

struct FreePracticeView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("practice.free.text") private var text = ""
    @AppStorage("practice.showIPA") private var showIPA = true
    @State private var readings: [String: PracticeChecker.Reading] = [:]

    /// One take per sentence, so a long passage is broken into speakable pieces.
    static func sentences(_ text: String) -> [String] {
        var out: [String] = []
        text.enumerateSubstrings(in: text.startIndex..., options: .bySentences) { s, _, _, _ in
            if let s = s?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty { out.append(s) }
        }
        return out
    }

    var body: some View {
        let language = model.language
        let sentences = Self.sentences(text)
        PracticePage(title: "Free Practice", subtitle: "Type or paste anything you want to memorise.",
                     startTitle: "Practise This Text",
                     groups: sentences.map { [SentencePrompt(text: $0, kind: .freeText, language: language.tag)] },
                     mode: .freeText) {
            Toggle("Show IPA", isOn: $showIPA)
        } content: {
            VStack(alignment: .leading, spacing: 10) {
                TextEditor(text: $text)
                    .font(.title2).frame(minHeight: 140).scrollContentBackground(.hidden).padding(6)
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(alignment: .topLeading) {
                        if text.isEmpty {
                            Text("Type or paste text in \(language.name)").font(.title2).foregroundStyle(.tertiary)
                                .padding(.horizontal, 11).padding(.vertical, 6).allowsHitTesting(false)
                        }
                    }
                    .accessibilityLabel("Practice text")
                Text(sentences.isEmpty ? "Each sentence becomes one take." : "\(sentences.count) sentence\(sentences.count == 1 ? "" : "s"). A take trains only when Whisper agrees with at least 85% of it; otherwise it stays practice.")
                    .font(.caption).foregroundStyle(.secondary)
                if !sentences.isEmpty && (showIPA || language.isChinese) {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(sentences, id: \.self) { s in
                            if let reading = readings[language.tag + s] {
                                VStack(alignment: .leading, spacing: 2) {
                                    if let pinyin = reading.pinyin, !pinyin.isEmpty { Text(pinyin).foregroundStyle(.secondary) }
                                    if showIPA { Text("/" + reading.ipa.joined(separator: " ") + "/").foregroundStyle(.secondary) }
                                }.textSelection(.enabled)
                            }
                        }
                    }
                }
            }
        }
        .task(id: "\(showIPA)|\(language.tag)|\(text)") {
            guard showIPA || language.isChinese else { return }
            try? await Task.sleep(for: .milliseconds(700))  // wait for typing to pause
            for s in sentences where readings[language.tag + s] == nil {
                guard !Task.isCancelled else { return }
                if let reading = await model.practiceChecker.phones(text: s, language: language.tag) { readings[language.tag + s] = reading }
            }
        }
    }
}

// MARK: - Sounds

/// Every sound of the practice language: hear it, read how to make it, then say example words
/// aloud and mouth them. Sounds are played from IPA by the system voice for the language.
struct SoundsView: View {
    @Environment(AppModel.self) private var model
    @State private var selection: String?
    @AppStorage("practice.showIPA") private var showIPA = true

    private var sounds: [PracticeCorpus.Sound] { model.practice.sounds[model.language.corpus] ?? [] }
    private var sound: PracticeCorpus.Sound? { sounds.first { $0.id == selection } ?? sounds.first }
    private var groups: [String] { sounds.map(\.group).reduce(into: []) { if !$0.contains($1) { $0.append($1) } } }

    var body: some View {
        let language = model.language
        PracticePage(title: "Sounds", subtitle: "Learn each sound: hear it, see how it's made, then say the example words.",
                     startTitle: "Say the Examples",
                     groups: sound.map { s in s.examples.map { [SentencePrompt($0, kind: .sound, language: language)] } } ?? [],
                     mode: .sound) {
            Toggle("Show IPA", isOn: $showIPA)
        } content: {
            if sounds.isEmpty {
                ContentUnavailableView("No sound guide for \(language.name) yet", systemImage: "waveform", description: Text("Choose another language from the toolbar."))
            } else {
                HStack(alignment: .top, spacing: 24) {
                    VStack(alignment: .leading, spacing: 14) {
                        ForEach(groups, id: \.self) { group in
                            Text(group).font(.headline).foregroundStyle(.secondary)
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 58), spacing: 6)], alignment: .leading, spacing: 6) {
                                ForEach(sounds.filter { $0.group == group }) { s in
                                    Button { selection = s.id; Task { await model.referenceSpeaker.speakIPA(s.say, language: language.tag) } } label: {
                                        VStack(spacing: 1) {
                                            Text(s.ipa).font(.title2)
                                            Text(s.name).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                                        }
                                        .frame(width: 58, height: 52)
                                        .background(s.id == sound?.id ? AnyShapeStyle(.tint.opacity(0.25)) : AnyShapeStyle(.quaternary.opacity(0.35)),
                                                    in: RoundedRectangle(cornerRadius: 8))
                                    }
                                    .buttonStyle(.plain)
                                    .accessibilityLabel("\(s.name), \(s.guide)")
                                    .accessibilityAddTraits(s.id == sound?.id ? .isSelected : [])
                                }
                            }
                        }
                    }.frame(width: 330)
                    if let sound { detail(sound, language: language) }
                }
            }
        }
        .onChange(of: model.language) { selection = nil }
    }

    private func detail(_ s: PracticeCorpus.Sound, language: AppLanguage) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 16) {
                Text("/\(s.ipa)/").font(.system(size: 56, weight: .medium))
                VStack(alignment: .leading, spacing: 2) {
                    Text(s.name).font(.title2.weight(.semibold))
                    Text(s.guide).foregroundStyle(.secondary)
                }
            }
            Button { Task { await model.referenceSpeaker.speakIPA(s.say, language: language.tag) } } label: {
                Label("Hear the Sound", systemImage: "speaker.wave.2.fill")
            }.controlSize(.large)
            if let tip = s.tip { Label(tip, systemImage: "mouth").foregroundStyle(.secondary) }
            Divider()
            Text("Examples").font(.headline)
            ForEach(s.examples, id: \.self) { line in
                HStack(alignment: .firstTextBaseline) {
                    PracticeLineView(line: line, language: language, showIPA: showIPA)
                    Spacer()
                    Button { Task { await model.referenceSpeaker.speak(line.text(language), language: language.tag) } } label: {
                        Label("Listen", systemImage: "speaker.wave.2")
                    }.accessibilityLabel("Listen to \(line.text(language))")
                }
            }
            Text("Each word plays before you say it aloud; then mouth it.").font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
