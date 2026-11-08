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

extension Font {
    /// IPA in Charis SIL (bundled, SIL Open Font Licence), whose symbols stay distinct at reading
    /// sizes: the system font draws ɾ almost exactly like r, so /peɾo/ and /pero/ looked the same.
    static func ipa(_ size: CGFloat, relativeTo style: Font.TextStyle) -> Font { .custom("Charis SIL", size: size, relativeTo: style) }
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
    /// What one take is ("words", "phrases", "parts", "sentences"), for the cost line under Start.
    var unit = "takes"
    /// Pages with their own scrolling panes (Sounds) turn the page scroll off.
    var scrolls = true
    @ViewBuilder var controls: () -> Controls
    @ViewBuilder var content: () -> Content
    @State private var showWhisperSetup = false

    private var workflow: SentenceTrainingWorkflow { model.sentenceTraining }
    private var motionReady: Bool { model.motion.status == .streaming }
    private var canStart: Bool {
        motionReady && model.localWhisper.isConfigured && !groups.isEmpty && !workflow.isBusy && !model.recorder.isRecording
    }
    /// "6 words · about 2 min": each is said aloud, then mouthed.
    private var cost: String {
        let pace = workflow.pace
        let seconds = groups.reduce(0.0) { $0 + 2 * (pace.baseline + workflow.cueDuration($1.map(\.text).joined(separator: " ")) + pace.recovery) }
        return "\(groups.count) \(unit) · about \(max(1, Int((seconds / 60).rounded()))) min"
    }
    /// Why Start is disabled, in the words of the thing to fix.
    private var blocker: String? {
        if !motionReady { return model.motion.status == .waiting ? "Needs AirPods with head tracking" : model.motion.statusText }
        if !model.localWhisper.isConfigured { return "Needs speech recognition set up" }
        if groups.isEmpty { return "Nothing to practise yet" }
        return nil
    }

    @ViewBuilder private var pageBody: some View {
                        if !model.localWhisper.isConfigured {
                            HStack {
                                Label("Practice needs speech recognition on this Mac (a local Whisper model) to check what you said. Nothing is uploaded.", systemImage: "exclamationmark.triangle")
                                    .foregroundStyle(.orange)
                                Button("Set Up Whisper…") { showWhisperSetup = true }
                            }
                        }
                        if let error = workflow.error { Label(error, systemImage: "xmark.octagon").foregroundStyle(.red) }
                        ProgressStrip()
                        PracticeFeedback(mode: mode, again: { model.startPractice(groups) })
                        HStack(spacing: 16) { controls() }
                        content()
        Text("Keyboard: ⌘↩ start · ⌘I IPA · ⌘1–6 pages · ⌥⌘1–6 language · while recording, Space next and ← redo")
            .font(.callout).foregroundStyle(.secondary)
    }

    var body: some View {
        AppPageLayout(showsMotion: false) {
            HStack(alignment: .center) {
                AppPageHeading(title: title, subtitle: subtitle)
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Button { model.startPractice(groups) } label: { Label(startTitle, systemImage: "record.circle") }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(!canStart)
                        .help(blocker ?? "Shows each one; you say it aloud, then mouth it. ⌘↩")
                        .accessibilityHint(blocker ?? "Command Return")
                    if let blocker, !motionReady {
                        Button(blocker + " ›") { model.screen = .devices }.buttonStyle(.link).font(.callout)
                    } else if !groups.isEmpty {
                        Text(cost + "  ·  ⌘↩").font(.callout).foregroundStyle(.secondary)
                    }
                }
            }
        } content: {
            if workflow.isBusy {
                VStack(spacing: 14) {
                    ProgressView()
                    Text(workflow.phase == .preparing ? "Starting…" : workflow.processingText.isEmpty ? "Saving…" : workflow.processingText)
                    Text("Processing stays on this Mac.").foregroundStyle(.secondary)
                    if workflow.phase == .preparing { Button("Cancel", action: workflow.stop) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if !scrolls {
                VStack(alignment: .leading, spacing: 16) { pageBody }
                    .frame(maxWidth: PageMetrics.contentWidth + 120, maxHeight: .infinity, alignment: .topLeading)
                    .padding(PageMetrics.padding).frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        pageBody
                    }
                    .frame(maxWidth: PageMetrics.contentWidth, alignment: .leading)
                    .padding(PageMetrics.padding).frame(maxWidth: .infinity)
                }
            }
        }
        .sheet(isPresented: $showWhisperSetup) { WhisperSetupSheet() }
        .onChange(of: model.startRequest) { if canStart { model.startPractice(groups) } }
    }
}

/// Takes today and the day streak, the same line on every practice page.
struct ProgressStrip: View {
    @Environment(AppModel.self) private var model
    var body: some View {
        let progress = PracticeProgress(pairs: model.sentenceTraining.data.pairs)
        HStack(spacing: 18) {
            Label(progress.today == 1 ? "1 take today" : "\(progress.today) takes today", systemImage: "checkmark.circle")
            Label(progress.streak == 1 ? "1-day streak" : "\(progress.streak)-day streak", systemImage: "flame")
                .foregroundStyle(progress.streak > 0 ? AnyShapeStyle(.orange) : AnyShapeStyle(.secondary))
        }
        .font(.callout)
        .accessibilityElement(children: .combine)
    }
}

/// Show IPA, with a plain-language explanation one click away.
struct IPAToggle: View {
    @Binding var isOn: Bool
    @State private var explain = false
    var body: some View {
        HStack(spacing: 4) {
            Toggle("Show IPA", isOn: $isOn).help("⌘I")
            Button { explain.toggle() } label: { Image(systemName: "questionmark.circle") }
                .buttonStyle(.borderless).accessibilityLabel("What is IPA?")
                .popover(isPresented: $explain) {
                    Text("IPA (the International Phonetic Alphabet) writes down exactly how a word sounds, one symbol per sound: /ʃ/ is the \"sh\" in \"ship\". You don't need to learn it — press Listen to hear any word. The Sounds page explains every symbol. English follows General American; transcriptions are broad.")
                        .padding(16).frame(width: 340).fixedSize(horizontal: false, vertical: true)
                }
        }
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
        let kept = pairs.filter { Self.verdict($0).kept }.count
        return "\(kept) of \(pairs.count) kept for training"
    }

    /// The decision stored with the pair, in words: kept, practice only, or not used and why.
    static func verdict(_ pair: SentencePair) -> (symbol: String, color: Color, text: String, kept: Bool) {
        let rate = pair.transcriptMatchRate.map { Int(($0 * 100).rounded()) }
        let recognised = rate.map { " · \($0)% of words recognised" } ?? ""
        if pair.transcript == nil && pair.error == nil { return ("clock", .secondary, "Waiting for speech recognition", false) }
        if pair.error == nil && pair.alignment != nil { return ("checkmark.circle.fill", .green, "Kept for training" + recognised, true) }
        if pair.practiceMode == .freeText, let rate, Double(rate) < SentenceTrainingWorkflow.freeTextAgreement * 100 {
            return ("pencil.circle.fill", .orange, "Practice only\(recognised); typed text needs 85%", false)
        }
        return ("xmark.circle.fill", .orange, "Not used for training\(recognised) · " + (pair.error ?? "word timing was missing"), false)
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
        let verdict = Self.verdict(pair)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Image(systemName: verdict.symbol).foregroundStyle(verdict.color).accessibilityHidden(true)
                Text(pair.text).font(.title3)
            }
            Text(verdict.text).font(.callout).foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                if let transcript = pair.transcript {
                    GridRow { Text("Recognised").foregroundStyle(.secondary); Text(transcript.text.trimmingCharacters(in: .whitespaces)) }
                }
                if let check = pair.phoneCheck {
                    GridRow { Text("Expected").foregroundStyle(.secondary); Text("/\(check.expectedPhones.joined())/").font(.ipa(15, relativeTo: .callout)) }
                    GridRow { Text("You said").foregroundStyle(.secondary); Text("/\(check.producedPhones.joined())/").font(.ipa(15, relativeTo: .callout)) }
                    if !check.mismatches.isEmpty {
                        GridRow {
                            Text("Work on").foregroundStyle(.secondary)
                            Text(check.mismatches.prefix(6).map { "\($0.expected ?? "–") → \($0.produced ?? "–")" }.joined(separator: ", "))
                        }
                    }
                }
            }.font(.callout)
            if let check = pair.phoneCheck, check.source != "phone-intervals" {
                Text("Sounds are compared word by word, so a sound inside a correctly recognised word isn't checked yet.")
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
            if let gloss = line.gloss { Text(gloss).font(.callout).foregroundStyle(.secondary) }
            if let pinyin = line.pinyin { Text(pinyin).font(.title3).foregroundStyle(.secondary) }
            if showIPA {
                // Large enough that look-alike symbols (ɾ / r, ɡ / g) are told apart; VoiceOver skips it.
                Text("/" + line.ipa.joined(separator: " ") + "/").font(.ipa(18, relativeTo: .title3)).foregroundStyle(.secondary)
                    .textSelection(.enabled).accessibilityHidden(true)
            }
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
                     startTitle: "Start Reading",
                     groups: piece.map { p in PracticeCorpus.takes(p.lines, language: language).map { $0.map { SentencePrompt($0, kind: .speech, language: language) } } } ?? [],
                     mode: .speech, unit: "parts") {
            IPAToggle(isOn: $showIPA)
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
                            Text("Read it aloud, then mouth it, a few lines at a time. Speech recognition on this Mac checks how much it understood.")
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

    private var drills: [PracticeCorpus.Drill] { model.practice.drills.filter { $0.language == model.language.corpus && model.language.shows(region: $0.region) } }
    private var drill: PracticeCorpus.Drill? { drills.first { $0.id == selection } ?? drills.first }
    private var topics: [String] { drills.compactMap(\.topic).reduce(into: []) { if !$0.contains($1) { $0.append($1) } } }

    var body: some View {
        let language = model.language
        PracticePage(title: "Language", subtitle: "Shadowing: see the phrase, hear it, say it aloud, then mouth it.",
                     startTitle: "Start Drill",
                     groups: drill.map { d in d.phrases.map { [SentencePrompt($0, kind: .drill, language: language)] } } ?? [],
                     mode: .drill, unit: "phrases") {
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
            IPAToggle(isOn: $showIPA)
        } content: {
            if let drill {
                VStack(alignment: .leading, spacing: 12) {
                    if let note = drill.note(language) { Text(note).foregroundStyle(.secondary) }
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
    @State private var roundSets: [[PracticeCorpus.Line]] = []
    @AppStorage("practice.showIPA") private var showIPA = true

    /// Contrasts that are tricky for the learner's own language come first.
    private var sets: [PracticeCorpus.MinimalPairs] {
        let tricky = Set(TrickySounds.ipa(practising: model.language.corpus, speaker: model.nativeLanguage))
        let all = model.practice.minimalPairs.filter { $0.language == model.language.corpus }
        return all.filter { !Set($0.contrast.split(separator: " ").map(String.init)).isDisjoint(with: tricky) }
            + all.filter { Set($0.contrast.split(separator: " ").map(String.init)).isDisjoint(with: tricky) }
    }
    private var contrast: PracticeCorpus.MinimalPairs? { sets.first { $0.id == selection } ?? sets.first }

    var body: some View {
        let language = model.language
        PracticePage(title: "Minimal Pairs", subtitle: "Words that differ by one sound and look alike on the lips. Hear each pair, then say every word.",
                     startTitle: "Start Pairs",
                     groups: round.map { [SentencePrompt($0, kind: .minimalPair, language: language)] },
                     mode: .minimalPair, unit: "words") {
            Picker("Sounds", selection: Binding(get: { contrast?.id }, set: { selection = $0; shuffle() })) {
                ForEach(sets) { Text($0.title).font(.ipa(15, relativeTo: .body)).tag(Optional($0.id)) }
            }.fixedSize()
            Button("Shuffle", systemImage: "shuffle", action: shuffle).disabled(contrast == nil)
            IPAToggle(isOn: $showIPA)
        } content: {
            if contrast == nil {
                ContentUnavailableView("No minimal pairs in \(language.name) yet", systemImage: "circle.grid.2x1", description: Text("Choose another language from the toolbar."))
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    // Each row is one contrast set, side by side, so the difference is easy to see and hear.
                    ForEach(Array(roundSets.enumerated()), id: \.offset) { _, set in
                        HStack(alignment: .top, spacing: 10) {
                            ForEach(Array(set.enumerated()), id: \.offset) { index, line in
                                if index > 0 { Image(systemName: "arrow.left.arrow.right").foregroundStyle(.tertiary).padding(.top, 14).accessibilityHidden(true) }
                                VStack(alignment: .leading, spacing: 8) {
                                    PracticeLineView(line: line, language: language, showIPA: showIPA)
                                    Button { Task { await model.referenceSpeaker.speak(line.text(language), language: language.tag) } } label: {
                                        Label("Listen", systemImage: "speaker.wave.2")
                                    }.accessibilityLabel("Listen to \(line.text(language))")
                                }
                                .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                    Text("In the round the words come in a shuffled order, so you can't predict the next one. Recorded as targeted data for these sounds.")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
        }
        .onAppear { if round.isEmpty { shuffle() } }
        .onChange(of: model.language) { selection = nil; shuffle() }
    }

    private func shuffle() {
        guard let contrast else { round = []; roundSets = []; return }
        var generator = SystemRandomNumberGenerator()
        roundSets = Array(contrast.sets.shuffled(using: &generator).prefix(2))
        round = roundSets.flatMap { $0 }.shuffled(using: &generator)
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
                     startTitle: "Start Practice",
                     groups: sentences.map { [SentencePrompt(text: $0, kind: .freeText, language: language.tag)] },
                     mode: .freeText, unit: "sentences") {
            IPAToggle(isOn: $showIPA)
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
                                    if showIPA { Text("/" + reading.ipa.joined(separator: " ") + "/").font(.ipa(18, relativeTo: .title3)).foregroundStyle(.secondary) }
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

    private var sounds: [PracticeCorpus.Sound] { (model.practice.sounds[model.language.corpus] ?? []).filter { !model.language.absentSounds.contains($0.ipa) } }
    private var sound: PracticeCorpus.Sound? { sounds.first { $0.id == selection } ?? sounds.first }
    private var groups: [String] { sounds.map(\.group).reduce(into: []) { if !$0.contains($1) { $0.append($1) } } }

    var body: some View {
        let language = model.language
        let progress = PracticeProgress(pairs: model.sentenceTraining.data.pairs)
        PracticePage(title: "Sounds", subtitle: "Learn each sound: hear it, see how it's made, then say the example words.",
                     startTitle: "Start Sounds",
                     groups: sound.map { s in s.examples.map { [SentencePrompt($0, kind: .sound, language: language)] } } ?? [],
                     mode: .sound, unit: "words", scrolls: false) {
            IPAToggle(isOn: $showIPA)
        } content: {
            if sounds.isEmpty {
                ContentUnavailableView("No sound guide for \(language.name) yet", systemImage: "waveform", description: Text("Choose another language from the toolbar."))
            } else {
                HStack(alignment: .top, spacing: 24) {
                    ScrollView { VStack(alignment: .leading, spacing: 14) {
                        Text("\(sounds.filter { progress.practised($0, language: language) }.count) of \(sounds.count) sounds practised")
                            .font(.headline)
                        let tricky = TrickySounds.ipa(practising: language.corpus, speaker: model.nativeLanguage).compactMap { ipa in sounds.first { $0.ipa == ipa } }
                        if !tricky.isEmpty {
                            Text("Start here: tricky for \(LanguageMenu.nativeName(model.nativeLanguage)) speakers").font(.headline).foregroundStyle(.orange)
                            chips(tricky, progress: progress, language: language)
                        }
                        ForEach(groups, id: \.self) { group in
                            Text(group).font(.headline).foregroundStyle(.secondary)
                            chips(sounds.filter { $0.group == group }, progress: progress, language: language)
                        }
                        Text("Broad IPA. English sounds follow General American.").font(.callout).foregroundStyle(.secondary)
                    }.padding(.trailing, 8) }.frame(width: 370)
                    // The selected sound stays in view while the grid scrolls.
                    if let sound { ScrollView { detail(sound, language: language) } }
                }
            }
        }
        .onChange(of: model.language) { selection = nil }
    }

    private func chips(_ list: [PracticeCorpus.Sound], progress: PracticeProgress, language: AppLanguage) -> some View {
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 76), spacing: 6)], alignment: .leading, spacing: 6) {
                                ForEach(list) { s in
                                    Button { selection = s.id; Task { await model.referenceSpeaker.speakIPA(s.say, language: language.tag) } } label: {
                                        VStack(spacing: 1) {
                                            Text(s.ipa).font(.ipa(24, relativeTo: .title2))
                                                .overlay(alignment: .topTrailing) {
                                                    if progress.practised(s, language: language) {
                                                        Image(systemName: "checkmark.circle.fill").font(.caption2).foregroundStyle(.green)
                                                            .offset(x: 14, y: -6).accessibilityHidden(true)
                                                    }
                                                }
                                            if s.name != s.ipa {
                                                Text(s.name).font(.caption).foregroundStyle(.secondary).lineLimit(2).multilineTextAlignment(.center)
                                            }
                                        }
                                        .frame(width: 72, height: 62).padding(.horizontal, 2)
                                        .background(s.id == sound?.id ? AnyShapeStyle(.tint.opacity(0.25)) : AnyShapeStyle(.quaternary.opacity(0.35)),
                                                    in: RoundedRectangle(cornerRadius: 8))
                                    }
                                    .buttonStyle(.plain)
                                    .accessibilityLabel("\(s.name), \(s.guide)" + (progress.practised(s, language: language) ? ", practised" : ""))
                                    .accessibilityAddTraits(s.id == sound?.id ? .isSelected : [])
                                }
                            }
    }

    private func detail(_ s: PracticeCorpus.Sound, language: AppLanguage) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 16) {
                Text("/\(s.ipa)/").font(.ipa(56, relativeTo: .largeTitle)).accessibilityLabel("Sound: \(s.name)")
                VStack(alignment: .leading, spacing: 2) {
                    Text(s.name).font(.title2.weight(.semibold))
                    Text(s.guide).foregroundStyle(.secondary)
                }
            }
            Button { Task { await model.referenceSpeaker.speakIPA(s.say, language: language.tag) } } label: {
                Label("Hear the Sound", systemImage: "speaker.wave.2.fill")
            }.controlSize(.large)
            if let tip = s.tip { Label(tip, systemImage: "mouth").font(.body) }
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
            Text("In a round, each word plays first; you say it aloud, then mouth it.").font(.callout).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
