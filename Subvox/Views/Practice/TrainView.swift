import AppKit
import SwiftUI

/// Train: the practice modes, chosen from the sidebar (docs/macos-ui-checklist.md). The paired
/// recording screen takes the whole page while recording; Reading's research tools show only while Reading is the selected mode.
struct TrainView: View {
    @Environment(AppModel.self) private var model

    private var readingOverlay: Bool {
        model.showObservability || model.showNonSpeechCollection || model.showSequenceLearning
            || model.showRepresentation || model.showSignalCheck
    }

    var body: some View {
        if model.sentenceTraining.isAcquiring || (model.trainMode == .reading && readingOverlay) { SentenceTrainingView() } else {
            switch model.trainMode {
            case .reading: SentenceTrainingView()
            case .sounds: SoundsView()
            case .speech, .poetry: LiteraturePracticeView(tab: model.trainMode).id(model.trainMode)
            case .language: LanguagePracticeView()
            case .minimalPairs: MinimalPairsView()
            case .free: FreePracticeView()
            case .memorization: MemorizationPracticeView()
            }
        }
    }
}

extension Font {
    /// IPA in Charis SIL (bundled, SIL Open Font Licence), whose symbols stay distinct at reading
    /// sizes: the system font draws ɾ almost exactly like r, so /peɾo/ and /pero/ looked the same.
    static func ipa(_ size: CGFloat, relativeTo style: Font.TextStyle) -> Font { .custom("Charis SIL", size: size, relativeTo: style) }
}

/// The last choice on each practice page, per practice language, so a session resumes where it left off.
enum SavedSelection {
    static func get(_ page: String, _ language: AppLanguage) -> String? {
        UserDefaults.standard.string(forKey: "practice.selection.\(page).\(language.tag)")
    }
    static func set(_ page: String, _ language: AppLanguage, _ id: String?) {
        UserDefaults.standard.set(id, forKey: "practice.selection.\(page).\(language.tag)")
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
    var title: LocalizedStringKey
    var subtitle: LocalizedStringKey
    var startTitle: LocalizedStringKey
    var groups: [[SentencePrompt]]
    var mode: SentencePromptKind
    /// What one take is ("words", "phrases", "parts", "sentences"), for the cost line under Start.
    var unit: LocalizedStringResource = "takes"
    /// Pages with their own scrolling panes (Sounds) turn the page scroll off.
    var scrolls = true
    var offersMicrophone = false
    var startRound: (() -> Void)? = nil
    @ViewBuilder var controls: () -> Controls
    @ViewBuilder var content: () -> Content
    @State private var showWhisperSetup = false
    /// IPA lines reveal on the disclosure role whether toggled here or with ⌘I.
    @AppStorage("practice.showIPA") private var showIPA = true

    private var workflow: SentenceTrainingWorkflow { model.sentenceTraining }
    private var motionReady: Bool { model.motion.status == .streaming }
    private func start() {
        if let startRound { startRound() }
        else { model.startPractice(groups) }
    }
    private var canStart: Bool {
        !groups.isEmpty && model.canRecordPractice
    }
    /// "6 words · about 2 min": each is said aloud, then mouthed.
    private var cost: String {
        let pace = workflow.pace
        let seconds = groups.reduce(0.0) { $0 + 2 * (pace.baseline + workflow.cueDuration($1.map(\.text).joined(separator: " ")) + pace.recovery) }
        return String(localized: "\(groups.count) \(String(localized: unit)) · about \(max(1, Int((seconds / 60).rounded()))) min")
    }
    /// Why Start is disabled, in the words of the thing to fix.
    private var blocker: String? {
        if !motionReady { return model.motion.status == .waiting ? String(localized: "Needs AirPods with head tracking") : model.motion.statusText }
        if !model.localWhisper.isConfigured { return String(localized: "Needs speech recognition set up") }
        if !workflow.storageReady { return String(localized: "Local recording storage is unavailable") }
        if workflow.data.reservedUnlockedAt != nil { return String(localized: "The recording collection is locked") }
        if groups.isEmpty { return String(localized: "Nothing to practise yet") }
        if workflow.phase == .processing { return String(localized: "Checking saved takes") }
        return nil
    }

    /// Rows that enter or leave the page flow (busy row, result card); the content below follows them.
    private var arrivalKey: [Bool] {
        [workflow.phase == .processing || workflow.phase == .preparing || workflow.isCheckingRecordings,
         !workflow.lastPracticePairIDs.isEmpty, workflow.error != nil]
    }

    @ViewBuilder private var pageBody: some View {
        if !offersMicrophone && !model.localWhisper.isConfigured {
            HStack {
                Label("Practice needs a local Whisper model to check your speech on this Mac.", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Button("Set Up Whisper…") { showWhisperSetup = true }
            }
        }
        if let error = workflow.error { Label(error, systemImage: "xmark.octagon").foregroundStyle(.red) }
        PracticeFeedback(mode: mode, again: {
            if let round = workflow.lastLiteratureRound, round.kind == mode, mode.isLiterature {
                model.startLiterature(round.groups, pieceID: round.pieceID, title: round.title, isSectionRetry: round.isSectionRetry == true)
            } else if let round = workflow.lastLiteratureRound, round.kind == .freeText, mode == .freeText {
                if let script = model.personalScripts.items.first(where: { $0.pieceID == round.pieceID }) {
                    model.startPersonalScript(script, groups: round.groups, isSectionRetry: round.isSectionRetry == true)
                }
            } else {
                model.startPractice(workflow.lastPracticePlan)
            }
        })
        HStack(spacing: 16) { controls(); Spacer() }
            .disabled(model.voicePractice.isBusy)
        content()
        if mode.isLiterature {
            MotionDisclosure("Word alignment settings") { ForcedAlignerSetup(aligner: workflow.forcedAligner) }
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    var body: some View {
        AppPageLayout(showsMotion: false) {
            HStack(alignment: .center) {
                AppPageHeading(title: title, subtitle: subtitle)
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Button(action: start) {
                        Label(offersMicrophone ? "Record with AirPods" : startTitle, systemImage: offersMicrophone ? "airpods" : "record.circle")
                    }
                        .tint(offersMicrophone ? .secondary : .accentColor)
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(!canStart)
                        .help(blocker ?? String(localized: "Starts a practice session. ⌘↩"))
                        .accessibilityHint(blocker ?? String(localized: "Starts a practice session. Command Return."))
                    if let blocker, !motionReady {
                        Button(blocker + " ›") { model.screen = .devices }.buttonStyle(.link).font(.callout)
                            .disabled(model.voicePractice.isBusy)
                            .accessibilityLabel(blocker + ". " + String(localized: "Opens Devices."))
                    } else if let blocker, !model.localWhisper.isConfigured {
                        Button(blocker + " ›") { showWhisperSetup = true }.buttonStyle(.link).font(.callout)
                            .disabled(model.voicePractice.isBusy)
                    } else if !groups.isEmpty {
                        Text(cost).font(.callout).foregroundStyle(.secondary)
                    }
                }
            }
        } content: {
            if !scrolls {
                VStack(alignment: .leading, spacing: 16) { pageBody }.motion(.arrive, value: arrivalKey).motion(.disclosure, value: showIPA)
                    .frame(maxWidth: PageMetrics.contentWidth + 120, maxHeight: .infinity, alignment: .topLeading)
                    .padding(PageMetrics.padding).frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        pageBody
                    }
                    .motion(.arrive, value: arrivalKey)
                    .motion(.disclosure, value: showIPA)
                    .frame(maxWidth: PageMetrics.contentWidth, alignment: .leading)
                    .padding(PageMetrics.padding).frame(maxWidth: .infinity)
                }
            }
        }
        .sheet(isPresented: $showWhisperSetup) { WhisperSetupSheet() }
        .onChange(of: model.startRequest) { if canStart { start() } }
    }
}

/// Daily practice progress for the window toolbar.
struct PracticeProgressBadge: View {
    @Environment(AppModel.self) private var model
    @State private var showsDetails = false

    var body: some View {
        let progress = PracticeProgress(pairs: model.sentenceTraining.data.pairs)
        let goal = PracticeProgress.dailyGoal
        let visible = progress.today > 0 || progress.streak > 0
        ZStack {
            if visible {
                let streak = progress.streak == 1 ? String(localized: "1 day in a row")
                    : String(localized: "\(progress.streak) days in a row")
                let details = String(localized: "\(progress.today) of \(goal) today · \(streak)")
                Button {
                    showsDetails.toggle()
                } label: {
                    Label {
                        Text("\(progress.today)/\(goal)")
                            .contentTransition(.numericText(value: Double(progress.today)))
                    } icon: {
                        Image(systemName: progress.today >= goal ? "trophy.fill" : "party.popper.fill")
                            .foregroundStyle(.orange)
                            .contentTransition(.symbolEffect(.replace))
                    }
                    .font(.callout)
                    .labelStyle(.titleAndIcon)
                    .fixedSize()
                }
                .help(details)
                .accessibilityLabel(details)
                .accessibilityHint("Shows practice progress")
                .popover(isPresented: $showsDetails, arrowEdge: .top) {
                    Text(details).padding(12)
                }
                .transition(.stateFade)
            }
        }
        .motion(.state, value: progress.today)
        .motion(.state, value: visible)
    }
}

/// Show IPA, with a plain-language explanation one click away.
struct IPAToggle: View {
    @Binding var isOn: Bool
    @State private var explain = false
    var body: some View {
        HStack(spacing: 4) {
            Toggle("Show IPA", isOn: $isOn).help("⌘I").fixedSize()
            Button("What's IPA?") { explain.toggle() }
                .buttonStyle(.link).font(.callout).lineLimit(1).fixedSize()
                .popover(isPresented: $explain) {
                    Text("IPA (the International Phonetic Alphabet) writes down exactly how a word sounds, one symbol per sound: /ʃ/ is the \"sh\" in \"ship\". You can listen to examples without learning these symbols. The Sounds page explains every symbol. English follows General American; transcriptions are broad.")
                        .padding(16).frame(width: 340).fixedSize(horizontal: false, vertical: true)
                }
        }
    }
}

struct ForcedAlignerSetup: View {
    var aligner: ForcedAligner

    var body: some View {
        if !aligner.isInstalled {
            if aligner.isInstalling {
                ProgressView("Setting up local word alignment…").controlSize(.small)
            } else {
                Button("Download word alignment model (1.26 GB)") {
                    Task { await aligner.install() }
                }
                Text("Matches the words you read to your audio on this Mac. Requires uv and eSpeak NG.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let error = aligner.installError {
                Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
        }
    }
}

struct ReferenceButton: View {
    @Environment(AppModel.self) private var model
    let text: String
    let language: AppLanguage
    var compact = true
    var phonetic = false
    var subject: String?
    var listenTitle: LocalizedStringKey = "Listen"

    var body: some View {
        let speaker = model.referenceSpeaker
        let active = speaker.playing == text
        Button {
            if active { speaker.stop() }
            else {
                Task {
                    if phonetic { await speaker.speakIPA(text, language: language.tag) }
                    else { await speaker.speak(text, language: language.tag) }
                }
            }
        } label: {
            if compact {
                Image(systemName: active ? "stop.fill" : "speaker.wave.2").frame(minWidth: 28, minHeight: 28)
            } else {
                // Both titles share one width so neighbours don't shift when playback starts.
                Label {
                    ZStack {
                        Text(listenTitle).opacity(active ? 0 : 1)
                        Text("Stop").opacity(active ? 1 : 0)
                    }
                } icon: {
                    Image(systemName: active ? "stop.fill" : "speaker.wave.2")
                }
            }
        }
        .contentTransition(.symbolEffect(.replace))
        .motion(.state, value: active)
        .help(active ? "Stop reference" : "Listen to reference")
        .accessibilityLabel(active ? "Stop reference" : "Listen to \(subject ?? text)")
        .disabled(model.voicePractice.isBusy)
    }
}

struct WhisperSetupSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    private func chooseFile(_ description: String) -> String? {
        let panel = NSOpenPanel()
        panel.message = description
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url?.path : nil
    }

    var body: some View {
        @Bindable var whisper = model.localWhisper
        VStack(alignment: .leading, spacing: 14) {
            Text("Local Whisper").font(.title2.weight(.semibold))
            Text("Practice uses a Whisper model already on this Mac to hear what you said. Nothing is downloaded.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            LabeledContent("Whisper model") {
                TextField("GGML model path", text: $whisper.modelPath).textFieldStyle(.roundedBorder)
                Button("Choose…") { if let path = chooseFile("Choose a ggml-*.bin model") { whisper.modelPath = path } }
                    .accessibilityLabel("Choose Whisper model")
            }
            if !whisper.hasModel { Text("Choose a readable ggml-*.bin model file.").font(.callout).foregroundStyle(.secondary) }
            LabeledContent("Whisper server") {
                TextField("Server path", text: $whisper.executablePath).textFieldStyle(.roundedBorder)
                Button("Choose…") { if let path = chooseFile("Choose the whisper-server executable") { whisper.executablePath = path } }
                    .accessibilityLabel("Choose Whisper server")
            }
            if !whisper.hasExecutable { Text("Choose the executable whisper-server file.").font(.callout).foregroundStyle(.secondary) }
            if whisper.isConfigured { Label("Ready", systemImage: "checkmark.circle") }
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
    @AccessibilityFocusState private var focusResults: Bool

    private var pairs: [SentencePair] {
        let ids = model.sentenceTraining.lastPracticePairIDs
        return model.sentenceTraining.data.pairs.filter { ids.contains($0.id) && $0.practiceMode == mode }
    }
    private var summary: String {
        let kept = pairs.filter { Self.verdict($0).kept }.count
        return String(localized: "\(kept) of \(pairs.count) kept for training")
    }

    /// The decision stored with the pair, in words: kept, practice only, or not used and why.
    static func verdict(_ pair: SentencePair) -> (symbol: String, color: Color, text: String, kept: Bool) {
        let recognised = pair.transcriptMatchRate.map { String(localized: " · \(Int(($0 * 100).rounded(.down)))% of words recognised") } ?? ""
        if pair.transcript == nil && pair.error == nil && pair.wordTimingSource == nil { return ("clock", .secondary, String(localized: "Waiting for speech recognition"), false) }
        // The same rule the training data uses (SentencePair.usesAlignedWords), not a guess from the view.
        if pair.usesAlignedWords {
            let label = pair.wordTimingSource == .forcedAlignment && (pair.transcriptMatchRate ?? 0) < 0.8
                ? String(localized: "Kept for training (words matched to your audio)")
                : String(localized: "Kept for training") + recognised
            return ("checkmark.circle.fill", .green, label, true)
        }
        if pair.error == SentenceTrainingWorkflow.Failure.practiceOnly.localizedDescription {
            return ("pencil.circle.fill", .orange, String(localized: "Practice only") + recognised + String(localized: "; typed text needs 85%"), false)
        }
        if pair.review == .excluded { return ("xmark.circle.fill", .orange, String(localized: "You excluded this take"), false) }
        if pair.error == nil, !pair.hasReliableTranscript {
            return ("questionmark.circle.fill", .orange, String(localized: "Needs review") + recognised + String(localized: "; training needs 85%"), false)
        }
        return ("xmark.circle.fill", .orange, String(localized: "Not used for training") + recognised + " · " + (pair.error ?? String(localized: "word timing was missing")), false)
    }


    var body: some View {
        if !pairs.isEmpty {
            let processing = pairs.contains { model.sentenceTraining.checkingPairIDs.contains($0.id) }
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    // The summary cross-fades in when checking ends; it never counts up.
                    (processing || mode.isLiterature ? Text("Last round") : Text("Last round: \(summary)")).font(.title3.weight(.semibold)).accessibilityAddTraits(.isHeader)
                        .contentTransition(.opacity)
                        .accessibilityFocused($focusResults)
                    Spacer()
                    Button("Done") { model.sentenceTraining.clearPracticeResults() }
                    Button("Practise Again", action: again).buttonStyle(.borderedProminent)
                        .disabled(!model.sentenceTraining.canStartRecording || model.motion.status != .streaming || !model.localWhisper.isConfigured || model.sentenceTraining.lastPracticePlan.isEmpty)
                }
                if mode.isLiterature {
                    Text("Play your spoken take beside each section in the library.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                if !mode.isLiterature || model.researchTools {
                    MotionDisclosure("Take feedback") {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 10) { ForEach(pairs) { row($0) } }
                        }.frame(maxHeight: 200)
                    }
                    LocalRecordingHistory()
                    Text("Speech recognition guesses what you meant, so it can accept sounds that aren't quite right. Use it to check you were understood, not to judge each sound. For a persistent difficulty, a teacher or speech and language therapist can hear what the app can't.")
                        .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(16)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
            .transition(.asymmetric(insertion: .opacity.combined(with: .move(edge: .top)), removal: .opacity))
            .motion(.state, value: processing)
            .onAppear { focusResults = true }  // VoiceOver lands on the results after a round

        }
    }

    private func row(_ pair: SentencePair) -> some View {
        let verdict = Self.verdict(pair)
        let heardRight: Bool? = pair.transcriptMatchRate.map { $0 >= SentenceTrainingWorkflow.freeTextAgreement }
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Image(systemName: heardRight == nil ? "clock" : heardRight! ? "checkmark.circle" : "arrow.uturn.backward.circle")
                    .foregroundStyle(heardRight == nil ? Color.secondary : heardRight! ? Color.green : Color.secondary).accessibilityHidden(true)
                Text(PracticeLineView.spoken(pair.text, AppLanguage(rawValue: pair.language ?? "en") ?? .en)).font(.title3)
            }
            // First what you'll care about as a learner, then what happens to the take.
            if let heardRight {
                Text(heardRight ? "Words recognised" : "Some words weren't recognised. Background noise or unclear speech can cause this; try once more, a little slower.")
                    .font(.body.weight(.medium)).fixedSize(horizontal: false, vertical: true)
            }
            Text(verdict.text).font(.callout).foregroundStyle(.secondary).contentTransition(.opacity).motion(.state, value: verdict.text)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                if let transcript = pair.transcript {
                    GridRow { Text("Recognised").foregroundStyle(.secondary); Text(transcript.text.trimmingCharacters(in: .whitespaces)) }
                }
                // Only when the recognised words differ: the target sounds, and the sounds of the words
                // that were recognised. Never presented as what the learner actually said.
                if let check = pair.phoneCheck, !check.mismatches.isEmpty {
                    GridRow { Text("Target sounds").foregroundStyle(.secondary); Text("/\(check.expectedPhones.joined())/").font(.ipa(15, relativeTo: .callout)) }
                    GridRow { Text("Recognised words").foregroundStyle(.secondary); Text("/\(check.producedPhones.joined())/").font(.ipa(15, relativeTo: .callout)) }
                }
            }.font(.callout)

        }
        .accessibilityElement(children: .combine)
    }
}

/// A line of practice text, with pinyin (Chinese) and IPA underneath.
struct PracticeLineView: View {
    @Environment(AppModel.self) private var model
    var line: PracticeCorpus.Line
    var language: AppLanguage
    var showIPA: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            if let reading = line.reading(language) {
                Text(reading).font(.callout).foregroundStyle(.secondary)
            }
            Text(Self.spoken(line.text(language), language)).font(.title2)
            if let gloss = line.translation(to: model.nativeLanguage), model.nativeLanguage != language.corpus {
                Text(gloss).font(.callout).foregroundStyle(.secondary)
            }
            if showIPA {
                // Large enough that look-alike symbols (ɾ / r, ɡ / g) are told apart; VoiceOver skips it.
                Text("/" + line.ipa(language).joined(separator: " ") + "/").font(.ipa(18, relativeTo: .title3)).foregroundStyle(.secondary)
                    .textSelection(.enabled).accessibilityHidden(true)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
    }

    /// Text tagged with its language, so VoiceOver reads French as French, not with an English voice.
    static func spoken(_ text: String, _ language: AppLanguage) -> AttributedString {
        var string = AttributedString(text)
        string.languageIdentifier = language.tag
        return string
    }
}

extension SentencePrompt {
    init(_ line: PracticeCorpus.Line, kind: SentencePromptKind, language: AppLanguage, translationLanguage: String = "en") {
        self.init(text: line.text(language), kind: kind, language: language.tag,
                  englishTranslation: translationLanguage == language.corpus ? nil : line.translation(to: translationLanguage),
                  pinyin: language.isChinese ? line.pinyin : nil)
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

enum LiteratureSectionRange {
    static func selection(start: Int, end: Int?, count: Int) -> Range<Int> {
        guard count > 0 else { return 0..<0 }
        let first = min(max(0, start), count - 1)
        let last = min(max(first, end ?? count - 1), count - 1)
        return first..<(last + 1)
    }
}

struct LiteraturePracticeView: View {
    var tab: TrainMode
    private var kind: SentencePromptKind { tab == .poetry ? .poetry : .speech }
    private var selectionKey: String { kind.rawValue }
    @Environment(AppModel.self) private var model
    @State private var selection: String?
    @State private var startPart = 0
    @State private var endPart: Int?
    @State private var repeatSelection = false
    @State private var rehearsalTask: Task<Void, Never>?
    @State private var search = ""
    @State private var showHistory = false
    @State private var playback = SpokenTakePlayer()
    @AppStorage("practice.showIPA") private var showIPA = true
    @AppStorage("practice.slowLiteratureReference") private var slowReference = false

    private var pieces: [PracticeCorpus.Piece] {
        (model.practice.pieces + model.personalPracticePieces).filter {
            $0.language == model.language.corpus && $0.practiceKind == kind
        }
    }
    private var piece: PracticeCorpus.Piece? { pieces.first { $0.id == selection } ?? pieces.first }

    var body: some View {
        let language = model.language
        let isSpeech = kind == .speech
        let parts = piece.map { $0.parts(language: language) } ?? []
        let partKey = "\(language.tag)/\(piece?.id ?? "")"
        let range = LiteratureSectionRange.selection(start: startPart, end: endPart, count: parts.count)
        let partial = range.count != parts.count
        let visiblePieces = search.isEmpty ? pieces : pieces.filter {
            $0.title(language).localizedStandardContains(search) || $0.author(language).localizedStandardContains(search)
        }
        let previewLines = piece?.lines ?? []
        let groups = parts[range].map { part in
            part.map { SentencePrompt($0, kind: kind, language: language, translationLanguage: model.nativeLanguage) }
        }
        let unfinished = model.sentenceTraining.resumableLiterature.first {
            $0.pieceID == piece?.id && $0.language == language.tag
        }
        let startTitle: LocalizedStringKey = partial ? "Record Selected Sections"
            : (unfinished == nil ? (isSpeech ? "Start Speech" : "Start Poem") : "Start New Round")
        return PracticePage(title: isSpeech ? "Speech" : "Poetry",
                     subtitle: isSpeech ? "Rehearse the whole speech or a selected range."
                         : "Rehearse the whole poem or a selected range.",
                     startTitle: startTitle,
                     groups: groups,
                     mode: kind, unit: groups.count == 1 ? "part" : "parts", scrolls: false, offersMicrophone: true,
                     startRound: {
                         if let piece {
                             model.startLiterature(groups, pieceID: piece.id, title: piece.title(language), isSectionRetry: partial)
                         }
                     }) {
            IPAToggle(isOn: $showIPA)
            Button("Recording history", systemImage: "clock.arrow.circlepath") { showHistory.toggle() }
                .labelStyle(.iconOnly).help("Recording history")
                .popover(isPresented: $showHistory) {
                    ScrollView { LocalRecordingHistory(expanded: true).padding(16) }
                        .frame(width: 400, height: 360)
                }
        } content: {
            if pieces.isEmpty {
                ContentUnavailableView("No pieces in \(language.localizedName) yet", systemImage: "text.quote", description: Text("Choose another language from the toolbar."))
            } else {
                HStack(alignment: .top, spacing: 20) {
                    VStack(alignment: .leading, spacing: 8) {
                        TextField("Search titles and authors", text: $search).textFieldStyle(.roundedBorder)
                        if visiblePieces.isEmpty { Text("No matching titles").foregroundStyle(.secondary) }
                        ScrollView { VStack(spacing: 4) {
                        ForEach(visiblePieces) { p in
                            Button { selection = p.id } label: { pieceRow(p, selected: p.id == piece?.id) }.buttonStyle(.plain)
                                .accessibilityLabel("\(p.title(language)), \(p.author(language)), \(p.when)")
                                .accessibilityAddTraits(p.id == piece?.id ? .isSelected : [])
                        }
                        } }
                    }.frame(width: 255)
                        .disabled(model.voicePractice.isBusy)
                    if let piece {
                        ScrollView { LazyVStack(alignment: .leading, spacing: 12) {
                            HStack(alignment: .firstTextBaseline) {
                                Text(piece.title(language)).font(.title2.weight(.semibold)).accessibilityAddTraits(.isHeader)
                                Spacer(minLength: 8)
                                pinButton(piece, language: language)
                            }
                            Text("\(piece.author(language)) · \(piece.when)").font(.subheadline).foregroundStyle(.secondary)
                            if !groups.isEmpty {
                                let target = VoicePracticeTarget(mode: tab.rawValue, language: language, itemID: piece.id,
                                                                 text: groups.map { $0.map(\.text).joined(separator: "\n") }.joined(separator: "\n\n"))
                                VoicePracticePanel(target: target, title: partial ? String(localized: "selected sections") : piece.title(language))
                            }
                            if !partial, model.librarySharingGrant != nil,
                               LibraryPilotCatalog.bundled.allows(pieceID: piece.id, language: language.tag, groups: groups) {
                                Label("Library sharing is on. Eligible sections are submitted in a public pull request.", systemImage: "arrow.up.circle")
                                    .font(.callout).foregroundStyle(.secondary)
                            }
                            if let unfinished {
                                Button("Continue recording", systemImage: "arrow.right.circle") {
                                    model.resumeLiterature(unfinished)
                                }
                                .buttonStyle(.borderedProminent)
                                .disabled(!model.canResumeLiterature)
                                .accessibilityLabel("Continue recording \(piece.title(language))")
                            }
                            HStack {
                                ReferenceButton(text: previewLines.map { $0.text(language) }.joined(separator: " "),
                                                language: language, compact: false,
                                                subject: String(localized: "the full text"))
                                Button("Memorize", systemImage: TrainMode.memorization.symbol) {
                                    MemorizationPracticeView.select(pieceID: piece.id, language: language)
                                    model.openPractice(.memorization)
                                }
                            }
                            .disabled(model.voicePractice.isBusy)
                            rehearsalControls(parts, range: range, language: language, key: partKey)
                                .disabled(model.voicePractice.isBusy)
                            if let error = playback.error {
                                Text(error).font(.callout).foregroundStyle(.orange)
                            }
                            ForEach(parts.indices, id: \.self) { index in
                                section(parts[index], index: index, language: language,
                                        piece: piece,
                                        usesStanzas: kind == .poetry && piece.stanzaEnds?.last == previewLines.count,
                                        canChooseStart: parts.count > 1)
                            }
                            Text(model.personalPracticePieces.contains(where: { $0.id == piece.id })
                                 ? "Personal library" : "Public domain in the UK and US.")
                                .font(.caption).foregroundStyle(.secondary)
                            Text("Each pass shows a few lines at a time. Speech recognition on this Mac checks the spoken pass.")
                                .font(.callout).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading) }
                        .id(piece.id)
                    }
                }
            }
        }
        .onAppear { if selection == nil { selection = SavedSelection.get(selectionKey, model.language) } }
        .modifier(SpokenPlaybackLifecycle(playback: playback))
        .onChange(of: partKey) {
            startPart = 0
            endPart = nil
            playback.stop()
            model.referenceSpeaker.stop()
        }
        .onChange(of: range) { rehearsalTask?.cancel() }
        .onDisappear { rehearsalTask?.cancel() }
        .onChange(of: model.literatureRequest) {
            guard let request = model.literatureRequest, request.mode == tab, request.language == model.language else { return }
            selection = request.pieceID
            search = ""
        }
        .onChange(of: selection) { SavedSelection.set(selectionKey, model.language, selection) }
        .onChange(of: model.language) {
            selection = SavedSelection.get(selectionKey, model.language)
            search = ""
        }
    }

    private func pinButton(_ piece: PracticeCorpus.Piece, language: AppLanguage) -> some View {
        let pin = LiteraturePin(pieceID: piece.id, language: language, mode: tab)
        let pinned = model.literaturePins.contains(pin)
        return Button(pinned ? "Unpin from Home" : "Pin to Home", systemImage: pinned ? "pin.fill" : "pin") {
            model.literaturePins.toggle(pin)
        }
        .labelStyle(.iconOnly)
        .help(pinned ? "Unpin from Home" : "Pin to Home")
        .accessibilityLabel(pinned ? "Unpin \(piece.title(language)) from Home" : "Pin \(piece.title(language)) to Home")
    }

    private func rehearsalControls(_ parts: [[PracticeCorpus.Line]], range: Range<Int>,
                                   language: AppLanguage, key: String) -> some View {
        let referenceKey = "rehearsal/\(key)"
        let isPlaying = model.referenceSpeaker.playing == referenceKey
        return MotionDisclosure("Rehearsal controls") {
            VStack(alignment: .leading, spacing: 10) {
                if parts.count > 1 {
                    HStack {
                        Picker("First section", selection: Binding(get: { range.lowerBound }, set: { first in
                            startPart = first
                            if let endPart, endPart < first { self.endPart = first }
                        })) {
                            ForEach(parts.indices, id: \.self) { index in Text("\(index + 1)").tag(index) }
                        }
                        Picker("Last section", selection: Binding(get: { range.upperBound - 1 }, set: { endPart = $0 })) {
                            ForEach(range.lowerBound..<parts.count, id: \.self) { index in Text("\(index + 1)").tag(index) }
                        }
                        Button("Whole piece") { startPart = 0; endPart = nil }
                            .disabled(range.count == parts.count)
                    }
                }
                Toggle("Slower example", isOn: $slowReference)
                Toggle("Repeat selection", isOn: $repeatSelection)
                Button(isPlaying ? "Stop example" : "Hear selected sections", systemImage: isPlaying ? "stop.fill" : "speaker.wave.2") {
                    if isPlaying {
                        rehearsalTask?.cancel()
                        model.referenceSpeaker.stop()
                    } else {
                        let text = parts[range].map { part in part.map { $0.text(language) }.joined(separator: "\n") }
                            .joined(separator: "\n\n")
                        rehearsalTask?.cancel()
                        rehearsalTask = Task {
                            await model.referenceSpeaker.rehearse(text, language: language.tag, slower: slowReference,
                                                                  repeats: repeatSelection, key: referenceKey)
                        }
                    }
                }
                .disabled(range.isEmpty)
                .onChange(of: slowReference) { if isPlaying { rehearsalTask?.cancel(); model.referenceSpeaker.stop() } }
                .onChange(of: repeatSelection) { if isPlaying { rehearsalTask?.cancel(); model.referenceSpeaker.stop() } }
            }
            .padding(.vertical, 4)
        }
    }

    private func section(_ lines: [PracticeCorpus.Line], index: Int, language: AppLanguage, piece: PracticeCorpus.Piece,
                         usesStanzas: Bool, canChooseStart: Bool) -> some View {
        let number = index + 1
        let title = usesStanzas ? String(localized: "Stanza \(number)") : String(localized: "Section \(number)")
        let takes = SpokenTakePlayer.recentTakes(for: lines, mode: kind, language: language, pieceID: piece.id,
                                                in: model.sentenceTraining.data.pairs)
        let latest = takes.first
        let previous = takes.dropFirst().first
        return VStack(alignment: .leading, spacing: 8) {
            Divider()
            Text(title).font(.headline).accessibilityAddTraits(.isHeader)
            ViewThatFits(in: .horizontal) {
                HStack { sectionPlayback(lines, language: language, title: title, latest: latest, previous: previous) }
                VStack(alignment: .leading, spacing: 6) {
                    sectionPlayback(lines, language: language, title: title, latest: latest, previous: previous)
                }
            }
            .disabled(model.voicePractice.isBusy)
            HStack {
                Button(latest == nil ? "Record section" : "Retry section", systemImage: latest == nil ? "record.circle" : "arrow.counterclockwise") {
                    let prompts = lines.map { SentencePrompt($0, kind: kind, language: language, translationLanguage: model.nativeLanguage) }
                    model.startLiterature([prompts], pieceID: piece.id, title: piece.title(language), isSectionRetry: true)
                }
                .disabled(!model.canResumeLiterature)
                .accessibilityLabel(latest == nil ? "Record \(title)" : "Retry \(title)")
                .help("Record a new silent and spoken take of this section.")
                if canChooseStart {
                    if startPart == index {
                        Label("Starts here", systemImage: "checkmark.circle")
                            .font(.callout).foregroundStyle(.secondary)
                    } else {
                        Button("Choose start") { startPart = index }
                            .help("Start the next session at this section.")
                            .accessibilityLabel("Choose \(title) as the start")
                    }
                }
            }
            .disabled(model.voicePractice.isBusy)
            ForEach(lines.indices, id: \.self) { index in
                PracticeLineView(line: lines[index], language: language, showIPA: showIPA)
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func sectionPlayback(_ lines: [PracticeCorpus.Line], language: AppLanguage, title: String,
                                 latest: (id: UUID, clip: SpokenTakePlayer.Clip)?,
                                 previous: (id: UUID, clip: SpokenTakePlayer.Clip)?) -> some View {
        ReferenceButton(text: lines.map { $0.text(language) }.joined(separator: "\n"),
                        language: language, compact: false, subject: title, listenTitle: "Hear example")
        takeButton(latest, title: previous == nil ? String(localized: "Play my take") : String(localized: "Play latest"), section: title)
        if let previous { takeButton(previous, title: String(localized: "Play previous"), section: title) }
    }

    private func takeButton(_ take: (id: UUID, clip: SpokenTakePlayer.Clip)?, title: String, section: String) -> some View {
        let isPlaying = take != nil && take?.id == playback.playing
        return Button(isPlaying ? String(localized: "Stop") : title, systemImage: isPlaying ? "stop.fill" : "play.fill") {
            guard let take else { return }
            model.referenceSpeaker.stop()
            playback.toggle(take.clip, id: take.id)
        }
        .disabled(take == nil || !model.sentenceTraining.canStartRecording)
        .help(take == nil ? "Record this section aloud to hear your take." : title)
        .accessibilityLabel(isPlaying ? "Stop playback for \(section)" : "\(title), \(section)")
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

enum LanguageDrillRounds {
    static func phrases(in drill: PracticeCorpus.Drill, language: AppLanguage) -> [[PracticeCorpus.Line]] {
        let available = drill.phrases.filter { language.shows(region: $0.region) }
        return stride(from: 0, to: available.count, by: 3).map { start in
            Array(available[start..<min(start + 3, available.count)])
        }
    }
}

struct LanguagePracticeView: View {
    @Environment(AppModel.self) private var model
    @State private var selection: String?
    @State private var selectedRound: Int?
    @State private var recall = false
    @State private var revealed: Set<Int> = []
    @State private var playback = SpokenTakePlayer()
    @State private var saved = SavedLanguagePhrases()
    @State private var showingSaved = false
    @State private var reviewCards: [LanguageReviewCard] = []
    @State private var reviewIndex = 0
    @State private var reviewRevealed = false
    @State private var reviewCompleted = false
    @AppStorage("practice.showIPA") private var showIPA = true

    private var drills: [PracticeCorpus.Drill] { model.practice.drills.filter { $0.language == model.language.corpus && model.language.shows(region: $0.region) } }
    private var drill: PracticeCorpus.Drill? { drills.first { $0.id == selection } ?? drills.first { $0.topic == "Everyday life" } ?? drills.first }
    /// Everyday phrases first: what a returning learner needs most, then the grammar topics.
    private var topics: [String] {
        let all = drills.compactMap(\.topic).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
        return all.filter { $0 == "Everyday life" } + all.filter { $0 != "Everyday life" }
    }

    var body: some View {
        Group {
            if showingSaved { savedPage }
            else { practicePage }
        }
        .modifier(SpokenPlaybackLifecycle(playback: playback))
        .onAppear { if selection == nil { selection = SavedSelection.get("drill", model.language) } }
        .onChange(of: selection) { SavedSelection.set("drill", model.language, selection) }
        .onChange(of: model.language) {
            selection = SavedSelection.get("drill", model.language)
            selectedRound = nil
            resetPreview()
            resetReview()
        }
        .onChange(of: showingSaved) {
            resetPreview()
            if !showingSaved { resetReview() }
        }
        .task(id: model.practiceReviewRequest) {
            guard model.takePracticeReview(.language) else { return }
            showingSaved = true
            startSavedReview()
        }
    }

    private var languageTabs: some View {
        Picker("Language activity", selection: $showingSaved) {
            Text("Practice").tag(false)
            Text("Saved phrases").tag(true)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .fixedSize()
    }

    private var practicePage: some View {
        let language = model.language
        let rounds = drill.map { LanguageDrillRounds.phrases(in: $0, language: language) } ?? []
        let savedRound = drill.flatMap { SavedSelection.get("drill-round-\($0.id)", language) }.flatMap(Int.init)
        let preferredRound = selectedRound ?? savedRound ?? 0
        let roundIndex = rounds.indices.contains(preferredRound) ? preferredRound : 0
        let phrases = rounds.indices.contains(roundIndex) ? rounds[roundIndex] : []
        return PracticePage(title: "Language", subtitle: "Listen to a few phrases, then try saying them aloud.",
                     startTitle: "Start Round",
                     groups: phrases.map { [SentencePrompt($0, kind: .drill, language: language, translationLanguage: model.nativeLanguage)] },
                     mode: .drill, unit: phrases.count == 1 ? "phrase" : "phrases", offersMicrophone: true) {
            languageTabs
        } content: {
            if let drill {
                VStack(alignment: .leading, spacing: 16) {
                    drillControls(rounds: rounds, roundIndex: roundIndex, language: language)
                        .disabled(model.voicePractice.isBusy)
                    if !phrases.isEmpty {
                        let target = VoicePracticeTarget(mode: TrainMode.language.rawValue, language: language, itemID: drill.id,
                                                         text: phrases.map { $0.text(language) }.joined(separator: "\n"))
                        VoicePracticePanel(target: target, title: drill.title(language))
                    }
                    Label(listeningTip(for: drill, language: language, roundIndex: roundIndex), systemImage: "ear")
                        .font(.callout).fixedSize(horizontal: false, vertical: true)
                    if let note = drill.note(language) {
                        MotionDisclosure("Usage note") {
                            Text(note).font(.callout).foregroundStyle(.secondary)
                        }
                    }
                    if !phrases.isEmpty {
                        Toggle("Try from memory", isOn: $recall)
                            .onChange(of: recall) { clearPreview() }
                            .disabled(model.voicePractice.isBusy)
                        if recall {
                            Text("Say the phrase from memory, then reveal it and listen.")
                                .font(.callout).foregroundStyle(.secondary)
                        }
                        if let error = playback.error {
                            Text(error).font(.callout).foregroundStyle(.orange)
                        }
                        ForEach(phrases.indices, id: \.self) { index in
                            phraseRow(phrases[index], drillID: drill.id, index: index, language: language)
                        }
                    }
                }
            } else {
                ContentUnavailableView("No drills in \(language.localizedName) yet", systemImage: "character.bubble", description: Text("Choose another language from the toolbar."))
            }
        }
    }

    private func drillControls(rounds: [[PracticeCorpus.Line]], roundIndex: Int, language: AppLanguage) -> some View {
        HStack(spacing: 16) {
            Picker("Practice goal", selection: Binding(get: { drill?.id }, set: { id in
                selection = id
                selectedRound = nil
                resetPreview()
            })) {
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
            if rounds.count > 1 {
                Picker("Round", selection: Binding(get: { roundIndex }, set: { index in
                    selectedRound = index
                    if let drill { SavedSelection.set("drill-round-\(drill.id)", language, String(index)) }
                    resetPreview()
                })) {
                    ForEach(rounds.indices, id: \.self) { index in
                        Text("\(index + 1) of \(rounds.count)").tag(index)
                    }
                }.fixedSize()
            }
            IPAToggle(isOn: $showIPA)
        }
    }

    private func phraseRow(_ line: PracticeCorpus.Line, drillID: String, index: Int, language: AppLanguage) -> some View {
        let id = LanguagePhraseID(line, drillID: drillID, language: language)
        return VStack(alignment: .leading, spacing: 8) {
            Divider()
            if recall && !revealed.contains(index) {
                if let clue = line.translation(to: model.nativeLanguage), clue != line.text(language) {
                    Text(clue).font(.headline)
                } else {
                    Text("Phrase \(index + 1)").font(.headline)
                }
                Button("Reveal phrase", systemImage: "eye") { revealed.insert(index) }
                    .accessibilityHint("Shows the phrase to compare with what you said")
            } else {
                phraseAnswer(line, language: language) {
                    let bookmarked = saved.contains(id)
                    Button(bookmarked ? "Saved" : "Save phrase",
                           systemImage: bookmarked ? "bookmark.fill" : "bookmark") { saved.toggle(id) }
                        .accessibilityLabel(bookmarked ? "Remove saved phrase \(id.text)" : "Save phrase \(id.text)")
                }
            }
        }
    }

    private func phraseAnswer<Actions: View>(_ line: PracticeCorpus.Line, language: AppLanguage,
                                             @ViewBuilder actions: () -> Actions) -> some View {
        let take = SpokenTakePlayer.latestTake(for: [line], mode: .drill, language: language,
                                               in: model.sentenceTraining.data.pairs)
        let playing = take != nil && take?.id == playback.playing
        return VStack(alignment: .leading, spacing: 8) {
            PracticeLineView(line: line, language: language, showIPA: showIPA)
            HStack {
                ReferenceButton(text: line.text(language), language: language,
                                compact: false, listenTitle: "Hear example")
                if let take {
                    Button(playing ? "Stop" : "Play my take", systemImage: playing ? "stop.fill" : "play.fill") {
                        model.referenceSpeaker.stop()
                        playback.toggle(take.clip, id: take.id)
                    }
                    .disabled(!model.sentenceTraining.canStartRecording)
                    .accessibilityLabel(playing ? Text("Stop playback") : Text("Play my take of \(line.text(language))"))
                    .accessibilityHint("Plays your latest spoken recording of this phrase")
                }
                actions()
            }
            .disabled(model.voicePractice.isBusy)
        }
    }

    private var savedPage: some View {
        let language = model.language
        let cards = saved.cards(in: drills, language: language)
        return AppPageLayout(showsMotion: false) {
            AppPageHeading(title: "Language", subtitle: "Revisit phrases you saved and try them from memory.")
        } content: {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    HStack(spacing: 16) {
                        languageTabs
                        IPAToggle(isOn: $showIPA)
                    }
                    .disabled(model.voicePractice.isBusy)
                    if reviewCards.indices.contains(reviewIndex) {
                        reviewCard(reviewCards[reviewIndex], language: language)
                    } else {
                        savedList(cards, language: language)
                    }
                }
                .frame(maxWidth: PageMetrics.contentWidth, alignment: .leading)
                .padding(PageMetrics.padding).frame(maxWidth: .infinity)
            }
        }
    }

    @ViewBuilder private func savedList(_ cards: [LanguageReviewCard], language: AppLanguage) -> some View {
        if cards.isEmpty {
            ContentUnavailableView("No saved phrases", systemImage: "bookmark",
                                   description: Text("Save a phrase in Practice to review it here."))
        } else {
            HStack {
                if reviewCompleted {
                    Label("Review complete", systemImage: "checkmark.circle")
                        .accessibilityAddTraits(.isHeader)
                }
                Spacer()
                Button(reviewCompleted ? "Start another review" : "Start review", systemImage: "arrow.clockwise", action: startSavedReview)
                .buttonStyle(.borderedProminent)
            }
            LazyVStack(alignment: .leading, spacing: 12) {
                ForEach(cards) { card in
                    Divider()
                    HStack(alignment: .firstTextBaseline) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(PracticeLineView.spoken(card.id.text, language)).font(.headline)
                            Text(card.drillTitle).font(.callout).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Remove", systemImage: "bookmark.slash") { saved.remove(card.id) }
                            .accessibilityLabel("Remove saved phrase \(card.id.text)")
                    }
                }
            }
        }
    }

    private func startSavedReview() {
        resetReview()
        reviewCards = saved.nextRound(in: drills, language: model.language)
    }

    private func reviewCard(_ card: LanguageReviewCard, language: AppLanguage) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Phrase \(reviewIndex + 1) of \(reviewCards.count)").font(.headline)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Button("Back to saved phrases") { resetReview() }
                    .disabled(model.voicePractice.isBusy)
            }
            Text(card.drillTitle).font(.callout).foregroundStyle(.secondary)
            if reviewRevealed {
                phraseAnswer(card.line, language: language) { EmptyView() }
                VoicePracticePanel(target: VoicePracticeTarget(mode: TrainMode.language.rawValue, language: language,
                                                               itemID: "review/\(card.id.drillID)", text: card.id.text),
                                   title: String(localized: "saved phrase"))
                if let error = playback.error { Text(error).font(.callout).foregroundStyle(.orange) }
                Button(reviewIndex + 1 == reviewCards.count ? "Finish review" : "Next phrase",
                       systemImage: "arrow.right") {
                    saved.markReviewed(card.id)
                    clearPreview()
                    reviewRevealed = false
                    reviewIndex += 1
                    reviewCompleted = reviewIndex == reviewCards.count
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.voicePractice.isBusy)
            } else {
                if let clue = card.translationCue(in: model.nativeLanguage) {
                    Text(clue).font(.title2)
                } else if let first = card.id.text.first(where: { $0.isLetter || $0.isNumber }) {
                    Text("Starts with \(String(first))").font(.title2)
                }
                Text("Say the phrase from memory, then reveal it to check.")
                    .font(.callout).foregroundStyle(.secondary)
                Button("Reveal phrase", systemImage: "eye") { reviewRevealed = true }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private func resetReview() {
        reviewCards = []
        reviewIndex = 0
        reviewRevealed = false
        reviewCompleted = false
        clearPreview()
    }

    private func clearPreview() {
        revealed.removeAll()
        model.referenceSpeaker.stop()
        playback.stop()
    }

    private func resetPreview() {
        recall = false
        clearPreview()
    }

    private func listeningTip(for drill: PracticeCorpus.Drill, language: AppLanguage,
                              roundIndex: Int) -> LocalizedStringKey {
        switch drill.id {
        case "en-coffee" where roundIndex == 0:
            "Listen for the emphasis on the drink name, then give it the same emphasis in your order."
        case "fr-cafe" where roundIndex == 0:
            "Listen to s’il vous plaît as one group. Try saying it without a pause between the words."
        case "es-cafe" where roundIndex == 0, "es-tacos-mx" where roundIndex == 0:
            "Listen to how por favor flows at the end of an order, then repeat it in one breath."
        case "zh-canting": "Listen to how the pitch changes through each syllable, then repeat the phrase aloud."
        default:
            switch language {
            case .en: "Listen for the word that stands out. Give that word a little extra emphasis when you repeat the phrase."
            case .fr: "Listen for words that run together. Repeat that group as one unit."
            case .es, .esMX: "Listen for the stressed syllable in a longer word, then keep the same stress when you say the phrase."
            case .zhHans, .zhHant: "Listen to the pitch of one syllable, then repeat it aloud before saying the full phrase."
            }
        }
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
        // Ordered by how high the contrast's sounds rank in the speaker's tricky list (b/v before p/b/m).
        let tricky = TrickySounds.ipa(practising: model.language.corpus, speaker: model.nativeLanguage)
        func rank(_ set: PracticeCorpus.MinimalPairs) -> Int {
            set.contrast.split(separator: " ").compactMap { tricky.firstIndex(of: String($0)) }.min() ?? Int.max
        }
        let all = model.practice.minimalPairs.filter { $0.language == model.language.corpus }
        return all.enumerated().sorted { (rank($0.element), $0.offset) < (rank($1.element), $1.offset) }.map(\.element)
    }
    private var contrast: PracticeCorpus.MinimalPairs? { sets.first { $0.id == selection } ?? sets.first }

    var body: some View {
        let language = model.language
        PracticePage(title: "Minimal Pairs", subtitle: "Words that differ by one sound and look alike on the lips. Hear each pair, then say every word.",
                     startTitle: "Start Pairs",
                     groups: round.map { [SentencePrompt($0, kind: .minimalPair, language: language, translationLanguage: model.nativeLanguage)] },
                     mode: .minimalPair, unit: "words") {
            Picker("Sounds", selection: Binding(get: { contrast?.id }, set: { selection = $0; withMotion(.reorder) { shuffle() } })) {
                ForEach(sets) { Text($0.title).font(.ipa(15, relativeTo: .body)).tag(Optional($0.id)) }
            }.fixedSize()
            Button("Shuffle", systemImage: "shuffle") { withMotion(.reorder) { shuffle() } }.disabled(contrast == nil)
            IPAToggle(isOn: $showIPA)
        } content: {
            if contrast == nil {
                ContentUnavailableView("No minimal pairs in \(language.localizedName) yet", systemImage: "circle.grid.2x1", description: Text("Choose another language from the toolbar."))
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    // Each row is one contrast set, side by side, so the difference is easy to see and hear.
                    // Rows are identified by their words, so a shuffle moves them rather than rewriting them in place.
                    ForEach(roundSets, id: \.self) { set in
                        HStack(alignment: .top, spacing: 10) {
                            ForEach(Array(set.enumerated()), id: \.element) { index, line in
                                if index > 0 { Image(systemName: "arrow.left.arrow.right").foregroundStyle(.tertiary).padding(.top, 14).accessibilityHidden(true) }
                                VStack(alignment: .leading, spacing: 8) {
                                    PracticeLineView(line: line, language: language, showIPA: showIPA)
                                    ReferenceButton(text: line.text(language), language: language)
                                }
                                .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                                // Opaque ground, so rows passing each other mid-shuffle don't show through.
                                .background(Color(nsColor: .windowBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                            }
                        }
                        .transition(.fadeInAfterReorder)
                    }
                    // What each contrasted sound is, in words, so the difference doesn't rest on the symbols alone.
                    let guide = (model.practice.sounds[language.corpus] ?? [])
                    ForEach(contrast?.contrast.split(separator: " ").map(String.init) ?? [], id: \.self) { symbol in
                        if let sound = guide.first(where: { $0.ipa == symbol }) {
                            let local = sound.text(for: model.nativeLanguage)
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text("/\(symbol)/").font(.ipa(20, relativeTo: .title3))
                                Text("\(local.name): \(local.guide)").font(.body)
                            }
                        }
                    }
                    Text("The round plays these words in a shuffled order.").font(.callout).foregroundStyle(.secondary)
                }
            }
        }
        .onAppear { if round.isEmpty { shuffle() } }
        .onAppear { if selection == nil { selection = SavedSelection.get("pairs", model.language) } }
        .onChange(of: selection) { SavedSelection.set("pairs", model.language, selection) }
        .onChange(of: model.language) { selection = SavedSelection.get("pairs", model.language); shuffle() }
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
    @Environment(\.undoManager) private var undoManager
    @AppStorage("practice.free.text") private var text = ""
    @AppStorage("practice.free.title") private var title = ""
    @AppStorage("practice.free.selectedScript") private var selectedID = ""
    @AppStorage("practice.showIPA") private var showIPA = true
    @State private var readings: [String: PracticeChecker.Reading] = [:]
    @State private var playback = SpokenTakePlayer()
    @State private var confirmDelete = false
    @State private var confirmDiscard = false
    @State private var nextScript: PersonalScript?

    private var selected: PersonalScript? { model.personalScripts.items.first { $0.id.uuidString == selectedID } }

    private var savedDraft: PersonalScript? {
        guard let selected, selected.language == model.language,
              selected.text.utf8.elementsEqual(text.trimmingCharacters(in: .whitespacesAndNewlines).utf8) else { return nil }
        return selected
    }

    private func load(_ script: PersonalScript) {
        model.referenceSpeaker.stop()
        playback.stop()
        model.language = script.language
        selectedID = script.id.uuidString
        title = script.title
        text = script.text
    }

    private func newDraft() {
        model.referenceSpeaker.stop()
        playback.stop()
        selectedID = ""
        title = ""
        text = ""
    }

    private func saveDraft() {
        guard let script = model.personalScripts.save(id: selected?.id, title: title, text: text, language: model.language) else { return }
        load(script)
    }

    private func switchDraft(to script: PersonalScript?) {
        if let script, savedDraft == script, title == script.title { return }
        let hasDraft = !text.isEmpty || !title.isEmpty
        if hasDraft && (savedDraft == nil || selected?.title != title) {
            nextScript = script
            confirmDiscard = true
        } else {
            applyScript(script)
        }
    }

    private func applyScript(_ script: PersonalScript?) {
        if let script { load(script) } else { newDraft() }
    }

    private func replaceDraft(with replacement: String) {
        let draft = $text
        let previous = draft.wrappedValue
        guard previous != replacement else { return }
        func register(_ value: String, restoring other: String, on manager: UndoManager) {
            manager.registerUndo(withTarget: manager) { target in
                draft.wrappedValue = value
                register(other, restoring: value, on: target)
            }
        }
        if let undoManager {
            register(previous, restoring: replacement, on: undoManager)
            undoManager.setActionName("Replace Practice Text")
        }
        draft.wrappedValue = replacement
    }

    static func example(_ language: AppLanguage) -> String {
        switch language {
        case .fr: "Bonjour, je m’appelle Claire. J’habite à Lyon depuis dix ans."
        case .es, .esMX: "Hola, me llamo Ana. Vivo en la ciudad desde hace diez años."
        case .zhHans: "你好，我叫小林。我在这个城市住了十年。"
        case .zhHant: "你好，我叫小林。我在這個城市住了十年。"
        case .en: "Hello, my name is Sam. I have lived in this city for ten years."
        }
    }

    /// One take per sentence, so a long passage is broken into speakable pieces.
    static func sentences(_ text: String) -> [String] {
        PersonalScript.sections(in: text)
    }

    var body: some View {
        let language = model.language
        let sentences = Self.sentences(text).filter { !Lexicon.words(in: $0).isEmpty }
        let groups = sentences.map { sentence in
            [SentencePrompt(text: sentence, kind: .freeText, language: language.tag,
                            pinyin: language.isChinese ? readings[language.tag + sentence]?.pinyin : nil)]
        }
        let scriptText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        PracticePage(title: "My Material", subtitle: "Practise a script, presentation or interview answer.",
                     startTitle: "Start Practice", groups: groups, mode: .freeText, unit: "sentences", offersMicrophone: true, startRound: {
                         if let script = savedDraft { model.startPersonalScript(script, groups: groups) }
                         else { model.startPractice(groups) }
                     }) {
            IPAToggle(isOn: $showIPA)
        } content: {
            VStack(alignment: .leading, spacing: 10) {
                scriptControls
                    .disabled(model.voicePractice.isBusy)
                if !scriptText.isEmpty {
                    let draftTarget = VoicePracticeTarget(mode: TrainMode.free.rawValue, language: language, itemID: "draft", text: scriptText)
                    let savedTarget = savedDraft.map {
                        VoicePracticeTarget(mode: TrainMode.free.rawValue, language: language, itemID: $0.pieceID, text: scriptText)
                    }
                    VoicePracticePanel(target: savedTarget ?? draftTarget, title: savedDraft?.title ?? String(localized: "script"),
                                       draftTarget: savedTarget == nil ? nil : draftTarget)
                }
                HStack {
                    Button("Paste", systemImage: "doc.on.clipboard") { replaceDraft(with: NSPasteboard.general.string(forType: .string) ?? text) }
                    Button("Try an Example", systemImage: "text.badge.plus") { replaceDraft(with: Self.example(language)) }
                    if !text.isEmpty { Button("Clear", systemImage: "xmark.circle") { replaceDraft(with: "") } }
                }
                .disabled(model.voicePractice.isBusy)
                TextField("Script title", text: $title).textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Script title")
                    .disabled(model.voicePractice.isBusy)
                Text("Practice text").font(.headline)
                if model.voicePractice.isBusy {
                    Text(scriptText).font(.title2).textSelection(.enabled)
                } else {
                    TextEditor(text: $text)
                        .font(.title2).frame(minHeight: 140).scrollContentBackground(.hidden).padding(6)
                        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                        .overlay(alignment: .topLeading) {
                            if text.isEmpty {
                                Text("Type or paste text in \(language.localizedName)").font(.title2).foregroundStyle(.tertiary)
                                    .padding(.horizontal, 11).padding(.vertical, 6).allowsHitTesting(false)
                            }
                        }
                        .accessibilityLabel("Practice text")
                }
                if !sentences.isEmpty {
                    ReferenceButton(text: sentences.joined(separator: " "), language: language, compact: false, listenTitle: "Hear script")
                        .disabled(model.voicePractice.isBusy)
                }
                if let script = savedDraft {
                    savedScriptPractice(script, groups: groups)
                        .disabled(model.voicePractice.isBusy)
                } else if !sentences.isEmpty {
                    Text("Save this script to resume it and replay individual takes.")
                        .font(.callout).foregroundStyle(.secondary)
                }
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
        .modifier(SpokenPlaybackLifecycle(playback: playback))
        .onAppear(perform: openRequestedScript)
        .onChange(of: model.personalScriptRequest) { openRequestedScript() }
        .onChange(of: language) { model.referenceSpeaker.stop(); playback.stop() }
        .confirmationDialog("Delete this saved script?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete Script", role: .destructive) {
                guard let selected else { return }
                model.personalScripts.remove(selected.id)
                newDraft()
            }
        } message: {
            Text("Saved recordings remain in your recording history.")
        }
        .confirmationDialog("Discard unsaved edits?", isPresented: $confirmDiscard, titleVisibility: .visible) {
            Button("Discard Edits", role: .destructive) { applyScript(nextScript); nextScript = nil }
            Button("Keep Editing", role: .cancel) { nextScript = nil }
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

    private func openRequestedScript() {
        if let request = model.takePersonalScriptRequest() { switchDraft(to: request) }
    }

    private var scriptControls: some View {
        HStack {
            if !model.personalScripts.items.isEmpty {
                Menu("Saved scripts", systemImage: "folder") {
                    ForEach(model.personalScripts.items) { script in
                        Button("\(script.title) (\(script.language.name))") { switchDraft(to: script) }
                    }
                }.fixedSize()
            }
            Button("New Script", systemImage: "plus") { switchDraft(to: nil) }
            Button(selected == nil ? "Save Script" : "Save Changes", systemImage: "checkmark", action: saveDraft)
                .buttonStyle(.borderedProminent)
                .disabled(title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || (savedDraft != nil && selected?.title == title))
            if selected != nil {
                Button("Delete Script…", systemImage: "trash") { confirmDelete = true }
            }
        }
    }

    private func savedScriptPractice(_ script: PersonalScript, groups: [[SentencePrompt]]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("Memorize Script", systemImage: TrainMode.memorization.symbol) {
                    MemorizationPracticeView.select(pieceID: script.pieceID, language: script.language)
                    model.openPractice(.memorization)
                }
                if let round = model.resumablePractice.first(where: { $0.pieceID == script.pieceID }) {
                    Button("Continue Recording") { model.resumeLiterature(round) }
                        .disabled(!model.canRecordPractice)
                }
            }
            if let error = playback.error { Text(error).foregroundStyle(.orange).font(.callout) }
            LazyVStack(alignment: .leading, spacing: 12) {
                ForEach(groups.indices, id: \.self) { index in
                    if let sentence = groups[index].first?.text {
                        let lines = [PracticeCorpus.Line(text: sentence, ipa: [])]
                        let takes = SpokenTakePlayer.recentTakes(for: lines, mode: .freeText, language: script.language,
                                                                pieceID: script.pieceID, in: model.sentenceTraining.data.pairs)
                        Divider()
                        Text(sentence).font(.body).textSelection(.enabled)
                        HStack {
                            ReferenceButton(text: sentence, language: script.language, compact: false, listenTitle: "Hear section")
                            if let latest = takes.first { playTake(latest, title: takes.count == 1 ? "Play my take" : "Play latest") }
                            if takes.count > 1 { playTake(takes[1], title: "Play previous") }
                            Button("Record Section", systemImage: "record.circle") {
                                model.startPersonalScript(script, groups: [groups[index]], isSectionRetry: true)
                            }
                            .disabled(!model.canRecordPractice)
                        }
                    }
                }
            }
        }
    }

    private func playTake(_ take: (id: UUID, clip: SpokenTakePlayer.Clip), title: LocalizedStringKey) -> some View {
        let playing = playback.playing == take.id
        return Button(playing ? "Stop" : title, systemImage: playing ? "stop.fill" : "play.fill") {
            model.referenceSpeaker.stop()
            playback.toggle(take.clip, id: take.id)
        }
        .disabled(!model.sentenceTraining.canStartRecording)
    }
}

// MARK: - Sounds

/// Every sound of the practice language: hear it, read how to make it, then say example words
/// aloud and mouth them. Sounds are played from IPA by the system voice for the language.
struct SoundsView: View {
    @Environment(AppModel.self) private var model
    @State private var selection: String?
    @AppStorage("practice.showIPA") private var showIPA = true
    /// One selection highlight travels between tiles instead of two tiles swapping colour.
    @Namespace private var selectionSpace

    private var sounds: [PracticeCorpus.Sound] { (model.practice.sounds[model.language.corpus] ?? []).filter { !model.language.absentSounds.contains($0.ipa) } }
    private var sound: PracticeCorpus.Sound? { sounds.first { $0.id == selection } ?? sounds.first }
    private var groups: [String] { sounds.map(\.group).reduce(into: []) { if !$0.contains($1) { $0.append($1) } } }

    var body: some View {
        let language = model.language
        let progress = PracticeProgress(pairs: model.sentenceTraining.data.pairs)
        PracticePage(title: "Sounds", subtitle: "Hear each sound, read how the lips and jaw make it, then say the example words.",
                     startTitle: sound.map { "Practise /\($0.ipa)/" } ?? "Start Sounds",
                     groups: sound.map { s in s.examples.map { [SentencePrompt($0, kind: .sound, language: language, translationLanguage: model.nativeLanguage)] } } ?? [],
                     mode: .sound, unit: "words", scrolls: false) {
            IPAToggle(isOn: $showIPA)
        } content: {
            if sounds.isEmpty {
                ContentUnavailableView("No sound guide for \(language.localizedName) yet", systemImage: "waveform", description: Text("Choose another language from the toolbar."))
            } else {
                HStack(alignment: .top, spacing: 24) {
                    ScrollView { VStack(alignment: .leading, spacing: 14) {
                        Text("\(sounds.filter { progress.practised($0, language: language) }.count) of \(sounds.count) sounds practised")
                            .font(.headline)
                        let tricky = TrickySounds.ipa(practising: language.corpus, speaker: model.nativeLanguage).compactMap { ipa in sounds.first { $0.ipa == ipa } }
                        let trickyIDs = Set(tricky.map(\.id))
                        if !tricky.isEmpty {
                            Text("Start here: tricky for \(LanguageMenu.nativeName(model.nativeLanguage)) speakers").font(.headline).foregroundStyle(.orange)
                                .accessibilityAddTraits(.isHeader)
                            chips(tricky, progress: progress, language: language, prefix: String(localized: "Tricky: "))
                        }
                        ForEach(groups.filter { group in sounds.contains { $0.group == group && !trickyIDs.contains($0.id) } }, id: \.self) { group in
                            Text(LocalizedStringKey(group)).font(.headline).foregroundStyle(.secondary).accessibilityAddTraits(.isHeader)
                            chips(sounds.filter { $0.group == group && !trickyIDs.contains($0.id) }, progress: progress, language: language)
                        }
                        Text("Broad IPA. English sounds follow General American.").font(.callout).foregroundStyle(.secondary)
                        Text("Keep it small and relaxed: don't strain your lips, jaw, tongue or throat, and rest if anything feels tight. Most of a sound happens inside the mouth, so go by feel and sound, not a mirror. This is practice, not speech therapy.")
                            .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }.padding(.trailing, 8) }.frame(width: 345)
                    // The selected sound stays in view while the grid scrolls.
                    if let sound {
                        ScrollView {
                            ZStack(alignment: .topLeading) {
                                detail(sound, language: language).id(sound.id)
                                    .transition(.asymmetric(insertion: .opacity.combined(with: .offset(y: 4)), removal: .opacity))
                            }
                        }
                        .motion(.state, value: sound.id)
                    }
                }
            }
        }
        .onAppear { if selection == nil { selection = SavedSelection.get("sounds", model.language) } }
        .onChange(of: selection) { SavedSelection.set("sounds", model.language, selection) }
        .onChange(of: model.language) { selection = SavedSelection.get("sounds", model.language) }
        .onChange(of: selection) {
            if let sound { AccessibilityNotification.Announcement(String(localized: "Selected \(sound.text(for: model.nativeLanguage).name). \(sound.text(for: model.nativeLanguage).guide)")).post() }
        }
    }

    private func chips(_ list: [PracticeCorpus.Sound], progress: PracticeProgress, language: AppLanguage, prefix: String = "") -> some View {
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 76), spacing: 6)], alignment: .leading, spacing: 6) {
                                ForEach(list) { s in
                                    Button { withMotion(.reorder) { selection = s.id }; model.referenceSpeaker.stop() } label: {
                                        VStack(spacing: 1) {
                                            Text(s.ipa).font(.ipa(24, relativeTo: .title2))
                                                .overlay(alignment: .topTrailing) {
                                                    if progress.practised(s, language: language) {
                                                        Image(systemName: "checkmark.circle.fill").font(.caption2).foregroundStyle(.green)
                                                            .offset(x: 14, y: -6).accessibilityHidden(true)
                                                    }
                                                }
                                            if s.text(for: model.nativeLanguage).name != s.ipa {
                                                Text(s.text(for: model.nativeLanguage).name).font(.callout).foregroundStyle(.primary.opacity(0.8)).lineLimit(2).minimumScaleFactor(0.85).multilineTextAlignment(.center)
                                            }
                                        }
                                        .frame(width: 74, height: 66)
                                        .background {
                                            ZStack {
                                                RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.35))
                                                if s.id == sound?.id {
                                                    RoundedRectangle(cornerRadius: 8).fill(.tint.opacity(0.25))
                                                        .matchedGeometryEffect(id: "soundSelection", in: selectionSpace)
                                                }
                                            }
                                        }
                                    }
                                    .buttonStyle(.plain)
                                    .accessibilityLabel(prefix + "\(s.text(for: model.nativeLanguage).name), \(s.text(for: model.nativeLanguage).guide)" + (progress.practised(s, language: language) ? String(localized: ", practised") : ""))
                                    .accessibilityAddTraits(s.id == sound?.id ? .isSelected : [])
                                }
                            }
    }

    private func detail(_ s: PracticeCorpus.Sound, language: AppLanguage) -> some View {
        let local = s.text(for: model.nativeLanguage)
        return VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 16) {
                Text("/\(s.ipa)/").font(.ipa(56, relativeTo: .largeTitle)).accessibilityLabel("Sound: \(local.name)")
                VStack(alignment: .leading, spacing: 2) {
                    Text(local.name).font(.title2.weight(.semibold)).accessibilityHidden(true)
                    Text(local.guide).foregroundStyle(.secondary)
                }
            }
            HStack {
                ReferenceButton(text: s.say, language: language, compact: false, phonetic: true).controlSize(.large)
            }
            Text("Computer voice: an approximation, especially for single sounds. Listen to the example words too.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if let tip = local.tip { Label(tip, systemImage: "mouth").font(.body) }
            Divider()
            Text("Examples").font(.headline).accessibilityAddTraits(.isHeader)
            ForEach(s.examples, id: \.self) { line in
                HStack(alignment: .firstTextBaseline) {
                    PracticeLineView(line: line, language: language, showIPA: showIPA)
                    Spacer()
                    ReferenceButton(text: line.text(language), language: language)
                }
            }
            Text("In a round, each word plays first; you say it aloud, then mouth it.").font(.callout).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
