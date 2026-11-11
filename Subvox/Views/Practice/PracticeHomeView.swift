import AppKit
import SwiftUI

struct PracticeHomeView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("practice.memorization.marks") private var memorizationMarks = Data()
    @State private var showingRecordings = false
    @State private var reviewRound: DueReviewRound?

    private var dueCards: [DueReviewCard] {
        DueReviewQueue.cards(languages: [model.language] + model.practiceLanguages.filter { $0 != model.language },
                             nativeLanguage: model.nativeLanguage, drills: model.practice.drills,
                             pieces: model.practice.pieces + model.personalPracticePieces + model.personalScripts.items.map(\.memoryPiece),
                             saved: model.savedLanguagePhrases, memorization: MemorizationReview(saved: memorizationMarks))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                let pieces = model.practice.pieces + model.personalPracticePieces
                let due = dueCards
                let suggestions = PracticeNext.suggestions(language: model.language, practiceLanguages: model.practiceLanguages,
                                                            drills: model.practice.drills, pieces: pieces, pins: model.literaturePins.items,
                                                            scripts: model.personalScripts.items, savedPhrases: model.savedLanguagePhrases,
                                                            memorization: MemorizationReview(saved: memorizationMarks),
                                                            recentRuns: model.voicePractice.runs.map(\.target), includeReviews: false)
                let resume = suggestions.first { suggestion in
                    if case .practice = suggestion.destination { return false }
                    return true
                } ?? suggestions.first { $0.id == .speech }
                let featured = due.isEmpty ? resume : PracticeSuggestion(
                    id: .language, title: String(localized: "Review due"),
                    detail: due.count == 1 ? String(localized: "1 item due") : String(localized: "\(due.count) items due"),
                    destination: .dueReview)

                if let featured {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Practise Next")
                                .font(.system(.largeTitle, design: .default, weight: .bold))
                                .accessibilityAddTraits(.isHeader)
                            Spacer()
                            Button("Recordings", systemImage: "waveform") {
                                model.voicePractice.clearError()
                                showingRecordings = true
                            }
                        }
                        featuredPractice(featured, pieces: pieces)
                    }
                }

                unfinishedPractice

                VStack(alignment: .leading, spacing: 12) {
                    sectionTitle("More ways to practise")
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 195), spacing: 12)], alignment: .leading, spacing: 12) {
                        ForEach(suggestions.filter { !due.isEmpty || $0.id != featured?.id }) { suggestion in
                            practiceTile(suggestion)
                        }
                    }
                }

                repertoire(pieces: pieces)
                shortcuts
            }
            .frame(maxWidth: 960, alignment: .leading)
            .padding(.horizontal, 32)
            .padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
        .disabled(model.navigationLocked)
        .sheet(isPresented: $showingRecordings, onDismiss: model.voicePractice.stopPlayback) {
            VoiceRecordingHistory()
        }
        .sheet(item: $reviewRound, onDismiss: model.referenceSpeaker.stop) { round in
            DuePracticeReviewView(round: round)
        }
    }

    private func sectionTitle(_ title: LocalizedStringKey) -> some View {
        Text(title).font(.title3.weight(.semibold))
            .accessibilityAddTraits(.isHeader)
    }

    private func featuredPractice(_ suggestion: PracticeSuggestion, pieces: [PracticeCorpus.Piece]) -> some View {
        let piece = featuredPiece(for: suggestion, in: pieces)
        let title = piece?.title ?? suggestion.title
        let detail = piece?.author ?? suggestion.detail
        let mode = piece?.mode ?? suggestion.id
        return HStack(spacing: 24) {
            VStack(alignment: .leading, spacing: 10) {
                Label(suggestion.destination == .dueReview ? String(localized: "Practice") : mode.title,
                      systemImage: suggestion.destination == .dueReview ? "clock.arrow.circlepath" : mode.symbol)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(Color(red: 0.69, green: 0.88, blue: 0.84))
                Text(title)
                    .font(.system(size: 41, weight: .semibold, design: .serif))
                    .lineSpacing(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)
                Text(detail)
                    .font(.body)
                    .foregroundStyle(.white.opacity(0.78))
                    .fixedSize(horizontal: false, vertical: true)
                Button { open(suggestion.destination) } label: {
                    HStack(spacing: 10) {
                        Text(primaryAction(for: suggestion))
                        Image(systemName: "arrow.right").accessibilityHidden(true)
                    }
                    .font(.callout.weight(.semibold))
                    .padding(.horizontal, 15)
                    .frame(minHeight: 36)
                    .foregroundStyle(Color(red: 0.08, green: 0.20, blue: 0.20))
                }
                .buttonStyle(FeaturedPracticeButtonStyle())
                .accessibilityHint(title)
                .padding(.top, 8)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .layoutPriority(1)

            if let image = piece?.image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 184, height: 196, alignment: .top)
                    .clipped()
                    .clipShape(.rect(cornerRadius: 12))
                    .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(.white.opacity(0.18)) }
                    .accessibilityHidden(true)
            }
        }
        .padding(28)
        .frame(minHeight: 246)
        .foregroundStyle(.white)
        .background(Color(red: 0.08, green: 0.16, blue: 0.19), in: .rect(cornerRadius: 18))
    }

    private func featuredPiece(for suggestion: PracticeSuggestion, in pieces: [PracticeCorpus.Piece])
        -> (title: String, author: String, image: NSImage?, mode: TrainMode)? {
        guard case .literature(let pin) = suggestion.destination,
              let piece = pieces.first(where: { $0.id == pin.pieceID }) else { return nil }
        let image = Bundle.main.url(forResource: piece.id, withExtension: "jpg", subdirectory: "Corpus/practice/portraits")
            .flatMap { NSImage(contentsOf: $0) }
        return (piece.title(pin.language), piece.author(pin.language), image, pin.mode)
    }

    private func primaryAction(for suggestion: PracticeSuggestion) -> LocalizedStringKey {
        switch suggestion.destination {
        case .review, .dueReview: "Start review"
        case .literature, .script: "Open rehearsal"
        case .memorize: "Continue memorizing"
        case .drill: "Open lesson"
        case .practice: "Start practising"
        }
    }

    @ViewBuilder private var unfinishedPractice: some View {
        let unfinished = model.resumablePractice
        if !unfinished.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                sectionTitle("Continue practising")
                ForEach(unfinished) { round in
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(round.title).font(.headline)
                            if let next = model.sentenceTraining.nextLiteratureTake(in: round) {
                                Text(next.condition == .silent
                                     ? "Silent pass, section \(next.group + 1) of \(round.groups.count)"
                                     : "Spoken pass, section \(next.group + 1) of \(round.groups.count)")
                                    .font(.callout).foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        Button("Continue") { model.resumeLiterature(round) }
                            .disabled(!model.canResumeLiterature)
                            .accessibilityLabel("Continue \(round.title)")
                    }
                    .padding(14)
                    .background(.quaternary.opacity(0.55), in: .rect(cornerRadius: 10))
                }
                if model.motion.status != .streaming || !model.localWhisper.isConfigured {
                    Button("Check device setup") { model.screen = .devices }
                        .buttonStyle(.link)
                }
            }
        }
    }

    @ViewBuilder private func repertoire(pieces: [PracticeCorpus.Piece]) -> some View {
        let pins = model.literaturePins.items.compactMap { pin in
            pieces.first { $0.id == pin.pieceID }.map { (pin, $0) }
        }
        let scripts = model.personalScripts.items.prefix(4)
        if !pins.isEmpty || !scripts.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                sectionTitle("My repertoire")
                ForEach(pins, id: \.0.id) { pin, piece in
                    Button {
                        model.openLiterature(pieceID: pin.pieceID, language: pin.language, mode: pin.mode)
                    } label: {
                        Label(piece.title(pin.language), systemImage: "pin")
                    }
                    .buttonStyle(.link)
                }
                ForEach(scripts) { script in
                    Button { model.openPersonalScript(script) } label: {
                        Label(script.title, systemImage: TrainMode.free.symbol)
                    }
                    .buttonStyle(.link)
                }
            }
        }
    }

    private var shortcuts: some View {
        VStack(alignment: .leading, spacing: 12) {
            Divider()
            HStack(spacing: 16) {
                ForEach([TrainMode.poetry, .reading, .sounds, .minimalPairs]) { mode in
                    Button(mode.title, systemImage: mode.symbol) { model.openPractice(mode) }
                }
                Button("Devices", systemImage: AppModel.Screen.devices.symbol) { model.screen = .devices }
                if model.researchTools {
                    Button("Try model", systemImage: AppModel.Screen.decode.symbol) { model.screen = .decode }
                }
            }
            .buttonStyle(.link)
            .controlSize(.regular)
        }
    }

    private func open(_ destination: PracticeSuggestion.Destination) {
        guard !model.navigationLocked else { return }
        switch destination {
        case .practice(let mode):
            model.openPractice(mode)
        case .review(let mode, let language):
            model.openPracticeReview(mode, language: language)
        case .dueReview:
            let due = dueCards
            if !due.isEmpty { reviewRound = DueReviewRound(due: due) }
        case .drill(let id, let language):
            model.language = language
            SavedSelection.set("drill", language, id)
            model.openPractice(.language)
        case .literature(let pin):
            model.openLiterature(pieceID: pin.pieceID, language: pin.language, mode: pin.mode)
        case .memorize(let pieceID, let language):
            model.language = language
            MemorizationPracticeView.select(pieceID: pieceID, language: language)
            model.openPractice(.memorization)
        case .script(let script):
            model.openPersonalScript(script)
        }
    }

    private func practiceTile(_ suggestion: PracticeSuggestion) -> some View {
        Button { open(suggestion.destination) } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: suggestion.id.symbol)
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 36, height: 36)
                        .background(Color.accentColor.opacity(0.11), in: .rect(cornerRadius: 8))
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .accessibilityHidden(true)
                Text(suggestion.title)
                    .font(.headline)
                    .lineLimit(2)
                Text(suggestion.detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer(minLength: 0)
            }
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 138, alignment: .topLeading)
            .multilineTextAlignment(.leading)
            .contentShape(.rect(cornerRadius: 12))
        }
        .buttonStyle(PracticeTileStyle())
    }
}

private struct VoiceRecordingHistory: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let voice = model.voicePractice
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Recordings").font(.title2.weight(.semibold)).accessibilityAddTraits(.isHeader)
                    Text("Microphone takes saved on this Mac.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            if let error = voice.error {
                Label(error, systemImage: "exclamationmark.circle").foregroundStyle(.orange)
            }
            if voice.runs.isEmpty {
                ContentUnavailableView("No microphone recordings yet", systemImage: "waveform",
                                       description: Text("Record a take from a practice page to listen to it here."))
                    .frame(maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(voice.runs) { run in
                            recordingRow(run)
                            if run.id != voice.runs.last?.id { Divider() }
                        }
                    }
                }
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
                if let active = voice.runs.first(where: { $0.id == voice.playingRunID }) {
                    VoiceRunTransport(title: Text(title(for: active.target)))
                }
            }
        }
        .padding(20)
        .frame(width: 570, height: 460)
    }

    private func recordingRow(_ run: VoicePracticeRun) -> some View {
        let voice = model.voicePractice
        let title = title(for: run.target)
        let mode = TrainMode(rawValue: run.target.mode)
        let active = voice.playingRunID == run.id
        return Button {
            if active {
                voice.togglePlaybackPause()
            } else {
                model.referenceSpeaker.stop()
                voice.play(run)
            }
        } label: {
            HStack(spacing: 12) {
                Image(systemName: active && !voice.playbackPaused ? "pause.fill" : "play.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 28, height: 28)
                    .background(Color.accentColor.opacity(0.11), in: .circle)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.headline).lineLimit(1)
                    Text("\(mode?.title ?? run.target.mode) · \(run.target.language.localizedName) · \(run.createdAt.formatted(date: .abbreviated, time: .shortened))")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
                Text(Duration.seconds(run.duration).formatted(.time(pattern: .minuteSecond)))
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .contentShape(.rect)
            .background(active ? Color.accentColor.opacity(0.09) : Color.clear)
        }
        .buttonStyle(.plain)
        .accessibilityHint(active && !voice.playbackPaused ? "Pauses this recording" : "Plays this recording")
    }

    private func title(for target: VoicePracticeTarget) -> String {
        if let script = model.personalScripts.items.first(where: { $0.pieceID == target.pieceID }) {
            return script.title
        }
        if let piece = (model.practice.pieces + model.personalPracticePieces).first(where: { $0.id == target.pieceID }) {
            return piece.title(target.language)
        }
        if let drill = model.practice.drills.first(where: { $0.id == target.drillID }) {
            return drill.title(target.language)
        }
        return String(target.text.prefix(while: { !$0.isNewline }))
    }
}

private struct FeaturedPracticeButtonStyle: ButtonStyle {
    @Environment(\.isFocused) private var focused
    @State private var hovered = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(Color(red: 0.75, green: 0.90, blue: 0.86)
                .opacity(configuration.isPressed ? 0.77 : hovered ? 0.9 : 1), in: .rect(cornerRadius: 8))
            .overlay { RoundedRectangle(cornerRadius: 8).strokeBorder(.white, lineWidth: focused ? 2 : 0) }
            .onHover { hovered = $0 }
    }
}

private struct PracticeTileStyle: ButtonStyle {
    @Environment(\.isFocused) private var focused
    @State private var hovered = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(.quaternary.opacity(configuration.isPressed ? 1 : hovered ? 0.85 : 0.55), in: .rect(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Color.accentColor.opacity(focused ? 0.85 : 0.15), lineWidth: focused ? 2 : 1)
            }
            .onHover { hovered = $0 }
    }
}
