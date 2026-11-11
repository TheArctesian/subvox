import SwiftUI

struct PracticeHomeView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("practice.memorization.marks") private var memorizationMarks = Data()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Subvox")
                        .font(.system(size: 52, weight: .semibold))
                        .accessibilityAddTraits(.isHeader)
                    Text("Choose what you’d like to practise today.")
                        .font(.title3).foregroundStyle(.secondary)
                }

                let unfinished = model.resumablePractice
                if !unfinished.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        sectionTitle("Continue practising")
                        ForEach(unfinished) { round in
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Label(round.title, systemImage: round.kind == .freeText ? TrainMode.free.symbol
                                          : round.kind == .poetry ? TrainMode.poetry.symbol : TrainMode.speech.symbol)
                                        .font(.headline)
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
                            .padding(.vertical, 4)
                        }
                        if model.motion.status != .streaming || !model.localWhisper.isConfigured {
                            Button("Check device setup") { model.screen = .devices }
                                .buttonStyle(.link)
                        }
                    }
                }

                let pieces = model.practice.pieces + model.personalPracticePieces
                practiseNext(pieces: pieces)

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

                HStack(spacing: 12) {
                    ForEach([TrainMode.poetry, .reading, .sounds, .minimalPairs]) { mode in
                        Button(mode.title, systemImage: mode.symbol) { model.openPractice(mode) }
                    }
                    Button("Devices", systemImage: AppModel.Screen.devices.symbol) { model.screen = .devices }
                    if model.researchTools {
                        Button("Try model", systemImage: AppModel.Screen.decode.symbol) { model.screen = .decode }
                    }
                }
            }
            .frame(maxWidth: 820, alignment: .leading)
            .padding(.horizontal, 36).padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
        .disabled(model.navigationLocked)
    }

    private func sectionTitle(_ title: LocalizedStringKey) -> some View {
        Text(title).font(.title3.weight(.semibold))
            .accessibilityAddTraits(.isHeader)
            .padding(.bottom, 8)
    }

    private func practiseNext(pieces: [PracticeCorpus.Piece]) -> some View {
        let suggestions = PracticeNext.suggestions(language: model.language, practiceLanguages: model.practiceLanguages,
                                                   drills: model.practice.drills, pieces: pieces, pins: model.literaturePins.items,
                                                   scripts: model.personalScripts.items, savedPhrases: SavedLanguagePhrases(),
                                                   memorization: MemorizationReview(saved: memorizationMarks),
                                                   recentRuns: model.voicePractice.runs.map(\.target))
        return VStack(alignment: .leading, spacing: 4) {
            sectionTitle("Practise Next")
            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 10) {
                GridRow { ForEach(suggestions.prefix(2)) { practiceRow($0) } }
                GridRow { ForEach(suggestions.dropFirst(2)) { practiceRow($0) } }
            }
        }
    }

    private func open(_ destination: PracticeSuggestion.Destination) {
        guard !model.navigationLocked else { return }
        switch destination {
        case .practice(let mode):
            model.openPractice(mode)
        case .review(let mode, let language):
            model.openPracticeReview(mode, language: language)
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

    private func practiceRow(_ suggestion: PracticeSuggestion) -> some View {
        VStack(spacing: 0) {
            Divider()
            Button {
                open(suggestion.destination)
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: suggestion.id.symbol)
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 24)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(suggestion.title).font(.title3.weight(.semibold)).lineLimit(2)
                        Text(suggestion.detail).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                    }
                    Spacer(minLength: 8)
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold)).foregroundStyle(.tertiary)
                        .accessibilityHidden(true)
                }
                .padding(.vertical, 14)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }
}
