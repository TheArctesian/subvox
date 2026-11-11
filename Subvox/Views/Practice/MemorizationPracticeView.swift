import SwiftUI

struct MemorizationPracticeView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("practice.memorization.marks") private var savedMarks = Data()
    @State private var pieceID: String?
    @State private var position = 0
    @State private var hideLines = false
    @State private var reviewSections: [MemorizationSection] = []
    @State private var reviewIndex = 0

    static func select(pieceID: String, language: AppLanguage) {
        SavedSelection.set("memorization", language, pieceID)
    }

    private var pieces: [PracticeCorpus.Piece] {
        (model.practice.pieces + model.personalPracticePieces + model.personalScriptPieces)
            .filter { $0.language == model.language.corpus && !$0.lines.isEmpty }
    }

    var body: some View {
        let language = model.language
        let pieces = pieces
        let piece = pieces.first { $0.id == pieceID } ?? pieces.first
        let sections = piece.map { MemorizationSection.sections(of: $0, language: language) } ?? []
        let studyIndex = sections.indices.contains(position) ? position : 0
        let marks = MemorizationReview(saved: savedMarks)
        let round = marks.round(in: pieces, language: language)

        AppPageLayout(showsMotion: false) {
            HStack {
                AppPageHeading(title: "Memorization", subtitle: "Learn a piece section by section.")
                Spacer()
                if !reviewSections.isEmpty {
                    Button("End review", action: endReview).disabled(model.voicePractice.isBusy)
                } else {
                    Button("Review due sections", systemImage: "arrow.clockwise") { beginReview(round) }
                        .buttonStyle(.borderedProminent)
                        .disabled(round.isEmpty || model.voicePractice.isBusy)
                        .help("Marked sections return when they’re due.")
                }
            }
        } content: {
            if !reviewSections.isEmpty {
                reviewBody(language: language)
            } else if let piece, !sections.isEmpty {
                let section = sections[studyIndex]
                studyBody(piece: piece, pieces: pieces, section: section, sections: sections, index: studyIndex,
                          marked: marks.isMarked(section.id), language: language)
            } else {
                ContentUnavailableView("No pieces in \(language.localizedName)", systemImage: "text.book.closed",
                                       description: Text("Choose another language or add your own material."))
            }
        }
        .onAppear {
            restoreSelection()
            openRequestedReview()
        }
        .onChange(of: language) {
            endReview()
            restoreSelection()
            openRequestedReview()
        }
        .onChange(of: model.practiceReviewRequest) { openRequestedReview() }
        .onDisappear { model.referenceSpeaker.stop() }
    }

    private func studyBody(piece: PracticeCorpus.Piece, pieces: [PracticeCorpus.Piece], section: MemorizationSection,
                           sections: [MemorizationSection], index: Int, marked: Bool, language: AppLanguage) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Piece", selection: Binding(get: { piece.id }, set: { selection in
                    pieceID = selection
                    position = 0
                    SavedSelection.set("memorization", language, selection)
                })) {
                    ForEach(pieces) { piece in Text(piece.title(language)).tag(piece.id) }
                }
                .frame(maxWidth: 460)
                .disabled(model.voicePractice.isBusy)

                HStack(spacing: 16) {
                    Picker("Section", selection: Binding(get: { index }, set: { position = $0 })) {
                        ForEach(sections.indices, id: \.self) { index in
                            Text(sections[index].label).tag(index)
                        }
                    }
                    .fixedSize()
                    Toggle("Hide lines", isOn: $hideLines)
                    Button(marked ? "Remove from review" : "Mark for review",
                           systemImage: marked ? "bookmark.fill" : "bookmark") {
                        updateMarks { $0.toggle(section.id) }
                    }
                }
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    MemorizationCard(section: section, language: language, hidesLines: hideLines)
                    HStack {
                        Button("Previous section") { position = index - 1 }
                            .disabled(index == 0)
                        Spacer()
                        Button("Next section") { position = index + 1 }
                            .disabled(index + 1 == sections.count)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .id(section.id)
            .onChange(of: section.id) { model.referenceSpeaker.stop() }

            let target = VoicePracticeTarget(mode: TrainMode.memorization.rawValue, language: language, itemID: piece.id,
                                             text: piece.lines.map { $0.text(language) }.joined(separator: "\n"))
            VoicePracticePanel(target: target, title: piece.title(language),
                               referenceTitle: hideLines ? nil : "Hear full piece", showsOtherRuns: !hideLines)
        }
        .frame(maxWidth: PageMetrics.contentWidth, maxHeight: .infinity, alignment: .topLeading)
        .padding(PageMetrics.padding)
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder private func reviewBody(language: AppLanguage) -> some View {
        if reviewSections.indices.contains(reviewIndex) {
            let section = reviewSections[reviewIndex]
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Review \(reviewIndex + 1) of \(reviewSections.count)")
                        .font(.subheadline).foregroundStyle(.secondary)
                    MemorizationCard(section: section, language: language, hidesLines: true)
                    HStack {
                        Button("Practise again") { finish(section, keepPractising: true) }
                        Button("I know it") { finish(section, keepPractising: false) }
                            .buttonStyle(.borderedProminent)
                    }
                }
                .frame(maxWidth: PageMetrics.contentWidth, alignment: .leading)
                .padding(PageMetrics.padding)
                .frame(maxWidth: .infinity)
            }
            .id(section.id)
        } else {
            ContentUnavailableView {
                Label("Review complete", systemImage: "checkmark.circle")
            } actions: {
                Button("Back to practice", action: endReview)
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private func restoreSelection() {
        pieceID = SavedSelection.get("memorization", model.language)
        position = 0
    }

    private func beginReview(_ sections: [MemorizationSection]) {
        model.referenceSpeaker.stop()
        reviewSections = sections
        reviewIndex = 0
    }

    private func openRequestedReview() {
        if model.takePracticeReview(.memorization) {
            beginReview(MemorizationReview(saved: savedMarks).round(in: pieces, language: model.language))
        }
    }

    private func finish(_ section: MemorizationSection, keepPractising: Bool) {
        model.referenceSpeaker.stop()
        updateMarks { $0.complete(section.id, keepPractising: keepPractising) }
        reviewIndex += 1
    }

    private func endReview() {
        model.referenceSpeaker.stop()
        reviewSections.removeAll()
        reviewIndex = 0
    }

    private func updateMarks(_ update: (inout MemorizationReview) -> Void) {
        var marks = MemorizationReview(saved: savedMarks)
        update(&marks)
        savedMarks = marks.saved
    }
}

struct MemorizationCard: View {
    @Environment(AppModel.self) private var model
    let section: MemorizationSection
    let language: AppLanguage
    let hidesLines: Bool
    @State private var hinted: Set<Int> = []
    @State private var revealed: Set<Int> = []

    var body: some View {
        let text = section.lines.map { $0.text(language) }.joined(separator: "\n")
        let playing = model.referenceSpeaker.playing == text

        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(section.pieceTitle).font(.title2.weight(.semibold)).accessibilityAddTraits(.isHeader)
                Text(section.label).foregroundStyle(.secondary)
            }
            Button(playing ? "Stop example" : hidesLines ? "Hear answer" : "Hear example",
                   systemImage: playing ? "stop.fill" : "speaker.wave.2") {
                if playing { model.referenceSpeaker.stop() }
                else { Task { await model.referenceSpeaker.speak(text, language: language.tag) } }
            }
            .disabled(model.voicePractice.isBusy)

            ForEach(section.lines.indices, id: \.self) { index in
                if !hidesLines || revealed.contains(index) {
                    HStack(alignment: .firstTextBaseline) {
                        PracticeLineView(line: section.lines[index], language: language, showIPA: false)
                        Spacer(minLength: 8)
                        if hidesLines {
                            Button("Hide line \(index + 1)", systemImage: "eye.slash") {
                                revealed.remove(index)
                                hinted.remove(index)
                            }
                                .labelStyle(.iconOnly)
                                .help("Hide line \(index + 1)")
                        }
                    }
                } else {
                    recallLine(index)
                }
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .onChange(of: hidesLines) {
            hinted.removeAll()
            revealed.removeAll()
        }
    }

    private func recallLine(_ index: Int) -> some View {
        let text = section.lines[index].text(language)
        let opening = text.drop(while: { !$0.isLetter && !$0.isNumber })
        let hint = language.isChinese ? String(opening.prefix(1))
            : opening.prefixMatch(of: /[\p{L}\p{N}]+(?:['’][\p{L}]+)*/).map { String($0.output) } ?? ""
        let hasMore = opening.dropFirst(hint.count).contains { $0.isLetter || $0.isNumber }
        return HStack(spacing: 10) {
            Text("Line \(index + 1)")
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)
            if hinted.contains(index) {
                Text("\(hint)…").font(.headline)
                    .accessibilityLabel("Starts with \(hint)")
            } else if !hint.isEmpty, hasMore {
                Button("Show hint", systemImage: "lightbulb") { hinted.insert(index) }
                    .accessibilityLabel("Show a hint for line \(index + 1)")
            }
            Button("Reveal line", systemImage: "eye") { revealed.insert(index) }
                .accessibilityLabel("Reveal line \(index + 1)")
        }
    }
}
