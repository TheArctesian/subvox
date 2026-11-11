import SwiftUI

struct DuePracticeReviewView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @AppStorage("practice.memorization.marks") private var memorizationMarks = Data()
    @State var round: DueReviewRound
    @State private var revealed = false
    @State private var exampleTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Due review").font(.title2.weight(.semibold)).accessibilityAddTraits(.isHeader)
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            if let card = round.current {
                Text("Item \(round.position + 1) of \(round.cards.count)")
                    .font(.callout).foregroundStyle(.secondary)
                ScrollView {
                    Group {
                        switch card {
                        case .phrase(let phrase, let listening):
                            phraseCard(phrase, listening: listening)
                        case .section(let section):
                            MemorizationCard(section: section, language: section.id.language, hidesLines: true)
                        }
                    }
                    .id(round.position)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                HStack {
                    Spacer()
                    Button("Practise again", systemImage: "arrow.counterclockwise") { advance(card, recalled: false) }
                    Button("I know it", systemImage: "checkmark") { advance(card, recalled: true) }
                        .buttonStyle(.borderedProminent)
                }
            } else {
                ContentUnavailableView("Review complete", systemImage: "checkmark.circle",
                                       description: Text("Reviewed items will return when they’re due."))
                    .frame(maxHeight: .infinity)
            }
        }
        .padding(24)
        .frame(width: 580, height: 430)
        .onDisappear { exampleTask?.cancel() }
    }

    private func phraseCard(_ card: LanguageReviewCard, listening: Bool) -> some View {
        let language = card.id.language
        let cue = card.translationCue(in: model.nativeLanguage)
        let playing = model.referenceSpeaker.playing == card.id.text
        return VStack(alignment: .leading, spacing: 12) {
            Text("\(language.localizedName) · \(card.drillTitle)").font(.headline)
            if listening || revealed {
                Button(playing ? "Stop example" : "Hear phrase", systemImage: playing ? "stop.fill" : "speaker.wave.2") {
                    exampleTask?.cancel()
                    if playing { model.referenceSpeaker.stop() }
                    else { exampleTask = Task { await model.referenceSpeaker.speak(card.id.text, language: language.tag) } }
                }
            }
            if revealed {
                PracticeLineView(line: card.line, language: language, showIPA: false)
            } else {
                if listening {
                    Text("Listen and say what it means.")
                } else if let cue {
                    Text(cue).font(.title2)
                } else if let first = card.id.text.first(where: { $0.isLetter || $0.isNumber }) {
                    Text("Starts with \(String(first))").font(.title2)
                }
                Button("Reveal answer", systemImage: "eye") { revealed = true }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: .rect(cornerRadius: 10))
    }

    private func advance(_ card: DueReviewCard, recalled: Bool) {
        exampleTask?.cancel()
        model.referenceSpeaker.stop()
        switch card {
        case .phrase(let phrase, _):
            model.savedLanguagePhrases.rate(phrase.id, recalled: recalled)
        case .section(let section):
            var review = MemorizationReview(saved: memorizationMarks)
            review.complete(section.id, keepPractising: !recalled)
            memorizationMarks = review.saved
        }
        revealed = false
        round.advance(recalled: recalled)
    }
}
