import Foundation

struct PracticeSuggestion: Identifiable, Equatable {
    enum Destination: Equatable {
        case practice(TrainMode)
        case review(TrainMode, AppLanguage)
        case dueReview
        case drill(id: String, language: AppLanguage)
        case literature(LiteraturePin)
        case memorize(pieceID: String, language: AppLanguage)
        case script(PersonalScript)
    }

    let id: TrainMode
    let title: String
    let detail: String
    let destination: Destination
}

/// Reuses the material and queues already saved by each practice mode.
@MainActor
enum PracticeNext {
    /// The poem Home features, and the Poetry page opens on, until the learner records or pins something.
    nonisolated static let featuredPoem: [String: String] = ["en": "invictus"]

    static func suggestions(language: AppLanguage, practiceLanguages: [AppLanguage],
                            drills: [PracticeCorpus.Drill], pieces: [PracticeCorpus.Piece],
                            pins: [LiteraturePin], scripts: [PersonalScript], savedPhrases: SavedLanguagePhrases,
                            memorization: MemorizationReview, recentRuns: [VoicePracticeTarget] = [],
                            includeReviews: Bool = true,
                            selection: (String, AppLanguage) -> String? = SavedSelection.get) -> [PracticeSuggestion] {
        let languages = [language] + practiceLanguages.filter { $0 != language }
        return [
            languageSuggestion(languages, drills: drills, saved: savedPhrases, recentRuns: recentRuns,
                               includeReviews: includeReviews, selection: selection),
            speechSuggestion(languages, pieces: pieces, pins: pins, recentRuns: recentRuns, selection: selection),
            memorizationSuggestion(languages, pieces: pieces, scripts: scripts, review: memorization,
                                   recentRuns: recentRuns, includeReviews: includeReviews, selection: selection),
            materialSuggestion(scripts, recentRuns: recentRuns)
        ]
    }

    private static func languageSuggestion(_ languages: [AppLanguage], drills: [PracticeCorpus.Drill],
                                           saved: SavedLanguagePhrases, recentRuns: [VoicePracticeTarget],
                                           includeReviews: Bool,
                                           selection: (String, AppLanguage) -> String?) -> PracticeSuggestion {
        for language in languages where includeReviews {
            let count = saved.nextRound(in: drills, language: language).count
            if count > 0 {
                let title = count == 1 ? String(localized: "Review 1 saved phrase")
                    : String(localized: "Review \(count) saved phrases")
                return PracticeSuggestion(id: .language, title: title, detail: language.localizedName,
                                          destination: .review(.language, language))
            }
        }
        for run in recentRuns where run.mode == TrainMode.language.rawValue && languages.contains(run.language) {
            if let drill = drills.first(where: {
                $0.id == run.drillID && $0.language == run.language.corpus && run.language.shows(region: $0.region)
            }) {
                return PracticeSuggestion(id: .language, title: String(localized: "Continue language practice"),
                                          detail: drill.title(run.language), destination: .drill(id: drill.id, language: run.language))
            }
        }
        if let language = languages.first, let id = selection("drill", language),
           let drill = drills.first(where: { $0.id == id && $0.language == language.corpus && language.shows(region: $0.region) }) {
            return PracticeSuggestion(id: .language, title: String(localized: "Continue language practice"),
                                      detail: drill.title(language), destination: .practice(.language))
        }
        return PracticeSuggestion(id: .language, title: String(localized: "Learn a language"),
                                  detail: String(localized: "Listen, speak and review saved phrases"), destination: .practice(.language))
    }

    private static func speechSuggestion(_ languages: [AppLanguage], pieces: [PracticeCorpus.Piece],
                                         pins: [LiteraturePin], recentRuns: [VoicePracticeTarget],
                                         selection: (String, AppLanguage) -> String?) -> PracticeSuggestion {
        let recent = recentRuns.compactMap { run -> LiteraturePin? in
            guard let mode = TrainMode(rawValue: run.mode), mode == .speech || mode == .poetry else { return nil }
            return LiteraturePin(pieceID: run.pieceID, language: run.language, mode: mode)
        }
        // What the learner recorded or pinned wins; then the featured poem; then the page they last browsed.
        let featured = languages.compactMap { language in
            featuredPoem[language.corpus].map { LiteraturePin(pieceID: $0, language: language, mode: .poetry) }
        }
        let candidates = (recent + pins).filter { languages.contains($0.language) } + featured + languages.flatMap { language in
            [("speech", TrainMode.speech), ("poetry", TrainMode.poetry)].compactMap { key, mode in
                selection(key, language).map { LiteraturePin(pieceID: $0, language: language, mode: mode) }
            }
        }
        for pin in candidates {
            let kind: SentencePromptKind = pin.mode == .poetry ? .poetry : .speech
            if let piece = pieces.first(where: { $0.id == pin.pieceID && $0.language == pin.language.corpus
                && $0.practiceKind == kind && !$0.lines.isEmpty }) {
                return PracticeSuggestion(id: .speech, title: String(localized: "Rehearse \(piece.title(pin.language))"),
                                          detail: String(localized: "Listen back or record another run"), destination: .literature(pin))
            }
        }
        return PracticeSuggestion(id: .speech, title: String(localized: "Practise a speech"),
                                  detail: String(localized: "Rehearse a section and compare takes"), destination: .practice(.speech))
    }

    private static func memorizationSuggestion(_ languages: [AppLanguage], pieces: [PracticeCorpus.Piece],
                                                scripts: [PersonalScript], review: MemorizationReview,
                                                recentRuns: [VoicePracticeTarget],
                                                includeReviews: Bool,
                                                selection: (String, AppLanguage) -> String?) -> PracticeSuggestion {
        for language in languages where includeReviews {
            let available = pieces + scripts.filter { $0.language == language }.map(\.memoryPiece)
            let round = review.round(in: available, language: language)
            if let first = round.first {
                let title = round.count == 1 ? String(localized: "Review 1 marked section")
                    : String(localized: "Review \(round.count) marked sections")
                return PracticeSuggestion(id: .memorization, title: title, detail: first.pieceTitle,
                                          destination: .review(.memorization, language))
            }
        }
        let recent = recentRuns.filter { $0.mode == TrainMode.memorization.rawValue && languages.contains($0.language) }
            .map { ($0.itemID, $0.language) }
        let selections = languages.compactMap { language in selection("memorization", language).map { ($0, language) } }
        for (id, language) in recent + selections {
            let available = pieces + scripts.filter { $0.language == language }.map(\.memoryPiece)
            if let piece = available.first(where: { $0.id == id && $0.language == language.corpus && !$0.lines.isEmpty }) {
                return PracticeSuggestion(id: .memorization, title: String(localized: "Continue memorizing"), detail: piece.title(language),
                                          destination: .memorize(pieceID: piece.id, language: language))
            }
        }
        return PracticeSuggestion(id: .memorization, title: String(localized: "Memorize a piece"),
                                  detail: String(localized: "Recall lines and revisit tricky sections"), destination: .practice(.memorization))
    }

    private static func materialSuggestion(_ scripts: [PersonalScript], recentRuns: [VoicePracticeTarget]) -> PracticeSuggestion {
        let recent = recentRuns.lazy.filter { $0.mode == TrainMode.free.rawValue }.compactMap { run in
            scripts.first { $0.pieceID == run.pieceID && $0.language == run.language }
        }.first
        if let script = recent ?? scripts.first {
            return PracticeSuggestion(id: .free, title: String(localized: "Rehearse \(script.title)"), detail: TrainMode.free.title,
                                      destination: .script(script))
        }
        return PracticeSuggestion(id: .free, title: String(localized: "Bring your own material"),
                                  detail: String(localized: "Save a script, presentation or interview answer"), destination: .practice(.free))
    }
}
