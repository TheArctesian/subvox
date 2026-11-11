import Foundation
import Observation

struct LanguagePhraseID: Codable, Hashable {
    let language: AppLanguage
    let drillID: String
    let text: String

    init(_ line: PracticeCorpus.Line, drillID: String, language: AppLanguage) {
        self.language = language
        self.drillID = drillID
        text = line.text(language)
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.language == rhs.language && lhs.drillID == rhs.drillID && lhs.text.utf8.elementsEqual(rhs.text.utf8)
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(language)
        hasher.combine(drillID)
        hasher.combine(Data(text.utf8))
    }
}

struct LanguageReviewCard: Identifiable {
    let id: LanguagePhraseID
    let line: PracticeCorpus.Line
    let drillTitle: String

    func translationCue(in language: String) -> String? {
        guard let translation = line.translation(to: language), translation != id.text else { return nil }
        return translation
    }
}

@MainActor @Observable
final class SavedLanguagePhrases {
    private struct Bookmark: Codable {
        let id: LanguagePhraseID
        var reviewedAt: Date?
        var schedule: PracticeReviewSchedule?

        var reviewSchedule: PracticeReviewSchedule {
            schedule ?? reviewedAt.map { PracticeReviewSchedule.new.rated(recalled: true, at: $0) } ?? .new
        }
    }

    private static let storageKey = "practice.savedLanguagePhrases"
    @ObservationIgnored private let defaults: UserDefaults
    private var bookmarks: [Bookmark]

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        bookmarks = defaults.data(forKey: Self.storageKey)
            .flatMap { try? JSONDecoder().decode([Bookmark].self, from: $0) } ?? []
    }

    func contains(_ id: LanguagePhraseID) -> Bool { bookmarks.contains { $0.id == id } }

    func toggle(_ id: LanguagePhraseID) {
        if contains(id) {
            remove(id)
        } else {
            bookmarks.append(Bookmark(id: id))
            save()
        }
    }

    func remove(_ id: LanguagePhraseID) {
        bookmarks.removeAll { $0.id == id }
        save()
    }

    func markReviewed(_ id: LanguagePhraseID, at date: Date = .now) {
        rate(id, recalled: true, at: date)
    }

    func rate(_ id: LanguagePhraseID, recalled: Bool, at date: Date = .now) {
        guard let index = bookmarks.firstIndex(where: { $0.id == id }) else { return }
        bookmarks[index].schedule = bookmarks[index].reviewSchedule.rated(recalled: recalled, at: date)
        bookmarks[index].reviewedAt = date
        save()
    }

    func cards(in drills: [PracticeCorpus.Drill], language: AppLanguage) -> [LanguageReviewCard] {
        cards(for: bookmarks, in: drills, language: language)
    }

    func nextRound(in drills: [PracticeCorpus.Drill], language: AppLanguage, now: Date = .now) -> [LanguageReviewCard] {
        Array(dueCards(in: drills, language: language, now: now).prefix(3)).map(\.card)
    }

    func dueCards(in drills: [PracticeCorpus.Drill], language: AppLanguage, now: Date) ->
        [(card: LanguageReviewCard, schedule: PracticeReviewSchedule)] {
        let ordered = bookmarks.filter { $0.id.language == language && $0.reviewSchedule.dueAt <= now }
            .sorted { $0.reviewSchedule.dueAt < $1.reviewSchedule.dueAt }
        let scheduleByID = Dictionary(ordered.map { ($0.id, $0.reviewSchedule) }, uniquingKeysWith: { first, _ in first })
        return cards(for: ordered, in: drills, language: language).compactMap { card in
            scheduleByID[card.id].map { (card, $0) }
        }
    }

    private func cards(for bookmarks: [Bookmark], in drills: [PracticeCorpus.Drill],
                       language: AppLanguage) -> [LanguageReviewCard] {
        let savedIDs = bookmarks.lazy.map(\.id).filter { $0.language == language }
        let byID = Dictionary(drills.lazy.filter {
            $0.language == language.corpus && language.shows(region: $0.region)
        }.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        return savedIDs.compactMap { id in
            guard let drill = byID[id.drillID], let line = drill.phrases.first(where: {
                language.shows(region: $0.region) && $0.text(language).utf8.elementsEqual(id.text.utf8)
            }) else { return nil }
            return LanguageReviewCard(id: id, line: line, drillTitle: drill.title(language))
        }
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(bookmarks) else { return }
        defaults.set(data, forKey: Self.storageKey)
    }
}
