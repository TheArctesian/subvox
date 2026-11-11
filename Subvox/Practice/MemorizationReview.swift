import CryptoKit
import Foundation

struct PracticeReviewSchedule: Codable, Equatable {
    private static let intervals = [1, 3, 7, 14, 30]
    var dueAt: Date
    private(set) var successes: Int
    private(set) var listening: Bool

    static let new = Self(dueAt: .distantPast, successes: 0, listening: false)

    func rated(recalled: Bool, at date: Date, calendar: Calendar = .current) -> Self {
        let next = recalled ? min(max(0, successes), Self.intervals.count - 1) + 1 : 0
        let days = Self.intervals[max(0, next - 1)]
        let due = calendar.date(byAdding: .day, value: days, to: date) ?? date.addingTimeInterval(Double(days) * 86_400)
        return Self(dueAt: due, successes: next, listening: recalled && !listening)
    }
}

struct MemorizationSection: Identifiable {
    struct ID: Codable, Hashable {
        let piece: String
        let language: AppLanguage
        let position: Int
        let revision: String
    }

    let id: ID
    let pieceTitle: String
    let isStanza: Bool
    let lines: [PracticeCorpus.Line]

    var label: String {
        isStanza ? String(localized: "Stanza \(id.position + 1)") : String(localized: "Section \(id.position + 1)")
    }

    static func sections(of piece: PracticeCorpus.Piece, language: AppLanguage) -> [Self] {
        let isStanza = piece.kind == "poem" && piece.stanzaEnds?.last == piece.lines.count
        return piece.parts(language: language).enumerated().map { position, lines in
            // Include line boundaries and exact text so edits never reuse old marks by position alone.
            let content = lines.map { line in
                let text = line.text(language)
                return "\(text.utf8.count):\(text)"
            }.joined()
            let revision = SHA256.hash(data: Data(content.utf8)).map { String(format: "%02x", $0) }.joined()
            let id = ID(piece: piece.id, language: language, position: position, revision: revision)
            return Self(id: id, pieceTitle: piece.title(language), isStanza: isStanza, lines: lines)
        }
    }
}

struct MemorizationReview {
    private struct Mark: Codable {
        let id: MemorizationSection.ID
        var schedule: PracticeReviewSchedule
    }

    private var marks: [Mark]

    init(saved: Data = Data()) {
        let decoder = JSONDecoder()
        if let current = try? decoder.decode([Mark].self, from: saved) {
            marks = current
        } else {
            marks = ((try? decoder.decode([MemorizationSection.ID].self, from: saved)) ?? [])
                .map { Mark(id: $0, schedule: .new) }
        }
    }

    var saved: Data { (try? JSONEncoder().encode(marks)) ?? Data() }

    func isMarked(_ id: MemorizationSection.ID) -> Bool { marks.contains { $0.id == id } }

    mutating func toggle(_ id: MemorizationSection.ID) {
        if isMarked(id) { marks.removeAll { $0.id == id } }
        else { marks.append(Mark(id: id, schedule: .new)) }
    }

    mutating func complete(_ id: MemorizationSection.ID, keepPractising: Bool, at date: Date = .now) {
        guard let index = marks.firstIndex(where: { $0.id == id }) else { return }
        marks[index].schedule = marks[index].schedule.rated(recalled: !keepPractising, at: date)
    }

    func round(in pieces: [PracticeCorpus.Piece], language: AppLanguage, now: Date = .now) -> [MemorizationSection] {
        Array(dueSections(in: pieces, language: language, now: now).prefix(3)).map(\.section)
    }

    func dueSections(in pieces: [PracticeCorpus.Piece], language: AppLanguage, now: Date) ->
        [(section: MemorizationSection, dueAt: Date)] {
        let localMarks = marks.filter { $0.id.language == language && $0.schedule.dueAt <= now }
        let markedPieces = Set(localMarks.map(\.id.piece))
        let sections = pieces.lazy.filter { $0.language == language.corpus && markedPieces.contains($0.id) }
            .flatMap { MemorizationSection.sections(of: $0, language: language) }
        let byID = Dictionary(sections.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        return localMarks.sorted { $0.schedule.dueAt < $1.schedule.dueAt }.compactMap { mark in
            byID[mark.id].map { ($0, mark.schedule.dueAt) }
        }
    }
}

enum DueReviewCard: Identifiable {
    enum ID: Hashable {
        case phrase(LanguagePhraseID)
        case section(MemorizationSection.ID)
    }

    case phrase(LanguageReviewCard, listening: Bool)
    case section(MemorizationSection)

    var id: ID {
        switch self {
        case .phrase(let card, _): .phrase(card.id)
        case .section(let section): .section(section.id)
        }
    }
}

@MainActor
enum DueReviewQueue {
    static func cards(languages: [AppLanguage], nativeLanguage: String, drills: [PracticeCorpus.Drill],
                      pieces: [PracticeCorpus.Piece], saved: SavedLanguagePhrases,
                      memorization: MemorizationReview, now: Date = .now) -> [DueReviewCard] {
        let phrases = languages.flatMap { saved.dueCards(in: drills, language: $0, now: now) }
            .sorted { $0.schedule.dueAt < $1.schedule.dueAt }
            .map { item in
                DueReviewCard.phrase(item.card, listening: item.schedule.listening
                    && item.card.translationCue(in: nativeLanguage) != nil)
            }
        let sections = languages.flatMap { memorization.dueSections(in: pieces, language: $0, now: now) }
            .sorted { $0.dueAt < $1.dueAt }.map { DueReviewCard.section($0.section) }
        var mixed: [DueReviewCard] = []
        for index in 0..<max(phrases.count, sections.count) {
            if phrases.indices.contains(index) { mixed.append(phrases[index]) }
            if sections.indices.contains(index) { mixed.append(sections[index]) }
        }
        return mixed
    }
}

struct DueReviewRound: Identifiable {
    let id = UUID()
    private(set) var cards: [DueReviewCard]
    private(set) var position = 0
    private var retried: Set<DueReviewCard.ID> = []

    init(due: [DueReviewCard]) { cards = Array(due.prefix(6)) }

    var current: DueReviewCard? { cards.indices.contains(position) ? cards[position] : nil }

    mutating func advance(recalled: Bool) {
        guard let current else { return }
        if !recalled && retried.insert(current.id).inserted { cards.append(current) }
        position += 1
    }
}
