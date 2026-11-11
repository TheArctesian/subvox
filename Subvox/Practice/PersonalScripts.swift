import Foundation
import Observation

struct PersonalScript: Codable, Identifiable, Equatable {
    let id: UUID
    var revision: UUID
    var title: String
    var text: String
    var language: AppLanguage

    var pieceID: String { "personal-\(revision.uuidString)" }
    var sections: [String] { Self.sections(in: text) }

    static func sections(in text: String) -> [String] {
        var sections: [String] = []
        text.enumerateSubstrings(in: text.startIndex..., options: .bySentences) { sentence, _, _, _ in
            if let sentence = sentence?.trimmingCharacters(in: .whitespacesAndNewlines), !sentence.isEmpty {
                sections.append(sentence)
            }
        }
        return sections
    }

    var memoryPiece: PracticeCorpus.Piece {
        let lines = sections.map { PracticeCorpus.Line(text: $0, ipa: []) }
        return PracticeCorpus.Piece(id: pieceID, language: language.corpus, kind: "prose", title: title,
                                    author: String(localized: "My Material"), lines: lines,
                                    stanzaEnds: lines.indices.map { $0 + 1 })
    }
}

@MainActor @Observable
final class PersonalScripts {
    private static let key = "practice.personalScripts"
    private let defaults: UserDefaults
    private(set) var items: [PersonalScript]

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        items = defaults.data(forKey: Self.key).flatMap { try? JSONDecoder().decode([PersonalScript].self, from: $0) } ?? []
    }

    @discardableResult
    func save(id: UUID?, title: String, text: String, language: AppLanguage) -> PersonalScript? {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, !text.isEmpty else { return nil }
        let previous = items.first { $0.id == id }
        let sameText = previous?.text.utf8.elementsEqual(text.utf8) == true && previous?.language == language
        let script = PersonalScript(id: previous?.id ?? UUID(), revision: (sameText ? previous?.revision : nil) ?? UUID(),
                                    title: title, text: text, language: language)
        let updated = [script] + items.filter { $0.id != script.id }
        return persist(updated) ? script : nil
    }

    func remove(_ id: UUID) {
        _ = persist(items.filter { $0.id != id })
    }

    private func persist(_ updated: [PersonalScript]) -> Bool {
        guard let data = try? JSONEncoder().encode(updated) else { return false }
        defaults.set(data, forKey: Self.key)
        items = updated
        return true
    }
}
