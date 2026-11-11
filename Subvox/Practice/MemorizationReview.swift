import CryptoKit
import Foundation

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
    private var marks: [MemorizationSection.ID]

    init(saved: Data = Data()) {
        marks = (try? JSONDecoder().decode([MemorizationSection.ID].self, from: saved)) ?? []
    }

    var saved: Data { (try? JSONEncoder().encode(marks)) ?? Data() }

    func isMarked(_ id: MemorizationSection.ID) -> Bool { marks.contains(id) }

    mutating func toggle(_ id: MemorizationSection.ID) {
        if isMarked(id) { marks.removeAll { $0 == id } }
        else { marks.append(id) }
    }

    mutating func complete(_ id: MemorizationSection.ID, keepPractising: Bool) {
        marks.removeAll { $0 == id }
        if keepPractising { marks.append(id) }
    }

    func round(in pieces: [PracticeCorpus.Piece], language: AppLanguage) -> [MemorizationSection] {
        let localMarks = marks.filter { $0.language == language }
        let markedPieces = Set(localMarks.map(\.piece))
        let sections = pieces.lazy.filter { $0.language == language.corpus && markedPieces.contains($0.id) }
            .flatMap { MemorizationSection.sections(of: $0, language: language) }
        let byID = Dictionary(sections.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        return Array(localMarks.lazy.compactMap { byID[$0] }.prefix(3))
    }
}
