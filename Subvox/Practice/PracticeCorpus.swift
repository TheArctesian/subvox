import Foundation

/// The Train tabs (docs/next-phase-plan.md, item 3). Every tab records through the same paired
/// interface: the text is shown first, spoken aloud, then mouthed.
enum TrainMode: String, CaseIterable, Identifiable {
    case reading = "Reading", speech = "Speech & Poetry", language = "Language", minimalPairs = "Minimal Pairs", free = "Free Practice"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .reading: "book"
        case .speech: "text.quote"
        case .language: "character.bubble"
        case .minimalPairs: "circle.grid.2x1"
        case .free: "square.and.pencil"
        }
    }
    var shortcut: Character { Character(String((TrainMode.allCases.firstIndex(of: self) ?? 0) + 1)) }
}

enum PracticeLanguage: String, CaseIterable, Identifiable, Codable {
    case en, fr, es, zh
    var id: String { rawValue }
    var name: String {
        switch self { case .en: "English"; case .fr: "Français"; case .es: "Español"; case .zh: "中文" }
    }
}

/// Corpus/practice/practice-v1.json, built by Tools/corpus/practice.py with IPA from ml/subvox_ml/ipa.py.
struct PracticeCorpus: Decodable {
    struct Line: Decodable, Hashable { var text: String; var ipa: [String]; var pinyin: String? }
    struct Piece: Decodable, Identifiable, Hashable {
        var id: String; var language: PracticeLanguage; var kind: String; var title: String; var author: String; var year: Int; var lines: [Line]
    }
    struct Drill: Decodable, Identifiable, Hashable { var id: String; var language: PracticeLanguage; var title: String; var phrases: [Line] }
    struct MinimalPairs: Decodable, Identifiable, Hashable {
        var language: PracticeLanguage; var contrast: String; var sets: [[Line]]
        var id: String { language.rawValue + ":" + contrast }
    }
    var version: Int
    var pieces: [Piece]
    var drills: [Drill]
    var minimalPairs: [MinimalPairs]
    enum CodingKeys: String, CodingKey { case version, pieces, drills, minimalPairs = "minimal_pairs" }

    static let empty = PracticeCorpus(version: 0, pieces: [], drills: [], minimalPairs: [])

    static let bundled: PracticeCorpus = {
        guard let url = Bundle.main.url(forResource: "practice-v1", withExtension: "json", subdirectory: "Corpus/practice"),
              let data = try? Data(contentsOf: url) else { return .empty }
        return (try? JSONDecoder().decode(PracticeCorpus.self, from: data)) ?? .empty
    }()
}

extension PracticeCorpus {
    /// A round of minimal pairs: every word of `count` sets, shuffled within the round so the
    /// next word can't be guessed from the previous one.
    static func round<G: RandomNumberGenerator>(_ pairs: MinimalPairs, count: Int, using generator: inout G) -> [Line] {
        Array(pairs.sets.shuffled(using: &generator).prefix(count).flatMap { $0 }).shuffled(using: &generator)
    }
}

/// Runs `subvox-ml phone-check` on a transcribed practice pair: expected IPA from the shown text,
/// produced IPA from what Whisper heard (or phone intervals, once a phone recogniser is configured).
@MainActor final class PracticeChecker {
    private let runtime: SequenceRuntime
    private let root: URL
    init(root: URL = SentenceStore.directory.appending(path: "Practice")) {
        self.root = root
        runtime = SequenceRuntime(root: SentenceStore.directory.appending(path: "SequenceLearning"))
    }

    func check(text: String, language: String, heard: String) async -> PhoneCheckResult? {
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let output = root.appending(path: "phone-check-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: output) }
        do {
            try await runtime.run(["phone-check", "--text", text, "--language", language, "--whisper-text", heard, "--output", output.path])
            return try JSONDecoder().decode(PhoneCheckResult.self, from: Data(contentsOf: output))
        } catch { return nil }
    }
}
