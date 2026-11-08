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

/// The practice language, chosen once for the whole app from the toolbar. Simplified and
/// Traditional Chinese share one corpus language ("zh") but show different characters.
enum AppLanguage: String, CaseIterable, Identifiable, Codable {
    case en, fr, es, zhHans = "zh-Hans", zhHant = "zh-Hant"
    var id: String { rawValue }
    var name: String {
        switch self { case .en: "English"; case .fr: "Français"; case .es: "Español"; case .zhHans: "简体中文"; case .zhHant: "繁體中文" }
    }
    var short: String {
        switch self { case .en: "EN"; case .fr: "FR"; case .es: "ES"; case .zhHans: "简"; case .zhHant: "繁" }
    }
    /// The language of the corpus entries and of Whisper.
    var corpus: String { rawValue.hasPrefix("zh") ? "zh" : rawValue }
    var isChinese: Bool { corpus == "zh" }
    /// The tag stored on prompts and takes (BCP-47; the script subtag keeps Traditional takes apart).
    var tag: String { rawValue }
}

/// Corpus/practice/practice-v1.json, built by Tools/corpus/practice.py with IPA from ml/subvox_ml/ipa.py.
struct PracticeCorpus: Decodable {
    struct Line: Decodable, Hashable {
        var text: String
        var ipa: [String]
        var pinyin: String? = nil
        var hant: String? = nil
        func text(_ language: AppLanguage) -> String { language == .zhHant ? (hant ?? text) : text }
    }
    struct Piece: Decodable, Identifiable, Hashable {
        var id: String; var language: String; var kind: String
        var title: String; var titleHant: String?; var author: String; var authorHant: String?
        var year: Int; var date: String?; var lines: [Line]
        enum CodingKeys: String, CodingKey {
            case id, language, kind, title, author, year, date, lines
            case titleHant = "title_hant", authorHant = "author_hant"
        }
        func title(_ language: AppLanguage) -> String { language == .zhHant ? (titleHant ?? title) : title }
        func author(_ language: AppLanguage) -> String { language == .zhHant ? (authorHant ?? author) : author }
        /// "Speech · 19 November 1863" or "Poem · 1609".
        var when: String { "\(kind.capitalized) · \(date ?? String(year))" }
    }
    struct Drill: Decodable, Identifiable, Hashable {
        var id: String; var language: String; var title: String; var phrases: [Line]
        var titleHant: String? = nil
        var level: String? = nil; var topic: String? = nil; var note: String? = nil
        enum CodingKeys: String, CodingKey { case id, language, title, phrases, level, topic, note, titleHant = "title_hant" }
        func title(_ language: AppLanguage) -> String { language == .zhHant ? (titleHant ?? title) : title }
    }
    struct MinimalPairs: Decodable, Identifiable, Hashable {
        var language: String; var contrast: String; var sets: [[Line]]
        var id: String { language + ":" + contrast }
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

    struct Reading: Hashable { var ipa: [String]; var pinyin: String? }

    /// IPA per word (and tone-marked pinyin for Chinese) for any text, from the trainer's converter.
    func phones(text: String, language: String) async -> Reading? {
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let output = root.appending(path: "phones-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: output) }
        struct Result: Decodable { var words: [[String]]; var pinyin: String? }
        do {
            try await runtime.run(["phones", "--text", text, "--language", language, "--output", output.path])
            let result = try JSONDecoder().decode(Result.self, from: Data(contentsOf: output))
            return Reading(ipa: result.words.map { $0.joined() }, pinyin: result.pinyin)
        } catch { return nil }
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
