import Foundation

/// The Train tabs (docs/next-phase-plan.md, item 3). Every tab records through the same paired
/// interface: the text is shown first, spoken aloud, then mouthed.
enum TrainMode: String, CaseIterable, Identifiable {
    case reading = "Reading", sounds = "Sounds", speech = "Speech & Poetry", language = "Language", minimalPairs = "Minimal Pairs", free = "Free Practice"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .reading: "book"
        case .sounds: "waveform"
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
    case en, fr, es, esMX = "es-MX", zhHans = "zh-Hans", zhHant = "zh-Hant"
    var id: String { rawValue }
    var name: String {
        switch self { case .en: "English"; case .fr: "Français"; case .es: "Español (España)"; case .esMX: "Español (México)"; case .zhHans: "简体中文"; case .zhHant: "繁體中文" }
    }
    var short: String {
        switch self { case .en: "EN"; case .fr: "FR"; case .es: "ES"; case .esMX: "MX"; case .zhHans: "简"; case .zhHant: "繁" }
    }
    /// The language of the corpus entries and of Whisper.
    var corpus: String { String(rawValue.prefix(2)) }
    /// Drills tagged for one region (Spain "ES", Mexico "MX") show only in that variant.
    func shows(region: String?) -> Bool {
        guard let region, corpus == "es" else { return true }
        return (region == "MX") == (self == .esMX)
    }
    /// Sounds a variant doesn't use: Latin American Spanish has no /θ/ (seseo) and merges /ʎ/ into /ʝ/.
    var absentSounds: Set<String> { self == .esMX ? ["θ", "ʎ"] : [] }
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
        /// English meaning (minimal pairs, everyday phrases).
        var gloss: String? = nil
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
        var level: String? = nil; var topic: String? = nil; var note: String? = nil; var noteHant: String? = nil
        var region: String? = nil
        enum CodingKeys: String, CodingKey { case id, language, title, phrases, level, topic, note, region, titleHant = "title_hant", noteHant = "note_hant" }
        func title(_ language: AppLanguage) -> String { language == .zhHant ? (titleHant ?? title) : title }
        func note(_ language: AppLanguage) -> String? { language == .zhHant ? (noteHant ?? note) : note }
    }
    /// One sound of a language: hear it, read how to make it, then say example words.
    struct Sound: Decodable, Identifiable, Hashable {
        var ipa: String; var group: String; var name: String; var guide: String
        var tip: String? = nil; var say: String; var examples: [Line]
        var id: String { ipa }
    }
    struct MinimalPairs: Decodable, Identifiable, Hashable {
        var language: String; var contrast: String; var sets: [[Line]]
        var label: String? = nil
        var id: String { language + ":" + contrast + (label ?? "") }
        /// "/b/ /v/", or a plain label where the contrast isn't one sound ("word endings p b").
        var title: String { label ?? "/" + contrast.replacingOccurrences(of: " ", with: "/ /") + "/" }
    }
    var version: Int
    var pieces: [Piece]
    var drills: [Drill]
    var minimalPairs: [MinimalPairs]
    var sounds: [String: [Sound]] = [:]
    enum CodingKeys: String, CodingKey { case version, pieces, drills, sounds, minimalPairs = "minimal_pairs" }

    init(version: Int, pieces: [Piece], drills: [Drill], minimalPairs: [MinimalPairs], sounds: [String: [Sound]] = [:]) {
        self.version = version; self.pieces = pieces; self.drills = drills; self.minimalPairs = minimalPairs; self.sounds = sounds
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = try c.decode(Int.self, forKey: .version)
        pieces = try c.decode([Piece].self, forKey: .pieces)
        drills = try c.decode([Drill].self, forKey: .drills)
        minimalPairs = try c.decode([MinimalPairs].self, forKey: .minimalPairs)
        sounds = try c.decodeIfPresent([String: [Sound]].self, forKey: .sounds) ?? [:]
    }

    static let empty = PracticeCorpus(version: 0, pieces: [], drills: [], minimalPairs: [])

    static let bundled: PracticeCorpus = {
        guard let url = Bundle.main.url(forResource: "practice-v1", withExtension: "json", subdirectory: "Corpus/practice"),
              let data = try? Data(contentsOf: url) else { return .empty }
        return (try? JSONDecoder().decode(PracticeCorpus.self, from: data)) ?? .empty
    }()
}

extension PracticeCorpus {
    /// A piece split into takes of whole lines, each short enough to read on the recording screen
    /// in one go (about 30 words, or 40 characters of Chinese).
    static func takes(_ lines: [Line], language: AppLanguage, limit: Int = 30) -> [[Line]] {
        func size(_ line: Line) -> Int {
            language.isChinese ? line.text(language).count * 3 / 4 : line.text(language).split(whereSeparator: \.isWhitespace).count
        }
        var takes: [[Line]] = [], current: [Line] = [], words = 0
        for line in lines {
            if !current.isEmpty && words + size(line) > limit { takes.append(current); current = []; words = 0 }
            current.append(line); words += size(line)
        }
        return current.isEmpty ? takes : takes + [current]
    }

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

/// Progress that's visible on every practice page: takes today, a day streak, and which sounds
/// have been practised. Counted from saved pairs with a mouthed take, so abandoned attempts don't count.
struct PracticeProgress: Equatable {
    var today = 0
    var streak = 0
    var practisedTexts: Set<String> = []

    init(pairs: [SentencePair], calendar: Calendar = .current, now: Date = Date()) {
        let done = pairs.filter { $0.silent != nil }
        today = done.filter { calendar.isDate($0.createdAt, inSameDayAs: now) }.count
        let days = Set(done.map { calendar.startOfDay(for: $0.createdAt) })
        var day = calendar.startOfDay(for: now)
        if !days.contains(day) { day = calendar.date(byAdding: .day, value: -1, to: day)! }
        while days.contains(day) { streak += 1; day = calendar.date(byAdding: .day, value: -1, to: day)! }
        practisedTexts = Set(done.filter { $0.practiceMode != nil }.map(\.text))
    }

    func practised(_ sound: PracticeCorpus.Sound, language: AppLanguage) -> Bool {
        sound.examples.contains { practisedTexts.contains($0.text(language)) }
    }
}

/// Sounds that speakers of one language most often find hard in another (learner-error
/// literature summarised; broad, not exhaustive). Keyed by practised language, then own language.
enum TrickySounds {
    static let table: [String: [String: [String]]] = [
        "en": ["es": ["ɪ", "i", "v", "b", "ʃ", "z", "h", "θ", "ð", "ʌ", "æ", "ʊ"],
               "fr": ["θ", "ð", "h", "ɪ", "ʊ", "ɹ", "ŋ", "æ", "ʌ", "oʊ"],
               "zh": ["θ", "ð", "v", "ɹ", "l", "n", "ɪ", "æ", "ɛ", "z", "dʒ"]],
        "fr": ["en": ["y", "u", "ʁ", "ɑ̃", "ɔ̃", "ɛ̃", "œ̃", "ø", "œ", "ɥ"],
               "es": ["y", "ø", "œ", "ə", "ʁ", "ʒ", "z", "v", "ɑ̃", "ɔ̃", "ɛ̃"],
               "zh": ["y", "ʁ", "ʒ", "v", "ɑ̃", "ɔ̃", "ɛ̃", "b", "d", "ɡ"]],
        "es": ["en": ["r", "ɾ", "β", "ð", "ɣ", "x", "θ", "e", "o", "ɲ"],
               "fr": ["r", "ɾ", "x", "θ", "β", "ð", "ɣ", "u"],
               "zh": ["r", "ɾ", "l", "β", "ð", "ɣ", "b", "d", "ɡ", "x"]],
        "zh": ["en": ["y", "ʈʂ", "ʂ", "ʐ", "tɕ", "ɕ", "tɕʰ", "ɤ", "ɻ̩", "ɹ̩", "ts"],
               "es": ["ʂ", "ʐ", "ɕ", "tɕ", "tɕʰ", "y", "ɤ", "pʰ", "tʰ", "kʰ", "ɻ̩"],
               "fr": ["ʂ", "ʐ", "ɕ", "tɕ", "pʰ", "tʰ", "kʰ", "ɤ", "ɻ̩", "ɹ̩"]],
    ]
    static func ipa(practising: String, speaker: String) -> [String] {
        practising == speaker ? [] : table[practising]?[speaker] ?? []
    }
}
