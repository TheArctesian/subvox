import CryptoKit
import Foundation

/// Public-domain reading prompts built by Tools/corpus/community.py (Corpus/community/*.jsonl).
/// Held-out passages are never loaded here, so they can never be assigned for recording.
struct CommunityPrompt: Codable, Hashable, Sendable {
    var promptID: String
    var passageID: String?
    var order: Int
    var text: String
    var sourceID: String
    var title: String
    var author: String
    var words: Int
    var phones: [String]
    var version: Int

    enum CodingKeys: String, CodingKey {
        case promptID = "prompt_id", passageID = "passage_id", sourceID = "source_id"
        case order, text, title, author, words, phones, version
    }
    var diphones: [String] {
        let flat = phones.flatMap { $0.split(separator: " ").map(String.init) }
        return zip(flat, flat.dropFirst()).map { "\($0) \($1)" }
    }
    var sentencePrompt: SentencePrompt { SentencePrompt(text: text, kind: passageID == nil ? .calibration : .community) }
}

struct CommunityPassage: Identifiable, Hashable, Sendable {
    var id: String
    var title: String
    var author: String
    var prompts: [CommunityPrompt]
    var words: Int { prompts.reduce(0) { $0 + $1.words } }
}

struct CommunityCorpus: Sendable {
    var passages: [CommunityPassage]
    var calibration: [CommunityPrompt]
    private var byID: [String: CommunityPrompt]

    init(prompts: [CommunityPrompt], calibration: [CommunityPrompt]) {
        var grouped: [String: [CommunityPrompt]] = [:]
        var order: [String] = []
        for prompt in prompts {
            guard let passage = prompt.passageID else { continue }
            if grouped[passage] == nil { order.append(passage) }
            grouped[passage, default: []].append(prompt)
        }
        passages = order.map { id in
            let items = grouped[id]!.sorted { $0.order < $1.order }
            return CommunityPassage(id: id, title: items[0].title, author: items[0].author, prompts: items)
        }
        self.calibration = calibration.sorted { $0.order < $1.order }
        byID = Dictionary((prompts + calibration).map { ($0.promptID, $0) }, uniquingKeysWith: { first, _ in first })
    }

    static let bundled: CommunityCorpus = load(version: 1)

    static func load(version: Int, bundle: Bundle = .main) -> CommunityCorpus {
        func rows(_ name: String) -> [CommunityPrompt] {
            guard let url = bundle.url(forResource: "\(name)-v\(version)", withExtension: "jsonl", subdirectory: "Corpus/community"),
                  let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
            return parse(text)
        }
        return CommunityCorpus(prompts: rows("prompts"), calibration: rows("calibration"))
    }

    static func parse(_ jsonl: String) -> [CommunityPrompt] {
        let decoder = JSONDecoder()
        return jsonl.split(separator: "\n").compactMap { try? decoder.decode(CommunityPrompt.self, from: Data($0.utf8)) }
    }

    func prompt(_ id: String) -> CommunityPrompt? { byID[id] }

    /// Diphone counts over community prompts that already have a mouthed take.
    func counts(recordedPromptIDs: Set<String>) -> [String: Int] {
        var result: [String: Int] = [:]
        for id in recordedPromptIDs { for pair in byID[id]?.diphones ?? [] { result[pair, default: 0] += 1 } }
        return result
    }

    /// Same greedy rule as `assign` in Tools/corpus/community.py: new-diphone gain per word, counts
    /// updated as passages are chosen, ties to the lowest passage ID. Recorded passages are skipped.
    func assign(counts: [String: Int], recordedPassages: Set<String>, count n: Int) -> [CommunityPassage] {
        var counts = counts
        var pool = passages.filter { !recordedPassages.contains($0.id) }
        var chosen: [CommunityPassage] = []
        while chosen.count < n, !pool.isEmpty {
            func gain(_ p: CommunityPassage) -> Double {
                let pairs = p.prompts.flatMap(\.diphones)
                return pairs.reduce(0) { $0 + 1 / Double(1 + counts[$1, default: 0]) } / Double(max(1, p.words))
            }
            var bestIndex = 0, bestGain = -Double.infinity
            for (i, p) in pool.enumerated() {
                let g = gain(p)
                if g > bestGain + 1e-12 || (abs(g - bestGain) <= 1e-12 && p.id < pool[bestIndex].id) { bestIndex = i; bestGain = g }
            }
            let best = pool.remove(at: bestIndex)
            chosen.append(best)
            for pair in best.prompts.flatMap(\.diphones) { counts[pair, default: 0] += 1 }
        }
        return chosen
    }
}

/// The public IDs refer to exact library sections, independently of local sentence IDs.
struct LibraryPilotPrompt: Decodable, Sendable {
    let promptID: String
    let passageID: String
    let pieceID: String
    let language: String
    let practiceKind: SentencePromptKind
    let section: Int
    let text: String
    let lines: [String]
    let title: String
    let author: String
    let sourceURL: URL

    enum CodingKeys: String, CodingKey {
        case promptID = "prompt_id", passageID = "passage_id", pieceID = "piece_id"
        case practiceKind = "kind", sourceURL = "source_url"
        case language, section, text, lines, title, author
    }
}

struct LibraryPilotCatalog: Sendable {
    static let version = "library-pilot-v1"
    static let bundled = load()
    private let prompts: [LibraryPilotPrompt]

    static func load(bundle: Bundle = .main) -> Self {
        guard let url = bundle.url(forResource: version, withExtension: "jsonl", subdirectory: "Corpus/community"),
              let jsonl = try? String(contentsOf: url, encoding: .utf8) else { return Self(prompts: []) }
        let decoder = JSONDecoder()
        do {
            let prompts = try jsonl.split(separator: "\n").map {
                try decoder.decode(LibraryPilotPrompt.self, from: Data($0.utf8))
            }
            let kinds: [String: SentencePromptKind] = ["invictus": .poetry, "gettysburg": .speech]
            let byPiece = Dictionary(grouping: prompts, by: \.pieceID)
            let expectedSections = ["invictus": [1, 2, 3, 4], "gettysburg": [1, 2, 3]]
            guard Set(prompts.map(\.promptID)).count == prompts.count,
                  byPiece.mapValues({ $0.map(\.section).sorted() }) == expectedSections,
                  prompts.allSatisfy({ prompt in
                      return prompt.language == "en" && kinds[prompt.pieceID] == prompt.practiceKind
                          && !prompt.lines.isEmpty
                          && prompt.passageID == "\(version):\(prompt.pieceID)"
                          && prompt.text.utf8.elementsEqual(prompt.lines.joined(separator: " ").utf8)
                          && prompt.promptID == identity(pieceID: prompt.pieceID, language: prompt.language, text: prompt.text)
                  }) else { return Self(prompts: []) }
            return Self(prompts: prompts)
        } catch {
            return Self(prompts: [])
        }
    }

    static func identity(pieceID: String, language: String, text: String) -> String {
        let material = [version, language, pieceID, text].joined(separator: "\n")
        return SHA256.hash(data: Data(material.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    func prompt(_ id: String) -> LibraryPilotPrompt? { prompts.first { $0.promptID == id } }

    func sections(pieceID: String, language: String) -> [LibraryPilotPrompt] {
        prompts.filter { $0.pieceID == pieceID && $0.language == language }.sorted { $0.section < $1.section }
    }

    func allows(pieceID: String, language: String, groups: [[SentencePrompt]]) -> Bool {
        guard let first = groups.first else { return false }
        let sections = sections(pieceID: pieceID, language: language)
        guard let start = sections.firstIndex(where: { Self.matches(first, $0) }),
              groups.count <= sections.count - start else { return false }
        return zip(groups, sections.dropFirst(start)).allSatisfy { Self.matches($0, $1) }
    }

    func prompt(for pair: SentencePair) -> LibraryPilotPrompt? {
        guard let pieceID = pair.literaturePieceID, let language = pair.language,
              let prompt = prompt(Self.identity(pieceID: pieceID, language: language, text: pair.text)),
              pair.practiceMode == prompt.practiceKind,
              pair.promptIDs == prompt.lines.map(SentencePrompt.identity) else { return nil }
        return prompt
    }

    private static func matches(_ group: [SentencePrompt], _ prompt: LibraryPilotPrompt) -> Bool {
        group.count == prompt.lines.count && zip(group, prompt.lines).allSatisfy { line, text in
            line.kind == prompt.practiceKind && line.language == prompt.language
                && line.text.utf8.elementsEqual(text.utf8)
        }
    }
}
