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
