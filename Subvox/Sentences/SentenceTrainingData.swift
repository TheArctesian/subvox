import Foundation
import CryptoKit

enum SentenceReference {
    enum Failure: LocalizedError {
        case ambiguousTiming, invalidClock
        var errorDescription: String? {
            switch self {
            case .ambiguousTiming: "Whisper assigned one interval to multiple words. Word boundaries are ambiguous; no duplicate timings were invented. Re-record with sentence prompts or review the raw audio."
            case .invalidClock: "Whisper word timing or the recording clock is invalid. Re-record this pair."
            }
        }
    }
    /// Whisper times already include the audio crop offset; add only the session audio clock.
    static func anchors(_ transcript: LocalWhisperTranscript, sessionStart: Double,
                        audioOffset: Double) throws -> [MotionWordAnchor] {
        guard sessionStart.isFinite, audioOffset.isFinite else { throw Failure.invalidClock }
        return try transcript.words.compactMap { token in
            let words = Lexicon.words(in: token.text)
            guard words.count <= 1 else { throw Failure.ambiguousTiming }
            guard let word = words.first else { return nil }
            guard token.start.isFinite, token.end.isFinite, token.start >= 0, token.end > token.start else {
                throw Failure.invalidClock
            }
            return MotionWordAnchor(word: word, start: token.start + sessionStart + audioOffset,
                                    end: token.end + sessionStart + audioOffset)
        }
    }
}

enum SentencePromptKind: String, Codable, Sendable { case sentence, wordRepair }

struct SentencePrompt: Codable, Identifiable, Hashable, Sendable {
    var text: String
    var kind: SentencePromptKind = .sentence
    /// Punctuation/case variants have the same persistent identity.
    var id: String { Self.identity(text) }
    static func identity(_ text: String) -> String {
        let normalized = Lexicon.words(in: text).joined(separator: " ")
        return SHA256.hash(data: Data(normalized.utf8)).map { String(format: "%02x", $0) }.joined()
    }
    static var bundled: [SentencePrompt] {
        guard let url = Bundle.main.url(forResource: "paired-sentences", withExtension: "txt", subdirectory: "Corpus"),
              let content = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return content.split(separator: "\n").map { SentencePrompt(text: String($0)) }
    }
}

struct SentenceRecording: Codable, Identifiable, Sendable {
    var id = UUID()
    var condition: SpeechCondition
    var samples: [MotionSample]
    var cueStart: Double
    var cueEnd: Double
    var cueSamples: [MotionSample] { samples.filter { $0.t >= cueStart && $0.t <= cueEnd } }
    var quality: TakeQuality { TakeQuality.inspect(cueSamples, maximumDuration: 48) }
}

struct SentencePair: Codable, Identifiable, Sendable {
    var id = UUID()
    var visitID: UUID
    var promptIDs: [String]
    var text: String
    var role: DataRole
    var createdAt = Date()
    var spoken: SentenceRecording?
    var silent: SentenceRecording?
    var directory: String?
    var sessionStartUptime: Double?
    var audioOffset: Double?
    var recordingSessionID: UUID?
    var transcript: LocalWhisperTranscript?
    var transcriptMatchRate: Double?
    var alignment: MotionTransferResult?
    var review: ReviewDecision = .pending
    var error: String?
    var complete: Bool { spoken != nil && silent != nil }
    var canAccept: Bool { complete && error == nil && alignment?.words.isEmpty == false }
}

struct SentenceVisit: Codable, Identifiable, Sendable {
    var id = UUID()
    var startedAt = Date()
    var endedAt: Date?
    var notes: String
    var directory: String?
    var pairIDs: [UUID] = []
    var audioDevice: String?
    var pace: String?
    var presentation: String?
    var motionRate: Double?
    var recordingSessionID: UUID?
}

/// Separate sentence experiment. The original word experiment is never reassigned.
struct SentenceTrainingData: Codable, Sendable {
    var schemaVersion = 1
    var id = UUID()
    var createdAt = Date()
    var prompts: [SentencePrompt] = []
    var roles: [String: DataRole] = [:]
    var visits: [SentenceVisit] = []
    var pairs: [SentencePair] = []
    var reservedUnlockedAt: Date?
    var validationReport: SentenceValidationReport?
    var decoderSettings: DecoderSettings?

    mutating func add<G: RandomNumberGenerator>(_ incoming: [SentencePrompt], using generator: inout G) {
        guard reservedUnlockedAt == nil else { return }
        let known = Set(prompts.map(\.id))
        let unique = Dictionary(incoming.filter { !known.contains($0.id) && !Lexicon.words(in: $0.text).isEmpty }
            .map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        for prompt in unique.values.sorted(by: { $0.id < $1.id }).shuffled(using: &generator) {
            if prompt.kind == .wordRepair {
                roles[prompt.id] = .training
            } else {
                let n = prompts.filter { $0.kind == .sentence }.count + 1
                let final = prompts.filter { $0.kind == .sentence && roles[$0.id] == .finalTest }.count
                let validation = prompts.filter { $0.kind == .sentence && roles[$0.id] == .validation }.count
                let development = n - 1 - final
                roles[prompt.id] = final < (n + 3) / 4 ? .finalTest
                    : validation < (development + 4) / 4 ? .validation : .training
            }
            prompts.append(prompt)
        }
    }

    func derivedWords(role: DataRole, lexicon: Lexicon) -> [Template] {
        guard role != .finalTest else { return [] }
        return pairs.filter { $0.role == role && $0.review == .accepted }.flatMap { pair -> [Template] in
            guard pair.canAccept, !pair.promptIDs.isEmpty,
                  pair.promptIDs.allSatisfy({ roles[$0] == role }), let silent = pair.silent,
                  let transfer = pair.alignment else { return [] }
            return transfer.words.enumerated().compactMap { index, word in
                guard lexicon.phones(word.word) != nil else { return nil }
                guard word.silentStart >= silent.cueStart - 0.04,
                      word.silentEnd <= silent.cueEnd + 0.04,
                      word.silentEnd > word.silentStart else { return nil }
                let samples = silent.cueSamples.filter { $0.t >= word.silentStart && $0.t <= word.silentEnd }
                guard samples.count >= 3, !TakeQuality.inspect(samples).invalid else { return nil }
                return Template(id: Self.wordID(pair: pair.id, index: index), label: word.word, samples: samples, createdAt: pair.createdAt, sessionID: pair.visitID,
                                condition: .silent, review: .accepted, purpose: .training,
                                sourceSentenceID: pair.promptIDs.joined(separator: ","), sourcePairID: pair.id)
            }
        }
    }

    static func wordID(pair: UUID, index: Int) -> UUID {
        let bytes = Array(SHA256.hash(data: Data("\(pair.uuidString):\(index)".utf8)).prefix(16))
        return UUID(uuid: (bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                           bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]))
    }

    /// Counts dictionary phone occurrences in accepted, aligned TRAINING words only.
    /// This is annotation coverage, never evidence that these phones were recognised.
    func coverage(lexicon: Lexicon) -> [String: Int] {
        var result: [String: Int] = [:]
        for word in derivedWords(role: .training, lexicon: lexicon) {
            for phone in lexicon.phones(word.label) ?? [] { result[phone, default: 0] += 1 }
        }
        return result
    }

    func plannedSentences(limit: Int, passage: Bool, lexicon: Lexicon) -> [[SentencePrompt]] {
        let acquired = Set(pairs.filter(\.complete).flatMap(\.promptIDs))
        let remaining = prompts.filter { $0.kind == .sentence && roles[$0.id] != .finalTest && !acquired.contains($0.id) }
        let training = remaining.filter { roles[$0.id] == .training }.prefix(limit)
        let validation = remaining.filter { roles[$0.id] == .validation }.prefix(max(1, limit / 3))
        let chosen = Array(training) + Array(validation)
        if !passage { return chosen.map { [$0] } }
        // Never mix train/validation/final sentence groups inside a passage.
        return [DataRole.training, .validation].flatMap { role in
            let group = chosen.filter { roles[$0.id] == role }
            return stride(from: 0, to: group.count, by: 3).map { Array(group[$0..<min($0 + 3, group.count)]) }
        }
    }

    func repairWords(lexicon: Lexicon, target: Int = 4) -> [String] {
        // Sentences get the first opportunity to fill gaps. A partial visit cannot trigger
        // a long isolated-word list before sentence collection/review is complete.
        let development = prompts.filter { $0.kind == .sentence && roles[$0.id] == .training }
        guard !development.isEmpty,
              development.allSatisfy({ prompt in pairs.contains {
                  $0.promptIDs.contains(prompt.id) && $0.role == .training && $0.review == .accepted && $0.canAccept
              } }) else { return [] }
        let counts = coverage(lexicon: lexicon)
        let needs = Dictionary(uniqueKeysWithValues: Phoneme.all.map { ($0.symbol, target - counts[$0.symbol, default: 0]) })
        var generator = SplitMix64(seed: 0x5E17)
        let candidates = Lexicon.vocabulary.prefix(4000).map(\.word).filter {
            let phones = lexicon.phones($0) ?? []
            return !phones.isEmpty && phones.filter(Lexicon.vowels.contains).count <= 3
        }
        return CoveragePlanner.plan(needs: needs, candidates: candidates, lexicon: lexicon, avoiding: [], using: &generator)
    }
}

enum SentenceStore {
    static var url: URL { TemplateStore.url.deletingLastPathComponent().appending(path: "sentence-training.json") }
    static func load(from location: URL? = nil) throws -> SentenceTrainingData {
        let location = location ?? url
        guard FileManager.default.fileExists(atPath: location.path) else { return SentenceTrainingData() }
        return try JSONDecoder().decode(SentenceTrainingData.self, from: Data(contentsOf: location))
    }
    static func save(_ data: SentenceTrainingData, to location: URL? = nil) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(data).write(to: location ?? url, options: .atomic)
    }
}
