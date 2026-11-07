import Foundation

/// One recording visit. Unknown legacy metadata is deliberately left unknown.
struct RecordingVisit: Codable, Identifiable, Sendable {
    var id = UUID()
    var startedAt = Date()
    var endedAt: Date?
    var condition: SpeechCondition
    var notes: String
    var sampleRate: Double
    var prompts: [String]
    var recordingDirectory: String?
}

enum SpeechCondition: String, Codable, CaseIterable, Identifiable, Sendable {
    case silent = "Silent mouthing", normal = "Normal speech", whisper = "Whisper"
    var id: String { rawValue }
}

enum ReviewDecision: String, Codable, CaseIterable, Sendable {
    case pending = "Needs review", accepted = "Accepted", excluded = "Excluded"
}

enum DataRole: String, Codable, Sendable {
    case training = "Training", validation = "Validation", finalTest = "Final test"
}

struct DecoderSettings: Codable, Equatable, Sendable {
    var method: LearningMethod = .legacyMedoids
    var vocabularyScope: VocabularyScope = .full
    // Sequence scores are length-normalized NLLs, not DTW distances. These thresholds are
    // independent heuristics until calibrated on validation; neither is a probability.
    var hmmMaxScore: Double = 20
    var hmmMinGap: Double = 0
    var condition: SpeechCondition = .silent
    var features: FeatureSet = .all
    var maxDistance: Double = 3
    var minConfidence: Double = 0
    var commonBias: Double = 0.05
}

/// Assignments are persisted by word so repeated takes never cross the split. Final test is
/// 25% of word groups (rounded up); a quarter of the remaining groups is validation.
struct TrainingExperiment: Codable, Sendable {
    var id = UUID()
    var createdAt = Date()
    var wordRoles: [String: DataRole] = [:]
    var historicalTakeCount = 0
    var settings = DecoderSettings()
    var frozenAt: Date?
    var finalManifest: DecoderManifest?
    var validationReport: DecoderReport?
    var comparisonReports: [DecoderReport]?
    var finalReport: DecoderReport?

    static func key(_ word: String) -> String { word.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }

    mutating func assign(_ words: [String]) {
        var generator = SystemRandomNumberGenerator()
        assign(words, using: &generator)
    }

    mutating func assign<G: RandomNumberGenerator>(_ words: [String], using generator: inout G) {
        guard frozenAt == nil else { return }
        let unseen = Set(words.map(Self.key)).subtracting(wordRoles.keys).sorted().shuffled(using: &generator)
        for word in unseen {
            let n = wordRoles.count + 1
            let finalCount = wordRoles.values.filter { $0 == .finalTest }.count
            let developmentCount = wordRoles.count - finalCount
            let validationCount = wordRoles.values.filter { $0 == .validation }.count
            if finalCount < (n + 3) / 4 {
                wordRoles[word] = .finalTest
            } else if validationCount < (developmentCount + 4) / 4 {
                wordRoles[word] = .validation
            } else {
                wordRoles[word] = .training
            }
        }
    }

    func role(of take: Template) -> DataRole? { wordRoles[Self.key(take.label)] }

    func matchesCondition(_ take: Template) -> Bool {
        take.condition == nil || take.condition == settings.condition
    }

    /// Validation is never used for fitting during development. Only once frozen may it be
    /// combined with training for the final fit. Final-test takes are never fitted.
    func fittingTakes(from takes: [Template]) -> [Template] {
        takes.filter { take in
            guard take.isUsable, matchesCondition(take) else { return false }
            return role(of: take) == .training || (frozenAt != nil && role(of: take) == .validation)
        }
    }
}

// Swift's synthesized Decodable does not apply property defaults to absent keys.
// Old experiments and historical reports must remain on their original decoder.
extension DecoderSettings {
    enum CodingKeys: String, CodingKey {
        case method, vocabularyScope, hmmMaxScore, hmmMinGap, condition, features, maxDistance, minConfidence, commonBias
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        method = try c.decodeIfPresent(LearningMethod.self, forKey: .method) ?? .legacyMedoids
        vocabularyScope = try c.decodeIfPresent(VocabularyScope.self, forKey: .vocabularyScope) ?? .full
        hmmMaxScore = try c.decodeIfPresent(Double.self, forKey: .hmmMaxScore) ?? 20
        hmmMinGap = try c.decodeIfPresent(Double.self, forKey: .hmmMinGap) ?? 0
        condition = try c.decodeIfPresent(SpeechCondition.self, forKey: .condition) ?? .silent
        features = try c.decodeIfPresent(FeatureSet.self, forKey: .features) ?? .all
        maxDistance = try c.decodeIfPresent(Double.self, forKey: .maxDistance) ?? 3
        minConfidence = try c.decodeIfPresent(Double.self, forKey: .minConfidence) ?? 0
        commonBias = try c.decodeIfPresent(Double.self, forKey: .commonBias) ?? 0.05
    }
}
