import Foundation

enum SpeechCondition: String, Codable, Identifiable, Sendable {
    case silent = "Silent mouthing", normal = "Normal speech"
    var id: String { rawValue }
}

enum ReviewDecision: String, Codable, Sendable {
    case pending = "Needs review", accepted = "Accepted", excluded = "Excluded"
}

enum DataRole: String, Codable, Sendable {
    case training = "Training", validation = "Validation", finalTest = "Final test"
}

enum TrainingPace: String, CaseIterable, Identifiable {
    case comfortable = "Comfortable", extraTime = "Extra time"
    var id: String { rawValue }
    var baseline: Double { self == .comfortable ? 2 : 3 }
    var recovery: Double { 2 }
}

struct DecoderSettings: Codable, Equatable, Sendable {
    var method: LearningMethod = .motionTemplates
    var vocabularyScope: VocabularyScope = .full
    var hmmMaxScore: Double = 20
    var hmmMinGap: Double = 0
    var features: FeatureSet = .all
    var maxDistance: Double = 3
    var minConfidence: Double = 0
    var commonBias: Double = 0.05
}

/// Permit new optional settings to be added without losing saved personal tuning.
extension DecoderSettings {
    enum CodingKeys: String, CodingKey {
        case method, vocabularyScope, hmmMaxScore, hmmMinGap, features, maxDistance, minConfidence, commonBias
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        method = try c.decodeIfPresent(LearningMethod.self, forKey: .method) ?? .motionTemplates
        vocabularyScope = try c.decodeIfPresent(VocabularyScope.self, forKey: .vocabularyScope) ?? .full
        hmmMaxScore = try c.decodeIfPresent(Double.self, forKey: .hmmMaxScore) ?? 20
        hmmMinGap = try c.decodeIfPresent(Double.self, forKey: .hmmMinGap) ?? 0
        features = try c.decodeIfPresent(FeatureSet.self, forKey: .features) ?? .all
        maxDistance = try c.decodeIfPresent(Double.self, forKey: .maxDistance) ?? 3
        minConfidence = try c.decodeIfPresent(Double.self, forKey: .minConfidence) ?? 0
        commonBias = try c.decodeIfPresent(Double.self, forKey: .commonBias) ?? 0.05
    }
}
