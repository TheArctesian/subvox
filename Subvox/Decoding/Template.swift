import Foundation

struct Template: Codable, Identifiable, Hashable, Sendable {
    var id = UUID()
    var label: String
    var samples: [MotionSample]
    var createdAt = Date()
    var sessionID: UUID?
    var condition: SpeechCondition?
    var measuredSampleRate: Double?
    var review: ReviewDecision?
    var purpose: RecordingPurpose?
    var cueStartUptime: TimeInterval?
    var cueEndUptime: TimeInterval?
    var sourceSentenceID: String?
    var sourcePairID: UUID?

    var duration: TimeInterval {
        guard let first = samples.first, let last = samples.last else { return 0 }
        return last.t - first.t
    }
}

struct RankedLabel: Identifiable, Hashable, Sendable {
    var label: String
    var distance: Double
    var id: String { label }
}

struct Prediction: Hashable, Sendable {
    var ranked: [RankedLabel]
    /// 0…1 margin between the best and second-best class.
    var confidence: Double
    var accepted: Bool

    var best: RankedLabel? { ranked.first }
}
