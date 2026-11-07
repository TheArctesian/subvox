import Foundation

/// Gestures for the mouth-observability check: can AirPods motion tell mouth shapes apart at all?
/// Everything is silent; the bat/pat pair differs only in voicing, which the sensor cannot see.
enum ObservabilityGesture: String, Codable, CaseIterable, Identifiable, Sendable {
    case rest, jawOpen, vowelAh, vowelEe, vowelOo, controlBat, controlPat
    var id: String { rawValue }
    var title: String {
        switch self {
        case .rest: "Rest"
        case .jawOpen: "Jaw open"
        case .vowelAh: "“ah”"
        case .vowelEe: "“ee”"
        case .vowelOo: "“oo”"
        case .controlBat: "“bat”"
        case .controlPat: "“pat”"
        }
    }
    var instruction: String {
        switch self {
        case .rest: "Keep your mouth relaxed and still"
        case .jawOpen: "Open your jaw wide, then close it"
        case .vowelAh: "Mouth “ah”, then relax"
        case .vowelEe: "Mouth “ee”, then relax"
        case .vowelOo: "Mouth “oo”, then relax"
        case .controlBat: "Mouth “bat”, then relax"
        case .controlPat: "Mouth “pat”, then relax"
        }
    }
    var symbol: String {
        switch self {
        case .rest: "pause.circle"
        case .jawOpen: "arrow.up.and.down"
        case .vowelAh, .vowelEe, .vowelOo: "mouth"
        case .controlBat, .controlPat: "equal.circle"
        }
    }
}

/// Fixed before any data is seen. Bonferroni across the three contrasts.
struct ObservabilityContrast: Codable, Identifiable, Sendable, Equatable {
    enum Kind: String, Codable, Sendable { case signal, control }
    var id: String
    var title: String
    var question: String
    var gestures: [ObservabilityGesture]
    var kind: Kind

    static let movement = Self(id: "movement", title: "Jaw movement vs rest",
        question: "Does any mouth movement reach the AirPods?", gestures: [.rest, .jawOpen], kind: .signal)
    static let vowels = Self(id: "vowels", title: "Vowel shape",
        question: "Can “ah”, “ee” and “oo” be told apart?", gestures: [.vowelAh, .vowelEe, .vowelOo], kind: .signal)
    static let control = Self(id: "control", title: "Control: bat vs pat",
        question: "These differ only in voicing and should NOT be separable.", gestures: [.controlBat, .controlPat], kind: .control)
    static let all = [movement, vowels, control]
    static let alpha = 0.05 / Double(all.count)
}

struct ObservabilityTake: Codable, Identifiable, Sendable {
    var id = UUID()
    var gesture: ObservabilityGesture
    var block: Int
    var samples: [MotionSample]
    var cueStart: Double
    var cueEnd: Double
    var discardedReason: String?
}

struct ObservabilityContrastResult: Codable, Identifiable, Sendable {
    var contrast: ObservabilityContrast
    var id: String { contrast.id }
    var takes: Int
    var blocks: Int
    var balancedAccuracy: Double
    var chance: Double
    var pValue: Double
    var permutations: Int
    /// confusion[truth][predicted], indexed like `contrast.gestures`.
    var confusion: [[Int]]
    var significant: Bool { pValue < ObservabilityContrast.alpha }
}

struct ObservabilityReport: Codable, Sendable {
    enum Verdict: String, Codable, Sendable { case insufficient, artifact, noSignal, jawOnly, articulation }
    var createdUnix = Date().timeIntervalSince1970
    var version = ObservabilityAnalysis.version
    var results: [ObservabilityContrastResult]
    var insufficient: [String]
    var verdict: Verdict
    var summary: String {
        switch verdict {
        case .insufficient: "Not enough valid takes per gesture to test. Record more blocks."
        case .artifact: "The control pair was separable, so something other than mouth shape (order, effort, timing) is leaking into the signal. Treat the other results as unreliable."
        case .noSignal: "Jaw movement could not be told apart from rest. With this placement the AirPods do not appear to carry a usable mouth signal."
        case .jawOnly: "Jaw movement is visible, but vowel shapes are not separable. Expect jaw-open/closed information at most, not phonemes."
        case .articulation: "Both jaw movement and vowel shape are separable above chance. There is an articulatory signal worth modelling."
        }
    }
}

struct ObservabilitySession: Codable, Identifiable, Sendable {
    struct Planned: Codable, Sendable, Equatable { var gesture: ObservabilityGesture; var block: Int }
    var id = UUID()
    var startedUnix = Date().timeIntervalSince1970
    var endedUnix: Double?
    var session: RecordingSession
    var seed: UInt64
    var planned: [Planned]
    var takes: [ObservabilityTake] = []
    var report: ObservabilityReport?

    /// Every block contains each gesture once in a shuffled order, so slow drift is never confounded with a gesture.
    static func plan(blocks: Int, seed: UInt64) -> [Planned] {
        var rng = SplitMix64(seed: seed)
        return (0..<blocks).flatMap { block in
            ObservabilityGesture.allCases.shuffled(using: &rng).map { Planned(gesture: $0, block: block) }
        }
    }
    var validTakes: [ObservabilityTake] {
        var latest: [Int: ObservabilityTake] = [:]
        for (i, item) in planned.enumerated() {
            if let take = takes.last(where: { $0.block == item.block && $0.gesture == item.gesture && $0.discardedReason == nil }) { latest[i] = take }
        }
        return latest.keys.sorted().compactMap { latest[$0] }
    }
}
