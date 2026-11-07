import Foundation

/// Diagnostic takes are kept with their plots but never used by the learning experiments.
enum RecordingPurpose: String, Codable, Hashable, Sendable {
    case training = "Training"
    case signalMouthed = "Signal check — mouth the word"
    case voicedReference = "Signal check — speak aloud"
    case cueOnly = "Signal check — keep mouth still"
}

enum SignalCheckPlan {
    static let seedWords = ["joy", "go", "good", "huge", "next", "back", "who", "oil", "father", "see", "cook", "shoe"]
    struct Trial: Equatable { var word: String; var purpose: RecordingPurpose }

    /// Selects only previously assigned TRAINING words. No reserved word is ever recruited.
    static func words(in experiment: TrainingExperiment) -> [String] {
        let preferred = ["joy", "go", "good", "huge", "next", "back", "who", "oil"]
        let available = experiment.wordRoles.filter { $0.value == .training }.map(\.key).sorted()
        let known = preferred.filter { experiment.wordRoles[$0] == .training }
        return Array((known + available.filter { !known.contains($0) }).prefix(6))
    }

    /// Independently shuffled rounds of six words × three instructions. This prevents
    /// word blocks and puts mouth/still conditions in the same visit instead of confounding
    /// condition with session. Repeated words remain separate trial positions.
    static func trials<G: RandomNumberGenerator>(words: [String], rounds: Int = 3,
                                                using generator: inout G) -> [Trial] {
        guard rounds > 0 else { return [] }
        let words = Array(Set(words)).sorted()
        return (0..<rounds).flatMap { _ in
            words.flatMap { word in
                [Trial(word: word, purpose: .signalMouthed), Trial(word: word, purpose: .cueOnly),
                 Trial(word: word, purpose: .voicedReference)]
            }.shuffled(using: &generator)
        }
    }
}

/// Converts monotonic cue times into the recorded audio file's clock. Missing clock anchors
/// cannot be replaced by an assumed zero offset without mislabelling voiced references.
enum SignalCueTiming {
    static func audioRange(for take: Template, startUptime: TimeInterval,
                           audioOffset: TimeInterval?) -> ClosedRange<Double>? {
        guard let from = take.cueStartUptime, let to = take.cueEndUptime, let audioOffset,
              [from, to, startUptime, audioOffset].allSatisfy(\.isFinite), to > from else { return nil }
        return (from - startUptime - audioOffset)...(to - startUptime - audioOffset)
    }
}

/// Explicit acquisition pacing, shared by the timer and duration estimate.
enum SignalCheckPace: String, CaseIterable, Identifiable, Sendable {
    case comfortable = "Comfortable", slow = "Extra time"
    var id: String { rawValue }
    var baseline: Double { self == .comfortable ? 2 : 3 }
    var cue: Double { self == .comfortable ? 5 : 7 }
    var recovery: Double { 2 }
    var trialSeconds: Double { baseline + cue + recovery }
    func maximumRounds(wordCount: Int) -> Int {
        // Leave a minute of headroom below the local transcription input limit of 15 min.
        min(8, max(2, Int(840 / (Double(max(1, wordCount)) * 3 * trialSeconds))))
    }
}
