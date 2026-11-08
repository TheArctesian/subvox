import Foundation

/// A fixed command vocabulary for closed-set recognition. Phrases vary in length and in jaw
/// pattern (open "ah"/"o" against closed "ee"/"m"), which is what the AirPods can sense.
enum CommandSet {
    static let version = "commands-v1"
    static let phrases = [
        "yes", "no", "stop", "pause", "play music", "next song", "go back", "volume up", "volume down", "call home",
        "send it", "never mind", "take a note", "set an alarm", "open the door", "what time is it", "read my messages",
        "turn on the lights", "remind me later", "how is the weather",
    ]

    /// Optional spoken enrolment (one take each), then `blocks` blocks where every phrase is
    /// mouthed once in a seeded order, so slow drift is never confounded with a phrase.
    static func plan(blocks: Int, enrolSpoken: Bool, seed: UInt64) -> [NeuralVisit.Planned] {
        var rng = SplitMix64(seed: seed)
        let spoken = enrolSpoken ? phrases.map { NeuralVisit.Planned(text: $0, condition: .normal, repetition: 0) } : []
        let mouthed = (0..<blocks).flatMap { block in
            phrases.shuffled(using: &rng).map { NeuralVisit.Planned(text: $0, condition: .silent, repetition: block + 1) }
        }
        return spoken + mouthed
    }
}

/// One modality's row from `subvox-ml closed-set`'s command protocols.
struct CommandScore: Identifiable, Sendable {
    var protocolName: String
    var modality: String
    var queries: Int
    var top1: Double
    var chance: Double
    var duration: Double
    var pValue: Double
    var id: String { protocolName + modality }
}
