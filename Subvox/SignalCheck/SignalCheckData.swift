import Foundation

struct SignalCheckPrompt: Codable, Hashable, Sendable {
    var sentence: Int
    var repetition: Int
    var condition: SpeechCondition
}

struct SignalCheckTake: Codable, Identifiable, Sendable {
    enum EndReason: String, Codable { case done, timeout, scheduled }
    var id = UUID()
    var prompt: SignalCheckPrompt
    var recording: SentenceRecording
    var endReason: EndReason
    var discarded = false
    var transcript: LocalWhisperTranscript?
    var audioError: String?
    var pacing: SignalPacing?
}

struct SignalCheckRun: Codable, Sendable {
    var id = UUID()
    var startedAt = Date()
    var endedAt: Date?
    var seed: UInt64
    var audioDevice: String?
    var session: RecordingSession?
    var takes: [SignalCheckTake] = []
    var report: SignalCheckReport?
    var reference: SignalReferenceLink?
    var reseatedAt: Date?
    var control: ControlledBlockCapture?
    var recordsAudio: Bool?

    // Independent diagnostic prompts: these never enter the sentence corpus or its splits.
    static let sentences = ["Pack my blue bag.", "See those green trees.", "Move the red cup."]
    static func plan(seed: UInt64) -> [SignalCheckPrompt] {
        var rng = SplitMix64(seed: seed)
        return (0..<3).flatMap { repetition in
            Array(sentences.indices).shuffled(using: &rng).flatMap { sentence in
                [SpeechCondition.normal, .silent].map { .init(sentence: sentence, repetition: repetition, condition: $0) }
            }
        }
    }
}

struct SignalTakeMetrics: Codable, Sendable {
    var takeID: UUID
    var usable: Bool
    var duration: Double
    var sampleRate: Double?
    var maximumGap: Double
    var rotationRMS: Double
    var baselineRMS: Double?
    var topOnePercentEnergyShare: Double
    var issues: [String]
}

struct SignalConditionResult: Codable, Sendable {
    var condition: SpeechCondition
    var correct: Int
    var evaluated: Int
    var shiftedCorrect: Int
    var shiftedEvaluated: Int
    var validTakes: Int
    var sameSentenceCost: Double?
    var otherSentenceCost: Double?
    var plannedTakes: Int?
    var collectedTakes: Int?
    var rejectedTakes: Int?
}

struct SignalCheckReport: Codable, Sendable {
    var metrics: [SignalTakeMetrics]
    var conditions: [SignalConditionResult]
    var warnings: [String]
    var reference: SignalReferenceLink?
    var predictions: [SignalSessionPrediction]?
}

enum SignalCheckAnalysis {
    static func metrics(_ take: SignalCheckTake) -> SignalTakeMetrics {
        let recording = take.recording, samples = recording.cueSamples
        let quality = recording.quality
        let duration = recording.cueEnd - recording.cueStart
        let internalGap = zip(samples, samples.dropFirst()).map { $1.t - $0.t }.max() ?? .infinity
        let startGap = samples.first.map { $0.t - recording.cueStart } ?? .infinity
        let endGap = samples.last.map { recording.cueEnd - $0.t } ?? .infinity
        let maxGap = max(internalGap, startGap, endGap)
        func energy(_ sample: MotionSample) -> Double { sample.rx * sample.rx + sample.ry * sample.ry + sample.rz * sample.rz }
        let energies = samples.map(energy)
        let total = energies.reduce(0, +)
        let finite = duration.isFinite && energies.allSatisfy(\.isFinite) && total.isFinite
        let rms = finite && !energies.isEmpty ? sqrt(total / Double(energies.count)) : 0
        // Avoid the start-key press and the last half-second before the visual START cue.
        let baseline = recording.samples.filter { $0.t < recording.cueStart - 0.5 && $0.t >= recording.cueStart - 2.5 }
        let baselineEnergy = baseline.map(energy)
        let baselineGaps = zip(baseline, baseline.dropFirst()).map { $1.t - $0.t }
        let hasBaseline = baseline.count >= 12 && (baseline.last!.t - baseline.first!.t) >= 1.5 &&
            baselineGaps.allSatisfy { $0 > 0 && $0 <= 0.12 } && baselineEnergy.allSatisfy(\.isFinite)
        let baselineRMS = hasBaseline
            ? sqrt(baselineEnergy.reduce(0, +) / Double(baseline.count)) : nil
        let share = finite && total > 0 ? energies.sorted(by: >).prefix(max(1, Int(ceil(Double(energies.count) * 0.01)))).reduce(0, +) / total : 0
        var issues = quality.issues
        if !hasBaseline { issues.append("Still baseline is missing or has sensor gaps; excluded from comparisons") }
        if startGap > 0.12 || endGap > 0.12 { issues.append("Motion does not cover the START or END marker; excluded from comparisons") }
        if take.endReason == .timeout { issues.append("No manual END marker; excluded from comparisons") }
        if share > 0.35 { issues.append("The strongest 1% of frames carry over 35% of rotation energy") }
        if let baselineRMS, baselineRMS >= rms { issues.append("Rotation is no stronger than the still baseline") }
        if take.discarded { issues.append("Discarded by Redo") }
        let usable = !take.discarded && (take.endReason == .done || take.endReason == .scheduled) && !quality.invalid && finite && hasBaseline &&
            samples.count >= 12 && duration >= 0.35 && duration <= 12.2 && maxGap <= max(0.12, 3 / max(1, quality.sampleRate ?? 1)) &&
            (quality.sampleRate ?? 0) >= 10
        return .init(takeID: take.id, usable: usable, duration: duration.isFinite ? duration : 0,
                     sampleRate: quality.sampleRate, maximumGap: maxGap.isFinite ? maxGap : 0,
                     rotationRMS: rms, baselineRMS: baselineRMS, topOnePercentEnergyShare: share, issues: issues)
    }

    static func run(_ run: SignalCheckRun) throws -> SignalCheckReport {
        let metrics = run.takes.map(metrics)
        let usable = Set(metrics.filter(\.usable).map(\.takeID))
        var results: [SignalConditionResult] = []
        for condition in [SpeechCondition.normal, .silent] {
            let takes = run.takes.filter { usable.contains($0.id) && $0.prompt.condition == condition }
            var correct = 0, evaluated = 0, shiftedCorrect = 0, shiftedEvaluated = 0
            var same: [Double] = [], other: [Double] = []
            for test in takes {
                try Task.checkCancellation()
                // Hold out the entire repetition block, including all three phrases.
                let fitting = takes.filter { $0.prompt.repetition != test.prompt.repetition }
                guard Set(fitting.map { $0.prompt.sentence }).count == SignalCheckRun.sentences.count else { continue }
                let clip = test.recording.cueSamples
                for shifted in [false, true] {
                    let input = shifted ? shiftedSamples(clip) : clip
                    let distances = fitting.compactMap { reference -> (Int, Double)? in
                        guard let cost = cost(input, reference.recording.cueSamples) else { return nil }
                        if !shifted {
                            if reference.prompt.sentence == test.prompt.sentence { same.append(cost) }
                            else { other.append(cost) }
                        }
                        return (reference.prompt.sentence, cost)
                    }
                    guard Set(distances.map { $0.0 }).count == SignalCheckRun.sentences.count,
                          let best = distances.min(by: { $0.1 < $1.1 }) else { continue }
                    // A tie across classes carries no evidence and is counted as incorrect.
                    let tied = distances.contains { $0.0 != best.0 && abs($0.1 - best.1) < 1e-9 }
                    if shifted { shiftedEvaluated += 1; if !tied && best.0 == test.prompt.sentence { shiftedCorrect += 1 } }
                    else { evaluated += 1; if !tied && best.0 == test.prompt.sentence { correct += 1 } }
                }
            }
            func mean(_ values: [Double]) -> Double? { values.isEmpty ? nil : values.reduce(0, +) / Double(values.count) }
            results.append(.init(condition: condition, correct: correct, evaluated: evaluated,
                shiftedCorrect: shiftedCorrect, shiftedEvaluated: shiftedEvaluated, validTakes: takes.count,
                sameSentenceCost: mean(same), otherSentenceCost: mean(other)))
        }
        return .init(metrics: metrics, conditions: results, warnings: [
            "Exploratory three-phrase similarity, with each repetition held out in turn. Chance is 1 in 3; nine takes per condition give a very uncertain estimate.",
            "The half-clip circular shift is a timing control, not a significance test. High repeatability can also reflect head movement or pacing.",
            "This check does not train the decoder, transfer word labels, or use the reserved final-test sentences."
        ])
    }

    static func shiftedSamples(_ samples: [MotionSample]) -> [MotionSample] {
        guard !samples.isEmpty else { return [] }
        return samples.indices.map { i in
            var sample = samples[(i + samples.count / 2) % samples.count]
            sample.t = samples[i].t
            return sample
        }
    }

    private static func cost(_ a: [MotionSample], _ b: [MotionSample]) -> Double? {
        guard let start = a.first?.t, let end = a.last?.t else { return nil }
        return try? SentenceMotionAlignment.align(voiced: a, silent: b,
            words: [.init(word: "whole sentence", start: start, end: end)]).normalizedCost
    }
}
