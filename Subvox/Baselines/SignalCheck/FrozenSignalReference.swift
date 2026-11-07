import CryptoKit
import Foundation

struct SignalReferenceLink: Codable, Equatable, Sendable {
    let id: UUID
    let sourceRunID: UUID
    let sha256: String
}

struct SignalSessionPrediction: Codable, Sendable {
    let takeID: UUID
    let expectedSentence: Int
    let predictedSentence: Int?
    let referenceTakeID: UUID?
    let cost: Double?
    let reason: String?
}

/// Immutable, source-only nearest-neighbor model. New takes are inputs to predict(), never
/// templates or fitting data. Per-clip mean subtraction is fixed preprocessing; all channel
/// scales, reference samples, features and DTW settings are frozen before acquisition.
struct FrozenSignalReference: Codable, Sendable {
    struct Template: Codable, Sendable {
        let takeID: UUID
        let sentence: Int
        let repetition: Int
        let condition: SpeechCondition
        let samples: [MotionSample]
    }
    struct Group: Codable, Sendable {
        let condition: SpeechCondition
        let scales: [Double]
    }
    struct FileRecord: Codable {
        let reference: FrozenSignalReference
        let sha256: String
    }
    struct Match {
        let sentence: Int?
        let takeID: UUID?
        let cost: Double?
        let reason: String?
    }

    let id: UUID
    let sourceRunID: UUID
    let sourceSessionID: UUID
    let sourceStartedAt: Date
    let frozenAt: Date
    let sentences: [String]
    let algorithmVersion: Int
    let features: FeatureSet
    let timeBand: Double
    let nonDiagonalPenalty: Double
    let groups: [Group]
    let templates: [Template]

    enum Failure: LocalizedError {
        case incomplete, invalidReference, incompatible, fingerprint, contaminated, missingReference
        var errorDescription: String? {
            switch self {
            case .incomplete: "Finish a signal check with three usable repetitions of every phrase, spoken and mouthed, before freezing it."
            case .invalidReference: "The reference contains invalid or uninformative motion. Keep the original recording and collect a complete usable reference check."
            case .incompatible: "This frozen reference uses different matcher settings. Its results cannot be evaluated with this matcher version."
            case .fingerprint: "The frozen reference fingerprint does not match. Restore the original reference file; it was not replaced."
            case .contaminated: "The second session must have a new recording identity, new takes, and a reseating confirmation. Reference recordings cannot be scored as a new session."
            case .missingReference: "The fixed reference for this test is unavailable. No within-session score was substituted."
            }
        }
    }

    static func freeze(_ run: SignalCheckRun) throws -> Self {
        guard run.reference == nil, run.endedAt != nil, let session = run.session else { throw Failure.incomplete }
        let usable = run.takes.filter { SignalCheckAnalysis.metrics($0).usable }
        guard usable.count == 18 else { throw Failure.incomplete }
        for condition in [SpeechCondition.normal, .silent] {
            for sentence in 0..<3 {
                let matching = usable.filter { $0.prompt.condition == condition && $0.prompt.sentence == sentence }
                guard matching.count == 3, Set(matching.map { $0.prompt.repetition }) == Set(0..<3),
                      matching.allSatisfy({ $0.recording.condition == condition }) else { throw Failure.incomplete }
            }
        }
        let templates = usable.map { Template(takeID: $0.id, sentence: $0.prompt.sentence,
            repetition: $0.prompt.repetition, condition: $0.prompt.condition, samples: $0.recording.cueSamples) }
        let features: FeatureSet = [.acceleration, .rotation]
        let floors = [1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]
        let groups = [SpeechCondition.normal, .silent].map { condition in
            let frames = templates.filter { $0.condition == condition }.map { MotionFeatures.frames($0.samples, set: features) }
            let scales = zip(MotionFeatures.channelScales(frames), floors).map { max($0, $1) }
            return Group(condition: condition, scales: scales)
        }
        let reference = Self(id: UUID(), sourceRunID: run.id, sourceSessionID: session.id, sourceStartedAt: run.startedAt,
            frozenAt: Date(), sentences: SignalCheckRun.sentences, algorithmVersion: SentenceMotionAlignment.algorithmVersion,
            features: features, timeBand: SentenceMotionAlignment.timeBand, nonDiagonalPenalty: SentenceMotionAlignment.nonDiagonalPenalty,
            groups: groups, templates: templates)
        try reference.validate()
        return reference
    }

    func validate() throws {
        guard algorithmVersion == SentenceMotionAlignment.algorithmVersion,
              features == [.acceleration, .rotation], sentences == SignalCheckRun.sentences,
              timeBand == SentenceMotionAlignment.timeBand, nonDiagonalPenalty == SentenceMotionAlignment.nonDiagonalPenalty else {
            throw Failure.incompatible
        }
        guard templates.count == 18, Set(templates.map(\.takeID)).count == 18, groups.count == 2,
              Set(groups.map(\.condition)) == Set([SpeechCondition.normal, .silent]) else { throw Failure.invalidReference }
        for condition in [SpeechCondition.normal, .silent] {
            guard let group = groups.first(where: { $0.condition == condition }), group.scales.count == 6 else { throw Failure.invalidReference }
            for sentence in 0..<3 {
                let matching = templates.filter { $0.condition == condition && $0.sentence == sentence }
                guard matching.count == 3, Set(matching.map(\.repetition)) == Set(0..<3) else { throw Failure.invalidReference }
                for template in matching {
                    guard cost(template.samples, template.samples, scales: group.scales) != nil else { throw Failure.invalidReference }
                }
            }
        }
    }

    func link() throws -> SignalReferenceLink {
        .init(id: id, sourceRunID: sourceRunID,
              sha256: SHA256.hash(data: try canonicalBytes()).map { String(format: "%02x", $0) }.joined())
    }

    /// Refuse overwrites: freezing another run cannot quietly replace this session's model.
    func saveNew(to url: URL) throws {
        try validate()
        let record = FileRecord(reference: self, sha256: try link().sha256)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(record).write(to: url, options: .withoutOverwriting)
    }

    static func load(from url: URL) throws -> Self {
        let record = try JSONDecoder().decode(FileRecord.self, from: Data(contentsOf: url))
        guard try record.reference.link().sha256 == record.sha256 else { throw Failure.fingerprint }
        try record.reference.validate()
        return record.reference
    }

    func predict(_ samples: [MotionSample], condition: SpeechCondition) -> Match {
        guard let group = groups.first(where: { $0.condition == condition }) else {
            return .init(sentence: nil, takeID: nil, cost: nil, reason: "No frozen reference for this condition")
        }
        let distances = templates.filter { $0.condition == condition }.compactMap { template -> (Template, Double)? in
            guard let value = cost(samples, template.samples, scales: group.scales) else { return nil }
            return (template, value)
        }
        guard Set(distances.map { $0.0.sentence }) == Set(0..<3),
              let best = distances.min(by: { $0.1 < $1.1 }) else {
            return .init(sentence: nil, takeID: nil, cost: nil, reason: "Motion could not be compared with all three frozen phrases")
        }
        guard !distances.contains(where: { $0.0.sentence != best.0.sentence && abs($0.1 - best.1) < 1e-9 }) else {
            return .init(sentence: nil, takeID: nil, cost: best.1, reason: "Ambiguous match between frozen phrases")
        }
        return .init(sentence: best.0.sentence, takeID: best.0.takeID, cost: best.1, reason: nil)
    }

    func evaluate(_ run: SignalCheckRun) throws -> SignalCheckReport {
        try validate()
        let link = try link()
        guard run.reference == link else { throw Failure.missingReference }
        let confirmed: Bool
        if let control = run.control {
            confirmed = control.pacing == .controlled && run.startedAt >= frozenAt &&
                control.blockConfirmedAt >= frozenAt && run.startedAt >= control.blockConfirmedAt &&
                (control.block == .unchanged ? run.reseatedAt == nil :
                    (run.reseatedAt.map { $0 >= frozenAt && run.startedAt >= $0 } ?? false))
        } else {
            confirmed = run.reseatedAt.map { $0 >= frozenAt && run.startedAt >= $0 } ?? false
        }
        guard run.id != sourceRunID, let session = run.session, session.id != sourceSessionID, confirmed,
              Set(run.takes.map(\.id)).isDisjoint(with: Set(templates.map(\.takeID))) else { throw Failure.contaminated }
        let metrics = run.takes.map(SignalCheckAnalysis.metrics)
        let usable = Set(metrics.filter(\.usable).map(\.takeID))
        var results: [SignalConditionResult] = [], predictions: [SignalSessionPrediction] = []
        for condition in [SpeechCondition.normal, .silent] {
            let collected = run.takes.filter { !$0.discarded && $0.prompt.condition == condition }
            let takes = collected.filter { usable.contains($0.id) && $0.recording.condition == condition }
            // Duplicate completed prompts would otherwise make the nine-take denominator misleading.
            let keys = collected.map { "\($0.prompt.sentence):\($0.prompt.repetition)" }
            guard Set(keys).count == keys.count, keys.count <= 9,
                  collected.allSatisfy({ (0..<3).contains($0.prompt.sentence) && (0..<3).contains($0.prompt.repetition) }) else { throw Failure.contaminated }
            var correct = 0, shiftedCorrect = 0, rejected = 0
            for take in collected {
                try Task.checkCancellation()
                guard usable.contains(take.id), take.recording.condition == condition else {
                    predictions.append(.init(takeID: take.id, expectedSentence: take.prompt.sentence, predictedSentence: nil,
                        referenceTakeID: nil, cost: nil, reason: "Invalid recording or missing manual END; excluded from scoring"))
                    continue
                }
                let match = predict(take.recording.cueSamples, condition: condition)
                if match.sentence == take.prompt.sentence { correct += 1 }
                if match.sentence == nil { rejected += 1 }
                let shifted = predict(SignalCheckAnalysis.shiftedSamples(take.recording.cueSamples), condition: condition)
                if shifted.sentence == take.prompt.sentence { shiftedCorrect += 1 }
                predictions.append(.init(takeID: take.id, expectedSentence: take.prompt.sentence, predictedSentence: match.sentence,
                    referenceTakeID: match.takeID, cost: match.cost, reason: match.reason))
            }
            results.append(.init(condition: condition, correct: correct, evaluated: takes.count,
                shiftedCorrect: shiftedCorrect, shiftedEvaluated: takes.count, validTakes: takes.count,
                sameSentenceCost: nil, otherSentenceCost: nil, plannedTakes: 9, collectedTakes: collected.count, rejectedTakes: rejected))
        }
        return .init(metrics: metrics, conditions: results, warnings: [
            "Test against frozen reference recordings and source-only channel scales. New takes never update this model.",
            "Scores identify three known phrases. Chance is 1 in 3; nine takes per condition remain a small sample. Head movement and pacing can contribute.",
            "Predictions are shown after acquisition ends. Missing and invalid takes are reported separately; unmatched valid takes remain in the scoring denominator.",
            "This check does not use or unlock the reserved final-test sentences."
        ], reference: link, predictions: predictions)
    }

    private func cost(_ a: [MotionSample], _ b: [MotionSample], scales: [Double]) -> Double? {
        guard let start = a.first?.t, let end = a.last?.t else { return nil }
        return try? SentenceMotionAlignment.align(voiced: a, silent: b,
            words: [.init(word: "whole phrase", start: start, end: end)], features: features, referenceScales: scales).normalizedCost
    }
    private func canonicalBytes() throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(self)
    }
}
