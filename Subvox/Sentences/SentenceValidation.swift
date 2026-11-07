import Foundation

/// Assisted word-boundary diagnostic, not end-to-end sentence transcription accuracy.
struct SentenceValidationReport: Codable, Sendable {
    var createdAt = Date()
    var trainingPairIDs: [UUID]
    var validationPairIDs: [UUID]
    var settings: DecoderSettings
    var manifest: DecoderManifest
    var expected: [String]
    var predicted: [[String]]
    var accepted: [Bool]
    var expectedValidationWordCount: Int?
    var top1: Double { expected.isEmpty ? 0 : Double(expected.indices.filter { predicted[$0].first == expected[$0] }.count) / Double(expected.count) }
    var top5: Double { expected.isEmpty ? 0 : Double(expected.indices.filter { predicted[$0].prefix(5).contains(expected[$0]) }.count) / Double(expected.count) }
    var acceptance: Double { expected.isEmpty ? 0 : Double(accepted.filter { $0 }.count) / Double(expected.count) }
    /// Fraction of all evaluated slices that Decode would emit correctly, including rejects.
    var acceptedCorrect: Double {
        expected.isEmpty ? 0 : Double(expected.indices.filter { accepted[$0] && predicted[$0].first == expected[$0] }.count) / Double(expected.count)
    }
}

enum SentenceValidation {
    static func run(data: SentenceTrainingData, settings: DecoderSettings, lexicon: Lexicon,
                    vocabulary: [(word: String, count: Int)]) throws -> SentenceValidationReport {
        let training = data.derivedWords(role: .training, lexicon: lexicon)
        let testing = data.derivedWords(role: .validation, lexicon: lexicon)
        guard !training.isEmpty, !testing.isEmpty else { throw Failure.notEnoughData }
        let trainGroups = Set(training.compactMap(\.sourceSentenceID).flatMap { $0.split(separator: ",").map(String.init) })
        guard testing.allSatisfy({ take in
            let groups = (take.sourceSentenceID ?? "").split(separator: ",").map(String.init)
            return !groups.isEmpty && trainGroups.isDisjoint(with: groups)
        }) else { throw Failure.overlap }
        guard let decoder = MotionWordDecoder(takes: training, lexicon: lexicon, vocabulary: vocabulary, settings: settings) else { throw Failure.notEnoughData }
        var guesses: [[String]] = []; var accepted: [Bool] = []
        for take in testing {
            try Task.checkCancellation()
            let p = decoder.predict(take.samples)
            guesses.append(p.ranked.map(\.label)); accepted.append(p.accepted)
        }
        return SentenceValidationReport(trainingPairIDs: Array(Set(training.compactMap(\.sourcePairID))),
            validationPairIDs: Array(Set(testing.compactMap(\.sourcePairID))), settings: settings,
            manifest: decoder.manifest, expected: testing.map(\.label), predicted: guesses, accepted: accepted,
            expectedValidationWordCount: data.pairs.filter {
                $0.role == .validation && $0.usesAlignedWords
                    && !$0.promptIDs.isEmpty && $0.promptIDs.allSatisfy { data.roles[$0] == .validation }
            }.reduce(0) { $0 + Lexicon.words(in: $1.text).count })
    }
    enum Failure: LocalizedError {
        case notEnoughData, overlap
        var errorDescription: String? {
            self == .overlap ? "Sentence groups overlap between fitting and validation."
                : "Collect usable aligned training and validation pairs before checking the word model."
        }
    }
}
