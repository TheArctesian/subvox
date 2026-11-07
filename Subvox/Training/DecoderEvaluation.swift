import Foundation

struct DecoderReport: Codable, Sendable {
    struct Row: Codable, Identifiable, Sendable {
        var id: UUID
        var sessionID: UUID?
        var expected: String
        var candidates: [String]
        var accepted: Bool
        var representable: Bool
        var inVocabulary: Bool?
        var top1: Bool { candidates.first == expected }
        var top5: Bool { candidates.prefix(5).contains(expected) }
    }
    var createdAt = Date()
    var role: DataRole
    var settings: DecoderSettings
    var trainingIDs: [UUID]
    var rows: [Row]
    var vocabularyCount: Int
    var missingPhones: [String]
    var manifest: DecoderManifest?
    var candidateCount: Int?
    var collisionCount: Int?
    var trainingSummary: String?
    var frequencyTop1: Double?
    var frequencyTop5: Double?
    var top1: Double { fraction(rows.filter(\.top1).count) }
    var top5: Double { fraction(rows.filter(\.top5).count) }
    var acceptance: Double { fraction(rows.filter(\.accepted).count) }
    var acceptedCorrect: Double { fraction(rows.filter { $0.accepted && $0.top1 }.count) }
    private func fraction(_ n: Int) -> Double { rows.isEmpty ? 0 : Double(n) / Double(rows.count) }
}

enum DecoderEvaluation {
    enum Failure: LocalizedError {
        case emptyTraining, emptyTest, overlap, cannotBuild, cancelled
        var errorDescription: String? {
            switch self {
            case .emptyTraining: "No accepted training takes. Review recordings or collect more data."
            case .emptyTest: "No accepted recordings in this evaluation split."
            case .overlap: "Training and evaluation word groups overlap. Evaluation stopped."
            case .cannotBuild: "The decoder could not be built from these training takes."
            case .cancelled: "Evaluation cancelled."
            }
        }
    }

    /// Uses the same builder, vocabulary, ranking and rejection settings as the Decode screen.
    static func run(training: [Template], testing: [Template], role: DataRole,
                    lexicon: Lexicon, vocabulary: [(word: String, count: Int)], settings: DecoderSettings,
                    progress: @Sendable (Int, Int) async -> Void = { _, _ in }) async throws -> DecoderReport {
        guard !training.isEmpty else { throw Failure.emptyTraining }
        guard !testing.isEmpty else { throw Failure.emptyTest }
        let learned = Set(training.map { TrainingExperiment.key($0.label) })
        guard testing.allSatisfy({ !learned.contains(TrainingExperiment.key($0.label)) }) else { throw Failure.overlap }
        guard let decoder = MotionWordDecoder(takes: training, lexicon: lexicon, vocabulary: vocabulary,
                                             settings: settings) else { throw Failure.cannotBuild }
        var rows: [DecoderReport.Row] = []
        for take in testing {
            try Task.checkCancellation()
            let prediction = decoder.predict(take.samples)
            let expected = TrainingExperiment.key(take.label)
            let phones = lexicon.phones(expected)
            let representable = phones.map { Set($0).isDisjoint(with: decoder.missingPhones) } ?? false
            rows.append(.init(id: take.id, sessionID: take.sessionID, expected: expected,
                              candidates: prediction.ranked.map(\.label), accepted: prediction.accepted,
                              representable: representable && decoder.candidateWords.contains(expected),
                              inVocabulary: decoder.candidateWords.contains(expected)))
            await progress(rows.count, testing.count)
        }
        let selected = settings.vocabularyScope.select(vocabulary)
        let frequencyWords = selected.prefix(5).map(\.word)
        let n = Double(rows.count)
        return DecoderReport(role: role, settings: settings, trainingIDs: training.map(\.id), rows: rows,
                             vocabularyCount: decoder.wordCount, missingPhones: decoder.missingPhones,
                             manifest: decoder.manifest, candidateCount: decoder.candidateWords.count,
                             collisionCount: decoder.collisionCount, trainingSummary: decoder.trainingSummary,
                             frequencyTop1: Double(rows.filter { $0.expected == frequencyWords.first }.count) / n,
                             frequencyTop5: Double(rows.filter { frequencyWords.contains($0.expected) }.count) / n)
    }
}
