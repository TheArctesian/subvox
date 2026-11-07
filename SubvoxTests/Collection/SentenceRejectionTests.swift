import XCTest
@testable import Subvox

final class SentenceRejectionTests: XCTestCase {
    func testCorrectRankedWordCanBeRejectedAndMustNotCountAsCorrectOutput() throws {
        let train = SentencePrompt(text: "a bee"), validation = SentencePrompt(text: "the bee")
        let lexicon = Lexicon(entries: ["bee": ["B", "IY"]])
        let samples = (0..<101).map { i in MotionSample(t: 100 + Double(i) / 50,
            accel: (0.02 * sin(Double(i) / 4), 0, 0), rotation: (0.1 * cos(Double(i) / 5), 0, 0)) }
        func pair(_ prompt: SentencePrompt, _ role: DataRole) -> SentencePair {
            var p = SentencePair(visitID: UUID(), promptIDs: [prompt.id], text: prompt.text, role: role)
            p.spoken = SentenceRecording(condition: .normal, samples: samples, cueStart: 100, cueEnd: 102)
            p.silent = SentenceRecording(condition: .silent, samples: samples, cueStart: 100, cueEnd: 102)
            p.review = .accepted
            p.alignment = MotionTransferResult(normalizedCost: 0, durationRatio: 1, path: [], words: [
                .init(word: "bee", voicedStart: 100, voicedEnd: 102, silentStart: 100, silentEnd: 102)], warnings: [])
            return p
        }
        let data = SentenceTrainingData(prompts: [train, validation], roles: [train.id: .training, validation.id: .validation],
                                        pairs: [pair(train, .training), pair(validation, .validation)])
        var settings = DecoderSettings(); settings.commonBias = 0; settings.minConfidence = 0.2
        // One candidate has no runner-up evidence; its correct rank must still be rejected.
        let rejected = try SentenceValidation.run(data: data, settings: settings, lexicon: lexicon, vocabulary: [("bee", 1)])
        XCTAssertEqual(rejected.top1, 1)
        XCTAssertEqual(rejected.top5, 1)
        XCTAssertEqual(rejected.accepted, [false])
        XCTAssertEqual(rejected.acceptance, 0)
        XCTAssertEqual(rejected.acceptedCorrect, 0)
        settings.minConfidence = 0
        let emitted = try SentenceValidation.run(data: data, settings: settings, lexicon: lexicon, vocabulary: [("bee", 1)])
        XCTAssertEqual(emitted.top1, rejected.top1)
        XCTAssertEqual(emitted.acceptance, 1)
        XCTAssertEqual(emitted.acceptedCorrect, 1)
        let restored = try JSONDecoder().decode(SentenceValidationReport.self, from: JSONEncoder().encode(rejected))
        XCTAssertEqual(restored.acceptance, 0)
        XCTAssertEqual(restored.acceptedCorrect, 0)
    }

    func testRejectionMetricsKeepIncorrectAndMissingCandidatesInDenominator() {
        let settings = DecoderSettings()
        let manifest = MotionWordDecoder.manifest(takes: [], lexicon: Lexicon(entries: [:]), vocabulary: [], settings: settings)
        var report = SentenceValidationReport(trainingPairIDs: [], validationPairIDs: [], settings: settings, manifest: manifest,
            expected: ["bee", "bob", "blue", "bird"], predicted: [["bee"], ["wrong"], ["blue"], []], accepted: [true, true, false, false])
        XCTAssertEqual(report.top1, 0.5)
        XCTAssertEqual(report.acceptance, 0.5)
        XCTAssertEqual(report.acceptedCorrect, 0.25) // Fraction of ALL evaluated slices, not only accepted slices.
        report.expected = []; report.predicted = []; report.accepted = []
        XCTAssertEqual(report.acceptance, 0)
        XCTAssertEqual(report.acceptedCorrect, 0)
    }
}
