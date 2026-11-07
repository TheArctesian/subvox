import XCTest
@testable import Subvox

final class DecoderEvaluationTests: XCTestCase {
    private let lexicon = Lexicon(tsv: "ah\tAA1\nee\tIY1\nahee\tAA1 IY1\nbee\tB IY1\n")
    private let vocabulary = [(word: "ah", count: 1), (word: "ee", count: 1), (word: "ahee", count: 1), (word: "bee", count: 1)]

    private func take(_ word: String, axes: [Int]) -> Template {
        let samples = (0..<(12 + axes.count * 12)).map { i -> MotionSample in
            var rotation = (0.0, 0.0, 0.0)
            let k = i - 6
            if k >= 0, k < axes.count * 12 {
                let value = 0.2 * sin(Double.pi * Double(k % 12) / 11)
                if axes[k / 12] == 0 { rotation.0 = value } else { rotation.2 = value }
            }
            return MotionSample(t: Double(i) / 50, rotation: rotation)
        }
        return Template(label: word, samples: samples, review: .accepted)
    }

    func testEvaluationMatchesLiveDecoderAndCountsUnbuildableWordsAsMisses() async throws {
        let training = [take("ah", axes: [0]), take("ee", axes: [2])]
        let testing = [take("ahee", axes: [0, 2]), take("bee", axes: [0, 2])]
        let settings = DecoderSettings(features: .rotation, maxDistance: 0, minConfidence: 1, commonBias: 0)
        let decoder = try XCTUnwrap(PhoneticDecoder(takes: training, lexicon: lexicon, vocabulary: vocabulary, featureSet: settings.features))
        let report = try await DecoderEvaluation.run(training: training, testing: testing, role: .validation,
                                                      lexicon: lexicon, vocabulary: vocabulary, settings: settings)
        XCTAssertEqual(report.rows.count, 2)
        for (row, take) in zip(report.rows, testing) {
            let actual = decoder.predict(take.samples, maxDistance: settings.maxDistance,
                                          minConfidence: settings.minConfidence, commonBias: settings.commonBias)
            XCTAssertEqual(row.candidates, actual.ranked.map(\.label))
            XCTAssertEqual(row.accepted, actual.accepted)
        }
        XCTAssertFalse(report.rows[1].representable)
        XCTAssertFalse(report.rows[1].top1)
        XCTAssertEqual(report.acceptance, 0)
        XCTAssertEqual(report.acceptedCorrect, 0)
        XCTAssertEqual(report.settings, settings)
        let restored = try JSONDecoder().decode(DecoderReport.self, from: JSONEncoder().encode(report))
        XCTAssertEqual(restored.trainingIDs, training.map(\.id))
        XCTAssertEqual(restored.rows.map(\.expected), ["ahee", "bee"])
    }

    func testSameWordInTrainingAndEvaluationIsRejected() async {
        do {
            _ = try await DecoderEvaluation.run(training: [take("ah", axes: [0])], testing: [take("AH", axes: [0])],
                                                 role: .finalTest, lexicon: lexicon, vocabulary: vocabulary,
                                                 settings: DecoderSettings())
            XCTFail("Repeated word must not cross the split")
        } catch { XCTAssertTrue(error is DecoderEvaluation.Failure) }
    }
}
