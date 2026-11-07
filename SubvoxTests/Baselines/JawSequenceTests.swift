import XCTest
@testable import Subvox

final class JawSequenceTests: XCTestCase {
    private let lexicon = Lexicon(tsv: """
    train\tAA IY EH
    reverse\tEH IY AA
    held\tAA EH IY
    other\tIY EH AA
    alpha\tAA IY
    beta\tAA IY
    double\tAA AA
    single\tAA
    unknown\tZZ
    """)

    /// Constant-energy signatures avoid trimming away a synthetic unit. Unequal durations
    /// deliberately violate initialization; jitter is deterministic, never random noise.
    private func signal(_ parts: [(Int, Int)]) -> [MotionSample] {
        var samples: [MotionSample] = []
        for (unit, count) in parts {
            for _ in 0..<count {
                let index = samples.count
                let jitter = 0.004 * sin(Double(index) * 1.7)
                let rotation: (Double, Double, Double)
                switch unit {
                case 0: rotation = (1 + jitter, 0, 0) // open
                case 1: rotation = (-1 + jitter, 0, 0) // close
                default: rotation = (0, 1 + jitter, 0) // mid
                }
                samples.append(MotionSample(t: Double(index) / 50, rotation: rotation))
            }
        }
        return samples
    }

    private var training: [Template] {
        [Template(label: "train", samples: signal([(0, 8), (1, 32), (2, 8)])),
         Template(label: "reverse", samples: signal([(2, 8), (1, 32), (0, 8)]))]
    }
    private var vocabulary: [(word: String, count: Int)] {
        ["held", "other", "train", "reverse", "double", "single"].map { ($0, 1) }
    }
    private func predict(_ decoder: JawSequenceDecoder, _ samples: [MotionSample], limit: Int = 8) -> Prediction {
        decoder.predict(samples, maxScore: 100, minScoreGap: 0, commonBias: 0, limit: limit)
    }

    func testHeldOutCompositionalWordAndUnequalDurationRecovery() throws {
        let initial = try XCTUnwrap(JawSequenceDecoder(takes: training, lexicon: lexicon,
            vocabulary: vocabulary, featureSet: .rotation, iterations: 0))
        let learned = try XCTUnwrap(JawSequenceDecoder(takes: training, lexicon: lexicon,
            vocabulary: vocabulary, featureSet: .rotation))
        // Forty-eight frames per take, three vowel positions: initialization allocates 16
        // frames each. Realignment must recover 8/32/8, not keep its equal-spread labels.
        XCTAssertTrue(initial.trainingSummary.contains("close=32, mid=32, open=32"))
        XCTAssertTrue(learned.trainingSummary.contains("close=64, mid=16, open=16"))
        let query = signal([(0, 8), (2, 8), (1, 32)])
        XCTAssertFalse(training.contains { $0.label == "held" })
        let result = predict(learned, query)
        XCTAssertEqual(result.best?.label, "held")
        XCTAssertTrue(result.accepted)
        XCTAssertLessThan(try XCTUnwrap(result.best?.distance),
                          try XCTUnwrap(predict(initial, query).best?.distance))
        XCTAssertEqual(learned.wordCount, vocabulary.count)
        // Coarse coverage: learning IY supplies the tied close unit for UH as well.
        XCTAssertFalse(learned.missingPhones.contains("UH"))
        XCTAssertTrue(learned.missingPhones.contains("B"))
    }

    func testInitializationOnlyAndIterationBoundsAreDeterministic() throws {
        func build(_ iterations: Int) throws -> JawSequenceDecoder {
            try XCTUnwrap(JawSequenceDecoder(takes: training, lexicon: lexicon,
                vocabulary: vocabulary, featureSet: .rotation, iterations: iterations))
        }
        let query = signal([(0, 8), (2, 8), (1, 32)])
        XCTAssertEqual(predict(try build(0), query), predict(try build(0), query))
        XCTAssertEqual(predict(try build(-5), query), predict(try build(0), query))
        XCTAssertEqual(predict(try build(100), query), predict(try build(6), query))
    }

    func testInvalidTakesAreSkippedAndConstantDataRemainsFinite() throws {
        let constant = (0..<12).map { MotionSample(t: Double($0) / 50) }
        var invalid = constant
        invalid[3].rx = .nan
        var nonmonotonic = constant
        nonmonotonic[3].t = 0
        let takes = [Template(label: "single", samples: constant),
                     Template(label: "single", samples: invalid),
                     Template(label: "single", samples: nonmonotonic),
                     Template(label: "unknown", samples: constant)]
        let decoder = try XCTUnwrap(JawSequenceDecoder(takes: takes, lexicon: lexicon,
            vocabulary: [("single", 1)], featureSet: .rotation))
        XCTAssertTrue(decoder.trainingSummary.contains("3 skipped takes"))
        let result = predict(decoder, constant)
        XCTAssertTrue(try XCTUnwrap(result.best?.distance).isFinite)
        XCTAssertTrue(result.confidence.isFinite)
        XCTAssertTrue(predict(decoder, invalid).ranked.isEmpty)
        XCTAssertTrue(predict(decoder, nonmonotonic).ranked.isEmpty)
        XCTAssertTrue(predict(decoder, []).ranked.isEmpty)
        XCTAssertTrue(predict(decoder, constant, limit: 0).ranked.isEmpty)
        XCTAssertFalse(decoder.predict(constant, maxScore: .nan, minScoreGap: 0, commonBias: 0).accepted)
        XCTAssertNil(JawSequenceDecoder(takes: Array(takes.dropFirst()), lexicon: lexicon,
            vocabulary: [("single", 1)], featureSet: .rotation))
        XCTAssertNil(JawSequenceDecoder(takes: takes, lexicon: lexicon,
            vocabulary: [("single", 1)], featureSet: []))
    }

    func testRepeatedIdenticalPositionsRequireACompletePath() throws {
        let two = [MotionSample(t: 0), MotionSample(t: 0.02)]
        let decoder = try XCTUnwrap(JawSequenceDecoder(takes: [Template(label: "double", samples: two)],
            lexicon: lexicon, vocabulary: [("double", 1)], featureSet: .rotation))
        XCTAssertEqual(decoder.wordCount, 1)
        XCTAssertTrue(predict(decoder, [two[0]]).ranked.isEmpty)
        XCTAssertEqual(predict(decoder, two).best?.label, "double")
        XCTAssertTrue(try XCTUnwrap(predict(decoder, two).best?.distance).isFinite)
    }

    func testCollisionsHaveDeterministicWordTiesAndNoFalseScoreGap() throws {
        let decoder = try XCTUnwrap(JawSequenceDecoder(takes: training, lexicon: lexicon,
            vocabulary: [("beta", 1), ("unknown", 1), ("alpha", 1), ("alpha", 1)], featureSet: .rotation))
        XCTAssertEqual(decoder.wordCount, 2)
        XCTAssertEqual(decoder.collisionCount, 2)
        let query = signal([(0, 12), (1, 12)])
        let result = predict(decoder, query)
        XCTAssertEqual(result.ranked.map(\.label), ["alpha", "beta"])
        XCTAssertEqual(result.ranked[0].distance, result.ranked[1].distance)
        XCTAssertEqual(result.confidence, 0)
        XCTAssertFalse(decoder.predict(query, maxScore: 100, minScoreGap: 0.001, commonBias: 0).accepted)
        // Asking for only one displayed result must not invent a confident single candidate.
        XCTAssertFalse(decoder.predict(query, maxScore: 100, minScoreGap: 0.001,
                                       commonBias: 0, limit: 1).accepted)
    }
}
