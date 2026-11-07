import XCTest
@testable import Subvox

final class PhoneEvaluationTests: XCTestCase {
    /// One-phone "words" so each take is a single sound with its own motion signature.
    private let lexicon = Lexicon(tsv: "ah\tAA1\nee\tIY1\n")

    private func take(_ word: String, axis: Int, noise: Double) -> Template {
        let samples = (0..<30).map { i -> MotionSample in
            let t = Double(i) / 25
            let burst = (8..<22).contains(i) ? 0.2 * sin(Double.pi * Double(i - 8) / 14) : 0
            let jitter = { Double.random(in: -noise...noise) }
            var rotation = (jitter(), jitter(), jitter())
            if axis == 0 { rotation.0 += burst } else { rotation.2 += burst }
            return MotionSample(t: t, rotation: rotation)
        }
        return Template(label: word, samples: samples)
    }

    func testDistinctSoundsAreRecognisedOnHeldOutTakes() throws {
        let takes = (0..<6).flatMap { _ in [take("ah", axis: 0, noise: 0.01), take("ee", axis: 2, noise: 0.01)] }
        let result = try XCTUnwrap(PhoneEvaluation.run(takes: takes, lexicon: lexicon, featureSet: .rotation))

        XCTAssertEqual(result.results["AA"]?.tested, 6)
        XCTAssertEqual(result.results["AA"]?.recall, 1)
        XCTAssertEqual(result.results["IY"]?.recall, 1)
        XCTAssertEqual(result.jawAccuracy, 1)  // open vs close
        XCTAssertEqual(result.chance, 0.5)
    }

    func testIdenticalMotionIsNotReportedAsRecognised() throws {
        let takes = (0..<8).flatMap { _ in [take("ah", axis: 0, noise: 0.05), take("ee", axis: 0, noise: 0.05)] }
        let result = try XCTUnwrap(PhoneEvaluation.run(takes: takes, lexicon: lexicon, featureSet: .rotation))
        XCTAssertLessThan(result.accuracy, 0.8)
    }
}
