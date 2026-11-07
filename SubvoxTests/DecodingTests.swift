import XCTest
@testable import Subvox

final class DecodingTests: XCTestCase {
    /// A synthetic "word": a burst of rotation at `frequency` cycles over the segment.
    private func take(_ frequency: Double, axis: Int, frames: Int = 30, noise: Double = 0.01) -> [MotionSample] {
        (0..<frames).map { i in
            let t = Double(i) / 25
            let envelope = sin(Double.pi * Double(i) / Double(frames - 1))
            let v = 0.2 * envelope * sin(2 * .pi * frequency * Double(i) / Double(frames))
            let jitter = { Double.random(in: -noise...noise) }
            var rotation = (jitter(), jitter(), jitter())
            switch axis {
            case 0: rotation.0 += v
            case 1: rotation.1 += v
            default: rotation.2 += v
            }
            return MotionSample(t: t, rotation: rotation)
        }
    }

    func testDTWIdenticalIsZeroAndWarpTolerant() {
        let a = MotionFeatures.frames(take(2, axis: 0, noise: 0), set: .rotation)
        let stretched = MotionFeatures.frames(take(2, axis: 0, frames: 40, noise: 0), set: .rotation)
        let other = MotionFeatures.frames(take(2, axis: 1, noise: 0), set: .rotation)
        XCTAssertEqual(DTW.distance(a, a), 0, accuracy: 1e-12)
        XCTAssertLessThan(DTW.distance(a, stretched), DTW.distance(a, other))
    }

    /// Quiet, then one rotation burst per axis in order, then quiet: a mouthed "word" whose
    /// sounds each have their own motion signature.
    private func word(_ axes: [Int], noise: Double = 0.005) -> [MotionSample] {
        let pad = 6, burst = 12
        let count = pad * 2 + burst * axes.count
        return (0..<count).map { i in
            let jitter = { Double.random(in: -noise...noise) }
            var rotation = (jitter(), jitter(), jitter())
            let k = i - pad
            if k >= 0, k < burst * axes.count {
                let v = 0.2 * sin(Double.pi * Double(k % burst) / Double(burst - 1))
                switch axes[k / burst] {
                case 0: rotation.0 += v
                case 1: rotation.1 += v
                default: rotation.2 += v
                }
            }
            return MotionSample(t: Double(i) / 25, rotation: rotation)
        }
    }

    func testDecoderSpellsWordsItWasNeverTrainedOn() throws {
        let lexicon = Lexicon(tsv: "ah\tAA1\nee\tIY1\noo\tUW1\nahee\tAA1 IY1\neeah\tIY1 AA1\nahoo\tAA1 UW1\n")
        var takes: [Template] = []
        for _ in 0..<4 {
            takes.append(Template(label: "ah", samples: word([0])))
            takes.append(Template(label: "ee", samples: word([2])))
            takes.append(Template(label: "oo", samples: word([1])))
        }
        let vocabulary = ["ah", "ee", "oo", "ahee", "eeah", "ahoo"].map { (word: $0, count: 1) }
        let decoder = try XCTUnwrap(PhoneticDecoder(takes: takes, lexicon: lexicon, vocabulary: vocabulary, featureSet: .rotation))
        XCTAssertEqual(decoder.wordCount, 6)
        XCTAssertTrue(decoder.missingPhones.contains("B"))

        // "ahee" was never recorded: it has to be assembled from the "ah" and "ee" sounds, in order.
        let prediction = decoder.predict(word([0, 2]), maxDistance: 10, minConfidence: 0, commonBias: 0)
        XCTAssertEqual(prediction.best?.label, "ahee")
    }

    func testTrimDropsQuietPadding() {
        let quiet = (0..<20).map { MotionSample(t: Double($0) / 25) }
        let burst = take(2, axis: 0, noise: 0).map { s -> MotionSample in
            var s = s; s.t += 0.8; return s
        }
        let trimmed = MotionFeatures.trim(quiet + burst + quiet.map { var s = $0; s.t += 2; return s })
        XCTAssertLessThan(trimmed.count, 40)
        XCTAssertGreaterThanOrEqual(trimmed.count, 20)
    }
}
