import XCTest
@testable import Subvox

final class SentenceMotionAlignmentTests: XCTestCase {
    private func motion(start: Double = 100, duration: Double = 4, hz: Double = 50,
                        warped: Bool = false) -> [MotionSample] {
        let count = Int((duration * hz).rounded()) + 1
        return (0..<count).map { i in
            let time = Double(i) / Double(count - 1)
            let phase = warped ? (time <= 0.5 ? 0.6 * time : 0.3 + 1.4 * (time - 0.5)) : time
            return MotionSample(t: start + time * duration,
                accel: (0.02 * sin(10 * .pi * phase), 0, 0),
                rotation: (0.1 * sin(6 * .pi * phase), 0.1 * cos(4 * .pi * phase), 0))
        }
    }

    private func checkFailure(_ expected: SentenceMotionAlignment.Failure,
                              voiced: [MotionSample], silent: [MotionSample],
                              words: [MotionWordAnchor] = [.init(word: "word", start: 100, end: 104)],
                              features: FeatureSet = [.acceleration, .rotation],
                              file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try SentenceMotionAlignment.align(voiced: voiced, silent: silent,
                                                               words: words, features: features), file: file, line: line) {
            XCTAssertEqual($0 as? SentenceMotionAlignment.Failure, expected, file: file, line: line)
        }
    }

    func testIdenticalMotionHasCompleteDiagonalPathAndAbsoluteWordTimes() throws {
        let a = motion(), b = motion(start: 500)
        let result = try SentenceMotionAlignment.align(voiced: a, silent: b,
            words: [.init(word: "zqx", start: 100.4, end: 103.6)])
        XCTAssertEqual(result.normalizedCost, 0, accuracy: 1e-10)
        XCTAssertEqual(result.durationRatio, 1)
        XCTAssertEqual(result.path.count, a.count)
        XCTAssertTrue(result.path.allSatisfy { $0.voicedIndex == $0.silentIndex })
        XCTAssertEqual(result.words[0].silentStart, 500.4, accuracy: 1e-8)
        XCTAssertEqual(result.words[0].silentEnd, 503.6, accuracy: 1e-8)
        XCTAssertEqual(result.words[0].word, "zqx") // No vocabulary filtering or text substitution.
        XCTAssertTrue(result.warnings.contains { $0.contains("not ground truth") })
        XCTAssertEqual(try JSONDecoder().decode(MotionTransferResult.self,
            from: JSONEncoder().encode(result)), result)
    }

    func testNonlinearWarpAcrossDifferentRatesRecoversKnownTimingAndRepeatedPositions() throws {
        let a = motion(), b = motion(start: 200, duration: 6, hz: 80, warped: true)
        let anchors: [MotionWordAnchor] = [
            .init(word: "go", start: 100, end: 101.2),
            .init(word: "go", start: 101.2, end: 102.8),
            .init(word: "again", start: 102.8, end: 104)]
        let result = try SentenceMotionAlignment.align(voiced: a, silent: b, words: anchors)
        XCTAssertEqual(result.durationRatio, 1.5, accuracy: 1e-10)
        XCTAssertEqual(result.words.map(\.word), ["go", "go", "again"])
        XCTAssertEqual(result.words[0].silentStart, 200)
        XCTAssertEqual(result.words[0].silentEnd, 203, accuracy: 0.2)
        XCTAssertEqual(result.words[1].silentEnd, 200 + 6 * (0.5 + 0.4 / 1.4), accuracy: 0.2)
        XCTAssertEqual(result.words[2].silentEnd, 206)
        XCTAssertEqual(result.words[0].silentEnd, result.words[1].silentStart)
        XCTAssertEqual(result.words[1].silentEnd, result.words[2].silentStart)
        XCTAssertEqual(Set(result.path.map(\.voicedIndex)), Set(a.indices))
        XCTAssertEqual(Set(result.path.map(\.silentIndex)), Set(b.indices))
        XCTAssertEqual(result.path.first, MotionAlignmentPoint(voicedIndex: 0, silentIndex: 0))
        XCTAssertEqual(result.path.last, MotionAlignmentPoint(voicedIndex: a.count - 1, silentIndex: b.count - 1))
        for (left, right) in zip(result.path, result.path.dropFirst()) {
            let dx = right.voicedIndex - left.voicedIndex, dy = right.silentIndex - left.silentIndex
            XCTAssertTrue((0...1).contains(dx) && (0...1).contains(dy) && dx + dy > 0)
        }
        XCTAssertTrue(result.normalizedCost.isFinite)
        // Anchor labels/times only annotate an independently computed path.
        let other = try SentenceMotionAlignment.align(voiced: a, silent: b,
            words: [.init(word: "different", start: 100.5, end: 103.5)])
        XCTAssertEqual(result.path, other.path)
        XCTAssertEqual(result.normalizedCost, other.normalizedCost)
    }

    func testFlatAndUnselectedMotionRejected() {
        let flat = (0..<201).map { MotionSample(t: 100 + Double($0) / 50) }
        checkFailure(.noMotion, voiced: flat, silent: motion())
        checkFailure(.noMotion, voiced: motion(), silent: flat)
        checkFailure(.noMotion, voiced: flat, silent: flat)
        checkFailure(.noMotion, voiced: motion(), silent: motion(), features: .attitude)
        checkFailure(.invalidFeatures, voiced: motion(), silent: motion(), features: [])
        checkFailure(.invalidFeatures, voiced: motion(), silent: motion(), features: .init(rawValue: 16))
    }

    func testBadSamplesGapsAndBoundsRejected() {
        var invalid = motion(); invalid[5].rx = .nan
        checkFailure(.invalidSamples, voiced: invalid, silent: motion())
        invalid = motion(); invalid[5].t = invalid[4].t
        checkFailure(.invalidSamples, voiced: motion(), silent: invalid)
        invalid = motion(); invalid[5].gz = .infinity
        checkFailure(.invalidSamples, voiced: invalid, silent: motion())
        invalid = motion(); invalid[5].rx = 1e300
        checkFailure(.invalidSamples, voiced: invalid, silent: motion())
        invalid = motion()
        for i in 20..<invalid.count { invalid[i].t += 0.3 }
        checkFailure(.timestampGap, voiced: invalid, silent: motion())
        checkFailure(.excessiveLength, voiced: [], silent: motion())
        checkFailure(.excessiveLength, voiced: Array(motion().prefix(5)), silent: motion())
        checkFailure(.excessiveLength, voiced: motion(duration: 48), silent: motion()) // 2,401 frames.
        checkFailure(.excessiveLength, voiced: motion(duration: 49, hz: 20), silent: motion())
        checkFailure(.timestampGap, voiced: motion(hz: 5), silent: motion())
        checkFailure(.excessiveWarp, voiced: motion(duration: 1), silent: motion(duration: 5))
    }

    func testInvalidAnchorsRejectedWithoutOmission() {
        let a = motion(), b = motion(start: 200)
        let cases: [[MotionWordAnchor]] = [[],
            [.init(word: "", start: 100, end: 101)],
            [.init(word: "bad", start: 99.95, end: 101)],
            [.init(word: "bad", start: 103, end: 104.05)],
            [.init(word: "bad", start: 102, end: 101)],
            [.init(word: "bad", start: .nan, end: 101)],
            [.init(word: "bad", start: 101, end: 101)],
            [.init(word: "one", start: 100, end: 102), .init(word: "two", start: 101, end: 103)],
            [.init(word: "one", start: 102, end: 103), .init(word: "two", start: 100, end: 101)]]
        for anchors in cases { checkFailure(.badAnchors, voiced: a, silent: b, words: anchors) }
    }

    func testBoundaryToleranceClipsAndWarnsAndLargeDurationWarpWarns() throws {
        let result = try SentenceMotionAlignment.align(voiced: motion(), silent: motion(start: 200, duration: 8),
            words: [.init(word: "all", start: 99.97, end: 104.03)])
        XCTAssertEqual(result.words[0].silentStart, 200)
        XCTAssertEqual(result.words[0].silentEnd, 208)
        XCTAssertTrue(result.warnings.contains { $0.contains("clipped") })
        XCTAssertTrue(result.warnings.contains { $0.contains("Large time warp") })
    }

    func testFortyFiveSecondPassagePreservesOriginalIndicesAndSparseAnchors() throws {
        let a = motion(duration: 45), b = motion(start: 200, duration: 45)
        XCTAssertEqual(a.count, 2251)
        // Unmatched transcript words simply have no supplied anchor; no words are invented.
        let result = try SentenceMotionAlignment.align(voiced: a, silent: b, words: [
            .init(word: "first", start: 99.98, end: 105),
            .init(word: "later", start: 130, end: 145.02)])
        XCTAssertEqual(result.path.count, a.count)
        XCTAssertEqual(result.path.last, MotionAlignmentPoint(voicedIndex: 2250, silentIndex: 2250))
        XCTAssertEqual(result.words.map(\.word), ["first", "later"])
        XCTAssertEqual(result.words[0].silentStart, 200)
        XCTAssertEqual(result.words[1].silentStart, 230, accuracy: 1e-8)
        XCTAssertEqual(result.words[1].silentEnd, 245)
        XCTAssertTrue(result.warnings.contains { $0.contains("not ground truth") })
    }

    func testMinimumClipAndUptimeTranslation() throws {
        // Same signal at different absolute clock offsets must transfer the same relative times.
        let a = motion(start: 1_000_000, duration: 0.1)
        let b = motion(start: 2_000_000, duration: 0.1)
        XCTAssertEqual(a.count, 6)
        let result = try SentenceMotionAlignment.align(voiced: a, silent: b,
            words: [.init(word: "short", start: 1_000_000.02, end: 1_000_000.08)])
        XCTAssertEqual(result.words[0].silentStart, 2_000_000.02, accuracy: 1e-7)
        XCTAssertEqual(result.words[0].silentEnd, 2_000_000.08, accuracy: 1e-7)
        XCTAssertTrue(result.normalizedCost.isFinite)
    }
}
