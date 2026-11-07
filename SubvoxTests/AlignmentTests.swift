import XCTest
@testable import Subvox

final class AlignmentTests: XCTestCase {
    func testRepeatedWordsKeepDistinctOrderedAnchors() {
        let matches = Alignment.commonSubsequence(["go", "go", "again"], ["go", "uh", "go", "again"])
        XCTAssertEqual(matches.map { $0.0 }, [0, 1, 2])
        XCTAssertEqual(matches.map { $0.1 }, [0, 2, 3])
    }
    func testMissingWordsAreNotInvented() {
        let matches = Alignment.commonSubsequence(["the", "blue", "boat"], ["the", "boat"])
        XCTAssertEqual(matches.map { $0.0 }, [0, 2])
        XCTAssertTrue(Alignment.commonSubsequence(["one"], []).isEmpty)
        XCTAssertTrue(Alignment.commonSubsequence([], ["one"]).isEmpty)
    }
    func testPhoneSpanWeightsPreserveTimeAndOrder() {
        let spans = Alignment.spread(weights: [1, 2, 1], over: 10...14)
        XCTAssertEqual(spans, [10...11, 11...13, 13...14])
    }
}
