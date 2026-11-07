import XCTest
@testable import Subvox

final class WordFocusGuideTests: XCTestCase {
    func testFocusWaitsForLeadInAdvancesOnceAndHoldsLastWordUntilDone() {
        let guide = WordFocusGuide.comfortable
        XCTAssertNil(guide.wordIndex(elapsed: 0, count: 3))
        XCTAssertNil(guide.wordIndex(elapsed: 0.399, count: 3))
        XCTAssertEqual(guide.wordIndex(elapsed: 0.4, count: 3), 0)
        XCTAssertEqual(guide.wordIndex(elapsed: 1.05, count: 3), 1)
        XCTAssertEqual(guide.wordIndex(elapsed: 1.7, count: 3), 2)
        XCTAssertEqual(guide.wordIndex(elapsed: 30, count: 3), 2)
        XCTAssertEqual(guide.wordIndex(elapsed: .greatestFiniteMagnitude, count: 3), 2)
        XCTAssertNil(guide.wordIndex(elapsed: .nan, count: 3))
        XCTAssertNil(guide.wordIndex(elapsed: 1, count: 0))
    }

    func testDisplayedWordTokensRetainPunctuationAndMatchAcrossWhitespace() {
        XCTAssertEqual(WordFocusGuide.words(in: "Her favourite\n song\tplayed softly."), ["Her", "favourite", "song", "played", "softly."])
        XCTAssertEqual(WordFocusGuide.words(in: "Don't rush."), ["Don't", "rush."])
        XCTAssertNil(WordFocusGuide(leadIn: 0, wordSeconds: 0).wordIndex(elapsed: 1, count: 3))
    }
}
