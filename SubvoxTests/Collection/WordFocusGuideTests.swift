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

    func testHanCharactersHaveIndividualBeatsAndMatchingPinyin() {
        let text = "「你好，世界！」"
        XCTAssertEqual(WordFocusGuide.fragments(in: text).joined(), text)
        XCTAssertEqual(WordFocusGuide.words(in: text), ["「你", "好，", "世", "界！」"])
        XCTAssertEqual(WordFocusGuide.pinyinByWord(in: text, pinyin: "nǐ hǎo shì jiè"), ["nǐ", "hǎo", "shì", "jiè"])
        XCTAssertEqual(WordFocusGuide.pinyinByWord(in: text, pinyin: "nǐ hǎo"), [nil, nil, nil, nil])
        XCTAssertEqual(WordFocusGuide.pinyinByWord(in: "你在哪兒？", pinyin: "nǐ zài nǎr"), ["nǐ", "zài", "nǎr", nil])
    }

    func testEmDashSeparatesAdjacentWordsWithoutChangingTheSource() {
        let text = "Forbid it, Almighty God!—I know"
        XCTAssertEqual(WordFocusGuide.fragments(in: text).joined(), text)
        XCTAssertEqual(WordFocusGuide.words(in: text), ["Forbid", "it,", "Almighty", "God!—", "I", "know"])
        XCTAssertEqual(WordFocusGuide.words(in: "A well-known word"), ["A", "well-known", "word"])
    }

    func testDisplayedWordTokensRetainPunctuationAndMatchAcrossWhitespace() {
        XCTAssertEqual(WordFocusGuide.words(in: "Her favourite\n song\tplayed softly."), ["Her", "favourite", "song", "played", "softly."])
        XCTAssertEqual(WordFocusGuide.words(in: "Don't rush."), ["Don't", "rush."])
        XCTAssertNil(WordFocusGuide(leadIn: 0, wordSeconds: 0).wordIndex(elapsed: 1, count: 3))
    }
}

final class WordFocusGuideDurationTests: XCTestCase {
    func testHighlightingEndsWhenTheLastWordHasHadItsTime() {
        let guide = WordFocusGuide.comfortable
        XCTAssertEqual(guide.duration(count: 7), 0.4 + 7 * 0.65, accuracy: 1e-9)
        XCTAssertEqual(guide.wordIndex(elapsed: guide.duration(count: 7) - 0.01, count: 7), 6)
        XCTAssertEqual(guide.duration(count: 0), 0.4)
    }
}

final class WordFocusObservationTests: XCTestCase {
    /// The recording screen redraws only for observed changes. The highlight and the cue timer are
    /// computed from the clock, so they must observe the 50 ms countdown or they freeze on screen.
    @MainActor func testHighlightAndTimerObserveTheTickingCountdown() {
        let workflow = SentenceTrainingWorkflow(storeURL: FileManager.default.temporaryDirectory.appending(path: "focus-\(UUID()).json"))
        for read in [{ _ = workflow.activeWordIndex }, { _ = workflow.guideSecondsRemaining }] {
            let changed = expectation(description: "redraw")
            withObservationTracking(read) { changed.fulfill() }
            workflow.tickForTesting()
            wait(for: [changed], timeout: 1)
        }
    }
}
