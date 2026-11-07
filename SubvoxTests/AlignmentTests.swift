import XCTest
@testable import Subvox

final class AlignmentTests: XCTestCase {
    private let lexicon = Lexicon(tsv: "the\tDH AH0\nboy\tB OY1\nsat\tS AE1 T\ndown\tD AW1 N\n")

    func testHeardWordsKeepSpeechTimingAndGapsAreSpreadByPhoneCount() {
        let heard = [
            RecognizedWord(text: "the", start: 1.0, end: 1.2),
            RecognizedWord(text: "uh", start: 1.2, end: 1.3),  // filler the script doesn't contain
            RecognizedWord(text: "down", start: 2.0, end: 2.5),
        ]
        let timed = Alignment.timeWords(expected: ["the", "boy", "sat", "down"], recognized: heard,
                                        window: 0.5...3, lexicon: lexicon)

        XCTAssertEqual(timed.map(\.source), [.speech, .estimated, .estimated, .speech])
        XCTAssertEqual(timed[0].start, 1.0)
        // "boy" (2 phones) and "sat" (3 phones) share the 1.2–2.0 gap in proportion.
        XCTAssertEqual(timed[1].start, 1.2, accuracy: 1e-9)
        XCTAssertEqual(timed[1].end, 1.2 + 0.8 * 2 / 5, accuracy: 1e-9)
        XCTAssertEqual(timed[2].end, 2.0, accuracy: 1e-9)
        XCTAssertEqual(timed[3].end, 2.5)
    }

    func testSilentLineIsSpreadOverTheWindow() {
        let timed = Alignment.timeWords(expected: ["the", "boy"], recognized: [], window: 0...4, lexicon: lexicon)
        XCTAssertEqual(timed.map(\.end), [2, 4])
        XCTAssertTrue(timed.allSatisfy { $0.source == .estimated })
    }

    func testLabelsSplitWordsIntoPhonesAndTagVowelJawClass() {
        let labels = Alignment.label([TimedWord(text: "boy", start: 0, end: 3, source: .speech)], lexicon: lexicon)
        let phones = labels.filter { $0.tier == .phone }
        XCTAssertEqual(phones.map(\.label), ["B", "OY"])
        XCTAssertEqual(labels.first { $0.tier == .word }?.source, .speech)
        XCTAssertTrue(labels.filter { $0.tier == .phone || $0.tier == .jaw }.allSatisfy { $0.source == .estimated })
        XCTAssertEqual(phones[0].end, 1, accuracy: 1e-9)  // vowels weigh double
        XCTAssertEqual(labels.filter { $0.tier == .jaw }.map(\.label), ["mid>close"])
    }

    func testCSVQuotesLabelsWithCommas() {
        let row = LabelInterval(tier: .line, start: 0, end: 1, label: "Yes, \"no\"", source: .prompt).csvRow
        XCTAssertEqual(row, "line,0.0000,1.0000,\"Yes, \"\"no\"\"\",prompt")
    }

    func testSentenceSplitKeepsQuotesAndAbbreviatedDialogueTogether() {
        let lines = ReadingScript.sentences("\"Look,\" he whispered, \"a perch.\" The fleet had stopped; only oars. Joy smiled!")
        XCTAssertEqual(lines, ["\"Look,\" he whispered, \"a perch.\"", "The fleet had stopped; only oars.", "Joy smiled!"])
    }
}
