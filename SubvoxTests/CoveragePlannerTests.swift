import XCTest
@testable import Subvox

final class CoveragePlannerTests: XCTestCase {
    private let lexicon = Lexicon(tsv: """
        bob\tB AA1 B
        bee\tB IY1
        see\tS IY1
        sob\tS AA1 B
        boss\tB AA1 S

        """)

    func testPlannerReachesEveryTargetWithFewWordsAndNoRepeats() {
        var generator = SplitMix64(seed: 1)
        let words = CoveragePlanner.plan(
            needs: ["B": 3, "AA": 2, "IY": 1, "S": 1], candidates: ["bob", "bee", "see", "sob", "boss"],
            lexicon: lexicon, avoiding: [], using: &generator)

        var counts: [String: Int] = [:]
        for word in words { for phone in lexicon.phones(word)! { counts[phone, default: 0] += 1 } }
        XCTAssertGreaterThanOrEqual(counts["B", default: 0], 3)
        XCTAssertGreaterThanOrEqual(counts["AA", default: 0], 2)
        XCTAssertGreaterThanOrEqual(counts["IY", default: 0], 1)
        XCTAssertGreaterThanOrEqual(counts["S", default: 0], 1)
        XCTAssertEqual(Set(words).count, words.count)
        XCTAssertLessThanOrEqual(words.count, 3)  // e.g. bob + boss + bee
    }

    func testPlannerSkipsWordsAlreadyRecordedAndStopsWhenNothingHelps() {
        var generator = SplitMix64(seed: 1)
        let words = CoveragePlanner.plan(
            needs: ["IY": 5], candidates: ["bee", "see", "bob"], lexicon: lexicon, avoiding: ["bee"], using: &generator)
        XCTAssertEqual(words, ["see"])
    }

    func testAccentVariableWordsAreLeftOut() {
        let unstable = [["K", "AA", "R"], ["AE", "F", "T", "ER"], ["D", "AE", "N", "S"], ["AO", "F"], ["L", "AO", "NG"]]
        let stable = [["R", "EH", "D"], ["V", "EH", "R", "IY"], ["K", "AE", "T"], ["TH", "AO", "T"], ["B", "ER", "D"]]
        XCTAssertTrue(unstable.allSatisfy { !CoveragePlanner.isAccentStable($0) })  // car, after, dance, off, long
        XCTAssertTrue(stable.allSatisfy(CoveragePlanner.isAccentStable))            // red, very, cat, thought, bird
    }

}
