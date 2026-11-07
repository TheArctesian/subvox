import XCTest
@testable import Subvox

final class JawUnitsTests: XCTestCase {
    private func samples() -> [MotionSample] {
        (0..<40).map { i in
            MotionSample(t: Double(i) / 50,
                         rotation: (0.1 * sin(Double(i) * 0.2), 0, 0))
        }
    }

    func testAll39KnownPhonesMapToExactlySixUnits() {
        XCTAssertEqual(JawUnits.version, "jaw6-v1")
        XCTAssertEqual(Phoneme.all.count, 39)
        let units = Phoneme.all.compactMap { JawUnits.unit(for: $0.symbol) }
        XCTAssertEqual(units.count, 39)
        XCTAssertEqual(Set(units), Set(["close", "mid", "open", "open>close", "mid>close", "consonant"]))
        for phone in Phoneme.all {
            XCTAssertEqual(JawUnits.unit(for: phone.symbol), Lexicon.jawClass(phone.symbol) ?? "consonant")
        }
        XCTAssertEqual(JawUnits.unit(for: "AA"), JawUnits.unit(for: "AE"))
        XCTAssertEqual(JawUnits.unit(for: "P"), JawUnits.unit(for: "NG"))
    }

    func testUnknownPhonesNeverBecomeConsonants() {
        for phone in ["", "QX", "SIL", "aa", "AA1"] {
            XCTAssertNil(JawUnits.unit(for: phone))
        }
        let lexicon = Lexicon(entries: ["unknown": ["QX"]])
        XCTAssertNil(PhoneticDecoder(takes: [Template(label: "unknown", samples: samples())],
                                    lexicon: lexicon, vocabulary: [("unknown", 1)],
                                    featureSet: .rotation, coarseUnits: true))
    }

    func testUnknownCannotUseAnExistingConsonantModel() throws {
        let lexicon = Lexicon(entries: ["known": ["P"], "unknown": ["QX"], "mixed": ["P", "QX"]])
        let takes = ["known", "unknown", "mixed"].map { Template(label: $0, samples: samples()) }
        let words = ["known", "unknown", "mixed"].map { (word: $0, count: 1) }
        let decoder = try XCTUnwrap(PhoneticDecoder(takes: takes, lexicon: lexicon,
                                                   vocabulary: words, featureSet: .rotation, coarseUnits: true))
        XCTAssertEqual(decoder.wordCount, 1)
        XCTAssertEqual(decoder.collisionCount, 0)
        XCTAssertEqual(decoder.predict(samples(), maxDistance: 10, minConfidence: 0,
                                       commonBias: 0).ranked.map(\.label), ["known"])
    }

    func testZeroVarianceChannelsProduceFiniteScoresAndZeroLimitIsSafe() throws {
        let still = (0..<40).map { MotionSample(t: Double($0) / 50) }
        let lexicon = Lexicon(entries: ["one": ["AA"], "two": ["AE"]])
        let decoder = try XCTUnwrap(PhoneticDecoder(takes: [Template(label: "one", samples: still)],
                                                   lexicon: lexicon, vocabulary: [("one", 2), ("two", 1)],
                                                   featureSet: .all, coarseUnits: true))
        let prediction = decoder.predict(still, maxDistance: 10, minConfidence: 0, commonBias: 0.05)
        XCTAssertEqual(prediction.ranked.count, 2)
        XCTAssertTrue(prediction.ranked.allSatisfy { $0.distance.isFinite && $0.distance >= 0 })
        XCTAssertTrue(prediction.confidence.isFinite)
        XCTAssertTrue((0...1).contains(prediction.confidence))
        let empty = decoder.predict(still, maxDistance: 10, minConfidence: 0, commonBias: 0, limit: 0)
        XCTAssertTrue(empty.ranked.isEmpty)
        XCTAssertFalse(empty.accepted)
    }

    func testUnrecordedVowelUsesRecordedJawClassOnlyInCoarseMode() throws {
        let lexicon = Lexicon(entries: ["father": ["AA"], "cat": ["AE"], "see": ["IY"], "unknown": ["QX"]])
        let takes = [Template(label: "father", samples: samples())]
        let words = ["father", "cat", "see", "unknown"].map { (word: $0, count: 1) }
        let legacy = try XCTUnwrap(PhoneticDecoder(takes: takes, lexicon: lexicon,
                                                  vocabulary: words, featureSet: .rotation))
        let coarse = try XCTUnwrap(PhoneticDecoder(takes: takes, lexicon: lexicon,
                                                  vocabulary: words, featureSet: .rotation, coarseUnits: true))
        XCTAssertEqual(legacy.wordCount, 1)
        XCTAssertEqual(coarse.wordCount, 2)
        XCTAssertTrue(legacy.missingPhones.contains("AE"))
        for phone in ["AA", "AE", "AO"] { XCTAssertFalse(coarse.missingPhones.contains(phone)) }
        XCTAssertTrue(coarse.missingPhones.contains("IY"))
        XCTAssertTrue(coarse.missingPhones.contains("P"))
        let prediction = coarse.predict(samples(), maxDistance: 10, minConfidence: 0, commonBias: 0)
        XCTAssertEqual(Set(prediction.ranked.map(\.label)), Set(["father", "cat"]))
        XCTAssertEqual(coarse.collisionCount, 2)
        XCTAssertEqual(legacy.collisionCount, 0)
    }

    func testCollisionsCountWordsInSharedBuildableSequencesAndPreservePositions() throws {
        let lexicon = Lexicon(entries: ["openone": ["AA"], "opentwo": ["AE"],
                                        "stop": ["P"], "nasal": ["M"],
                                        "repeat": ["AA", "AA"], "unknown": ["QX"]])
        let takes = ["openone", "stop"].map { Template(label: $0, samples: samples()) }
        let words = ["openone", "opentwo", "stop", "nasal", "repeat", "unknown"].map { (word: $0, count: 1) }
        let decoder = try XCTUnwrap(PhoneticDecoder(takes: takes, lexicon: lexicon,
                                                   vocabulary: words, featureSet: .rotation, coarseUnits: true))
        XCTAssertEqual(decoder.wordCount, 5)
        // Two shared buckets of two words each; the repeated open unit is a different sequence.
        XCTAssertEqual(decoder.collisionCount, 4)
        XCTAssertFalse(decoder.missingPhones.contains("NG"))
    }

    func testDefaultMatchesExplicitLegacyIncludingTieOrderAndHomophones() throws {
        let lexicon = Lexicon(entries: ["first": ["AA"], "alias": ["AA"], "different": ["AE"]])
        let takes = [Template(label: "first", samples: samples())]
        let words = ["first", "alias", "different"].map { (word: $0, count: 1) }
        let implicit = try XCTUnwrap(PhoneticDecoder(takes: takes, lexicon: lexicon,
                                                    vocabulary: words, featureSet: .rotation))
        let explicit = try XCTUnwrap(PhoneticDecoder(takes: takes, lexicon: lexicon,
                                                    vocabulary: words, featureSet: .rotation, coarseUnits: false))
        let prediction = implicit.predict(samples(), maxDistance: 10, minConfidence: 0, commonBias: 0)
        XCTAssertEqual(prediction, explicit.predict(samples(), maxDistance: 10, minConfidence: 0, commonBias: 0))
        XCTAssertEqual(prediction.ranked.map(\.label), ["first", "alias"])
        XCTAssertEqual(implicit.missingPhones, explicit.missingPhones)
        XCTAssertEqual(implicit.wordCount, 2)
        XCTAssertEqual(implicit.collisionCount, 2)
    }
}
