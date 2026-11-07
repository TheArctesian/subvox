import XCTest
@testable import Subvox

final class MotionWordDecoderTests: XCTestCase {
    func testExperimentalSettingsRoundTripWithoutLosingIndependentHMMThresholds() throws {
        var settings = DecoderSettings()
        settings.method = .jawSequence
        settings.vocabularyScope = .common1000
        settings.hmmMaxScore = 7.5
        settings.hmmMinGap = 0.2
        settings.maxDistance = 1.1
        settings.minConfidence = 0.4
        let copy = try JSONDecoder().decode(DecoderSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(copy, settings)
        XCTAssertEqual(copy.method, .jawSequence)
        XCTAssertEqual(copy.vocabularyScope, .common1000)
    }

    func testDefaultFacadePreservesTemplatePredictionsAndRejection() throws {
        let fixture = makeFixture()
        var settings = DecoderSettings()
        settings.features = .rotation
        settings.maxDistance = 0.75
        settings.minConfidence = 0.2
        XCTAssertEqual(settings.method, .motionTemplates)
        XCTAssertEqual(settings.vocabularyScope, .full)
        let baseline = try XCTUnwrap(PhoneticDecoder(takes: fixture.training, lexicon: fixture.lexicon,
            vocabulary: fixture.vocabulary, featureSet: settings.features))
        let facade = try XCTUnwrap(MotionWordDecoder(takes: fixture.training, lexicon: fixture.lexicon,
            vocabulary: fixture.vocabulary, settings: settings))
        let samples = take("inside", phones: ["AA", "IY"]).samples
        let expected = baseline.predict(samples, maxDistance: settings.maxDistance,
            minConfidence: settings.minConfidence, commonBias: settings.commonBias)
        let actual = facade.predict(samples)
        XCTAssertFalse(expected.ranked.isEmpty)
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(facade.wordCount, baseline.wordCount)
        XCTAssertEqual(facade.missingPhones, baseline.missingPhones)
    }

    func testVocabularyScopesPreserveFixedPrefixesIncludingTheirCounts() {
        // Deliberately non-monotonic counts: scope selection must use the supplied manifest,
        // not rerank it or add words from an evaluation transcript.
        let words = (0..<1002).map { (word: "word\($0)", count: ($0 * 17) % 101 + 1) }
        for (scope, count) in [(VocabularyScope.common100, 100), (.common1000, 1000), (.full, 1002)] {
            let selected = scope.select(words)
            XCTAssertEqual(selected.map(\.word), words.prefix(count).map(\.word))
            XCTAssertEqual(selected.map(\.count), words.prefix(count).map(\.count))
        }
        XCTAssertFalse(VocabularyScope.common100.select(words).contains { $0.word == "word100" })
        XCTAssertFalse(VocabularyScope.common1000.select(words).contains { $0.word == "word1000" })
        XCTAssertEqual(VocabularyScope.common100.select(Array(words.prefix(3))).count, 3)
        XCTAssertTrue(VocabularyScope.full.select([]).isEmpty)
    }

    func testManifestIncludesTrainingPronunciationsOutsideCandidateVocabulary() {
        let training = [take("learn", phones: ["AA"])]
        let vocabulary = [(word: "inside", count: 10)]
        let originalLexicon = Lexicon(entries: ["learn": ["AA"], "inside": ["AA"]])
        let changedLexicon = Lexicon(entries: ["learn": ["IY"], "inside": ["AA"]])
        for method in LearningMethod.allCases {
            var settings = DecoderSettings()
            settings.method = method
            settings.vocabularyScope = .common100
            let original = MotionWordDecoder.manifest(takes: training, lexicon: originalLexicon,
                vocabulary: vocabulary, settings: settings)
            let changed = MotionWordDecoder.manifest(takes: training, lexicon: changedLexicon,
                vocabulary: vocabulary, settings: settings)
            XCTAssertNotEqual(original, changed,
                "\(method): training pronunciation changes alter fitting even outside the candidate list")
        }
    }

    func testSameManifestCannotHideOrderDependentTemplateMedoidSelection() throws {
        let first = take("learn", phones: ["AA"])
        var second = first
        second.id = UUID()
        second.samples = first.samples.map { sample in
            var opposite = sample
            opposite.rx = -sample.rx
            opposite.ry = -sample.ry
            opposite.rz = -sample.rz
            return opposite
        }
        let lexicon = Lexicon(entries: ["learn": ["AA"], "inside": ["AA"]])
        let vocabulary = [(word: "inside", count: 10)]
        var settings = DecoderSettings()
        settings.features = .rotation
        settings.commonBias = 0
        let forward = try XCTUnwrap(MotionWordDecoder(takes: [first, second], lexicon: lexicon,
            vocabulary: vocabulary, settings: settings))
        let reverse = try XCTUnwrap(MotionWordDecoder(takes: [second, first], lexicon: lexicon,
            vocabulary: vocabulary, settings: settings))
        let firstPrediction = forward.predict(first.samples)
        let secondPrediction = reverse.predict(first.samples)
        XCTAssertFalse(firstPrediction.ranked.isEmpty)
        XCTAssertFalse(secondPrediction.ranked.isEmpty)
        // Either preserve fitting order in the hash or fit in a canonical order. Identical
        // manifests must not conceal a different medoid chosen from two repeated takes.
        XCTAssertTrue(forward.manifest != reverse.manifest || firstPrediction == secondPrediction,
            "Frozen identity must capture any order dependence that changes fitted predictions")
    }

    private func makeFixture() -> (training: [Template], lexicon: Lexicon, vocabulary: [(word: String, count: Int)]) {
        // Cover every vowel and a consonant, so a new unit map cannot make the positive
        // control unbuildable merely by grouping vowels differently.
        let phones = Lexicon.vowels.sorted() + ["B"]
        var entries = Dictionary(uniqueKeysWithValues: phones.map { ("train" + $0.lowercased(), [$0]) })
        let training = phones.map { take("train" + $0.lowercased(), phones: [$0]) }
        let words = ["inside"] + (1..<100).map { "distractor\($0)" } + ["outside"]
        for word in words { entries[word] = ["AA", "IY"] }
        let vocabulary = words.enumerated().map { (word: $0.element, count: 1000 - $0.offset) }
        return (training, Lexicon(entries: entries), vocabulary)
    }

    private func take(_ word: String, phones: [String]) -> Template {
        let allPhones = Lexicon.vowels.sorted() + ["B"]
        let framesPerPhone = 24
        let samples = (0..<(12 + phones.count * framesPerPhone)).map { i -> MotionSample in
            let k = i - 6
            var x = 0.0, y = 0.0, z = 0.0
            if k >= 0, k < phones.count * framesPerPhone {
                let phone = phones[k / framesPerPhone]
                let index = allPhones.firstIndex(of: phone) ?? 0
                let value = 0.2 * sin(Double.pi * Double(k % framesPerPhone) / Double(framesPerPhone - 1))
                x = value * (1 + Double(index % 3) * 0.1)
                y = value * Double(index % 5) * 0.15
                z = value * Double(index % 7) * 0.1
            }
            return MotionSample(t: Double(i) / 50, rotation: (x, y, z))
        }
        return Template(label: word, samples: samples)
    }
}
