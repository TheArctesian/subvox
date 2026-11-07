import XCTest
@testable import Subvox

final class LearningWorkflowTests: XCTestCase {
    // This is the on-disk settings shape from before learning methods were added.
    private let legacySettingsJSON = """
    {"condition":"Whisper","features":2,"maxDistance":1.25,
     "minConfidence":0.35,"commonBias":0.08}
    """

    func testLegacySettingsMigrationPreservesThresholdsAndDefaultDecoder() throws {
        let settings = try JSONDecoder().decode(DecoderSettings.self, from: Data(legacySettingsJSON.utf8))
        XCTAssertEqual(settings.method, .legacyMedoids)
        XCTAssertEqual(settings.vocabularyScope, .full)
        XCTAssertEqual(settings.condition, .whisper)
        XCTAssertEqual(settings.features, .rotation)
        XCTAssertEqual(settings.maxDistance, 1.25)
        XCTAssertEqual(settings.minConfidence, 0.35)
        XCTAssertEqual(settings.commonBias, 0.08)
        XCTAssertEqual(settings.hmmMaxScore, 20)
        XCTAssertEqual(settings.hmmMinGap, 0)
    }

    func testFrozenLegacyExperimentDoesNotSilentlySwitchMethodsOrReassignHoldout() throws {
        let json = """
        {"id":"00000000-0000-0000-0000-000000000001","createdAt":1,
         "wordRoles":{"learn":"Training","tune":"Validation","sealed":"Final test"},
         "historicalTakeCount":0,"settings":\(legacySettingsJSON),"frozenAt":2}
        """
        var experiment = try JSONDecoder().decode(TrainingExperiment.self, from: Data(json.utf8))
        let roles = experiment.wordRoles
        experiment.assign(["sealed", "newword"])
        XCTAssertEqual(experiment.wordRoles, roles)
        XCTAssertEqual(experiment.settings.method, .legacyMedoids)
        XCTAssertEqual(experiment.settings.vocabularyScope, .full)
        XCTAssertEqual(experiment.frozenAt, Date(timeIntervalSinceReferenceDate: 2))

        let copy = try JSONDecoder().decode(TrainingExperiment.self, from: JSONEncoder().encode(experiment))
        XCTAssertEqual(copy.settings, experiment.settings)
        XCTAssertEqual(copy.wordRoles, roles)
        XCTAssertEqual(copy.frozenAt, experiment.frozenAt)
    }

    func testExperimentalSettingsRoundTripWithoutLosingIndependentHMMThresholds() throws {
        var settings = DecoderSettings()
        settings.method = .jawHMM
        settings.vocabularyScope = .common1000
        settings.hmmMaxScore = 7.5
        settings.hmmMinGap = 0.2
        settings.maxDistance = 1.1
        settings.minConfidence = 0.4
        let copy = try JSONDecoder().decode(DecoderSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(copy, settings)
        XCTAssertEqual(copy.method, .jawHMM)
        XCTAssertEqual(copy.vocabularyScope, .common1000)
    }

    func testDefaultFacadePreservesLegacyPredictionsAndRejection() throws {
        let fixture = makeFixture()
        var settings = DecoderSettings()
        settings.features = .rotation
        settings.maxDistance = 0.75
        settings.minConfidence = 0.2
        XCTAssertEqual(settings.method, .legacyMedoids)
        XCTAssertEqual(settings.vocabularyScope, .full)
        let legacy = try XCTUnwrap(PhoneticDecoder(takes: fixture.training, lexicon: fixture.lexicon,
            vocabulary: fixture.vocabulary, featureSet: settings.features))
        let facade = try XCTUnwrap(MotionWordDecoder(takes: fixture.training, lexicon: fixture.lexicon,
            vocabulary: fixture.vocabulary, settings: settings))
        let samples = take("inside", phones: ["AA", "IY"]).samples
        let expected = legacy.predict(samples, maxDistance: settings.maxDistance,
            minConfidence: settings.minConfidence, commonBias: settings.commonBias)
        let actual = facade.predict(samples)
        XCTAssertFalse(expected.ranked.isEmpty)
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(facade.wordCount, legacy.wordCount)
        XCTAssertEqual(facade.missingPhones, legacy.missingPhones)
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

    func testSameManifestCannotHideOrderDependentLegacyMedoidSelection() throws {
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

    func testCollisionSummaryCountsParticipatingWordsConsistentlyAcrossMethods() throws {
        let fixture = makeFixture()
        // All candidate pronunciations are identical. Every word participates in a collision,
        // rather than all-but-one words or one bucket, matching the report's user-facing count.
        for method in LearningMethod.allCases {
            var settings = DecoderSettings()
            settings.method = method
            settings.features = .rotation
            let decoder = try XCTUnwrap(MotionWordDecoder(takes: fixture.training, lexicon: fixture.lexicon,
                vocabulary: fixture.vocabulary, settings: settings), "\(method)")
            XCTAssertEqual(decoder.wordCount, fixture.vocabulary.count)
            XCTAssertEqual(decoder.collisionCount, decoder.wordCount, "\(method)")
        }
    }

    func testCoverageAndFrequencyBaselineDistinguishListedButUnbuildableTargets() async throws {
        let fixture = makeFixture()
        let vocabulary = [(word: "inside", count: 10), (word: "unknown", count: 9)]
        let testing = ["inside", "unknown", "outside"].map { take($0, phones: ["AA", "IY"]) }
        var settings = DecoderSettings()
        settings.features = .rotation
        settings.commonBias = 0
        let decoder = try XCTUnwrap(MotionWordDecoder(takes: fixture.training, lexicon: fixture.lexicon,
            vocabulary: vocabulary, settings: settings))
        let report = try await DecoderEvaluation.run(training: fixture.training, testing: testing,
            role: .validation, lexicon: fixture.lexicon, vocabulary: vocabulary, settings: settings)
        XCTAssertEqual(report.rows.count, 3)
        XCTAssertEqual(report.rows.map(\.inVocabulary), [true, true, false])
        XCTAssertEqual(report.rows.map(\.representable), [true, false, false])
        XCTAssertEqual(report.candidateCount, 2)
        XCTAssertEqual(report.vocabularyCount, 1)
        XCTAssertEqual(try XCTUnwrap(report.frequencyTop1), 1.0 / 3, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(report.frequencyTop5), 2.0 / 3, accuracy: 1e-12)
        XCTAssertEqual(report.manifest, decoder.manifest)
        XCTAssertEqual(report.collisionCount, decoder.collisionCount)
        XCTAssertEqual(report.trainingSummary, decoder.trainingSummary)
        let restored = try JSONDecoder().decode(DecoderReport.self, from: JSONEncoder().encode(report))
        XCTAssertEqual(restored.manifest, report.manifest)
        XCTAssertEqual(restored.rows.map(\.inVocabulary), report.rows.map(\.inVocabulary))
    }

    func testHistoricalReportDoesNotInventVocabularyCoverageOrManifest() throws {
        let json = """
        {"createdAt":1,"role":"Validation","settings":\(legacySettingsJSON),
         "trainingIDs":["00000000-0000-0000-0000-000000000001"],
         "rows":[{"id":"00000000-0000-0000-0000-000000000002","expected":"inside",
                  "candidates":["inside"],"accepted":true,"representable":true}],
         "vocabularyCount":1,"missingPhones":[]}
        """
        let report = try JSONDecoder().decode(DecoderReport.self, from: Data(json.utf8))
        XCTAssertEqual(report.settings.method, .legacyMedoids)
        XCTAssertEqual(report.settings.vocabularyScope, .full)
        XCTAssertEqual(report.top1, 1)
        XCTAssertEqual(report.acceptedCorrect, 1)
        XCTAssertNil(report.rows[0].inVocabulary)
        XCTAssertNil(report.manifest)
        XCTAssertNil(report.frequencyTop1)
        XCTAssertNil(report.candidateCount)
    }

    func testOutOfScopeAndUnpronounceableWordsRemainMissesForEveryMethod() async throws {
        let fixture = makeFixture()
        let testing = [take("outside", phones: ["AA", "IY"]), take("unknown", phones: ["AA", "IY"])]
        for method in LearningMethod.allCases {
            var settings = DecoderSettings()
            settings.method = method
            settings.features = .rotation
            settings.commonBias = 0
            settings.vocabularyScope = .common100
            let decoder = try XCTUnwrap(MotionWordDecoder(takes: fixture.training, lexicon: fixture.lexicon,
                                                         vocabulary: fixture.vocabulary, settings: settings), "\(method)")
            let report = try await DecoderEvaluation.run(training: fixture.training, testing: testing,
                role: .validation, lexicon: fixture.lexicon, vocabulary: fixture.vocabulary, settings: settings)
            XCTAssertEqual(report.rows.map(\.id), testing.map(\.id), "\(method): evaluation must not drop impossible targets")
            XCTAssertEqual(report.rows.map(\.expected), ["outside", "unknown"])
            XCTAssertEqual(report.vocabularyCount, decoder.wordCount)
            XCTAssertEqual(report.settings, settings)
            XCTAssertEqual(report.top1, 0, "\(method)")
            XCTAssertEqual(report.top5, 0, "\(method)")
            XCTAssertEqual(report.acceptedCorrect, 0, "\(method)")
            for row in report.rows {
                XCTAssertFalse(row.representable, "\(method): scope and pronunciation coverage must both count")
                XCTAssertFalse(row.candidates.contains("outside"), "\(method): do not inject test truth")
                XCTAssertFalse(row.candidates.contains("unknown"))
            }
        }
    }

    func testEvaluationUsesLiveFacadeForEveryLearningMethod() async throws {
        let fixture = makeFixture()
        let testing = [take("inside", phones: ["AA", "IY"]), take("outside", phones: ["AA", "IY"])]
        for method in LearningMethod.allCases {
            var settings = DecoderSettings()
            settings.method = method
            settings.features = .rotation
            settings.commonBias = 0
            settings.vocabularyScope = .common100
            let decoder = try XCTUnwrap(MotionWordDecoder(takes: fixture.training, lexicon: fixture.lexicon,
                                                         vocabulary: fixture.vocabulary, settings: settings), "\(method)")
            let report = try await DecoderEvaluation.run(training: fixture.training, testing: testing,
                role: .validation, lexicon: fixture.lexicon, vocabulary: fixture.vocabulary, settings: settings)
            XCTAssertEqual(report.rows.count, testing.count)
            XCTAssertEqual(Set(report.trainingIDs), Set(fixture.training.map(\.id)))
            XCTAssertTrue(report.rows[0].representable, "\(method): positive coverage control")
            for (row, testTake) in zip(report.rows, testing) {
                let live = decoder.predict(testTake.samples)
                XCTAssertFalse(live.ranked.isEmpty, "\(method): parity must not pass vacuously")
                XCTAssertEqual(row.candidates, live.ranked.map(\.label), "\(method)")
                XCTAssertEqual(row.accepted, live.accepted, "\(method)")
            }
            // Use every test take in the denominator, even when the expected word is outside scope.
            XCTAssertEqual(report.top1, Double(report.rows.filter(\.top1).count) / Double(testing.count))
            XCTAssertEqual(report.top5, Double(report.rows.filter(\.top5).count) / Double(testing.count))
            XCTAssertEqual(report.acceptance, Double(report.rows.filter(\.accepted).count) / Double(testing.count))
            XCTAssertEqual(report.acceptedCorrect,
                Double(report.rows.filter { $0.accepted && $0.top1 }.count) / Double(testing.count))
        }
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
        return Template(label: word, samples: samples, condition: .silent, review: .accepted)
    }
}
