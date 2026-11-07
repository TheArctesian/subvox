import XCTest
@testable import Subvox

final class SentenceTrainingTests: XCTestCase {
    private let lexicon = Lexicon(entries: ["bee": ["B", "IY"], "bob": ["B", "AA", "B"]])

    private func pair(_ prompt: SentencePrompt, role: DataRole, visitID: UUID = UUID()) -> SentencePair {
        let samples = (0..<101).map { i in
            MotionSample(t: 200 + Double(i) / 50,
                         accel: (0.02 * sin(Double(i) / 4), 0, 0),
                         rotation: (0.1 * cos(Double(i) / 4), 0, 0))
        }
        var pair = SentencePair(visitID: visitID, promptIDs: [prompt.id], text: prompt.text, role: role)
        pair.spoken = SentenceRecording(condition: .normal, samples: samples, cueStart: 200, cueEnd: 202)
        pair.silent = SentenceRecording(condition: .silent, samples: samples, cueStart: 200, cueEnd: 202)
        pair.alignment = MotionTransferResult(normalizedCost: 0.1, durationRatio: 1, path: [], words: [
            .init(word: "bee", voicedStart: 200, voicedEnd: 201, silentStart: 200, silentEnd: 201),
            .init(word: "bob", voicedStart: 201, voicedEnd: 202, silentStart: 201, silentEnd: 202)], warnings: [])
        pair.review = .accepted
        return pair
    }

    func testNormalizedIdentityAndPersistentQuarterSentenceSplit() throws {
        XCTAssertEqual(SentencePrompt(text: "Bee, BOB!").id, SentencePrompt(text: "bee bob").id)
        var data = SentenceTrainingData(), rng = SplitMix64(seed: 7)
        let prompts = (0..<32).map { SentencePrompt(text: "sentence " + String(repeating: "bee ", count: $0 + 1)) }
        data.add(prompts, using: &rng)
        XCTAssertEqual(data.roles.values.filter { $0 == .finalTest }.count, 8)
        XCTAssertEqual(data.roles.values.filter { $0 == .validation }.count, 6)
        XCTAssertEqual(data.roles.values.filter { $0 == .training }.count, 18)
        let roles = data.roles
        data.add(prompts.reversed(), using: &rng)
        XCTAssertEqual(data.prompts.count, 32)
        XCTAssertEqual(data.roles, roles)
        let restored = try JSONDecoder().decode(SentenceTrainingData.self, from: JSONEncoder().encode(data))
        XCTAssertEqual(restored.roles, roles)
        data.add([SentencePrompt(text: "bob", kind: .wordRepair)], using: &rng)
        XCTAssertEqual(data.roles[SentencePrompt.identity("bob")], .training)
        XCTAssertEqual(data.roles.values.filter { $0 == .finalTest }.count, 8)
        for (id, role) in roles { XCTAssertEqual(data.roles[id], role) }
    }

    func testPassagesNeverCrossRolesOrPromptReservedText() {
        var data = SentenceTrainingData(), rng = SplitMix64(seed: 6)
        data.add((0..<32).map { SentencePrompt(text: String(repeating: "bee ", count: $0 + 1)) }, using: &rng)
        let first = data.plannedSentences(limit: 8, passage: true, lexicon: lexicon)
        XCTAssertFalse(first.isEmpty)
        for group in first {
            XCTAssertTrue((1...3).contains(group.count))
            XCTAssertEqual(Set(group.compactMap { data.roles[$0.id]?.rawValue }).count, 1)
            XCTAssertTrue(group.allSatisfy { data.roles[$0.id] != .finalTest })
        }
        let acquired = first[0]
        var complete = pair(acquired[0], role: data.roles[acquired[0].id]!)
        complete.promptIDs = acquired.map(\.id)
        data.pairs = [complete]
        let nextIDs = Set(data.plannedSentences(limit: 8, passage: true, lexicon: lexicon).flatMap { $0.map(\.id) })
        XCTAssertTrue(nextIDs.isDisjoint(with: acquired.map(\.id)))
    }

    func testOnlyCompleteAcceptedValidTrainingPairsSupplyCoverage() {
        let prompt = SentencePrompt(text: "bee bob")
        var data = SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training])
        let complete = pair(prompt, role: .training)
        data.pairs = [complete]
        XCTAssertEqual(data.derivedWords(role: .training, lexicon: lexicon).map(\.label), ["bee", "bob"])
        XCTAssertEqual(data.coverage(lexicon: lexicon), ["B": 3, "IY": 1, "AA": 1])
        let ids = data.derivedWords(role: .training, lexicon: lexicon).map(\.id)
        XCTAssertEqual(data.derivedWords(role: .training, lexicon: lexicon).map(\.id), ids)
        XCTAssertEqual(Set(ids).count, 2)
        let mutations: [(inout SentencePair) -> Void] = [
            { $0.review = .pending }, { $0.review = .excluded }, { $0.spoken = nil },
            { $0.silent = nil }, { $0.alignment = nil }, { $0.error = "bad" }, { $0.promptIDs = [] }
        ]
        for mutation in mutations {
            var invalid = complete
            mutation(&invalid); data.pairs = [invalid]
            XCTAssertTrue(data.derivedWords(role: .training, lexicon: lexicon).isEmpty)
            XCTAssertTrue(data.coverage(lexicon: lexicon).isEmpty)
        }
    }

    func testFinalAndValidationCannotSupplyCoverageEvenWithAcceptedOrForgedRecords() {
        for role in [DataRole.validation, .finalTest] {
            let prompt = SentencePrompt(text: "bee bob")
            var data = SentenceTrainingData(prompts: [prompt], roles: [prompt.id: role])
            data.pairs = [pair(prompt, role: role)]
            XCTAssertTrue(data.coverage(lexicon: lexicon).isEmpty)
            XCTAssertTrue(data.derivedWords(role: .finalTest, lexicon: lexicon).isEmpty)
            data.pairs[0].role = .training // Stored pair cannot override the persistent prompt assignment.
            XCTAssertTrue(data.derivedWords(role: .training, lexicon: lexicon).isEmpty)
        }
    }

    func testRepeatedWordPositionsRetainSeparateStableIDsAndLineage() throws {
        let prompt = SentencePrompt(text: "bee bee")
        var data = SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training])
        var item = pair(prompt, role: .training)
        item.alignment!.words[1].word = "bee"
        data.pairs = [item]
        let takes = data.derivedWords(role: .training, lexicon: lexicon)
        XCTAssertEqual(takes.map(\.label), ["bee", "bee"])
        XCTAssertNotEqual(takes[0].id, takes[1].id)
        XCTAssertTrue(takes.allSatisfy { $0.sourcePairID == item.id && $0.sourceSentenceID == prompt.id && $0.sessionID == item.visitID })
        let restored = try JSONDecoder().decode(SentenceTrainingData.self, from: JSONEncoder().encode(data))
        XCTAssertEqual(restored.derivedWords(role: .training, lexicon: lexicon).map(\.id), takes.map(\.id))
    }

    func testOutOfCueAnnotationsCannotBecomeWordTrainingTakes() {
        let prompt = SentencePrompt(text: "bee bob")
        var data = SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training])
        var item = pair(prompt, role: .training)
        item.alignment!.words[0].silentStart = 190
        item.alignment!.words[1].silentEnd = 210
        data.pairs = [item]
        XCTAssertTrue(data.derivedWords(role: .training, lexicon: lexicon).isEmpty)
    }

    func testRepairPlanWaitsForSentenceReviewAndIgnoresValidationAndReservedPairs() {
        let train = SentencePrompt(text: "bee bob"), validation = SentencePrompt(text: "bob bee"), final = SentencePrompt(text: "bee bee")
        var data = SentenceTrainingData(prompts: [train, validation, final], roles: [train.id: .training, validation.id: .validation, final.id: .finalTest])
        XCTAssertTrue(data.repairWords(lexicon: .bundled).isEmpty)
        var pending = pair(train, role: .training); pending.review = .pending
        data.pairs = [pending]
        XCTAssertTrue(data.repairWords(lexicon: .bundled).isEmpty)
        data.pairs[0].review = .accepted
        let repairs = data.repairWords(lexicon: .bundled)
        XCTAssertFalse(repairs.isEmpty)
        data.pairs.append(pair(validation, role: .validation))
        data.pairs.append(pair(final, role: .finalTest))
        XCTAssertEqual(data.repairWords(lexicon: .bundled), repairs)
    }

    func testValidationUsesSentenceGroupsAndNeverScoresFinalWords() throws {
        let train = SentencePrompt(text: "bee bob"), validation = SentencePrompt(text: "bob bee"), final = SentencePrompt(text: "bee bee")
        var data = SentenceTrainingData(prompts: [train, validation, final], roles: [train.id: .training, validation.id: .validation, final.id: .finalTest])
        data.pairs = [pair(train, role: .training), pair(validation, role: .validation), pair(final, role: .finalTest)]
        var settings = DecoderSettings(); settings.commonBias = 0
        let report = try SentenceValidation.run(data: data, settings: settings, lexicon: lexicon, vocabulary: [("bee", 1), ("bob", 1)])
        XCTAssertEqual(report.trainingPairIDs, [data.pairs[0].id])
        XCTAssertEqual(report.validationPairIDs, [data.pairs[1].id])
        XCTAssertEqual(report.expected, ["bee", "bob"])
        XCTAssertEqual(report.predicted.count, 2)
        XCTAssertFalse(report.validationPairIDs.contains(data.pairs[2].id))
        XCTAssertTrue(report.top1.isFinite && report.top5.isFinite)
    }

    func testBundledSentenceCorpusIsPresentAndHasKnownPronunciations() {
        XCTAssertEqual(SentencePrompt.bundled.count, 32)
        let words = SentencePrompt.bundled.flatMap { Lexicon.words(in: $0.text) }
        XCTAssertTrue(words.allSatisfy { Lexicon.bundled.phones($0) != nil }, "Missing: \(words.filter { Lexicon.bundled.phones($0) == nil })")
    }

    func testReferenceClockConversionPreservesRepeatedWordsWithoutAddingCropTwice() throws {
        let transcript = LocalWhisperTranscript(text: "Bee bee", words: [
            .init(text: "Bee", start: 22, end: 22.5), .init(text: "bee", start: 23, end: 23.5)], sourceAudioStartSeconds: 20)
        let anchors = try SentenceReference.anchors(transcript, sessionStart: 1000, audioOffset: 0.25)
        XCTAssertEqual(anchors.map(\.word), ["bee", "bee"])
        XCTAssertEqual(anchors.map(\.start), [1022.25, 1023.25])
        XCTAssertEqual(anchors.map(\.end), [1022.75, 1023.75])
    }

    func testAmbiguousMultiwordIntervalsAndInvalidClocksAreRejected() {
        let ambiguous = LocalWhisperTranscript(text: "twenty-one", words: [.init(text: "twenty-one", start: 1, end: 2)])
        XCTAssertThrowsError(try SentenceReference.anchors(ambiguous, sessionStart: 100, audioOffset: 0))
        let invalid = LocalWhisperTranscript(text: "bee", words: [.init(text: "bee", start: 1, end: 1)])
        XCTAssertThrowsError(try SentenceReference.anchors(invalid, sessionStart: 100, audioOffset: 0))
        XCTAssertThrowsError(try SentenceReference.anchors(ambiguous, sessionStart: .nan, audioOffset: 0))
    }
}
