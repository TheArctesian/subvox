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

    private func automaticPair() -> SentencePair {
        var item = pair(SentencePrompt(text: "bee bob"), role: .training)
        item.review = .pending
        item.transcriptMatchRate = 1
        item.recordingSessionID = UUID()
        return item
    }

    func testAutomaticAlignedPairSuppliesWordsWithoutChangingStoredReview() throws {
        let item = automaticPair(), prompt = SentencePrompt(text: item.text)
        let data = SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training], pairs: [item])
        XCTAssertEqual(data.derivedWords(role: .training, lexicon: lexicon).map(\.label), ["bee", "bob"])
        XCTAssertEqual(data.coverage(lexicon: lexicon), ["B": 3, "IY": 1, "AA": 1])
        XCTAssertTrue(item.automaticUse.wordModel)
        XCTAssertTrue(item.automaticUse.takes.allSatisfy(\.included))
        XCTAssertEqual(item.automaticUse.state, .included)
        let restored = try JSONDecoder().decode(SentenceTrainingData.self, from: JSONEncoder().encode(data))
        XCTAssertEqual(restored.pairs[0].review, .pending, "Automatic inclusion must not overwrite an explicit review decision")
        XCTAssertEqual(restored.derivedWords(role: .training, lexicon: lexicon).count, 2)
    }

    func testWordTimingFailureStillAdmitsSentenceTakesAutomatically() {
        var item = automaticPair()
        item.alignment = nil; item.error = "Too few words have usable audio-aligned times."
        XCTAssertEqual(item.automaticUse.state, .warnings)
        XCTAssertFalse(item.usesAlignedWords)
        XCTAssertTrue(item.automaticUse.takes.allSatisfy(\.included))
        XCTAssertTrue(item.canIncludeManually)
    }

    func testAutomaticGapExclusionIsPerTakeAndDoesNotDiscardValidPartner() {
        var item = automaticPair()
        item.silent!.samples.removeSubrange(30..<45)
        item.alignment = nil; item.error = "Motion has a sampling gap or a sample rate below 10 Hz."
        let use = item.automaticUse
        XCTAssertEqual(use.state, .warnings)
        XCTAssertTrue(use.takes[0].included)
        XCTAssertFalse(use.takes[1].included)
        XCTAssertTrue(use.takes[1].reason?.contains("Sensor gap") == true)
        XCTAssertEqual(item.silent!.samples.count, 86, "Assessment must preserve the raw take")
        item.spoken = item.silent
        XCTAssertEqual(item.automaticUse.state, .attention)
        XCTAssertFalse(item.canIncludeManually, "Confirmation must not bypass invalid motion")
    }

    func testSourceSwitchCannotEnterWordModelThroughCachedAlignmentOrManualConfirmation() {
        var item = automaticPair()
        item.silent!.samples = item.silent!.samples.map { var sample = $0; sample.sensorLocation = 1; return sample }
        item.silent!.samples[40].sensorLocation = 2
        XCTAssertFalse(item.usesAlignedWords)
        XCTAssertTrue(item.automaticUse.takes[0].included)
        XCTAssertFalse(item.automaticUse.takes[1].included)
        item.review = .accepted
        XCTAssertFalse(item.usesAlignedWords)
        XCTAssertFalse(item.automaticUse.takes[1].included)
    }

    func testAutomaticUseRetainsSubtleAndLargeMotionAsWarnings() {
        var item = automaticPair()
        item.spoken!.samples = item.spoken!.samples.map { MotionSample(t: $0.t) }
        item.silent!.samples[40].rx = 3
        let use = item.automaticUse
        XCTAssertEqual(use.state, .warnings)
        XCTAssertTrue(use.takes.allSatisfy(\.included))
        XCTAssertTrue(use.warnings.contains { $0.contains("Very little motion") })
        XCTAssertTrue(use.warnings.contains { $0.contains("Large motion") })
    }

    func testAutomaticUseSeparatesWordAndNeuralSamplingRequirements() {
        var item = automaticPair()
        item.spoken!.samples = item.spoken!.samples.enumerated().filter { $0.offset.isMultiple(of: 3) }.map(\.element)
        item.silent!.samples = item.spoken!.samples
        XCTAssertTrue(item.usesAlignedWords, "The existing word alignment supports lower rates")
        XCTAssertTrue(item.automaticUse.takes.allSatisfy { !$0.included && $0.reason == "Motion rate below 20 Hz" })
        XCTAssertEqual(item.automaticUse.state, .warnings)
    }

    func testSlowerAirPodsAtTwentyFiveHertzStayEligible() {
        var item = automaticPair()
        item.spoken!.samples = item.spoken!.samples.enumerated().filter { $0.offset.isMultiple(of: 2) }.map(\.element)
        item.silent!.samples = item.spoken!.samples
        XCTAssertTrue(item.automaticUse.takes.allSatisfy(\.included), "Some AirPods report at about 25 Hz")
    }

    func testAutomaticUseHoldsWeakTranscriptAndPreservesManualOverrides() {
        var item = automaticPair()
        item.transcriptMatchRate = 0.8
        XCTAssertEqual(item.automaticUse.state, .attention)
        XCTAssertFalse(item.usesAlignedWords)
        item.review = .accepted
        XCTAssertTrue(item.usesAlignedWords)
        XCTAssertTrue(item.automaticUse.takes.allSatisfy(\.included))
        item.review = .excluded
        XCTAssertEqual(item.automaticUse.state, .excluded)
        XCTAssertFalse(item.usesAlignedWords)
        XCTAssertTrue(item.automaticUse.takes.allSatisfy { !$0.included })
        item.review = .pending; item.transcriptMatchRate = .nan
        XCTAssertEqual(item.automaticUse.state, .attention)
    }

    func testAutomaticUseKeepsFinalTestReservedDespiteManualAcceptance() {
        var item = automaticPair()
        item.role = .finalTest; item.review = .accepted
        XCTAssertEqual(item.automaticUse.state, .reserved)
        XCTAssertFalse(item.canIncludeManually)
        XCTAssertFalse(item.automaticUse.wordModel)
        XCTAssertTrue(item.automaticUse.takes.allSatisfy { !$0.included })
    }

    @MainActor func testManualConfirmationAndReturnToAutomaticPersistWithoutInventingWordTimes() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "sentence-auto-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        var item = automaticPair()
        item.transcriptMatchRate = 0.8; item.alignment = nil; item.error = "Word timing unavailable"
        let prompt = SentencePrompt(text: item.text), url = root.appending(path: "sentences.json")
        let data = SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .training], pairs: [item])
        try SentenceStore.save(data, to: url)
        let workflow = SentenceTrainingWorkflow(storeURL: url)
        var changes = 0; workflow.onTrainingChanged = { changes += 1 }
        workflow.review(item.id, decision: .accepted)
        XCTAssertEqual(workflow.data.pairs[0].review, .accepted)
        XCTAssertEqual(workflow.use(for: workflow.data.pairs[0]).state, .warnings)
        XCTAssertTrue(workflow.data.pairs[0].automaticUse.takes.allSatisfy(\.included))
        XCTAssertFalse(workflow.data.pairs[0].usesAlignedWords)
        workflow.review(item.id, decision: .excluded)
        XCTAssertEqual(workflow.data.pairs[0].automaticUse.state, .excluded)
        XCTAssertEqual(workflow.pairUses[item.id]?.state, .excluded)
        // This fixture has no audio directory: even a failed retry must retain exclusion.
        workflow.retryProcessing(item.id, whisper: LocalWhisper())
        for _ in 0..<200 {
            if !workflow.isBusy { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertFalse(workflow.isBusy)
        XCTAssertEqual(workflow.data.pairs[0].review, .excluded)
        XCTAssertEqual(workflow.pairUses[item.id]?.state, .excluded)
        workflow.review(item.id, decision: .pending)
        XCTAssertEqual(workflow.data.pairs[0].automaticUse.state, .attention)
        XCTAssertEqual(try SentenceStore.load(from: url).pairs[0].review, .pending)
        XCTAssertEqual(changes, 4)
    }

    @MainActor func testReservedPromptAssignmentOverridesPairRoleAndCannotBeManuallyPromoted() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "sentence-auto-reserved-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let item = automaticPair(), prompt = SentencePrompt(text: item.text), url = root.appending(path: "sentences.json")
        try SentenceStore.save(SentenceTrainingData(prompts: [prompt], roles: [prompt.id: .finalTest], pairs: [item]), to: url)
        let workflow = SentenceTrainingWorkflow(storeURL: url)
        XCTAssertEqual(workflow.use(for: item).state, .reserved)
        workflow.review(item.id, decision: .accepted)
        XCTAssertEqual(workflow.data.pairs[0].review, .pending)
        XCTAssertTrue(workflow.fittingWords.isEmpty)
    }

    func testAutomaticValidationIncludesEligiblePairsWithoutApprovalClicks() throws {
        let train = SentencePrompt(text: "bee bob"), validation = SentencePrompt(text: "bob bee")
        var a = pair(train, role: .training), b = pair(validation, role: .validation)
        a.review = .pending; a.transcriptMatchRate = 1
        b.review = .pending; b.transcriptMatchRate = 1
        let data = SentenceTrainingData(prompts: [train, validation], roles: [train.id: .training, validation.id: .validation], pairs: [a, b])
        let result = try SentenceValidation.run(data: data, settings: DecoderSettings(), lexicon: lexicon, vocabulary: [("bee", 1), ("bob", 1)])
        XCTAssertEqual(result.trainingPairIDs, [a.id])
        XCTAssertEqual(result.validationPairIDs, [b.id])
        XCTAssertEqual(result.expectedValidationWordCount, 2)
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
        for method in LearningMethod.allCases {
            settings.method = method
            let vocabulary = [(word: "bee", count: 1), (word: "bob", count: 1)]
            let report = try SentenceValidation.run(data: data, settings: settings, lexicon: lexicon, vocabulary: vocabulary)
            XCTAssertEqual(report.trainingPairIDs, [data.pairs[0].id])
            XCTAssertEqual(report.validationPairIDs, [data.pairs[1].id])
            XCTAssertEqual(report.expected, ["bee", "bob"])
            XCTAssertEqual(report.predicted.count, 2)
            XCTAssertFalse(report.validationPairIDs.contains(data.pairs[2].id))
            XCTAssertTrue(report.top1.isFinite && report.top5.isFinite)
            let live = try XCTUnwrap(MotionWordDecoder(takes: data.derivedWords(role: .training, lexicon: lexicon),
                                                       lexicon: lexicon, vocabulary: vocabulary, settings: settings))
            let predictions = data.derivedWords(role: .validation, lexicon: lexicon).map { live.predict($0.samples) }
            XCTAssertEqual(report.predicted, predictions.map { $0.ranked.map(\.label) })
            XCTAssertEqual(report.accepted, predictions.map(\.accepted))
        }
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
