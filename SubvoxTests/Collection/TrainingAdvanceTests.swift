import AVFoundation
import XCTest
@testable import Subvox

final class TrainingAdvanceTests: XCTestCase {
    @MainActor
    private func fixture() throws -> (SentenceTrainingWorkflow, URL) {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-advance-\(UUID())")
        let workflow = SentenceTrainingWorkflow(storeURL: root.appending(path: "sentences.json"))
        XCTAssertTrue(workflow.storageReady)
        return (workflow, root)
    }

    @MainActor
    private func waitFor(_ phase: SentenceTrainingWorkflow.Phase, in workflow: SentenceTrainingWorkflow) async throws {
        for _ in 0..<200 {
            if workflow.phase == phase { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("The capture did not reach \(phase)")
    }

    @MainActor
    func testReadingDestinationsFromDecodeSelectReadingAndCorrectOverlay() throws {
        let (workflow, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(sentenceTraining: workflow)
        XCTAssertEqual(model.screen, .home)
        for showModels in [true, false] {
            model.trainMode = .poetry
            model.screen = .decode
            model.openReading(showModels: showModels)
            XCTAssertEqual(model.screen, .train)
            XCTAssertEqual(model.trainMode, .reading)
            XCTAssertEqual(model.showSequenceLearning, showModels)
        }
    }

    @MainActor
    func testDoneEndsSpokenAndMouthedTakesAtThePressAndExcludesLateSamples() async throws {
        for condition in [SpeechCondition.normal, .silent] {
            let (workflow, root) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let capture = Task { try await workflow.captureTake(condition: condition, maximumDuration: 30) }
            try await waitFor(condition == .normal ? .speak : .mouth, in: workflow)
            let sampleTime = ProcessInfo.processInfo.systemUptime
            workflow.ingest(MotionSample(t: sampleTime, rotation: (0.1, 0, 0)))
            XCTAssertTrue(workflow.canFinishTake)
            let beforePress = ProcessInfo.processInfo.systemUptime
            XCTAssertTrue(workflow.finishCurrentTake())
            let afterPress = ProcessInfo.processInfo.systemUptime
            XCTAssertFalse(workflow.canFinishTake)
            XCTAssertFalse(workflow.finishCurrentTake()) // A double click cannot queue another advance.
            let lateTime = afterPress + 0.001
            workflow.ingest(MotionSample(t: lateTime, rotation: (1, 0, 0)))
            let take = try await capture.value
            XCTAssertEqual(take.condition, condition)
            XCTAssertGreaterThanOrEqual(take.cueEnd, beforePress)
            XCTAssertLessThanOrEqual(take.cueEnd, afterPress)
            XCTAssertLessThan(take.cueEnd - take.cueStart, 5)
            XCTAssertEqual(take.cueSamples.map(\.t), [sampleTime])
            XCTAssertFalse(take.samples.contains { $0.t == lateTime })
            XCTAssertFalse(workflow.canFinishTake)
        }
    }

    @MainActor
    func testRightAndDownAdvanceButRepeatsModifiersAndOtherKeysDoNot() async throws {
        for forward: UInt16 in [124, 125, 49] {
            let (workflow, root) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let model = AppModel(sentenceTraining: workflow)
            model.openPractice(.reading)
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false)) // Setup stays interactive.
            let capture = Task { try await workflow.captureTake(condition: .normal, maximumDuration: 30) }
            try await waitFor(.speak, in: workflow)
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: forward, isRepeat: true))
            XCTAssertTrue(workflow.canFinishTake)
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: 0, isRepeat: false))
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false, modifiers: .command))
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false, isEditingText: true))
            XCTAssertTrue(workflow.canFinishTake)
            model.captureShortcutsSuspended = true
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false))
            XCTAssertTrue(workflow.canFinishTake)
            model.captureShortcutsSuspended = false
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false))
            _ = try await capture.value
            // Holding the same key into the next cue must not finish the silent take.
            let silent = Task { try await workflow.captureTake(condition: .silent, maximumDuration: 30) }
            try await waitFor(.mouth, in: workflow)
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: forward, isRepeat: true))
            XCTAssertTrue(workflow.canFinishTake)
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false))
            _ = try await silent.value
        }
    }

    @MainActor
    func testLiteratureCompletesTheSilentPassBeforeStartingTheSpokenPass() {
        let literature = SentenceTrainingWorkflow.takeOrder(groupCount: 3, conditions: [.normal, .silent], fullPiece: true)
        XCTAssertEqual(literature.map(\.group), [0, 1, 2, 0, 1, 2])
        XCTAssertEqual(literature.map(\.condition), [.silent, .silent, .silent, .normal, .normal, .normal])

        let drill = SentenceTrainingWorkflow.takeOrder(groupCount: 2, conditions: [.normal, .silent], fullPiece: false)
        XCTAssertEqual(drill.map(\.group), [0, 0, 1, 1])
        XCTAssertEqual(drill.map(\.condition), [.normal, .silent, .normal, .silent])
    }

    @MainActor
    func testSectionRetryKeepsTheOriginalResumeCheckpoint() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "literature-section-retry-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = root.appending(path: "sentences.json")
        let groups = ["Opening.", "Closing."].map { [SentencePrompt(text: $0, kind: .speech, language: "en")] }
        var data = SentenceTrainingData()
        data.addPractice(groups.flatMap { $0 })
        try SentenceStore.save(data, to: store)

        let workflow = SentenceTrainingWorkflow(storeURL: store)
        let session = RecordingSession(id: UUID(), directory: root, startUptime: 100, audioOffset: 0)
        let visit = try workflow.beginVisit(session: session, audioDevice: nil, timing: .comfortable, presentation: .sentences)
        let pair = try workflow.beginPair(prompts: groups[0], visit: visit, session: session, literaturePieceID: "fixture")
        let round = LiteraturePracticeRound(pieceID: "fixture", title: "Fixture", language: "en", kind: .speech, groups: groups)
        let silent = SentenceRecording(condition: .silent, samples: [], cueStart: 101, cueEnd: 102)
        _ = try workflow.saveLiteratureHalf(silent, pairID: pair.id, session: session, round: round, section: 0)

        let retrySession = RecordingSession(id: UUID(), directory: root, startUptime: 200, audioOffset: 0)
        let retryVisit = try workflow.beginVisit(session: retrySession, audioDevice: nil, timing: .comfortable, presentation: .sentences)
        let retryPair = try workflow.beginPair(prompts: groups[0], visit: retryVisit, session: retrySession, literaturePieceID: "fixture")
        let retry = LiteraturePracticeRound(pieceID: "fixture", title: "Fixture", language: "en", kind: .speech,
                                           groups: [groups[0]], isSectionRetry: true)
        let newSilent = SentenceRecording(condition: .silent, samples: [], cueStart: 201, cueEnd: 202)
        let progress = try workflow.saveLiteratureHalf(newSilent, pairID: retryPair.id, session: retrySession, round: retry, section: 0)
        let newSpoken = SentenceRecording(condition: .normal, samples: [], cueStart: 203, cueEnd: 204)
        _ = try workflow.saveLiteratureHalf(newSpoken, pairID: retryPair.id, session: retrySession, round: progress, section: 0)

        let reopened = SentenceTrainingWorkflow(storeURL: store)
        let resumable = try XCTUnwrap(reopened.resumableLiterature.first)
        XCTAssertEqual(resumable.id, round.id)
        XCTAssertEqual(reopened.nextLiteratureTake(in: resumable)?.group, 1)
        XCTAssertEqual(reopened.data.pairs.first { $0.id == pair.id }?.silent?.id, silent.id)
        XCTAssertEqual(reopened.data.pairs.first { $0.id == retryPair.id }?.spoken?.id, newSpoken.id)
    }

    @MainActor
    func testResumeUsesSavedHalvesAndPreservesEachPhysicalSessionClock() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "literature-resume-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = root.appending(path: "sentences.json")
        let groups = ["Mouth this line.", "Then move onward."].map {
            [SentencePrompt(text: $0, kind: .poetry, language: "en")]
        }
        var data = SentenceTrainingData()
        data.addPractice(groups.flatMap { $0 })
        try SentenceStore.save(data, to: store)

        let initial = SentenceTrainingWorkflow(storeURL: store)
        let firstSession = RecordingSession(id: UUID(), directory: root.appending(path: "first"),
                                            startUptime: 100, audioOffset: 0.5)
        let firstVisit = try initial.beginVisit(session: firstSession, audioDevice: nil,
                                               timing: .comfortable, presentation: .sentences)
        let first = try initial.beginPair(prompts: groups[0], visit: firstVisit, session: firstSession,
                                          literaturePieceID: "fixture")
        _ = try initial.beginPair(prompts: groups[1], visit: firstVisit, session: firstSession,
                                  literaturePieceID: "fixture")
        let draft = LiteraturePracticeRound(pieceID: "fixture", title: "Fixture", language: "en",
                                            kind: .poetry, groups: groups)
        let silent = SentenceRecording(condition: .silent, samples: [], cueStart: 101, cueEnd: 102)
        _ = try initial.saveLiteratureHalf(silent, pairID: first.id, session: firstSession,
                                           round: draft, section: 0)

        let reopened = SentenceTrainingWorkflow(storeURL: store)
        var round = try XCTUnwrap(reopened.resumableLiterature.first)
        XCTAssertEqual(round.id, draft.id)
        XCTAssertEqual(reopened.nextLiteratureTake(in: round)?.group, 1)
        XCTAssertEqual(reopened.nextLiteratureTake(in: round)?.condition, .silent)

        let nextSession = RecordingSession(id: UUID(), directory: root.appending(path: "second"),
                                           startUptime: 200, audioOffset: 0.25)
        let nextVisit = try reopened.beginVisit(session: nextSession, audioDevice: nil,
                                                timing: .comfortable, presentation: .sentences)
        let second = try reopened.beginPair(prompts: groups[1], visit: nextVisit, session: nextSession,
                                            literaturePieceID: "fixture")
        let resumedSilent = SentenceRecording(condition: .silent, samples: [], cueStart: 201, cueEnd: 202)
        round = try reopened.saveLiteratureHalf(resumedSilent, pairID: second.id, session: nextSession,
                                                round: round, section: 1)
        XCTAssertEqual(reopened.nextLiteratureTake(in: round)?.group, 0)
        XCTAssertEqual(reopened.nextLiteratureTake(in: round)?.condition, .normal)

        let spoken = SentenceRecording(condition: .normal, samples: [], cueStart: 203, cueEnd: 204)
        XCTAssertThrowsError(try reopened.saveLiteratureHalf(spoken, pairID: first.id, session: nextSession,
                                                             round: round, section: 0))
        let spokenFirst = try reopened.beginPair(prompts: groups[0], visit: nextVisit, session: nextSession,
                                                 literaturePieceID: "fixture")
        round = try reopened.saveLiteratureHalf(spoken, pairID: spokenFirst.id, session: nextSession,
                                                round: round, section: 0)
        XCTAssertEqual(reopened.nextLiteratureTake(in: round)?.group, 1)
        round = try reopened.saveLiteratureHalf(spoken, pairID: second.id, session: nextSession,
                                                round: round, section: 1)

        let completed = SentenceTrainingWorkflow(storeURL: store)
        XCTAssertTrue(completed.resumableLiterature.isEmpty)
        let firstHalf = try XCTUnwrap(completed.data.pairs.first { $0.id == first.id })
        let secondHalf = try XCTUnwrap(completed.data.pairs.first { $0.id == spokenFirst.id })
        XCTAssertEqual(firstHalf.recordingSessionID, firstSession.id)
        XCTAssertEqual(firstHalf.sessionStartUptime, 100)
        XCTAssertEqual(firstHalf.silent?.id, silent.id)
        XCTAssertNil(firstHalf.spoken)
        XCTAssertEqual(secondHalf.recordingSessionID, nextSession.id)
        XCTAssertEqual(secondHalf.sessionStartUptime, 200)
        XCTAssertEqual(secondHalf.audioOffset, 0.25)
        XCTAssertEqual(secondHalf.spoken?.id, spoken.id)
        XCTAssertNil(secondHalf.silent)
        XCTAssertFalse(firstHalf.canAccept)
        XCTAssertFalse(secondHalf.canAccept)
        XCTAssertTrue(completed.data.pairs.first { $0.id == second.id }?.complete == true)
    }
    @MainActor
    func testInvictusStanzaTakesAreAvailableAfterReopening() throws {
        let poem = try XCTUnwrap(PracticeCorpus.bundled.pieces.first { $0.id == "invictus" })
        let stanzas = poem.parts(language: .en)
        XCTAssertEqual(stanzas.map(\.count), [4, 4, 4, 4])
        let root = FileManager.default.temporaryDirectory.appending(path: "invictus-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = root.appending(path: "sentences.json")
        let prompts = stanzas.map { $0.map { SentencePrompt($0, kind: .poetry, language: .en) } }
        var data = SentenceTrainingData()
        data.addPractice(prompts.flatMap { $0 })
        try SentenceStore.save(data, to: store)

        let audioURL = root.appending(path: "audio.caf")
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 8_000, channels: 1))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 80_000))
        buffer.frameLength = buffer.frameCapacity
        do {
            let audio = try AVAudioFile(forWriting: audioURL, settings: format.settings)
            try audio.write(from: buffer)
        }

        let workflow = SentenceTrainingWorkflow(storeURL: store)
        let session = RecordingSession(id: UUID(), directory: root, startUptime: 100, audioOffset: 0.5)
        let visit = try workflow.beginVisit(session: session, audioDevice: nil, timing: .comfortable, presentation: .sentences)
        let pairs = try prompts.map { try workflow.beginPair(prompts: $0, visit: visit, session: session) }
        for (index, pair) in pairs.enumerated() {
            let start = 101.5 + Double(index * 2)
            let spoken = SentenceRecording(condition: .normal, samples: [], cueStart: start, cueEnd: start + 1)
            try workflow.saveCompletedHalf(spoken, pairID: pair.id, session: session)
        }
        let reopened = SentenceTrainingWorkflow(storeURL: store)
        for (index, stanza) in stanzas.enumerated() {
            let take = try XCTUnwrap(SpokenTakePlayer.latestPoetryTake(for: stanza, language: .en, in: reopened.data.pairs))
            XCTAssertEqual(take.id, pairs[index].id)
            XCTAssertEqual(take.clip.url, audioURL)
            XCTAssertEqual(take.clip.range, Double(1 + index * 2)...Double(2 + index * 2))
        }
    }

    @MainActor
    func testMotionGapRetriesWithoutSettingASessionError() async throws {
        let (workflow, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        workflow.guidedRecording = true
        let take = Task {
            try await workflow.collectTake(condition: .normal, maximumDuration: 30,
                                           baselineSeconds: 0, recoverySeconds: 0,
                                           onCompleted: { _ in }, onRedo: { _ in })
        }
        defer { take.cancel() }
        try await waitFor(.ready, in: workflow)
        XCTAssertTrue(workflow.advance())
        try await waitFor(.speak, in: workflow)

        let now = ProcessInfo.processInfo.systemUptime
        workflow.ingest(MotionSample(t: now - 0.2, rotation: (0, 0, 0)))
        workflow.ingest(MotionSample(t: now, rotation: (0, 0, 0)))
        try await waitFor(.ready, in: workflow)
        XCTAssertNil(workflow.error)

        XCTAssertTrue(workflow.advance())
        try await waitFor(.speak, in: workflow)
        XCTAssertNil(workflow.error)
        XCTAssertTrue(workflow.finishCurrentTake())
        try await waitFor(.ended, in: workflow)
        XCTAssertTrue(workflow.advance())
        try await take.value
    }

    @MainActor
    func testAdvanceCannotShortenRelaxOrRecovery() async throws {
        for phase in [SentenceTrainingWorkflow.Phase.relax, .recovery] {
            let (workflow, root) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let model = AppModel(sentenceTraining: workflow)
            model.openPractice(.reading)
            let start = ProcessInfo.processInfo.systemUptime
            let resting = Task { try await workflow.rest(phase, seconds: 0.15) }
            try await waitFor(phase, in: workflow)
            XCTAssertFalse(workflow.finishCurrentTake())
            XCTAssertFalse(workflow.redoCurrentTake())
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: 124, isRepeat: false))
            try await resting.value
            XCTAssertGreaterThanOrEqual(ProcessInfo.processInfo.systemUptime - start, 0.15)
        }
    }

    @MainActor
    func testLeftAndUpRedoTheSameTakeAndHeldKeysDoNotRedoAgain() async throws {
        for backward: UInt16 in [123, 126] {
            let (workflow, root) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let model = AppModel(sentenceTraining: workflow)
            model.openPractice(.reading)
            var completed: [SentenceRecording] = [], discarded = 0
            let collecting = Task {
                try await workflow.collectTake(condition: .normal, maximumDuration: 30, baselineSeconds: 0.03,
                    recoverySeconds: 0.03, onCompleted: { completed.append($0) }, onRedo: { take in
                        XCTAssertNil(take) // The failed attempt never became a completed take.
                        discarded += 1
                    })
            }
            try await waitFor(.speak, in: workflow)
            let badStart = ProcessInfo.processInfo.systemUptime
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: backward, isRepeat: true))
            XCTAssertTrue(workflow.canFinishTake)
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: backward, isRepeat: false, modifiers: .command))
            XCTAssertTrue(workflow.finishCurrentTake())
            // A quick redo overrides Done before its completed take has been saved.
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: backward, isRepeat: false))
            try await waitFor(.relax, in: workflow)
            XCTAssertEqual(discarded, 1)
            XCTAssertTrue(completed.isEmpty)
            try await waitFor(.speak, in: workflow)
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: backward, isRepeat: true))
            XCTAssertTrue(workflow.canFinishTake)
            workflow.finishCurrentTake()
            try await collecting.value
            XCTAssertEqual(completed.count, 1)
            XCTAssertEqual(completed[0].condition, .normal)
            XCTAssertGreaterThan(completed[0].cueStart, badStart)
            XCTAssertEqual(discarded, 1)
        }
    }

    @MainActor
    func testRecoveryRedoReplacesSilentTakeAndPreservesSpokenReference() async throws {
        let (workflow, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = SessionRecorder(root: root.appending(path: "raw"))
        defer { recorder.stop() }
        let session = try XCTUnwrap(recorder.start(audioSink: nil))
        let visit = try workflow.beginVisit(session: session, audioDevice: nil, timing: .comfortable, presentation: .sentences)
        let prompt = try XCTUnwrap(workflow.data.prompts.first { workflow.data.roles[$0.id] == .training })
        let pair = try workflow.beginPair(prompts: [prompt], visit: visit, session: session)
        let reference = SentenceRecording(condition: .normal, samples: [], cueStart: 1, cueEnd: 2)
        try workflow.saveCompletedHalf(reference, pairID: pair.id, session: session)
        var attempts: [UUID] = [], discarded: [UUID] = []
        let collecting = Task {
            try await workflow.collectTake(condition: .silent, maximumDuration: 30, baselineSeconds: 0.03,
                recoverySeconds: 0.2, onCompleted: { recording in
                    attempts.append(recording.id)
                    try workflow.saveCompletedHalf(recording, pairID: pair.id, session: session)
                }, onRedo: { take in
                    discarded.append(try XCTUnwrap(take).id)
                    try workflow.discardHalf(condition: .silent, pairID: pair.id)
                })
        }
        try await waitFor(.mouth, in: workflow)
        workflow.finishCurrentTake()
        try await waitFor(.recovery, in: workflow)
        let unwantedID = try XCTUnwrap(workflow.data.pairs.first?.silent?.id)
        XCTAssertTrue(workflow.canRedoTake)
        workflow.redoCurrentTake()
        try await waitFor(.relax, in: workflow)
        let disk = try SentenceStore.load(from: root.appending(path: "sentences.json"))
        XCTAssertNil(disk.pairs[0].silent)
        XCTAssertEqual(disk.pairs[0].spoken?.id, reference.id)
        try await waitFor(.mouth, in: workflow)
        workflow.finishCurrentTake()
        try await collecting.value
        XCTAssertEqual(attempts.count, 2)
        XCTAssertNotEqual(attempts[0], attempts[1])
        XCTAssertEqual(discarded, [unwantedID])
        let final = try SentenceStore.load(from: root.appending(path: "sentences.json"))
        XCTAssertEqual(final.pairs[0].spoken?.id, reference.id)
        XCTAssertEqual(final.pairs[0].silent?.id, attempts[1])
        XCTAssertTrue(final.pairs[0].complete)
    }

    @MainActor
    func testRedoThenStopOrQuitCannotKeepTheUnwantedSavedHalf() async throws {
        for quit in [false, true] {
            let (workflow, root) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let recorder = SessionRecorder(root: root.appending(path: "raw"))
            defer { recorder.stop() }
            let session = try XCTUnwrap(recorder.start(audioSink: nil))
            let visit = try workflow.beginVisit(session: session, audioDevice: nil, timing: .comfortable, presentation: .sentences)
            let prompt = try XCTUnwrap(workflow.data.prompts.first { workflow.data.roles[$0.id] == .training })
            let pair = try workflow.beginPair(prompts: [prompt], visit: visit, session: session)
            var discarded = 0
            let collecting = Task {
                try await workflow.collectTake(condition: .normal, maximumDuration: 30, baselineSeconds: 0.03,
                    recoverySeconds: 30, onCompleted: {
                        try workflow.saveCompletedHalf($0, pairID: pair.id, session: session)
                    }, onRedo: { _ in
                        discarded += 1
                        try workflow.discardHalf(condition: .normal, pairID: pair.id)
                    })
            }
            try await waitFor(.speak, in: workflow)
            workflow.finishCurrentTake()
            try await waitFor(.recovery, in: workflow)
            XCTAssertNotNil(workflow.data.pairs[0].spoken)
            workflow.redoCurrentTake()
            if quit {
                workflow.prepareForTermination(recorder: recorder)
                // Quit cannot wait for the asynchronous capture to resume.
                XCTAssertNil(try SentenceStore.load(from: root.appending(path: "sentences.json")).pairs[0].spoken)
            }
            collecting.cancel()
            do { try await collecting.value; XCTFail("Interrupted collection must stop") }
            catch is CancellationError { }
            XCTAssertEqual(discarded, 1)
            XCTAssertNil(try SentenceStore.load(from: root.appending(path: "sentences.json")).pairs[0].spoken)
        }
    }

    @MainActor
    func testTimeoutStillCompletesAndAnEarlyFinishDoesNotCarryIntoAnotherCue() async throws {
        let (workflow, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let spoken = Task { try await workflow.captureTake(condition: .normal, maximumDuration: 30) }
        try await waitFor(.speak, in: workflow)
        XCTAssertTrue(workflow.finishCurrentTake())
        _ = try await spoken.value
        let silent = try await workflow.captureTake(condition: .silent, maximumDuration: 0.15)
        XCTAssertGreaterThanOrEqual(silent.cueEnd - silent.cueStart, 0.15)
        XCTAssertEqual(workflow.secondsRemaining, 0)
    }

    @MainActor
    func testCancellationStillDiscardsTheCurrentTakeAfterDoneWasRequested() async throws {
        let (workflow, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let capture = Task { try await workflow.captureTake(condition: .normal, maximumDuration: 30) }
        try await waitFor(.speak, in: workflow)
        XCTAssertTrue(workflow.finishCurrentTake())
        capture.cancel()
        do { _ = try await capture.value; XCTFail("Cancellation must not produce a completed take") }
        catch is CancellationError { }
        XCTAssertFalse(workflow.canFinishTake)
        XCTAssertTrue(workflow.data.pairs.isEmpty)
    }
}
