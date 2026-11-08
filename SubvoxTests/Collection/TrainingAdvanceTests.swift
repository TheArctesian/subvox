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
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false)) // Setup stays interactive.
            let capture = Task { try await workflow.captureTake(condition: .normal, maximumDuration: 30) }
            try await waitFor(.speak, in: workflow)
            XCTAssertTrue(model.handleTrainingArrowKey(keyCode: forward, isRepeat: true))
            XCTAssertTrue(workflow.canFinishTake)
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: 0, isRepeat: false))
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false, modifiers: .command))
            XCTAssertFalse(model.handleTrainingArrowKey(keyCode: forward, isRepeat: false, isEditingText: true))
            XCTAssertTrue(workflow.canFinishTake)
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
    func testAdvanceCannotShortenRelaxOrRecovery() async throws {
        for phase in [SentenceTrainingWorkflow.Phase.relax, .recovery] {
            let (workflow, root) = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let model = AppModel(sentenceTraining: workflow)
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
