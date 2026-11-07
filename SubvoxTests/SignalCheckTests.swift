import XCTest
@testable import Subvox

final class SignalCheckTests: XCTestCase {
    private func take(sentence: Int = 0, repetition: Int = 0, condition: SpeechCondition = .silent) -> SignalCheckTake {
        let samples = (0..<251).map { i -> MotionSample in
            let time = Double(i) / 50
            if time < 3 { return MotionSample(t: 100 + time, rotation: (0.001, 0, 0)) }
            let phase = time - 3
            // Three distinct 3D motion patterns; repeats share the pattern, not absolute time.
            let a = 0.1 * sin(phase * (3 + Double(sentence) * 4))
            let b = 0.07 * cos(phase * (7 + Double(sentence) * 3))
            return MotionSample(t: 100 + time, accel: (a / 10, b / 10, 0), rotation: (a, b, 0.03 * sin(phase * 5)))
        }
        return .init(prompt: .init(sentence: sentence, repetition: repetition, condition: condition),
            recording: .init(condition: condition, samples: samples, cueStart: 103, cueEnd: 105), endReason: .done)
    }

    func testPlanBalancesAllSentencesConditionsAndRepetitionsWithoutFinalTestPrompts() {
        let a = SignalCheckRun.plan(seed: 42), b = SignalCheckRun.plan(seed: 42)
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.count, 18)
        for repetition in 0..<3 {
            for sentence in 0..<3 {
                let pair = a.filter { $0.repetition == repetition && $0.sentence == sentence }
                XCTAssertEqual(pair.map(\.condition), [.normal, .silent])
            }
        }
        XCTAssertFalse(Set(SignalCheckRun.sentences.map(SentencePrompt.identity)).isEmpty)
        XCTAssertTrue(Set(SignalCheckRun.sentences.map(SentencePrompt.identity)).isDisjoint(with: Set(SentencePrompt.bundled.map(\.id))))
    }

    func testMetricsCatchGapsTimeoutsAndDominantSpikes() {
        let normal = take()
        XCTAssertTrue(SignalCheckAnalysis.metrics(normal).usable)
        var timeout = normal; timeout.endReason = .timeout
        XCTAssertFalse(SignalCheckAnalysis.metrics(timeout).usable)
        var discarded = normal; discarded.discarded = true
        XCTAssertFalse(SignalCheckAnalysis.metrics(discarded).usable)
        var gapped = normal
        gapped.recording.samples.removeAll { $0.t > 103.5 && $0.t < 104 }
        XCTAssertFalse(SignalCheckAnalysis.metrics(gapped).usable)
        var missingEnd = normal
        missingEnd.recording.samples.removeAll { $0.t > 104.6 }
        XCTAssertFalse(SignalCheckAnalysis.metrics(missingEnd).usable)
        var noBaseline = normal
        noBaseline.recording.samples.removeAll { $0.t < 103 }
        XCTAssertFalse(SignalCheckAnalysis.metrics(noBaseline).usable)
        var spiked = normal; spiked.recording.samples[210].rx = 10
        let metrics = SignalCheckAnalysis.metrics(spiked)
        XCTAssertGreaterThan(metrics.topOnePercentEnergyShare, 0.9)
        XCTAssertTrue(metrics.issues.contains { $0.contains("35%") })
        XCTAssertGreaterThan(metrics.rotationRMS, try! XCTUnwrap(metrics.baselineRMS))
    }

    func testHeldOutRepetitionsRecoverDistinctSyntheticSentencesAndReportSeparateConditions() throws {
        var run = SignalCheckRun(seed: 1)
        run.takes = SignalCheckRun.plan(seed: 1).map { take(sentence: $0.sentence, repetition: $0.repetition, condition: $0.condition) }
        let report = try SignalCheckAnalysis.run(run)
        XCTAssertEqual(report.conditions.count, 2)
        for result in report.conditions {
            XCTAssertEqual(result.validTakes, 9)
            XCTAssertEqual(result.evaluated, 9)
            XCTAssertEqual(result.correct, 9)
            XCTAssertEqual(result.shiftedEvaluated, 9)
            XCTAssertLessThan(result.shiftedCorrect, result.correct)
            XCTAssertLessThan(try XCTUnwrap(result.sameSentenceCost), try XCTUnwrap(result.otherSentenceCost))
        }
        // No other repetition may masquerade as held out if just one repetition exists.
        run.takes = run.takes.filter { $0.prompt.repetition == 0 }
        XCTAssertTrue(try SignalCheckAnalysis.run(run).conditions.allSatisfy { $0.evaluated == 0 })
    }

    func testStillMotionAndIncompleteClassesCannotYieldPositiveAccuracy() throws {
        var run = SignalCheckRun(seed: 1)
        run.takes = (0..<3).map { take(sentence: 0, repetition: $0) }
        XCTAssertTrue(try SignalCheckAnalysis.run(run).conditions.allSatisfy { $0.evaluated == 0 })
        run.takes = SignalCheckRun.plan(seed: 1).map { prompt in
            var value = take(sentence: prompt.sentence, repetition: prompt.repetition, condition: prompt.condition)
            value.recording.samples = value.recording.samples.map { MotionSample(t: $0.t) }
            return value
        }
        XCTAssertTrue(try SignalCheckAnalysis.run(run).conditions.allSatisfy { $0.evaluated == 0 })
    }

    @MainActor func testExplicitBoundariesExcludeLateSamplesAndRedoPersistsDiscardedAttempt() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-signal-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = SessionRecorder(root: root.appending(path: "raw"))
        recorder.condition = .signalCheck
        let session = try XCTUnwrap(recorder.start(audioSink: nil))
        let store = root.appending(path: "signal.json")
        let workflow = SignalCheckWorkflow(storeURL: store)
        workflow.begin(session: session, audioDevice: "Fixture Mac microphone", seed: 1, recorder: recorder)
        let start = session.startUptime + 3
        workflow.beginCapture(at: start)
        for i in -100...100 { workflow.ingest(MotionSample(t: start + Double(i) / 50, rotation: (0.01, 0, 0))) }
        workflow.endCapture(at: start + 1, reason: .done)
        workflow.endCapture(at: start + 2, reason: .done) // No duplicate END.
        XCTAssertEqual(workflow.run?.takes.count, 1)
        let take = try XCTUnwrap(workflow.latestTake)
        XCTAssertEqual(take.recording.cueStart, start)
        XCTAssertEqual(take.recording.cueEnd, start + 1)
        XCTAssertFalse(take.recording.samples.contains { $0.t > start + 1 })
        XCTAssertTrue(recorder.isRecording)
        workflow.redo()
        XCTAssertTrue(try XCTUnwrap(workflow.run?.takes.first).discarded)
        XCTAssertEqual(workflow.index, 0)
        let restored = SignalCheckWorkflow(storeURL: store)
        XCTAssertEqual(restored.run?.audioDevice, "Fixture Mac microphone")
        XCTAssertTrue(try XCTUnwrap(restored.run?.takes.first).discarded)
        workflow.prepareForTermination()
        XCTAssertFalse(recorder.isRecording)
        let events = try String(contentsOf: session.directory.appending(path: "events.csv"), encoding: .utf8)
        XCTAssertTrue(events.contains("signal-start")); XCTAssertTrue(events.contains("signal-end"))
        XCTAssertTrue(events.contains("signal-redo"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: session.directory.appending(path: "signal-check.json").path))
    }

    @MainActor func testStopDuringUnfinishedCaptureDoesNotSaveAnIncompleteTakeOrStopAReplacementSession() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-signal-stop-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = SessionRecorder(root: root.appending(path: "raw"))
        let session = try XCTUnwrap(recorder.start(audioSink: nil))
        let workflow = SignalCheckWorkflow(storeURL: root.appending(path: "signal.json"))
        workflow.begin(session: session, audioDevice: nil, seed: 3, recorder: recorder)
        workflow.beginCapture(at: session.startUptime + 3)
        recorder.stop(sessionID: session.id)
        let replacement = try XCTUnwrap(recorder.start(audioSink: nil))
        workflow.prepareForTermination()
        XCTAssertEqual(workflow.run?.takes.count, 0)
        XCTAssertEqual(recorder.activeSessionID, replacement.id)
        recorder.stop()
    }

    @MainActor func testDiagnosticArrowKeysIgnoreRepeatsAndRouteRedoOnlyToTheCheck() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-signal-keys-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let workflow = SignalCheckWorkflow(storeURL: root.appending(path: "signal.json"))
        let training = SentenceTrainingWorkflow(storeURL: root.appending(path: "training.json"))
        let session = RecordingSession(id: UUID(), directory: root, startUptime: ProcessInfo.processInfo.systemUptime, audioOffset: nil)
        workflow.begin(session: session, audioDevice: nil, seed: 1)
        let model = AppModel(sentenceTraining: training, signalCheck: workflow)
        model.showSignalCheck = true
        workflow.beginCapture(at: session.startUptime - 1)
        XCTAssertTrue(model.handleTrainingArrowKey(keyCode: 124, isRepeat: true))
        XCTAssertEqual(workflow.phase, .recording)
        XCTAssertFalse(model.handleTrainingArrowKey(keyCode: 124, isRepeat: false, isEditingText: true))
        XCTAssertEqual(workflow.phase, .recording)
        XCTAssertTrue(model.handleTrainingArrowKey(keyCode: 124, isRepeat: false))
        XCTAssertEqual(workflow.phase, .ended)
        XCTAssertTrue(model.handleTrainingArrowKey(keyCode: 124, isRepeat: true))
        XCTAssertEqual(workflow.phase, .ended)
        XCTAssertTrue(model.handleTrainingArrowKey(keyCode: 123, isRepeat: false))
        XCTAssertEqual(workflow.phase, .ready)
        XCTAssertTrue(try XCTUnwrap(workflow.run?.takes.first).discarded)
        XCTAssertFalse(training.isBusy)
        workflow.prepareForTermination()
    }

    @MainActor func testRefreshingDevicesPreservesExplicitMicrophoneChoices() {
        let audio = AudioService()
        audio.selectedDeviceID = 123456
        audio.refreshDevices()
        XCTAssertEqual(audio.selectedDeviceID, 123456)
        audio.selectedDeviceID = nil // Explicit system default after initial discovery.
        audio.refreshDevices()
        XCTAssertNil(audio.selectedDeviceID)
    }

    func testMacMicSelectionUsesTransportAndFallsBackWithoutChangingSystemDefault() {
        let devices: [AudioDevices.Device] = [.init(id: 1, name: "AirPods"), .init(id: 2, name: "Ordinateur", isBuiltIn: true)]
        XCTAssertEqual(AudioDevices.preferredInput(in: devices), 2)
        XCTAssertNil(AudioDevices.preferredInput(in: [devices[0]]))
    }
}
