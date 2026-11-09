import XCTest
@testable import Subvox

final class ControlledSignalTests: XCTestCase {
    private let neutral = CameraHeadPose(yaw: 2, pitch: 3, roll: -1, x: 0.5, y: 0.5, size: 0.3)
    private func readings(start: Double, seconds: Double, pose: CameraHeadPose? = nil) -> [CameraPoseReading] {
        (0...Int((seconds * 10).rounded())).map { .init(t: start + Double($0) / 10, cameraID: "fixture-camera", pose: pose ?? neutral) }
    }
    func testCameraTargetRequiresStableSingleFaceAndEnforcesAnglesPositionScaleAndFreshness() throws {
        let samples = readings(start: 100, seconds: 1.6)
        let target = try XCTUnwrap(CameraPoseTarget.capture(samples, now: 101.6))
        XCTAssertTrue(target.isStable(samples, now: 101.6))
        XCTAssertFalse(target.isStable(samples, now: 102.2))
        var drift = neutral; drift.yaw += 6
        XCTAssertFalse(target.accepts(.init(t: 101, cameraID: target.cameraID, pose: drift)))
        drift = neutral; drift.x += 0.06
        XCTAssertFalse(target.accepts(.init(t: 101, cameraID: target.cameraID, pose: drift)))
        drift = neutral; drift.size *= 1.2
        XCTAssertFalse(target.accepts(.init(t: 101, cameraID: target.cameraID, pose: drift)))
        XCTAssertFalse(target.accepts(.init(t: 101, cameraID: "other-camera", pose: neutral)))
        XCTAssertFalse(target.accepts(.init(t: 101, cameraID: target.cameraID, pose: nil)))
        var bad = samples; bad[5].pose = nil
        XCTAssertNil(CameraPoseTarget.capture(bad, now: 101.6))
        bad = samples; bad.removeSubrange(4...10)
        XCTAssertNil(CameraPoseTarget.capture(bad, now: 101.6))
    }

    func testPacingIsFixedAcrossFourWordsAndCannotHideEarlyOrLateCaptures() throws {
        let pacing = SignalPacing.controlled
        XCTAssertEqual(pacing.duration(words: 4), 3.4, accuracy: 1e-8)
        XCTAssertNil(pacing.wordIndex(elapsed: 0.2, words: 4))
        XCTAssertEqual(pacing.wordIndex(elapsed: 0.4, words: 4), 0)
        XCTAssertEqual(pacing.wordIndex(elapsed: 1.05, words: 4), 1)
        XCTAssertEqual(pacing.wordIndex(elapsed: 2.35, words: 4), 3)
        XCTAssertNil(pacing.wordIndex(elapsed: 3.1, words: 4))
        XCTAssertNil(pacing.wordIndex(elapsed: 3.4, words: 4))
    }

    func testTimeWeightedPoseQualityCountsMissingCameraDriftAndTimerErrors() throws {
        let target = try XCTUnwrap(CameraPoseTarget.capture(readings(start: 98, seconds: 1.6), now: 99.6))
        var run = SignalCheckTestData.source()
        run.control = .init(testID: UUID(), block: .unchanged, target: target, pacing: .controlled, blockConfirmedAt: Date())
        run.takes = [run.takes[0]]
        run.takes[0].recording.cueStart = 100; run.takes[0].recording.cueEnd = 103.4; run.takes[0].endReason = .scheduled
        run.control?.readings = readings(start: 99.9, seconds: 3.5)
        XCTAssertTrue(try XCTUnwrap(ControlledSignalComparison.qualities(run).first).passes)
        run.control?.readings.removeSubrange(10...20)
        let gap = try XCTUnwrap(ControlledSignalComparison.qualities(run).first)
        XCTAssertLessThan(gap.coverage, 0.9); XCTAssertFalse(gap.passes)
        var wrong = neutral; wrong.roll += 12
        run.control?.readings = readings(start: 99.9, seconds: 3.5, pose: wrong)
        let drift = try XCTUnwrap(ControlledSignalComparison.qualities(run).first)
        XCTAssertGreaterThan(drift.coverage, 0.9); XCTAssertLessThan(drift.withinTarget, 0.1); XCTAssertFalse(drift.passes)
        run.control?.readings = readings(start: 99.9, seconds: 3.5)
        run.takes[0].recording.cueEnd = 103.8
        XCTAssertFalse(try XCTUnwrap(ControlledSignalComparison.qualities(run).first).passes)
    }

    @MainActor private func fixture() throws -> (SignalCheckWorkflow, URL, TestClock) {
        let root = FileManager.default.temporaryDirectory.appending(path: "subvox-controlled-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = SignalCheckTestData.source(directory: root)
        try JSONEncoder().encode(source).write(to: root.appending(path: "signal.json"))
        try FrozenSignalReference.freeze(source).saveNew(to: root.appending(path: "frozen-signal-reference.json"))
        let clock = TestClock()
        let workflow = SignalCheckWorkflow(storeURL: root.appending(path: "signal.json"), clock: { clock.now })
        workflow.prepareControlled(policy: .strictPose)
        XCTAssertTrue(workflow.isAcquiring)
        XCTAssertFalse(workflow.showsCaptureCue, "Setup must expose the controls that start the first block")
        aim(workflow, clock: clock)
        workflow.savePoseTarget()
        XCTAssertNotNil(workflow.comparison?.target)
        return (workflow, root, clock)
    }
    @MainActor private func aim(_ workflow: SignalCheckWorkflow, clock: TestClock) {
        for sample in readings(start: clock.now - 1.6, seconds: 1.6) { workflow.camera.ingest(sample) }
        workflow.ingest(MotionSample(t: clock.now))
    }
    @MainActor private func collect(_ workflow: SignalCheckWorkflow, clock: TestClock) throws {
        let patterns = SignalCheckTestData.source().takes
        for index in workflow.plan.indices {
            aim(workflow, clock: clock)
            let prompt = workflow.plan[index], start = clock.now
            workflow.beginCapture(at: start)
            XCTAssertEqual(workflow.phase, .recording)
            XCTAssertTrue(workflow.showsCaptureCue)
            workflow.finishTake() // Manual early advance must not shorten a controlled take.
            XCTAssertEqual(workflow.phase, .recording)
            let pattern = try XCTUnwrap(patterns.first { $0.prompt.sentence == prompt.sentence && $0.prompt.condition == prompt.condition })
            for original in pattern.recording.samples {
                var s = original
                let rel = original.t - pattern.recording.cueStart
                s.t = start + (rel < 0 ? rel : rel * 1.7)
                workflow.ingest(s)
            }
            for reading in readings(start: start, seconds: 3.4) { workflow.ingestPose(reading) }
            clock.now += 3.4
            workflow.endCapture(at: clock.now, reason: .scheduled)
            if index + 1 < workflow.plan.count { workflow.next(); clock.now += 2 }
        }
    }
    @MainActor private func wait(_ workflow: SignalCheckWorkflow, phase: SignalCheckWorkflow.Phase) async throws {
        for _ in 0..<500 {
            if workflow.phase == phase { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("Did not reach \(phase); error \(workflow.error ?? "none")")
    }

    @MainActor func testBothBlocksUseSameModelPoseAndOrderWhileKeepingResultsSeparate() async throws {
        let (workflow, root, clock) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let referenceBytes = try Data(contentsOf: root.appending(path: "frozen-signal-reference.json"))
        let comparison = try XCTUnwrap(workflow.comparison), target = try XCTUnwrap(comparison.target)
        let recorder = SessionRecorder(root: root.appending(path: "raw"))
        let first = try XCTUnwrap(recorder.start(audioSink: nil))
        workflow.begin(session: first, audioDevice: "Fixture Mac microphone", seed: comparison.seed, recorder: recorder,
            reference: comparison.reference, control: .init(testID: comparison.id, block: .unchanged, target: target,
                pacing: .controlled, blockConfirmedAt: Date()))
        XCTAssertTrue(workflow.isPaced)
        let firstPlan = workflow.plan
        try collect(workflow, clock: clock)
        workflow.next()
        try await wait(workflow, phase: .controlBreak)
        XCTAssertEqual(workflow.comparison?.blocks.count, 1)
        XCTAssertFalse(workflow.canStartControlledBlock)
        XCTAssertFalse(workflow.canSavePose) // The camera target cannot be recaptured after block A.
        let savedTarget = workflow.comparison?.target
        workflow.savePoseTarget(); XCTAssertEqual(workflow.comparison?.target, savedTarget)
        clock.now += 31; aim(workflow, clock: clock)
        XCTAssertTrue(workflow.canStartControlledBlock)
        let second = try XCTUnwrap(recorder.start(audioSink: nil)), reseated = Date()
        workflow.begin(session: second, audioDevice: "Fixture Mac microphone", seed: comparison.seed, recorder: recorder,
            reference: comparison.reference, reseatedAt: reseated,
            control: .init(testID: comparison.id, block: .reseated, target: target, pacing: .controlled, blockConfirmedAt: reseated))
        XCTAssertEqual(workflow.plan, firstPlan)
        try collect(workflow, clock: clock)
        workflow.next()
        try await wait(workflow, phase: .finished)
        let result = try XCTUnwrap(workflow.comparison)
        XCTAssertEqual(result.blocks.count, 2)
        try result.validate()
        for block in result.blocks {
            XCTAssertEqual(block.reference, comparison.reference)
            XCTAssertEqual(block.control?.target, target)
            XCTAssertEqual(block.control?.pacing, .controlled)
            XCTAssertEqual(block.takes.filter { !$0.discarded }.count, 18)
            XCTAssertTrue(ControlledSignalComparison.qualities(block).allSatisfy(\.passes))
            XCTAssertEqual(block.report?.reference, comparison.reference)
            XCTAssertTrue(block.report?.conditions.allSatisfy { $0.plannedTakes == 9 && $0.evaluated == 9 } == true)
        }
        XCTAssertNil(result.blocks[0].reseatedAt); XCTAssertNotNil(result.blocks[1].reseatedAt)
        XCTAssertNotEqual(result.blocks[0].session?.id, result.blocks[1].session?.id)
        XCTAssertEqual(try Data(contentsOf: root.appending(path: "frozen-signal-reference.json")), referenceBytes)
        let restored = SignalCheckWorkflow(storeURL: root.appending(path: "signal.json"))
        XCTAssertEqual(restored.comparison?.id, result.id)
        XCTAssertEqual(restored.comparison?.blocks.count, 2)
    }

    @MainActor func testTerminationSavesCompletedControlledTakesAndRecoversInterruptedBlock() throws {
        let (workflow, root, clock) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let comparison = try XCTUnwrap(workflow.comparison), target = try XCTUnwrap(comparison.target)
        let recorder = SessionRecorder(root: root.appending(path: "raw")), session = try XCTUnwrap(recorder.start(audioSink: nil))
        workflow.begin(session: session, audioDevice: nil, seed: comparison.seed, recorder: recorder,
            reference: comparison.reference, control: .init(testID: comparison.id, block: .unchanged, target: target, pacing: .controlled, blockConfirmedAt: Date()))
        workflow.beginCapture(at: clock.now)
        workflow.ingest(MotionSample(t: clock.now + 0.1))
        workflow.endCapture(at: clock.now + 3.4, reason: .scheduled)
        workflow.next()
        workflow.beginCapture(at: clock.now + 4)
        workflow.prepareForTermination()
        XCTAssertFalse(recorder.isRecording)
        let restored = SignalCheckWorkflow(storeURL: root.appending(path: "signal.json"))
        XCTAssertEqual(restored.comparison?.blocks.count, 1)
        XCTAssertEqual(restored.comparison?.blocks.first?.takes.count, 1)
        XCTAssertFalse(restored.controlledActive)
    }

    func testComparisonRejectsChangedTargetsPacingAndReferenceIdentity() throws {
        let target = try XCTUnwrap(CameraPoseTarget.capture(readings(start: 100, seconds: 1.6), now: 101.6))
        let source = SignalCheckTestData.source(), reference = try FrozenSignalReference.freeze(source)
        var comparison = ControlledSignalComparison(reference: try reference.link(), seed: source.seed)
        comparison.target = target
        var run = source; run.reference = comparison.reference
        run.control = .init(testID: comparison.id, block: .unchanged, target: target, pacing: .controlled, blockConfirmedAt: Date())
        comparison.blocks = [run]; try comparison.validate()
        comparison.pacing = .init(leadIn: 0, wordSeconds: 1, tail: 0)
        XCTAssertThrowsError(try comparison.validate())
        comparison.pacing = .controlled
        comparison.target = .init(id: UUID(), cameraID: target.cameraID, pose: neutral, angleTolerance: 5, positionTolerance: 0.05, sizeTolerance: 0.12)
        XCTAssertThrowsError(try comparison.validate())
    }
}

private final class TestClock { var now = ProcessInfo.processInfo.systemUptime }
