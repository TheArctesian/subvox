import XCTest
@testable import Subvox

final class CameraMotionCompensationTests: XCTestCase {
    private let testID=UUID()
    private let target = CameraPoseTarget(id: UUID(), cameraID: "camera", pose: .init(yaw: 0,pitch: 0,roll: 0,x: 0.5,y: 0.5,size: 0.3),
        angleTolerance: 5,positionTolerance: 0.05,sizeTolerance: 0.12,estimator: "apple-vision-face-landmarks-v1")

    private func mouth() -> CameraMouthObservation {
        let outer:[CameraLandmarkPoint]=[.init(x:0.4,y:0.4),.init(x:0.5,y:0.43),.init(x:0.6,y:0.4),.init(x:0.5,y:0.37)]
        let inner:[CameraLandmarkPoint]=[.init(x:0.42,y:0.4),.init(x:0.5,y:0.41),.init(x:0.58,y:0.4),.init(x:0.5,y:0.39)]
        return CameraMouthObservation.measure(outer:outer,inner:inner,leftEye:.init(x:0.4,y:0.6),rightEye:.init(x:0.6,y:0.6),imageWidth:1000,imageHeight:1000,confidence:0.9)!
    }
    private func fixture(start:Double=100, block:ControlledSignalBlock = .unchanged, headPhase:Double=0,seed:UInt64=42) -> SignalCheckRun {
        var run=SignalCheckRun(seed:seed)
        run.session = .init(id:UUID(),directory:URL(fileURLWithPath:"/tmp/camera-compensation-fixture"),startUptime:start)
        run.startedAt=Date();run.endedAt=Date();run.reseatedAt=block == .reseated ? Date() : nil
        run.control = .init(testID:testID,block:block,target:target,pacing:.controlled,blockConfirmedAt:Date(),motionPolicy:.naturalMovement)
        let lips=mouth()
        for (index,prompt) in SignalCheckRun.plan(seed:seed).enumerated() {
            let begin=start+Double(index)*9, cue=begin+3, phase=Double(index)*0.7+headPhase
            let camera=(0...220).map { i -> CameraPoseReading in
                let t=begin-0.3+Double(i)/30, u=t-begin
                let pose=CameraHeadPose(yaw:12*sin(u*0.9+phase),pitch:8*cos(u*1.3+phase),roll:5*sin(u*0.7+phase),
                    x:0.5+0.04*sin(u*0.5+phase),y:0.5+0.03*cos(u*1.1+phase),size:0.3*exp(0.1*sin(u*0.4+phase)))
                return .init(t:t,cameraID:target.cameraID,pose:pose,usedReceiptTime:false,mouth:lips,estimator:target.estimator)
            }
            run.control?.readings += camera
            let timeline=CameraHeadTimeline(camera,cameraID:target.cameraID,estimator:target.estimator)
            let samples=(0...320).map { i -> MotionSample in
                let t=begin+Double(i)/50, x=timeline.features(at:t)!, u=t-cue
                var rotation=[0.7*x[0]+0.2*x[1]+x[3], -0.3*x[0]+0.8*x[2]+x[4], 0.6*x[1]+0.2*x[2]+x[5]]
                if u>=0 {
                    rotation[0] += 0.035*sin(u*(3+Double(prompt.sentence)*3))
                    rotation[1] += 0.025*cos(u*(5+Double(prompt.sentence)*2))
                    rotation[2] += 0.015*sin(u*7)
                }
                return .init(t:t,rotation:(rotation[0],rotation[1],rotation[2]))
            }
            run.takes.append(.init(prompt:prompt,recording:.init(condition:prompt.condition,samples:samples,cueStart:cue,cueEnd:cue+3.4),endReason:.scheduled))
        }
        return run
    }

    func testCalibrationRemovesKnownHeadComponentWhilePreservingIndependentJawMotion() throws {
        let run=fixture(), correction=try CameraHeadCorrection.fit(run)
        XCTAssertGreaterThan(correction.heldOutReduction,0.9)
        let take=run.takes[0], corrected=correction.correct(take.recording.cueSamples,readings:run.control!.readings)
        XCTAssertEqual(corrected.count,take.recording.cueSamples.count)
        var rawError=0.0,correctedError=0.0
        for (original,sample) in zip(take.recording.cueSamples,corrected) {
            let u=sample.t-take.recording.cueStart, expected=0.035*sin(u*(3+Double(take.prompt.sentence)*3))
            rawError += pow(original.rx-expected,2);correctedError += pow(sample.rx-expected,2)
            XCTAssertEqual(sample.ax,original.ax);XCTAssertEqual(sample.t,original.t)
        }
        // Ridge regularization can leave some head component: require a substantial reduction,
        // then independently verify that adding jaw motion survives the fixed subtraction.
        XCTAssertLessThan(correctedError,rawError*0.2)
        XCTAssertGreaterThan(corrected.map { abs($0.rx) }.max()!,0.025)
        let headOnly=take.recording.cueSamples.map { s -> MotionSample in
            var result=s;let u=s.t-take.recording.cueStart
            result.rx -= 0.035*sin(u*(3+Double(take.prompt.sentence)*3));return result
        }
        let headResidual=correction.correct(headOnly,readings:run.control!.readings)
        for (sample,head) in zip(corrected,headResidual) {
            let u=sample.t-take.recording.cueStart
            XCTAssertEqual(sample.rx-head.rx,0.035*sin(u*(3+Double(take.prompt.sentence)*3)),accuracy:1e-10)
        }
    }

    func testCalibrationCannotLearnFromWordsOrLipFeatures() throws {
        let run=fixture(), before=try CameraHeadCorrection.fit(run)
        var changed=run
        for i in changed.takes.indices {
            changed.takes[i].prompt.sentence=(changed.takes[i].prompt.sentence+1)%3
            for j in changed.takes[i].recording.samples.indices where changed.takes[i].recording.samples[j].t>=changed.takes[i].recording.cueStart {
                changed.takes[i].recording.samples[j].rx += 100
            }
        }
        for i in changed.control!.readings.indices { changed.control?.readings[i].mouth=nil }
        let after=try CameraHeadCorrection.fit(changed)
        XCTAssertEqual(before.coefficients,after.coefficients)
        XCTAssertEqual(before.heldOutReduction,after.heldOutReduction)
    }

    func testTrackingGapsOtherCamerasClockFallbackAndAngleWrapAreHandled() throws {
        let run=fixture(), correction=try CameraHeadCorrection.fit(run), take=run.takes[0]
        var camera=run.control!.readings
        let middle=take.recording.cueStart+1.5
        camera.removeAll { abs($0.t-middle)<0.3 }
        let corrected=correction.correct(take.recording.cueSamples,readings:camera)
        XCTAssertFalse(CameraMotionReference.covered(corrected,recording:take.recording))
        XCTAssertFalse(corrected.contains { abs($0.t-middle)<0.25 })
        let wrong=camera.map { r -> CameraPoseReading in var x=r;x.cameraID="different";return x }
        XCTAssertTrue(correction.correct(take.recording.cueSamples,readings:wrong).isEmpty)
        let fallback=camera.map { r -> CameraPoseReading in var x=r;x.usedReceiptTime=true;return x }
        XCTAssertTrue(correction.correct(take.recording.cueSamples,readings:fallback).isEmpty)
        let wrapped=(0...30).map { i -> CameraPoseReading in
            var pose=target.pose;pose.yaw=179+Double(i)/15;if pose.yaw>180 { pose.yaw-=360 }
            return .init(t:Double(i)/30,cameraID:target.cameraID,pose:pose,estimator:target.estimator)
        }
        let feature=try XCTUnwrap(CameraHeadTimeline(wrapped,cameraID:target.cameraID,estimator:target.estimator).features(at:0.5))
        XCTAssertEqual(feature[0],2 * .pi/180,accuracy:1e-7)
    }

    func testUninformativeOrUnpredictiveCalibrationNeverProducesCorrection() throws {
        var still=fixture()
        for i in still.control!.readings.indices { still.control?.readings[i].pose=target.pose }
        XCTAssertThrowsError(try CameraHeadCorrection.fit(still))
        let rawReference=try CameraMotionReference.freeze(still)
        XCTAssertNil(rawReference.correction);XCTAssertNotNil(rawReference.correctionNote)
        XCTAssertEqual(rawReference.rawTemplates.count,18)
        var inconsistent=fixture()
        for i in inconsistent.takes.indices where i%2==1 {
            for j in inconsistent.takes[i].recording.samples.indices {
                inconsistent.takes[i].recording.samples[j].rx *= -1
                inconsistent.takes[i].recording.samples[j].ry *= -1
                inconsistent.takes[i].recording.samples[j].rz *= -1
            }
        }
        XCTAssertThrowsError(try CameraHeadCorrection.fit(inconsistent))
    }

    func testNewReferenceIsFrozenBeforeReseatingWithMatchedRawAndCorrectedTimestamps() throws {
        let a=fixture(), reference=try CameraMotionReference.freeze(a), b=fixture(start:400,block:.reseated,headPhase:1)
        XCTAssertNotNil(reference.correction)
        for (raw,corrected) in zip(reference.rawTemplates,reference.correctedTemplates) { XCTAssertEqual(raw.samples.map(\.t),corrected.samples.map(\.t)) }
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let before=try encoder.encode(reference)
        let scores=try reference.evaluate(b)
        XCTAssertEqual(scores.count,2)
        for score in scores { XCTAssertEqual(score.rawEvaluated,9);XCTAssertEqual(score.correctedEvaluated,9);XCTAssertGreaterThanOrEqual(score.correctedCorrect!,8) }
        XCTAssertEqual(try encoder.encode(reference),before)
        XCTAssertThrowsError(try reference.evaluate(a))
        XCTAssertThrowsError(try CameraMotionReference.freeze(b))
        var partial=a;partial.takes.removeLast()
        XCTAssertThrowsError(try CameraMotionReference.freeze(partial))
        let restored=try JSONDecoder().decode(CameraMotionReference.self,from:before)
        XCTAssertEqual(restored.correction?.coefficients,reference.correction?.coefficients)
        XCTAssertEqual(try restored.evaluate(b).map(\.correctedCorrect),scores.map(\.correctedCorrect))
    }

    func testNaturalQualityAllowsPoseChangesButStillRejectsTrackingLossAndKeepsLegacyRules() throws {
        var run=fixture();run.takes=[run.takes[0]]
        XCTAssertTrue(ControlledSignalComparison.qualities(run)[0].passes)
        XCTAssertLessThan(ControlledSignalComparison.qualities(run)[0].withinTarget,0.9)
        run.control?.motionPolicy=nil
        XCTAssertFalse(ControlledSignalComparison.qualities(run)[0].passes)
        run.control?.motionPolicy = .naturalMovement
        for i in run.control!.readings.indices { run.control?.readings[i].mouth=nil }
        XCTAssertFalse(ControlledSignalComparison.qualities(run)[0].passes)
    }

    @MainActor func testNaturalWorkflowStartsWhileHeadMovesAndOnlyBlocksMissingTracking() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"subvox-natural-\(UUID())")
        defer { try? FileManager.default.removeItem(at:root) }
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        try FrozenSignalReference.freeze(SignalCheckTestData.source()).saveNew(to:root.appending(path:"frozen-signal-reference.json"))
        var now=100.0
        let workflow=SignalCheckWorkflow(storeURL:root.appending(path:"signal.json"),clock:{ now })
        workflow.prepareControlled(policy:.naturalMovement)
        XCTAssertTrue(workflow.allowsHeadMovement);XCTAssertFalse(workflow.canStartControlledBlock)
        for i in 0...8 {
            now=100+Double(i)/30
            var pose=target.pose;pose.yaw=Double(i)*5;pose.x += Double(i)*0.01
            workflow.ingest(MotionSample(t:now))
            workflow.camera.ingest(.init(t:now,cameraID:target.cameraID,pose:pose,mouth:mouth(),estimator:target.estimator))
        }
        XCTAssertNotNil(workflow.comparison?.target)
        XCTAssertTrue(workflow.poseReady);XCTAssertTrue(workflow.canStartControlledBlock)
        XCTAssertFalse(workflow.comparison!.target!.accepts(workflow.camera.latest!))
        workflow.camera.ingest(.init(t:now+0.01,cameraID:target.cameraID,estimator:target.estimator));now += 0.01
        XCTAssertFalse(workflow.canStartControlledBlock)
        workflow.prepareForTermination()
    }

    @MainActor func testNaturalWorkflowPersistsCorrectionAfterAAndKeepsItFixedThroughB() async throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"subvox-natural-cycle-\(UUID())")
        defer { try? FileManager.default.removeItem(at:root) }
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let original=try FrozenSignalReference.freeze(SignalCheckTestData.source())
        let referenceURL=root.appending(path:"frozen-signal-reference.json")
        try original.saveNew(to:referenceURL)
        let originalBytes=try Data(contentsOf:referenceURL)
        var now=100.0
        let workflow=SignalCheckWorkflow(storeURL:root.appending(path:"signal.json"),clock:{ now })
        workflow.prepareControlled(policy:.naturalMovement)
        let c=try XCTUnwrap(workflow.comparison), a=fixture(start:110,seed:c.seed)
        workflow.camera.ingest(.init(t:now,cameraID:target.cameraID,pose:target.pose,mouth:mouth(),estimator:target.estimator))
        let savedTarget=try XCTUnwrap(workflow.comparison?.target)
        let recorder=SessionRecorder(root:root.appending(path:"raw"))
        func begin(_ block:ControlledSignalBlock) throws {
            let session=try XCTUnwrap(recorder.start(audioSink:nil,cameraPoints:true)),date=Date()
            workflow.begin(session:session,audioDevice:nil,seed:c.seed,recorder:recorder,reference:c.reference,
                reseatedAt:block == .reseated ? date : nil,
                control:.init(testID:c.id,block:block,target:savedTarget,pacing:.controlled,blockConfirmedAt:date,motionPolicy:.naturalMovement))
        }
        func acquire(_ data:SignalCheckRun) {
            for (index,take) in data.takes.enumerated() {
                now=take.recording.cueStart
                for reading in data.control!.readings where reading.t>=now-0.12 && reading.t<=now { workflow.camera.ingest(reading) }
                workflow.ingest(MotionSample(t:now))
                workflow.beginCapture(at:now)
                XCTAssertEqual(workflow.phase,.recording)
                for sample in take.recording.samples { workflow.ingest(sample) }
                let beforeEnd=data.control!.readings.filter { $0.t>=now-3 && $0.t<=now+3.3 }
                for reading in beforeEnd { workflow.ingestPose(reading) }
                now += 3.4
                workflow.endCapture(at:now,reason:.scheduled)
                // Camera frames captured near END arrive after the take has ended.
                for reading in data.control!.readings where reading.t>beforeEnd.last!.t && reading.t<=now+0.2 { workflow.ingestPose(reading) }
                if index+1<data.takes.count { workflow.next() }
            }
        }
        func waitFor(_ phase:SignalCheckWorkflow.Phase) async throws {
            for _ in 0..<1000 {
                if workflow.phase==phase { return }
                try await Task.sleep(for:.milliseconds(5))
            }
            XCTFail("Did not reach \(phase): \(workflow.error ?? "none")")
        }
        // Preflight readiness is tracking-only; no return to a saved head angle.
        for i in 1...4 { now += 0.03;workflow.camera.ingest(.init(t:now,cameraID:target.cameraID,pose:target.pose,mouth:mouth(),estimator:target.estimator)) }
        workflow.ingest(MotionSample(t:now))
        try begin(.unchanged);acquire(a);workflow.next()
        try await waitFor(.controlBreak)
        let fitted=try XCTUnwrap(workflow.comparison?.cameraReference)
        XCTAssertNotNil(fitted.correction,fitted.correctionNote ?? "Missing correction without an explanation")
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let fittedBytes=try encoder.encode(fitted)
        now += 31
        let b=fixture(start:now+1,block:.reseated,headPhase:1,seed:c.seed)
        for i in 0...4 { workflow.camera.ingest(.init(t:now-0.12+Double(i)*0.03,cameraID:target.cameraID,pose:target.pose,mouth:mouth(),estimator:target.estimator)) }
        workflow.ingest(MotionSample(t:now))
        XCTAssertTrue(workflow.canStartControlledBlock)
        try begin(.reseated);acquire(b);workflow.next()
        try await waitFor(.finished)
        let comparison=try XCTUnwrap(workflow.comparison)
        try comparison.validate()
        XCTAssertEqual(comparison.blocks.count,2)
        XCTAssertEqual(comparison.cameraScores?.count,2)
        XCTAssertTrue(comparison.cameraScores!.allSatisfy { $0.correctedEvaluated==9 })
        XCTAssertEqual(try encoder.encode(comparison.cameraReference!),fittedBytes)
        XCTAssertEqual(try Data(contentsOf:referenceURL),originalBytes)
        let restored=SignalCheckWorkflow(storeURL:root.appending(path:"signal.json"))
        XCTAssertEqual(restored.comparison?.cameraReference?.id,fitted.id)
        XCTAssertEqual(restored.comparison?.cameraScores?.map(\.correctedCorrect),comparison.cameraScores?.map(\.correctedCorrect))
    }
}
