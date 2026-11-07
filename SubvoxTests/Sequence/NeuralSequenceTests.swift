import CryptoKit
import XCTest
@testable import Subvox

final class NeuralSequenceTests:XCTestCase {
    @MainActor func testNonSpeechTimerSavesUsableTakeAndContinueAdvances() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"negative-timer-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var now=100.0
        let workflow=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var sample=MotionSample(t:now);sample.sensorLocation=1;workflow.ingest(sample)
        workflow.mode = .nonSpeech;workflow.startVisit(data:SentenceTrainingData(),audio:AudioService(),recorder:recorder)
        XCTAssertEqual(workflow.captureDuration,8)
        workflow.beginCapture(at:now)
        for i in 0...400 {
            now=100+Double(i)/50;sample.t=now;workflow.ingest(sample);recorder.record(sample)
        }
        workflow.finishAtTimeLimit()
        XCTAssertEqual(workflow.phase,.ended);XCTAssertTrue(workflow.canAdvance);XCTAssertFalse(workflow.requiresRedo)
        let take=try XCTUnwrap(workflow.visit?.takes.last)
        XCTAssertNil(take.discardedReason);XCTAssertTrue(take.text.isEmpty)
        XCTAssertEqual(take.recording.cueEnd-take.recording.cueStart,8,accuracy:0.001)
        workflow.next();XCTAssertEqual(workflow.phase,.ready);XCTAssertEqual(workflow.index,1)
        workflow.stop()
        let saved=try JSONDecoder().decode(NeuralVisit.self,from:Data(contentsOf:workflow.collectionDirectory!.appending(path:"session.json")))
        XCTAssertNil(saved.takes.last?.discardedReason)
        XCTAssertEqual(workflow.resumableNonSpeechCount,15)
    }

    @MainActor func testSentenceTimeoutRequiresRedoAndContinueKeepsOriginalReason() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"sentence-timer-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var now=100.0,data=SentenceTrainingData(),rng=SplitMix64(seed:4)
        data.add(SentencePrompt.bundled,using:&rng)
        let workflow=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var sample=MotionSample(t:now);sample.sensorLocation=1;workflow.ingest(sample)
        workflow.startVisit(data:data,audio:AudioService(),recorder:recorder)
        XCTAssertEqual(workflow.captureDuration,35)
        workflow.beginCapture(at:now);now+=35;workflow.finishAtTimeLimit()
        let reason=try XCTUnwrap(workflow.takeIssue)
        XCTAssertTrue(reason.contains("Timed out"));XCTAssertTrue(workflow.requiresRedo);XCTAssertFalse(workflow.canAdvance)
        workflow.next();XCTAssertEqual(workflow.index,0);XCTAssertEqual(workflow.error,reason)
        workflow.redo();XCTAssertEqual(workflow.phase,.ready);XCTAssertNil(workflow.takeIssue)
        XCTAssertTrue(workflow.canAdvance);XCTAssertNotNil(workflow.visit?.takes.last?.discardedReason)
        workflow.stop()
    }

    @MainActor func testNonSpeechSignalInterruptionCannotBeAcceptedByTimerOrContinue() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"negative-gap-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var now=100.0
        let workflow=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var sample=MotionSample(t:now);sample.sensorLocation=1;workflow.ingest(sample)
        workflow.mode = .nonSpeech;workflow.startVisit(data:SentenceTrainingData(),audio:AudioService(),recorder:recorder)
        workflow.beginCapture(at:now);workflow.ingest(sample)
        now+=0.2;sample.t=now;workflow.ingest(sample)
        let reason=try XCTUnwrap(workflow.takeIssue)
        workflow.finishAtTimeLimit();workflow.next()
        XCTAssertTrue(workflow.requiresRedo);XCTAssertFalse(workflow.canAdvance);XCTAssertEqual(workflow.error,reason)
        XCTAssertEqual(workflow.index,0);XCTAssertEqual(workflow.visit?.takes.count,1)
        workflow.stop()
    }

    @MainActor func testStoppedNonSpeechSessionResumesOnlyUnfinishedCellsWithoutRewritingOriginal() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"resume-negative-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var now=100.0
        let original=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var sample=MotionSample(t:now);sample.sensorLocation=1;original.ingest(sample)
        original.mode = .nonSpeech;original.startVisit(data:SentenceTrainingData(),audio:AudioService(),recorder:recorder)
        let item=try XCTUnwrap(original.visit?.planned.first)
        let take=RepresentationFixture.take(.init(text:item.text),index:0,session:original.visit!.session.id,condition:item.condition)
        now=take.recording.cueStart;original.beginCapture(at:now)
        for sample in take.recording.samples {original.ingest(sample);recorder.record(sample)}
        now=take.recording.cueEnd;original.next();original.stop()
        let file=original.collectionDirectory!.appending(path:"session.json"),before=try Data(contentsOf:file)
        let resumed=SequenceLearningWorkflow(root:root,clock:{now})
        XCTAssertEqual(resumed.resumableNonSpeechCount,15)
        resumed.resumeNonSpeech(audio:AudioService(),recorder:recorder)
        XCTAssertEqual(resumed.phase,.ready);XCTAssertEqual(resumed.visit?.planned.count,15)
        resumed.next();XCTAssertEqual(resumed.phase,.ready,"Offline Ready must not start a recording cue")
        XCTAssertNotNil(resumed.error)
        XCTAssertEqual(resumed.visit?.continuedFrom,original.visit?.id)
        XCTAssertNotEqual(resumed.visit?.session.id,original.visit?.session.id)
        XCTAssertEqual(resumed.visit?.planned.first?.condition,.silent)
        XCTAssertEqual(resumed.visit?.movementGuideVersion,"illustrated-v1")
        XCTAssertEqual(resumed.displayIndex,2);XCTAssertEqual(resumed.displayTotal,16)
        XCTAssertEqual(resumed.currentMovement,.nodAndTurn)
        XCTAssertEqual(try Data(contentsOf:file),before)
        resumed.stop()
        XCTAssertEqual(resumed.resumableNonSpeech?.id,resumed.visit?.id,"Resume must be available without relaunching the app")
        XCTAssertEqual(resumed.resumableNonSpeechCount,15)
        let loaded=SequenceLearningWorkflow(root:root,clock:{now})
        XCTAssertEqual(loaded.resumableNonSpeech?.id,resumed.visit?.id,"Do not reopen an ancestor after it has a continuation")
        XCTAssertEqual(try Data(contentsOf:file),before)
    }
    @MainActor func testNonSpeechClipsSaveEmptyLabelsAndMotionOnlyModalities() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"negative-visit-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var now=100.0
        let workflow=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var sample=MotionSample(t:now);sample.sensorLocation=1;workflow.ingest(sample)
        workflow.mode = .nonSpeech;workflow.negativeRole = .validation
        workflow.startVisit(data:SentenceTrainingData(),audio:AudioService(),recorder:recorder)
        XCTAssertEqual(workflow.visit?.role,"negativeDevelopment")
        let planned=try XCTUnwrap(workflow.visit?.planned);XCTAssertEqual(planned.count,16)
        for (i,item) in planned.enumerated() {
            let take=RepresentationFixture.take(.init(text:item.text),index:i,session:workflow.visit!.session.id,condition:item.condition)
            now=take.recording.cueStart;workflow.beginCapture(at:now)
            for sample in take.recording.samples {workflow.ingest(sample);recorder.record(sample)}
            now=take.recording.cueEnd;workflow.next();workflow.next()
        }
        XCTAssertEqual(workflow.visit?.modalities,["AirPods motion"])
        XCTAssertTrue(workflow.visit?.complete==true)
        XCTAssertNil(workflow.resumableNonSpeech,"A completed movement collection must not offer Resume")
        XCTAssertTrue(workflow.visit!.takes.allSatisfy {$0.text.isEmpty && $0.camera.isEmpty})
        XCTAssertEqual(Set(workflow.visit!.takes.map(\.sentenceID)).count,4)
        XCTAssertFalse(FileManager.default.fileExists(atPath:workflow.visit!.session.directory.appending(path:"audio.caf").path))
    }
    @MainActor func testOlderDatasetRemainsLoadedButCannotStartCrossValidationTraining() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"neural-schema-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        let dataset=root.appending(path:"old-dataset")
        try FileManager.default.createDirectory(at:dataset,withIntermediateDirectories:true)
        try Data("{\"schema\":1}".utf8).write(to:dataset.appending(path:"manifest.json"))
        try JSONEncoder().encode(["dataset":dataset.path,"run":""]).write(to:root.appending(path:"latest-ui.json"))
        let workflow=SequenceLearningWorkflow(root:root)
        XCTAssertEqual(workflow.datasetURL?.path,dataset.path);XCTAssertFalse(workflow.datasetCompatible)
        workflow.train()
        XCTAssertEqual(workflow.phase,.idle);XCTAssertNotNil(workflow.error)
        XCTAssertEqual(try Data(contentsOf:dataset.appending(path:"manifest.json")),Data("{\"schema\":1}".utf8))
    }
    @MainActor func testCurrentPythonDatasetSchemaCanStartTraining() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"neural-schema-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        for schema in [2,3] {
            let dataset=root.appending(path:"dataset-\(schema)")
            try FileManager.default.createDirectory(at:dataset,withIntermediateDirectories:true)
            try Data("{\"schema\":\(schema)}".utf8).write(to:dataset.appending(path:"manifest.json"))
            try JSONEncoder().encode(["dataset":dataset.path,"run":""]).write(to:root.appending(path:"latest-ui.json"))
            XCTAssertTrue(SequenceLearningWorkflow(root:root).datasetCompatible,"schema \(schema)")
        }
    }
    @MainActor func testMotionOnlyNeuralVisitKeepsAllSentenceGroupsOutOfFinalTestAndSavesBothConditions() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"neural-visit-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var data=SentenceTrainingData(),rng=SplitMix64(seed:81)
        data.add(SentencePrompt.bundled,using:&rng)
        let roles=data.roles
        var now=100.0
        let workflow=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw")),audio=AudioService()
        var latest=MotionSample(t:now);latest.sensorLocation=1;workflow.ingest(latest)
        workflow.startVisit(data:data,audio:audio,recorder:recorder)
        let plan=try XCTUnwrap(workflow.visit?.planned)
        XCTAssertEqual(plan.count,16)
        XCTAssertTrue(plan.allSatisfy {roles[SentencePrompt.identity($0.text)] == .training})
        for (i,item) in plan.enumerated() {
            let fixture=RepresentationFixture.take(.init(text:item.text),index:i,session:workflow.visit!.session.id,condition:item.condition)
            now=fixture.recording.cueStart;workflow.beginCapture(at:now)
            for sample in fixture.recording.samples {workflow.ingest(sample);recorder.record(sample)}
            now=fixture.recording.cueEnd;workflow.next()
            XCTAssertEqual(workflow.phase,.ended)
            workflow.next()
        }
        XCTAssertEqual(workflow.phase,.idle)
        let visit=try XCTUnwrap(workflow.visit)
        XCTAssertTrue(visit.complete);XCTAssertEqual(visit.takes.count,16)
        XCTAssertTrue(visit.takes.allSatisfy {$0.camera.isEmpty && $0.wordFocusGuide == .comfortable})
        let meta=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:visit.session.directory.appending(path:"meta.json"))) as? [String:Any])
        XCTAssertEqual(meta["audioRecordingEnabled"] as? Bool,false);XCTAssertEqual(meta["videoRecordingEnabled"] as? Bool,false)
        XCTAssertEqual(meta["cameraPointSamples"] as? Int,0)
        XCTAssertFalse(FileManager.default.fileExists(atPath:visit.session.directory.appending(path:"audio.caf").path))
        XCTAssertEqual(data.roles,roles);XCTAssertNil(data.reservedUnlockedAt)
        let restored=try JSONDecoder().decode(NeuralVisit.self,from:Data(contentsOf:workflow.collectionDirectory!.appending(path:"session.json")))
        XCTAssertEqual(restored.id,visit.id);XCTAssertEqual(restored.takes.count,16)
    }

    @MainActor func testNeuralRedoResetsFocusAndPreservesDiscardedAttempt() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"neural-redo-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var data=SentenceTrainingData(),rng=SplitMix64(seed:4);data.add(SentencePrompt.bundled,using:&rng)
        var now=100.0
        let workflow=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var sample=MotionSample(t:now);sample.sensorLocation=1;workflow.ingest(sample)
        workflow.startVisit(data:data,audio:AudioService(),recorder:recorder)
        workflow.beginCapture(at:now);now += 1.1
        XCTAssertEqual(workflow.activeWordIndex,1)
        workflow.redo()
        XCTAssertEqual(workflow.phase,.ready);XCTAssertNil(workflow.activeWordIndex)
        XCTAssertEqual(workflow.index,0);XCTAssertNotNil(workflow.visit?.takes[0].discardedReason)
        workflow.beginCapture(at:now)
        XCTAssertNil(workflow.activeWordIndex)
        workflow.stop()
    }

    @MainActor func testExplicitTeacherVisitsStayInOneRoleAndNormalPairedPlanCanBeRestored() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"teacher-roles-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        let workflow=SentenceTrainingWorkflow(storeURL:root.appending(path:"sentences.json"),corpus:SentencePrompt.bundled)
        let original=workflow.plan
        workflow.collectionRole = .training
        XCTAssertFalse(workflow.plan.isEmpty)
        XCTAssertTrue(workflow.plan.flatMap {$0}.allSatisfy {workflow.data.roles[$0.id] == .training})
        workflow.collectionRole = .validation
        XCTAssertFalse(workflow.plan.isEmpty)
        XCTAssertTrue(workflow.plan.flatMap {$0}.allSatisfy {workflow.data.roles[$0.id] == .validation})
        workflow.collectionRole=nil
        XCTAssertEqual(workflow.plan,original)
    }

    @MainActor func testNoNeuralModelStartsInferenceAndIdleStopNeverWritesLoadedData() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"neural-no-model-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var now=100.0
        let workflow=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var sample=MotionSample(t:now);sample.sensorLocation=1;workflow.ingest(sample)
        workflow.mode = .inference
        workflow.startVisit(data:SentenceTrainingData(),audio:AudioService(),recorder:recorder)
        XCTAssertFalse(recorder.isRecording);XCTAssertNil(workflow.visit);XCTAssertNotNil(workflow.error)
        workflow.stop();now += 1
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.path))
    }
    @MainActor func testCameraModelsRequireTheCameraAndMotionModelsDoNot() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"neural-fusion-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        let run=root.appending(path:"runs/fusion")
        let weights=Data("frozen".utf8),fingerprint=SHA256.hash(data:weights).map {String(format:"%02x",$0)}.joined()
        for (experiment,modalities) in [("fused","fused"),("imu","imu")] {
            for slug in ["spoken","mouthed"] {
                let model=run.appending(path:"\(experiment)-\(slug)")
                try FileManager.default.createDirectory(at:model,withIntermediateDirectories:true)
                try weights.write(to:model.appending(path:"weights.pt"))
                try JSONSerialization.data(withJSONObject:["modalities":modalities,"weights_sha256":fingerprint]).write(to:model.appending(path:"model.json"))
            }
        }
        try JSONEncoder().encode(["dataset":"","run":run.path]).write(to:root.appending(path:"latest-ui.json"))
        var now=100.0
        let workflow=SequenceLearningWorkflow(root:root,clock:{now}),recorder=SessionRecorder(root:root.appending(path:"raw"))
        defer {recorder.stop()}
        workflow.mode = .inference;workflow.inferenceCondition = .silent;workflow.expectedText="pack my blue bag"
        workflow.experiment="fused"
        XCTAssertEqual(workflow.modalities(condition:.silent),"fused")
        XCTAssertTrue(workflow.needsCamera)
        var sample=MotionSample(t:now);sample.sensorLocation=1;workflow.ingest(sample)
        workflow.startVisit(data:SentenceTrainingData(),audio:AudioService(),recorder:recorder)
        XCTAssertNil(workflow.visit,"A camera model never silently runs on motion alone")
        XCTAssertFalse(recorder.isRecording)
        XCTAssertNotNil(workflow.error)

        workflow.experiment="imu"
        XCTAssertFalse(workflow.needsCamera)
        now+=0.05;sample.t=now;workflow.ingest(sample)
        workflow.startVisit(data:SentenceTrainingData(),audio:AudioService(),recorder:recorder)
        XCTAssertEqual(workflow.visit?.modalities,["AirPods motion"])
        workflow.mode = .training
        XCTAssertFalse(workflow.needsCamera,"Training collection stays motion-only")
        workflow.stop()
    }
    func testCameraReadingsWithFacePointsRoundTripAndOlderReadingsStillDecode() throws {
        let point=CameraLandmarkPoint(x:0.4,y:0.3)
        var reading=CameraPoseReading(t:1,cameraID:"front")
        reading.face=CameraFaceLandmarks(faceContour:[point,point,point],noseCrest:[point])
        let decoded=try JSONDecoder().decode(CameraPoseReading.self,from:JSONEncoder().encode(reading))
        XCTAssertEqual(decoded.face,reading.face)
        let older=try JSONDecoder().decode(CameraPoseReading.self,from:Data("{\"t\":2,\"cameraID\":\"front\"}".utf8))
        XCTAssertNil(older.face)
        XCTAssertFalse(String(decoding:try JSONEncoder().encode(older),as:UTF8.self).contains("face"),"Older readings re-encode unchanged")
        XCTAssertFalse(CameraFaceLandmarks(faceContour:[point],noseCrest:[point]).valid)
    }
}
