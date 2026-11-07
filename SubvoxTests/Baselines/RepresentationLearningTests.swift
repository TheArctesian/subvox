import XCTest
@testable import Subvox

enum RepresentationFixture {
    static let train=["pack bag","map cup","cat pack","bat map"].map {SentencePrompt(text:$0)}
    static let test=["cap bag","mat cup","cap map","mat cat"].map {SentencePrompt(text:$0)}
    static let lexicon=Lexicon(entries:["pack":["P","AE1","K"],"bag":["B","AE1","G"],"map":["M","AE1","P"],"cup":["K","AH1","P"],
        "cat":["K","AE1","T"],"bat":["B","AE1","T"],"cap":["K","AE1","P"],"mat":["M","AE1","T"]])
    static func mouth(_ u:Double)->CameraMouthObservation {
        let outer:[CameraLandmarkPoint]=[.init(x:0.4,y:0.4),.init(x:0.5,y:0.43),.init(x:0.6,y:0.4),.init(x:0.5,y:0.37)]
        let inner:[CameraLandmarkPoint]=[.init(x:0.42,y:0.4),.init(x:0.5,y:0.41),.init(x:0.58,y:0.4),.init(x:0.5,y:0.39)]
        var m=CameraMouthObservation.measure(outer:outer,inner:inner,leftEye:.init(x:0.4,y:0.6),rightEye:.init(x:0.6,y:0.6),imageWidth:1000,imageHeight:1000,confidence:0.95)!
        m.opening=0.1+0.04*sin(u*5);m.width=0.7+0.03*cos(u*7);m.outerAspect=0.3+0.02*sin(u*9)
        return m
    }
    static func take(_ prompt:SentencePrompt,index:Int,session:UUID,source:Int=1,condition:SpeechCondition = .silent)->RepresentationTake {
        let start=100+Double(index)*10,cue=start+3,end=cue+2.4,frequency=Double(3+index%4)
        let samples=(0...270).map {i -> MotionSample in
            let t=start+Double(i)/50,u=max(0,t-cue)
            var s=MotionSample(t:t,accel:(0.02*sin(u*frequency),0.03*cos(u*frequency),0.02*sin(u*7)),rotation:(0.1*sin(u*frequency),0.12*cos(u*frequency),0.08*sin(u*5)))
            s.sensorLocation=source;return s
        }
        let camera=(0...177).map {i -> CameraPoseReading in
            let t=start-0.2+Double(i)/30
            return .init(t:t,cameraID:"fixture-camera",pose:nil,usedReceiptTime:false,mouth:mouth(max(0,t-cue)),imageWidth:1000,imageHeight:1000,estimator:"fixture-v1")
        }
        return .init(sentenceID:prompt.id,text:prompt.text,sessionID:session,repetition:index/4,recording:.init(condition:condition,samples:samples,cueStart:cue,cueEnd:end),camera:camera)
    }
    static func training()->[RepresentationTake] {
        let session=UUID()
        return [SpeechCondition.normal,.silent].flatMap {condition in
            (0..<8).map {take(train[$0%4],index:$0,session:session,condition:condition)}
        }
    }
    static func freeze(_ takes:[RepresentationTake]) throws -> RepresentationReference {
        try RepresentationReference.freeze(studyID:UUID(),takes:takes,training:train,evaluation:test,forbidden:[SentencePrompt.identity("reserved sentence")],
            profile:PronunciationProfile(),lexicon:lexicon,cameraID:"fixture-camera",estimator:"fixture-v1")
    }
}

final class RepresentationLearningTests:XCTestCase {
    @MainActor func testReloadRecoversOnlyMissingEmbeddedPolicyFromVerifiedFrozenReference() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"representation-compat-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        let takes=RepresentationFixture.training(),reference=try RepresentationFixture.freeze(takes)
        var embedded=reference;embedded.featurePolicy=nil
        let study=RepresentationStudy(id:reference.studyID,training:RepresentationFixture.train,evaluation:RepresentationFixture.test,
            forbiddenFinalSentences:reference.forbiddenFinalSentences,profile:reference.profile,takes:takes,
            reference:embedded,fingerprint:try reference.fingerprint())
        let dir=root.appending(path:study.id.uuidString)
        try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
        let bytes=try JSONEncoder().encode(study),frozen=try JSONEncoder().encode(reference)
        try bytes.write(to:root.appending(path:"latest.json"));try bytes.write(to:dir.appending(path:"study.json"))
        try frozen.write(to:dir.appending(path:"frozen-reference.json"))
        let restored=RepresentationWorkflow(root:root)
        XCTAssertTrue(restored.storageReady,restored.error ?? "")
        XCTAssertEqual(restored.phase,.reseat)
        XCTAssertEqual(restored.study?.reference?.featurePolicy,.observedTailV2)
        XCTAssertEqual(try restored.study?.reference?.fingerprint(),try reference.fingerprint())
        restored.stop()
        XCTAssertEqual(try Data(contentsOf:root.appending(path:"latest.json")),bytes)
        XCTAssertEqual(try Data(contentsOf:dir.appending(path:"study.json")),bytes)
        XCTAssertEqual(try Data(contentsOf:dir.appending(path:"frozen-reference.json")),frozen)
        var changed=study;changed.takes[0].recording.samples[0].ax += 1
        try JSONEncoder().encode(changed).write(to:root.appending(path:"latest.json"))
        XCTAssertFalse(RepresentationWorkflow(root:root).storageReady,"A changed training recording must still fail verification")
        changed=study;changed.reference?.heads[0].linguistic.mean[0] += 1
        try JSONEncoder().encode(changed).write(to:root.appending(path:"latest.json"))
        XCTAssertFalse(RepresentationWorkflow(root:root).storageReady,"Missing policy must not permit changed model coefficients")
        var changedFrozen=reference;changedFrozen.featurePolicy=nil
        try bytes.write(to:root.appending(path:"latest.json"))
        try JSONEncoder().encode(changedFrozen).write(to:dir.appending(path:"frozen-reference.json"))
        XCTAssertFalse(RepresentationWorkflow(root:root).storageReady,"The separate reference must match the original fingerprint")
    }

    func testDeliveryLagAtEndUsesOnlyObservedFramesWithoutRelaxingGapChecks() throws {
        var take=RepresentationFixture.take(RepresentationFixture.train[0],index:0,session:UUID())
        let end=take.recording.cueEnd
        take.recording.samples.removeAll {$0.t>end-0.06}
        XCTAssertNotNil(MotionSourceSafety.source(take.recording))
        XCTAssertNil(RepresentationFrames(take,cameraID:"fixture-camera",estimator:"fixture-v1",edgePolicy:.strictV1))
        let frames=try XCTUnwrap(RepresentationFrames(take,cameraID:"fixture-camera",estimator:"fixture-v1"))
        let supported=take.recording.samples.last!.t-take.recording.cueStart
        XCTAssertEqual(frames.raw.count,Int(floor(supported/0.04))+1)
        XCTAssertGreaterThanOrEqual(Double(frames.raw.count-1)*0.04,(take.recording.cueEnd-take.recording.cueStart)*0.9)
        take.recording.samples.removeAll {$0.t>end-0.2}
        XCTAssertNil(RepresentationFrames(take,cameraID:"fixture-camera",estimator:"fixture-v1"))
    }
    @MainActor func testClosingAnIdleLoadedStudyCannotOverwriteAnActiveWritersProgress() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"representation-reader-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var data=SentenceTrainingData()
        data.prompts=RepresentationFixture.train+RepresentationFixture.test
        for p in RepresentationFixture.train {data.roles[p.id] = .training}
        for p in RepresentationFixture.test {data.roles[p.id] = .validation}
        let writer=RepresentationWorkflow(root:root);writer.prepare(data:data)
        let reader=RepresentationWorkflow(root:root)
        var progress=try XCTUnwrap(writer.study)
        progress.takes.append(RepresentationFixture.take(progress.training[0],index:0,session:UUID()))
        let bytes=try JSONEncoder().encode(progress),url=root.appending(path:"latest.json")
        try bytes.write(to:url)
        reader.stop()
        XCTAssertEqual(try Data(contentsOf:url),bytes)
    }
    func testRawChannelsSurviveEveryAuxiliaryBranchAndFrozenInferenceNeedsNoTestCamera() throws {
        let reference=try RepresentationFixture.freeze(RepresentationFixture.training())
        XCTAssertEqual(reference.featurePolicy,.observedTailV2)
        let head=try XCTUnwrap(reference.heads.first {$0.condition == .silent && $0.source==1})
        let take=RepresentationFixture.take(RepresentationFixture.test[0],index:0,session:UUID())
        let frames=try XCTUnwrap(RepresentationFrames(take,cameraID:reference.cameraID,estimator:reference.estimator))
        XCTAssertEqual(frames.raw[0].count,39);XCTAssertEqual(frames.invariant[0].count,21)
        XCTAssertGreaterThanOrEqual(frames.teacherBaseline.filter {$0}.count,35)
        XCTAssertTrue(frames.teacher.indices.filter {frames.teacherBaseline[$0]}.allSatisfy {frames.teacher[$0]==[0,0,0]})
        XCTAssertEqual(RepresentationReference.features(frames.raw,head:head,arm:.raw)?.count,156)
        XCTAssertEqual(RepresentationReference.features(frames.raw,head:head,arm:.cameraAugmented)?.count,168)
        XCTAssertEqual(RepresentationReference.features(frames.raw,head:head,arm:.linguisticAugmented)?.count,176)
        let fingerprint=try reference.fingerprint(),date=Date()
        let score=try reference.evaluate([take],startedAt:date,reseatedAt:date,studyID:reference.studyID)
        var blind=take;blind.camera=[]
        let again=try reference.evaluate([blind],startedAt:date,reseatedAt:date,studyID:reference.studyID)
        XCTAssertEqual(score.map(\.predicted),again.map(\.predicted));XCTAssertEqual(score.map(\.margin),again.map(\.margin))
        // A display target is provenance, never measured alignment or a decoder feature.
        blind.wordFocusGuide = .init(leadIn:3,wordSeconds:0.2)
        let differentGuide=try reference.evaluate([blind],startedAt:date,reseatedAt:date,studyID:reference.studyID)
        XCTAssertEqual(score.map(\.predicted),differentGuide.map(\.predicted))
        XCTAssertEqual(score.map(\.margin),differentGuide.map(\.margin))
        XCTAssertEqual(score.count,4);XCTAssertTrue(score.allSatisfy {!$0.ranked.isEmpty})
        XCTAssertEqual(try reference.fingerprint(),fingerprint)
        let restored=try JSONDecoder().decode(RepresentationReference.self,from:JSONEncoder().encode(reference))
        XCTAssertEqual(try restored.fingerprint(),fingerprint)
        var tampered=reference;tampered.heads[0].decoders[RepresentationArm.raw.rawValue]?.scale[0]=0
        XCTAssertThrowsError(try tampered.validate())
        var changedTraining=RepresentationFixture.training()
        changedTraining[0].recording.samples[0].ax=999
        XCTAssertThrowsError(try reference.verifyTraining(changedTraining))
    }
    func testUnseenAirPodSideUsesFrozenInvariantFallbackAndMidTakeSwitchCannotBeScored() throws {
        let reference=try RepresentationFixture.freeze(RepresentationFixture.training()),date=Date()
        var take=RepresentationFixture.take(RepresentationFixture.test[0],index:0,session:UUID(),source:2)
        let scores=try reference.evaluate([take],startedAt:date,reseatedAt:date,studyID:reference.studyID)
        XCTAssertTrue(scores.allSatisfy(\.invariantFallback));XCTAssertTrue(scores.allSatisfy {$0.source==2 && !$0.ranked.isEmpty})
        take.recording.samples[200].sensorLocation=1
        let invalid=try reference.evaluate([take],startedAt:date,reseatedAt:date,studyID:reference.studyID)
        XCTAssertTrue(invalid.allSatisfy {$0.ranked.isEmpty && $0.predicted==nil})
        XCTAssertNil(MotionSourceSafety.source(take.recording))
    }
    func testFullSentencesSessionsAndFinalSentenceGroupsCannotLeakAcrossSplits() throws {
        let training=RepresentationFixture.training(),reference=try RepresentationFixture.freeze(training),date=Date()
        XCTAssertThrowsError(try reference.evaluate([training[0]],startedAt:date,reseatedAt:date,studyID:reference.studyID))
        var overlap=RepresentationFixture.take(RepresentationFixture.test[0],index:0,session:training[0].sessionID)
        XCTAssertThrowsError(try reference.evaluate([overlap],startedAt:date,reseatedAt:date,studyID:reference.studyID))
        overlap.sessionID=UUID();overlap.sentenceID=RepresentationFixture.train[0].id
        XCTAssertThrowsError(try reference.evaluate([overlap],startedAt:date,reseatedAt:date,studyID:reference.studyID))
        XCTAssertThrowsError(try RepresentationReference.freeze(studyID:UUID(),takes:training,training:RepresentationFixture.train,evaluation:RepresentationFixture.test,
            forbidden:[RepresentationFixture.test[0].id],profile:PronunciationProfile(),lexicon:RepresentationFixture.lexicon,cameraID:"fixture-camera",estimator:"fixture-v1"))
        let test=RepresentationFixture.take(RepresentationFixture.test[0],index:0,session:UUID())
        XCTAssertThrowsError(try reference.evaluate([test],startedAt:.distantPast,reseatedAt:date,studyID:reference.studyID))
        XCTAssertThrowsError(try reference.evaluate([test,test],startedAt:date,reseatedAt:date,studyID:reference.studyID))
        var duplicateCell=test;duplicateCell.id=UUID()
        XCTAssertThrowsError(try reference.evaluate([test,duplicateCell],startedAt:date,reseatedAt:date,studyID:reference.studyID))
        var anotherSession=RepresentationFixture.take(RepresentationFixture.test[1],index:1,session:UUID())
        XCTAssertThrowsError(try reference.evaluate([test,anotherSession],startedAt:date,reseatedAt:date,studyID:reference.studyID))
        anotherSession.sessionID=test.sessionID
        XCTAssertNoThrow(try reference.evaluate([test,anotherSession],startedAt:date,reseatedAt:date,studyID:reference.studyID))
    }
    func testUnknownSourceAndCaptureGapsAreNotInterpolatedIntoUsableMotion() {
        var take=RepresentationFixture.take(RepresentationFixture.train[0],index:0,session:UUID())
        XCTAssertTrue(MotionSourceSafety.stable(take.recording.samples,now:take.recording.cueEnd))
        for i in take.recording.samples.indices {take.recording.samples[i].sensorLocation=nil}
        XCTAssertFalse(MotionSourceSafety.stable(take.recording.samples,now:take.recording.cueEnd))
        XCTAssertNil(MotionSourceSafety.source(take.recording))
        take=RepresentationFixture.take(RepresentationFixture.train[0],index:0,session:UUID())
        take.recording.samples.removeSubrange(180...200)
        XCTAssertNil(RepresentationFrames(take,cameraID:"fixture-camera",estimator:"fixture-v1"))
    }

    func testRelaxedBaselineHeadPenaltySuppressesKnownNuisanceAndRejectsMalformedPairing() throws {
        let times=(0..<180).map {Double($0)*0.04},baseline=times.indices.map {$0<80}
        let head=times.map {sin($0*2)},jaw=times.indices.map {baseline[$0] ? 0:cos(times[$0]*7)}
        let x=times.indices.map {[jaw[$0]+head[$0],head[$0]]},y=jaw.map {[$0]}
        let ordinary=try RepresentationRidge.fit(x,y)
        let constrained=try RepresentationRidge.fit(x,y,baseline:baseline,headMotion:head.map {[$0,0,0,0]})
        XCTAssertLessThanOrEqual(abs(constrained.predict([1,1])![0]),abs(ordinary.predict([1,1])![0]))
        XCTAssertThrowsError(try RepresentationRidge.fit(x,y,baseline:[true],headMotion:nil))
    }
    func testPlannerUsesOnlyDevelopmentRolesAndPreservesFinalReservation() throws {
        var data=SentenceTrainingData(),rng=SplitMix64(seed:42)
        let prompts=SentencePrompt.bundled
        data.add(prompts,using:&rng)
        let original=data.roles
        let planned=try XCTUnwrap(RepresentationStudy.plan(data:data,profile:PronunciationProfile(),lexicon:.bundled))
        XCTAssertEqual(planned.training.count,4);XCTAssertEqual(planned.evaluation.count,4)
        XCTAssertTrue(planned.training.allSatisfy {data.roles[$0.id] == .training})
        XCTAssertTrue(planned.evaluation.allSatisfy {data.roles[$0.id] == .validation})
        XCTAssertEqual(planned.forbiddenFinalSentences.count,8)
        XCTAssertTrue(planned.forbiddenFinalSentences.isDisjoint(with:Set((planned.training+planned.evaluation).map(\.id))))
        XCTAssertEqual(data.roles,original);XCTAssertNil(data.reservedUnlockedAt)
        XCTAssertFalse(planned.novelWords.isEmpty)
    }

    func testReviewedAcousticAnnotationsReachSharedFramesAndDurationSpreadLabelsCannotEnter() throws {
        var take=RepresentationFixture.take(RepresentationFixture.train[0],index:0,session:UUID())
        let start=take.recording.cueStart
        let provenance=PhoneAlignmentProvenance(method:.cameraWarpFromAcoustic,backend:"reviewed fixture warp",modelSHA256:nil,
            pronunciationSHA256:"profile",audioSHA256:nil,rawOutputPath:nil,sourceAlignmentID:UUID(),warpCost:0.1,note:"estimated")
        take.phoneAlignment=PhoneAlignment(condition:.silent,phones:[.init(ipa:"æ",start:start+0.1,end:start+2,confidence:nil,backendScore:nil)],provenance:provenance,reviewed:true)
        let frames=try XCTUnwrap(RepresentationFrames(take,cameraID:"fixture-camera",estimator:"fixture-v1"))
        XCTAssertGreaterThan(frames.phoneTargets.count,35)
        XCTAssertEqual(frames.phoneTargets[0],ArticulatoryInventory.byIPA["æ"]?.gesturePrior)
        XCTAssertEqual(Set(SensorObservability.examples(take).map(\.modality)),["camera","AirPods invariant"])
        take.phoneAlignment?.reviewed=false
        XCTAssertTrue(try XCTUnwrap(RepresentationFrames(take,cameraID:"fixture-camera",estimator:"fixture-v1")).phoneTargets.isEmpty)
        XCTAssertTrue(SensorObservability.examples(take).isEmpty)
    }

    func testAcceptedSentencePhoneSidecarsRequireTrainingRoleMatchingProfileAndExplicitReview() throws {
        let take=RepresentationFixture.take(RepresentationFixture.train[0],index:0,session:UUID(),condition:.normal)
        let profile=PronunciationProfile(),hash=try profile.fingerprint()
        let provenance=PhoneAlignmentProvenance(method:.localAcousticAlignment,backend:"fixture acoustic alignment",modelSHA256:"model",
            pronunciationSHA256:hash,audioSHA256:"audio",rawOutputPath:nil,note:"estimate")
        var pair=SentencePair(visitID:UUID(),promptIDs:[take.sentenceID],text:take.text,role:.training,
            spoken:take.recording,recordingSessionID:take.sessionID,review:.accepted,pronunciationProfile:profile,
            spokenPhones:.init(condition:.normal,phones:[.init(ipa:"p",start:103.1,end:104,confidence:nil,backendScore:nil)],provenance:provenance,reviewed:true),cameraReadings:take.camera)
        XCTAssertEqual(try pair.representationTakes(profile:profile).count,1)
        pair.role = .finalTest;XCTAssertTrue(try pair.representationTakes(profile:profile).isEmpty)
        pair.role = .training;pair.spokenPhones?.reviewed=false
        XCTAssertTrue(try pair.representationTakes(profile:profile).isEmpty)
        pair.spokenPhones?.reviewed=true
        var changed=profile;try changed.set(word:"pack",ipaLines:"p ɑː k")
        XCTAssertTrue(try pair.representationTakes(profile:changed).isEmpty)
        pair.promptIDs.append(SentencePrompt.identity("another sentence"))
        XCTAssertTrue(try pair.representationTakes(profile:profile).isEmpty)
    }

    func testReviewedTimeLocalSupervisionIsAppendedWithoutReplacingRawChannels() throws {
        var takes=RepresentationFixture.training()
        let hash=try PronunciationProfile().fingerprint()
        for i in takes.indices {
            let condition=takes[i].recording.condition,start=takes[i].recording.cueStart
            let provenance=PhoneAlignmentProvenance(method:condition == .normal ? .localAcousticAlignment:.cameraWarpFromAcoustic,
                backend:"reviewed fixture",modelSHA256:nil,pronunciationSHA256:hash,audioSHA256:nil,rawOutputPath:nil,
                sourceAlignmentID:condition == .silent ? UUID():nil,warpCost:condition == .silent ? 0.1:nil,note:"fixture estimate")
            takes[i].phoneAlignment = .init(condition:condition,phones:[.init(ipa:"æ",start:start+0.1,end:start+2,confidence:nil,backendScore:nil)],provenance:provenance,reviewed:true)
        }
        let reference=try RepresentationFixture.freeze(takes)
        let head=try XCTUnwrap(reference.heads.first {$0.condition == .silent && $0.source==1})
        XCTAssertNotNil(head.frameLinguistic)
        let frames=try XCTUnwrap(RepresentationFrames(takes[8],cameraID:"fixture-camera",estimator:"fixture-v1"))
        XCTAssertEqual(RepresentationReference.features(frames.raw,head:head,arm:.linguisticAugmented)?.count,200)
        try reference.validate()
    }

    @MainActor func testFullAcquisitionCycleKeepsPointsOnlyFreezesAAndUsesFreshSessionAndSentencesInB() async throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"representation-cycle-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var now=100.0
        let workflow=RepresentationWorkflow(root:root,clock:{now}),audio=AudioService(),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var data=SentenceTrainingData()
        data.prompts=RepresentationFixture.train+RepresentationFixture.test+[SentencePrompt(text:"reserved sentence")]
        for prompt in RepresentationFixture.train {data.roles[prompt.id] = .training}
        for prompt in RepresentationFixture.test {data.roles[prompt.id] = .validation}
        data.roles[SentencePrompt.identity("reserved sentence")] = .finalTest
        let roles=data.roles
        workflow.prepare(data:data)
        XCTAssertNotNil(workflow.study)
        func prime(_ source:Int) {
            for i in 0...85 {
                var s=MotionSample(t:now-1.7+Double(i)/50);s.sensorLocation=source;workflow.ingest(s)
            }
            for i in 0...4 {workflow.camera.ingest(.init(t:now-0.12+Double(i)*0.03,cameraID:"fixture-camera",pose:nil,mouth:RepresentationFixture.mouth(0),imageWidth:1000,imageHeight:1000,estimator:"fixture-v1"))}
        }
        func acquire(_ block:Int) throws {
            now=block==0 ? 100:500
            prime(block==0 ? 1:2)
            workflow.startBlock(audio:audio,recorder:recorder)
            XCTAssertEqual(workflow.phase,.ready)
            let plan=workflow.plan
            for (i,item) in plan.enumerated() {
                var fixture=RepresentationFixture.take(item.0,index:i+(block==0 ? 0:40),session:workflow.study!.sessions.last!.id,source:block==0 ? 1:2,condition:item.2)
                fixture.repetition=item.1
                now=fixture.recording.cueStart;workflow.beginCapture(at:now)
                XCTAssertEqual(workflow.phase,.recording)
                for s in fixture.recording.samples {recorder.record(s);workflow.ingest(s)}
                for reading in fixture.camera {workflow.ingestCamera(reading)}
                now=fixture.recording.cueEnd;workflow.next()
                XCTAssertEqual(workflow.phase,.ended)
                XCTAssertNil(workflow.study?.takes.last?.discardedReason)
                workflow.next()
            }
        }
        func wait(_ phase:RepresentationWorkflow.Phase) async throws {
            for _ in 0..<1000 {if workflow.phase==phase{return};try await Task.sleep(for:.milliseconds(10))}
            XCTFail("Expected \(phase), got \(workflow.phase): \(workflow.error ?? "no error")")
        }
        try acquire(0);try await wait(.reseat)
        let fingerprint=try XCTUnwrap(workflow.study?.fingerprint)
        let frozen=root.appending(path:workflow.study!.id.uuidString).appending(path:"frozen-reference.json")
        let bytes=try Data(contentsOf:frozen)
        let restored=RepresentationWorkflow(root:root,clock:{now})
        XCTAssertEqual(restored.phase,.reseat);XCTAssertEqual(restored.study?.fingerprint,fingerprint)
        try acquire(1);try await wait(.finished)
        XCTAssertNil(workflow.error);XCTAssertTrue(workflow.study?.complete==true)
        XCTAssertEqual(workflow.study?.sessions.count,2)
        XCTAssertEqual(workflow.study?.predictions?.count,64)
        XCTAssertTrue(workflow.study!.predictions!.allSatisfy(\.invariantFallback))
        XCTAssertTrue(workflow.study!.takes.allSatisfy {$0.wordFocusGuide == .comfortable})
        XCTAssertEqual(try Data(contentsOf:frozen),bytes)
        XCTAssertEqual(try workflow.study?.reference?.fingerprint(),fingerprint)
        XCTAssertEqual(data.roles,roles);XCTAssertNil(data.reservedUnlockedAt)
        for session in workflow.study!.sessions {
            let meta=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:session.directory.appending(path:"meta.json"))) as? [String:Any])
            XCTAssertEqual(meta["audioRecordingEnabled"] as? Bool,false)
            XCTAssertEqual(meta["videoRecordingEnabled"] as? Bool,false)
            XCTAssertFalse(FileManager.default.fileExists(atPath:session.directory.appending(path:"audio.caf").path))
            XCTAssertGreaterThan(meta["cameraPointSamples"] as? Int ?? 0,0)
        }
    }

    @MainActor func testSavedCompleteACanBeRefittedWithoutChangingAnyRecording() async throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"representation-refit-\(UUID())")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        var takes=RepresentationFixture.training()
        for i in takes.indices {
            let end=takes[i].recording.cueEnd
            takes[i].recording.samples.removeAll {$0.t>end-0.06}
        }
        let study=RepresentationStudy(training:RepresentationFixture.train,evaluation:RepresentationFixture.test,forbiddenFinalSentences:[],profile:PronunciationProfile(),
            takes:takes,sessions:[.init(id:takes[0].sessionID,directory:root.appending(path:"raw"),startUptime:100)])
        let dir=root.appending(path:study.id.uuidString)
        try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
        let bytes=try JSONEncoder().encode(study)
        try bytes.write(to:dir.appending(path:"study.json"));try bytes.write(to:root.appending(path:"latest.json"))
        let workflow=RepresentationWorkflow(root:root)
        XCTAssertTrue(workflow.canFitSavedTraining)
        workflow.fitSavedTraining()
        for _ in 0..<1000 {if workflow.phase != .processing {break};try await Task.sleep(for:.milliseconds(10))}
        XCTAssertEqual(workflow.phase,.reseat,workflow.error ?? "no error")
        XCTAssertNotNil(workflow.study?.reference)
        XCTAssertNil(workflow.study?.predictions)
        let takeEncoder=JSONEncoder();takeEncoder.outputFormatting=[.sortedKeys]
        XCTAssertEqual(try takeEncoder.encode(workflow.study!.takes),try takeEncoder.encode(takes))
        let backups=try FileManager.default.contentsOfDirectory(at:dir,includingPropertiesForKeys:nil).filter {$0.lastPathComponent.hasPrefix("before-refit-")}
        XCTAssertEqual(backups.count,1)
        XCTAssertEqual(try Data(contentsOf:backups[0]),bytes)
        try workflow.study?.reference?.verifyTraining(takes)
        XCTAssertFalse(workflow.canFitSavedTraining)
    }

    @MainActor func testSourceSwitchAbortsCurrentAttemptAndRequiresStableSourceBeforeRetry() throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"representation-switch-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        var now=100.0
        let workflow=RepresentationWorkflow(root:root,clock:{now}),audio=AudioService(),recorder=SessionRecorder(root:root.appending(path:"raw"))
        var data=SentenceTrainingData()
        data.prompts=RepresentationFixture.train+RepresentationFixture.test
        for p in RepresentationFixture.train {data.roles[p.id] = .training}
        for p in RepresentationFixture.test {data.roles[p.id] = .validation}
        workflow.prepare(data:data)
        for i in 0...85 {var s=MotionSample(t:now-1.7+Double(i)/50);s.sensorLocation=1;workflow.ingest(s)}
        for i in 0...4 {workflow.camera.ingest(.init(t:now-0.12+Double(i)*0.03,cameraID:"fixture-camera",pose:nil,mouth:RepresentationFixture.mouth(0),imageWidth:1000,imageHeight:1000,estimator:"fixture-v1"))}
        workflow.startBlock(audio:audio,recorder:recorder)
        workflow.beginCapture(at:now)
        now += 0.02
        var switched=MotionSample(t:now);switched.sensorLocation=2;workflow.ingest(switched)
        XCTAssertEqual(workflow.phase,.ended)
        XCTAssertNotNil(workflow.study?.takes.last?.discardedReason)
        workflow.next();XCTAssertEqual(workflow.index,0)
        workflow.redo();XCTAssertEqual(workflow.phase,.ready)
        XCTAssertFalse(workflow.sourceStable)
        workflow.startTake();XCTAssertEqual(workflow.phase,.ready)
        workflow.stop()
    }
}
