import XCTest
@testable import Subvox

final class MouthTeacherTests: XCTestCase {
    private let target=CameraPoseTarget(id:UUID(),cameraID:"teacher-camera",pose:.init(yaw:0,pitch:0,roll:0,x:0.5,y:0.5,size:0.3),angleTolerance:5,positionTolerance:0.05,sizeTolerance:0.12,estimator:"apple-vision-face-landmarks-v1")
    private let testID=UUID()
    private func mouth(_ t:Double,phase:Double=0)->CameraMouthObservation {
        let outer:[CameraLandmarkPoint]=[.init(x:0.4,y:0.4),.init(x:0.5,y:0.43),.init(x:0.6,y:0.4),.init(x:0.5,y:0.37)]
        let inner:[CameraLandmarkPoint]=[.init(x:0.42,y:0.4),.init(x:0.5,y:0.41),.init(x:0.58,y:0.4),.init(x:0.5,y:0.39)]
        var result=CameraMouthObservation.measure(outer:outer,inner:inner,leftEye:.init(x:0.4,y:0.6),rightEye:.init(x:0.6,y:0.6),imageWidth:1000,imageHeight:1000,confidence:0.9)!
        result.opening=0.1+0.03*sin(t*6);result.width=0.7+0.02*cos(t*5);result.outerAspect=0.3+0.02*sin(t*4)
        let angle=0.2*t+phase,span=0.2*exp(0.1*t),x=0.5+0.05*t,y=0.6+0.03*t
        result.leftEye = .init(x:x-span*cos(angle)/2,y:y-span*sin(angle)/2)
        result.rightEye = .init(x:x+span*cos(angle)/2,y:y+span*sin(angle)/2)
        return result
    }
    private func learningData()->[TeacherLearningTake] {
        (0..<9).map { trial in
            let rows=(0..<140).map { i -> TeacherLearningRow in
                let t=Double(i)*0.04,baseline=i<50
                let jaw=baseline ? [0,0,0] : [sin(t*(4+Double(trial%3))),cos(t*7),sin(t*9)]
                let h=[sin(t*0.8+Double(trial)),cos(t*0.5+Double(trial)),sin(t*0.4+Double(trial)),cos(t*0.3)]
                let s=MotionSample(t:t,accel:(jaw[0]+3*h[0],jaw[1]+4*h[1],jaw[2]+2*h[2]),rotation:(h[0],h[1],h[2]))
                let b=TeacherIMUFeatures.base(s,mode:.body)!,v=TeacherIMUFeatures.base(s,mode:.invariant)!
                return .init(body:b+b+b,invariant:v+v+v,target:jaw,head:h,baseline:baseline)
            }
            return .init(repetition:trial/3,rows:rows)
        }
    }

    func testGravityFeaturesAreInvariantToCommonSensorRotationAndBodyFeaturesKeepDirection() throws {
        var original=MotionSample(t:1,accel:(0.1,-0.2,0.3),rotation:(-0.5,0.7,0.2))
        original.gx=0.2;original.gy=0.1;original.gz = -0.97
        var rotated=original
        // An independent 90-degree rotation about z, applied to every measured vector.
        rotated.ax = -original.ay;rotated.ay=original.ax
        rotated.rx = -original.ry;rotated.ry=original.rx
        rotated.gx = -original.gy;rotated.gy=original.gx
        let a=try XCTUnwrap(TeacherIMUFeatures.base(original,mode:.invariant)),b=try XCTUnwrap(TeacherIMUFeatures.base(rotated,mode:.invariant))
        for (x,y) in zip(a,b) { XCTAssertEqual(x,y,accuracy:1e-12) }
        XCTAssertNotEqual(TeacherIMUFeatures.base(original,mode:.body),TeacherIMUFeatures.base(rotated,mode:.body))
    }

    func testCausalWindowDoesNotReadFutureAndRejectsGapsAndSensorSwitches() throws {
        var samples=(0...50).map { i -> MotionSample in var s=MotionSample(t:Double(i)/50,rotation:(Double(i)/50,0,0));s.sensorLocation=1;return s }
        let original=try XCTUnwrap(TeacherMotionTimeline(samples).context(at:0.6))
        for i in samples.indices where samples[i].t>0.6 { samples[i].rx=10000 }
        XCTAssertEqual(TeacherMotionTimeline(samples).context(at:0.6)?.body,original.body)
        samples.removeAll { $0.t>0.4 && $0.t<0.56 }
        XCTAssertNil(TeacherMotionTimeline(samples).context(at:0.6))
        samples=(0...50).map { i in var s=MotionSample(t:Double(i)/50);s.sensorLocation=i<25 ? 1:2;return s }
        XCTAssertNil(TeacherMotionTimeline(samples).context(at:0.6))
        XCTAssertNotNil(TeacherMotionTimeline(samples).context(at:0.9))
        XCTAssertNil(TeacherMotionTimeline([]).context(at:0))
        XCTAssertNil(TeacherMotionTimeline([MotionSample(t:0)]).context(at:0.01))
    }

    func testEyeReferenceAndMouthVelocitiesDoNotDependOnNoisyFaceAngles() throws {
        var readings=(0...100).map { i -> CameraPoseReading in
            let t=Double(i)/100
            return .init(t:t,cameraID:target.cameraID,pose:.init(yaw:999*t,pitch:-888*t,roll:777*t,x:0.5,y:0.5,size:0.3),mouth:mouth(t),imageWidth:1000,imageHeight:1000,estimator:target.estimator)
        }
        let timeline=TeacherCameraTimeline(readings,cameraID:target.cameraID,estimator:target.estimator)
        let velocity=try XCTUnwrap(timeline.velocity(at:0.5))
        XCTAssertEqual(velocity.mouth[0],0.03*(sin(0.58*6)-sin(0.42*6))/0.16,accuracy:1e-10)
        for (actual,expected) in zip(velocity.head,[0.05,0.03,0.1,0.2]) { XCTAssertEqual(actual,expected,accuracy:1e-10) }
        for i in readings.indices { readings[i].pose=nil }
        XCTAssertEqual(TeacherCameraTimeline(readings,cameraID:target.cameraID,estimator:target.estimator).velocity(at:0.5)?.mouth,velocity.mouth)
        readings.removeAll { $0.t>=0.44 && $0.t<=0.56 }
        XCTAssertNil(TeacherCameraTimeline(readings,cameraID:target.cameraID,estimator:target.estimator).velocity(at:0.5))
    }

    func testStudentLearnsKnownJawComponentAcrossWholeHeldOutRepetitionsAndBeatsShiftedTeacher() throws {
        let model=try MouthTeacherModel.learn(learningData())
        try model.validate()
        XCTAssertGreaterThan(model.sourceValidationGain,0.25)
        XCTAssertGreaterThan(model.sourceValidationGain,model.shiftedTeacherGain+0.1)
        XCTAssertEqual(model.candidates.count,6)
        let bytes=try JSONEncoder().encode(model),restored=try JSONDecoder().decode(MouthTeacherModel.self,from:bytes)
        let input=learningData()[0].rows.last!.input(model.mode)
        XCTAssertEqual(model.predict(input),restored.predict(input))
        XCTAssertNil(model.predict([]))
        var constant=learningData()
        for i in constant.indices { for j in constant[i].rows.indices { constant[i].rows[j].target=[0,0,0] } }
        XCTAssertThrowsError(try MouthTeacherModel.learn(constant))
    }

    func testGenericDTWHasNoPhantomMatchFromMalformedInputsAndRetainsTimingControl() {
        let times=(0..<60).map { Double($0)*0.04 }
        let sequence=TeacherSequence(times:times,values:times.map { [sin($0*4),cos($0*7),sin($0*9)] })
        XCTAssertEqual(sequence.distance(to:sequence,scales:[1,1,1]),0)
        XCTAssertGreaterThan(sequence.distance(to:sequence.shifted(),scales:[1,1,1])!,0.1)
        XCTAssertNil(sequence.distance(to:sequence,scales:[0,1,1]))
        XCTAssertNil(TeacherSequence(times:[0,0,1],values:[[1],[2],[3]]).distance(to:sequence,scales:[1]))
    }

    func testLinearDecoderUsesFixedSourceStatisticsAndRejectsUninformativeTies() throws {
        let times=(0..<85).map {Double($0)*0.04}
        let examples=(0..<9).map { i -> (Int,TeacherSequence) in
            let label=i%3
            let values=times.map {t in [sin(t*Double(3+label*3)),cos(t*Double(5+label*2)),sin(t*Double(7+label))]}
            return (label,.init(times:times,values:values))
        }
        let classifier=try TeacherLinearClassifier.fit(examples);try classifier.validate()
        for example in examples {XCTAssertEqual(classifier.predict(example.1),example.0)}
        let raw=examples.map {label,seq in (label,TeacherSequence(times:seq.times,values:seq.values.map {$0+$0}))}
        let rawClassifier=try TeacherLinearClassifier.fit(raw);try rawClassifier.validate()
        for example in raw {XCTAssertEqual(rawClassifier.predict(example.1),example.0)}
        let zero=TeacherSequence(times:times,values:times.map {_ in [0,0,0]})
        let tied=try TeacherLinearClassifier.fit((0..<9).map {($0%3,zero)})
        XCTAssertNil(tied.predict(zero))
        XCTAssertThrowsError(try TeacherLinearClassifier.fit(examples+[(3,zero)]))
    }

    func testNewPacingVariesByRepetitionAndChangesAfterReseatingWhileLegacyScheduleIsFixed() {
        for sentence in 0..<3 {
            var rates=Set<Double>()
            for repetition in 0..<3 {
                let prompt=SignalCheckPrompt(sentence:sentence,repetition:repetition,condition:.silent)
                let a=SignalPacing.teacher(prompt:prompt,block:.unchanged),b=SignalPacing.teacher(prompt:prompt,block:.reseated)
                rates.insert(a.wordSeconds);XCTAssertNotEqual(a.wordSeconds,b.wordSeconds)
                XCTAssertEqual(a,SignalPacing.teacher(prompt:prompt,block:.unchanged))
                XCTAssertGreaterThan(a.duration(words:4),2.5);XCTAssertLessThan(a.duration(words:4),4.6)
            }
            XCTAssertEqual(rates.count,3)
        }
        XCTAssertEqual(SignalPacing.controlled.duration(words:4),3.4,accuracy:1e-10)
    }

    private func fixture(start:Double=100,block:ControlledSignalBlock = .unchanged,seed:UInt64=42,
                         target suppliedTarget:CameraPoseTarget?=nil,testID suppliedID:UUID?=nil)->SignalCheckRun {
        let target=suppliedTarget ?? self.target
        var run=SignalCheckRun(seed:seed)
        run.startedAt=Date();run.endedAt=Date();run.reseatedAt=block == .reseated ? Date():nil
        run.session = .init(id:UUID(),directory:URL(fileURLWithPath:"/tmp/teacher-fixture"),startUptime:start)
        run.control = .init(testID:suppliedID ?? testID,block:block,target:target,pacing:.controlled,blockConfirmedAt:Date(),motionPolicy:.cameraTeacher)
        for (index,prompt) in SignalCheckRun.plan(seed:seed).enumerated() {
            let begin=start+Double(index)*10,cue=begin+3,pacing=SignalPacing.teacher(prompt:prompt,block:block),duration=pacing.duration(words:4)
            let camera=(0...Int(ceil((3+duration+0.8)*30))).map { i -> CameraPoseReading in
                let t=begin-0.4+Double(i)/30,u=min(duration,max(0,t-cue)),phase=Double(index)*0.3
                var m=mouth(0)
                m.opening=0.1+0.03*sin(u*(4+Double(prompt.sentence)*2))
                m.width=0.7+0.03*cos(u*(5+Double(prompt.sentence)*3))
                m.outerAspect=0.3+0.02*sin(u*(7+Double(prompt.sentence)))
                let h=t-begin,angle=0.05*sin(h*0.7+phase),span=0.2*exp(0.02*sin(h*0.5+phase)),x=0.5+0.02*sin(h*0.8+phase),y=0.6+0.02*cos(h*0.6+phase)
                m.leftEye = .init(x:x-span*cos(angle)/2,y:y-span*sin(angle)/2);m.rightEye = .init(x:x+span*cos(angle)/2,y:y+span*sin(angle)/2)
                return .init(t:t,cameraID:target.cameraID,pose:target.pose,usedReceiptTime:false,mouth:m,imageWidth:1000,imageHeight:1000,estimator:target.estimator)
            }
            run.control?.readings += camera
            let timeline=TeacherCameraTimeline(camera,cameraID:target.cameraID,estimator:target.estimator)
            let samples=(0...Int(floor((3+duration)*50))).map { i -> MotionSample in
                let t=begin+Double(i)/50,v=timeline.velocity(at:t)!,y=v.mouth,h=v.head
                var s=MotionSample(t:t,accel:(0.1*y[0]+h[0],0.1*y[1]+h[1],0.1*y[2]+0.05*h[2]),rotation:(0.2*y[0]+h[3],0.2*y[1]+0.1*h[2],0.2*y[2]+0.1*h[1]))
                s.sensorLocation=block == .reseated ? 2:1;return s
            }
            run.takes.append(.init(prompt:prompt,recording:.init(condition:prompt.condition,samples:samples,cueStart:cue,cueEnd:cue+duration),endReason:.scheduled,pacing:pacing))
        }
        return run
    }

    func testReferenceIsFrozenAndStudentUsesNoTestCameraWhileControlsRemainSeparate() throws {
        let source=fixture(),reference=try CameraTeacherReference.freeze(source),test=fixture(start:500,block:.reseated)
        let fingerprint=try reference.fingerprint(),scores=try reference.evaluate(test)
        XCTAssertEqual(scores.count,2)
        for score in scores {
            XCTAssertEqual(score.arms.count,7);XCTAssertTrue(score.arms.allSatisfy { $0.evaluated==9 })
            XCTAssertTrue(score.sensorSourceChanged)
        }
        var blind=test;blind.control?.readings=[]
        let blindScores=try reference.evaluate(blind)
        for (normal,blind) in zip(scores,blindScores) {
            for arm in [TeacherArm.raw,.rawClassifier,.context,.student,.studentClassifier] {
                XCTAssertEqual(normal.arms.first {$0.arm==arm}?.correct,blind.arms.first {$0.arm==arm}?.correct)
                XCTAssertEqual(blind.arms.first {$0.arm==arm}?.evaluated,9)
            }
            XCTAssertEqual(blind.arms.first {$0.arm == .camera}?.evaluated,0)
            XCTAssertEqual(blind.arms.first {$0.arm == .head}?.evaluated,0)
        }
        XCTAssertEqual(try reference.fingerprint(),fingerprint)
        XCTAssertThrowsError(try reference.evaluate(source));XCTAssertThrowsError(try CameraTeacherReference.freeze(test))
        let restored=try JSONDecoder().decode(CameraTeacherReference.self,from:JSONEncoder().encode(reference))
        XCTAssertEqual(try restored.fingerprint(),fingerprint)
        var relabelled=source
        for i in relabelled.takes.indices { relabelled.takes[i].prompt.sentence=(relabelled.takes[i].prompt.sentence+1)%3 }
        // Different labels would prescribe a different pace; keep this isolated fit check on legacy pacing metadata.
        relabelled.control?.motionPolicy = .naturalMovement
        let newReference=try CameraTeacherReference.freeze(relabelled)
        for (a,b) in zip(reference.groups,newReference.groups) { XCTAssertEqual(a.model?.weights,b.model?.weights) }
    }

    func testOldMotionDataDecodesWithoutSensorLocationAndNewRowsPreserveSensorIdentity() throws {
        var sample=MotionSample(t:1)
        let old=try JSONEncoder().encode(sample),decoded=try JSONDecoder().decode(MotionSample.self,from:old)
        XCTAssertNil(decoded.sensorLocation)
        sample.sensorLocation=2
        XCTAssertEqual(sample.csvRow(relativeTo:0).split(separator:",").last,"2")
        XCTAssertEqual(try JSONDecoder().decode(MotionSample.self,from:JSONEncoder().encode(sample)).sensorLocation,2)
    }

    func testTrainingDigestIgnoresDerivedReportsButDetectsChangedSensorInputs() throws {
        var source=fixture()
        let hash=try CameraTeacherReference.trainingHash(source)
        source.report = .init(metrics:[],conditions:[],warnings:["Derived after the source snapshot"])
        XCTAssertEqual(try CameraTeacherReference.trainingHash(source),hash)
        source.takes[0].recording.samples[0].rx += 0.01
        XCTAssertNotEqual(try CameraTeacherReference.trainingHash(source),hash)
    }

    func testSessionGroupsOverrideRepetitionSplitsAndHistoryCannotReenterFinalTest() throws {
        let original=learningData()
        var grouped=original
        for i in grouped.indices { grouped[i].validationGroup=grouped[i].repetition;grouped[i].repetition=99 }
        let a=try MouthTeacherModel.learn(original),b=try MouthTeacherModel.learn(grouped)
        XCTAssertEqual(a.weights,b.weights);XCTAssertEqual(a.sourceValidationGain,b.sourceValidationGain)
        let oldA=fixture(),oldB=fixture(start:500,block:.reseated),current=fixture(start:1000)
        let reference=try CameraTeacherReference.freeze(current,history:[oldA,oldB])
        XCTAssertEqual(reference.trainingSources.count,3)
        XCTAssertTrue(reference.groups.allSatisfy {$0.validationScope == "whole recording sessions held out"})
        XCTAssertTrue(reference.trainingSources.allSatisfy {$0.sha256.count==64})
        let fresh=fixture(start:1500,block:.reseated)
        XCTAssertTrue(try reference.evaluate(fresh).allSatisfy(\.sensorSourceChanged))
        var reused=fixture(start:1500,block:.reseated);reused.id=oldB.id
        XCTAssertThrowsError(try reference.evaluate(reused))
    }

    @MainActor func testFullTeacherWorkflowFreezesAAndScoresFreshBWithRecordedVariablePacing() async throws {
        let root=FileManager.default.temporaryDirectory.appending(path:"subvox-teacher-cycle-\(UUID())")
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let oldReferenceURL=root.appending(path:"frozen-signal-reference.json")
        try FrozenSignalReference.freeze(SignalCheckTestData.source()).saveNew(to:oldReferenceURL)
        let originalReference=try Data(contentsOf:oldReferenceURL)
        var now=100.0
        let workflow=SignalCheckWorkflow(storeURL:root.appending(path:"signal.json"),clock:{now})
        workflow.prepareControlled()
        XCTAssertTrue(workflow.usesCameraTeacher)
        let comparison=try XCTUnwrap(workflow.comparison),camera=mouth(0)
        for i in 0...4 {
            now=100+Double(i)/30;workflow.ingest(MotionSample(t:now))
            workflow.camera.ingest(.init(t:now,cameraID:target.cameraID,pose:target.pose,mouth:camera,imageWidth:1000,imageHeight:1000,estimator:target.estimator))
        }
        let savedTarget=try XCTUnwrap(workflow.comparison?.target),recorder=SessionRecorder(root:root.appending(path:"raw"))
        func begin(_ block:ControlledSignalBlock) throws {
            let session=try XCTUnwrap(recorder.start(audioSink:nil,cameraPoints:true)),date=Date()
            workflow.begin(session:session,audioDevice:nil,seed:comparison.seed,recorder:recorder,reference:comparison.reference,
                reseatedAt:block == .reseated ? date:nil,
                control:.init(testID:comparison.id,block:block,target:savedTarget,pacing:.controlled,blockConfirmedAt:date,motionPolicy:.cameraTeacher))
        }
        func acquire(_ run:SignalCheckRun) {
            for (index,take) in run.takes.enumerated() {
                now=take.recording.cueStart
                for reading in run.control!.readings where reading.t>=now-0.13 && reading.t<=now {workflow.camera.ingest(reading)}
                workflow.ingest(MotionSample(t:now));workflow.beginCapture(at:now)
                XCTAssertEqual(workflow.phase,.recording)
                for sample in take.recording.samples {workflow.ingest(sample)}
                let end=take.recording.cueEnd
                let before=run.control!.readings.filter {$0.t>=now-3 && $0.t<=end-0.1}
                for reading in before {workflow.ingestPose(reading)}
                now=end;workflow.endCapture(at:end,reason:.scheduled)
                XCTAssertEqual(workflow.latestTake?.pacing,take.pacing)
                for reading in run.control!.readings where reading.t>before.last!.t && reading.t<=end+0.2 {workflow.ingestPose(reading)}
                if index+1<run.takes.count {workflow.next()}
            }
        }
        func wait(_ phase:SignalCheckWorkflow.Phase) async throws {
            for _ in 0..<2000 {if workflow.phase==phase{return};try await Task.sleep(for:.milliseconds(5))}
            XCTFail("Missing phase \(phase); \(workflow.error ?? "no error")")
        }
        let a=fixture(start:110,seed:comparison.seed,target:savedTarget,testID:comparison.id)
        try begin(.unchanged);acquire(a);workflow.next();try await wait(.controlBreak)
        let frozen=try XCTUnwrap(workflow.comparison?.teacherReference),fingerprint=try frozen.fingerprint()
        XCTAssertTrue(frozen.groups.allSatisfy {$0.model != nil})
        now += 31
        for i in 0...4 {workflow.camera.ingest(.init(t:now-0.12+Double(i)*0.03,cameraID:target.cameraID,pose:target.pose,mouth:camera,imageWidth:1000,imageHeight:1000,estimator:target.estimator))}
        workflow.ingest(MotionSample(t:now));XCTAssertTrue(workflow.canStartControlledBlock)
        let b=fixture(start:now+1,block:.reseated,seed:comparison.seed,target:savedTarget,testID:comparison.id)
        try begin(.reseated);acquire(b);workflow.next();try await wait(.finished)
        let completed=try XCTUnwrap(workflow.comparison);try completed.validate()
        XCTAssertEqual(completed.blocks.count,2);XCTAssertEqual(completed.teacherScores?.count,2)
        XCTAssertTrue(completed.teacherScores!.allSatisfy {$0.arms.first(where:{$0.arm == .student})?.evaluated==9})
        XCTAssertEqual(try completed.teacherReference?.fingerprint(),fingerprint)
        XCTAssertEqual(try Data(contentsOf:oldReferenceURL),originalReference)
        let restored=SignalCheckWorkflow(storeURL:root.appending(path:"signal.json"))
        XCTAssertEqual(restored.comparison?.teacherReference?.id,frozen.id)
        XCTAssertEqual(restored.comparison?.teacherScores?.count,2)
    }
}
