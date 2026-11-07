import CryptoKit
import Foundation

/// No absolute heading is assumed. Gravity projections and Gram quantities are invariant
/// to a common rotation of acceleration, angular velocity and gravity at each timestamp.
enum TeacherIMUFeatures {
    static func base(_ s: MotionSample, mode: TeacherFeatureMode) -> [Double]? {
        let a = [s.ax,s.ay,s.az], r = [s.rx,s.ry,s.rz], g = [s.gx,s.gy,s.gz]
        guard (a+r+g).allSatisfy(\.isFinite) else { return nil }
        if mode == .body { return a+r }
        func dot(_ x: [Double], _ y: [Double]) -> Double { zip(x,y).reduce(0) { $0 + $1.0*$1.1 } }
        let gn = sqrt(dot(g,g)); guard gn > 0.1 else { return nil }
        let unit = g.map { $0/gn }, ap = dot(a,unit), rp = dot(r,unit), aa = dot(a,a), rr = dot(r,r)
        return [ap,rp,sqrt(max(0,aa-ap*ap)),sqrt(max(0,rr-rp*rp)),sqrt(aa),sqrt(rr),dot(a,r)]
    }
}

struct TeacherMotionTimeline {
    let samples: [MotionSample]
    init(_ samples: [MotionSample]) { self.samples = samples }
    private func index(_ t: Double) -> Int {
        var lo=0,hi=samples.count
        while lo<hi { let m=(lo+hi)/2; if samples[m].t<t { lo=m+1 } else { hi=m } }
        return lo
    }
    private func sample(at t: Double) -> MotionSample? {
        let i=index(t)
        if i<samples.count, abs(samples[i].t-t)<1e-7 { return samples[i] }
        guard i>0,i<samples.count else { return nil }
        let a=samples[i-1],b=samples[i],delta=b.t-a.t
        guard delta>0,delta<=0.12,a.sensorLocation==b.sensorLocation else { return nil }
        let f=(t-a.t)/delta
        func lerp(_ x:Double,_ y:Double)->Double { x+(y-x)*f }
        var result=MotionSample(t:t,accel:(lerp(a.ax,b.ax),lerp(a.ay,b.ay),lerp(a.az,b.az)),
            rotation:(lerp(a.rx,b.rx),lerp(a.ry,b.ry),lerp(a.rz,b.rz)))
        result.gx=lerp(a.gx,b.gx);result.gy=lerp(a.gy,b.gy);result.gz=lerp(a.gz,b.gz)
        result.sensorLocation=a.sensorLocation
        return result
    }
    func context(at t:Double) -> (body:[Double],invariant:[Double])? {
        guard samples.count>=2 else { return nil }
        let lo=index(t-0.24),hi=index(t)
        guard lo<samples.count,hi<=samples.count else { return nil }
        if hi>lo {
            for i in max(1,lo)...min(hi,samples.count-1) {
                guard samples[i].t>samples[i-1].t,samples[i].t-samples[i-1].t<=0.12,
                      samples[i].sensorLocation==samples[i-1].sensorLocation else { return nil }
            }
        }
        var body:[Double]=[],invariant:[Double]=[]
        for lag in [-0.24,-0.12,0.0] {
            guard let s=sample(at:t+lag),let b=TeacherIMUFeatures.base(s,mode:.body),let v=TeacherIMUFeatures.base(s,mode:.invariant) else { return nil }
            body += b;invariant += v
        }
        return (body,invariant)
    }
}

/// Eye centres, eye spacing and eye-line roll provide a continuous rigid reference.
/// Face-box yaw/pitch/roll are not differentiated or treated as exact head motion.
struct TeacherCameraTimeline {
    let readings:[CameraPoseReading]
    let cameraID:String
    let estimator:String?
    init(_ readings:[CameraPoseReading],cameraID:String,estimator:String?) {
        self.readings=readings;self.cameraID=cameraID;self.estimator=estimator
    }
    private func index(_ t:Double)->Int {
        var lo=0,hi=readings.count
        while lo<hi { let m=(lo+hi)/2;if readings[m].t<t {lo=m+1}else{hi=m} }
        return lo
    }
    private func state(_ r:CameraPoseReading)->[Double]? {
        guard r.cameraID==cameraID,r.estimator==estimator,r.usedReceiptTime != true,
              let mouth=r.mouth,mouth.confidence>=0.5,let w=r.imageWidth,let h=r.imageHeight,w>0,h>0 else { return nil }
        let l=mouth.leftEye,q=mouth.rightEye,dx=(q.x-l.x)*Double(w),dy=(q.y-l.y)*Double(h),span=hypot(dx,dy)
        guard l.valid,q.valid,span>=10 else { return nil }
        let result=[mouth.opening,mouth.width,mouth.outerAspect,(l.x+q.x)/2,(l.y+q.y)/2,log(span),atan2(dy,dx)]
        return result.allSatisfy(\.isFinite) ? result : nil
    }
    private func state(at t:Double)->[Double]? {
        let i=index(t)
        if i<readings.count,abs(readings[i].t-t)<1e-7 { return state(readings[i]) }
        guard i>0,i<readings.count,let a=state(readings[i-1]),let b=state(readings[i]),
              readings[i].t>readings[i-1].t,readings[i].t-readings[i-1].t<=0.12 else { return nil }
        let f=(t-readings[i-1].t)/(readings[i].t-readings[i-1].t)
        var delta=zip(a,b).map { $1-$0 };delta[6]=atan2(sin(delta[6]),cos(delta[6]))
        return a.indices.map { a[$0]+delta[$0]*f }
    }
    func velocity(at t:Double)->(mouth:[Double],head:[Double])? {
        guard let a=state(at:t-0.08),let b=state(at:t+0.08) else { return nil }
        let lo=index(t-0.08),hi=index(t+0.08)
        guard hi<readings.count,lo<=hi else { return nil }
        for i in lo...hi {
            guard state(readings[i]) != nil,
                  i==lo || (readings[i].t>readings[i-1].t && readings[i].t-readings[i-1].t<=0.12) else { return nil }
        }
        var delta=zip(a,b).map { ($1-$0)/0.16 }
        delta[6]=atan2(sin(b[6]-a[6]),cos(b[6]-a[6]))/0.16
        return (Array(delta.prefix(3)),Array(delta.suffix(4)))
    }
}

struct TeacherTakeFrames {
    let takeID:UUID
    let prompt:SignalCheckPrompt
    let cueStart:Double
    let cueEnd:Double
    var times:[Double]=[]
    var body:[[Double]]=[]
    var invariant:[[Double]]=[]
    var learning:[TeacherLearningRow]=[]
    var cameraTimes:[Double]=[]
    var camera:[[Double]]=[]
    var head:[[Double]]=[]
    var validationGroup:Int?

    init(_ take:SignalCheckTake,camera:TeacherCameraTimeline?) {
        takeID=take.id;prompt=take.prompt;cueStart=take.recording.cueStart;cueEnd=take.recording.cueEnd
        guard Set(take.recording.samples.compactMap(\.sensorLocation).filter({$0>0})).count<=1 else { return }
        let motion=TeacherMotionTimeline(take.recording.samples)
        let steps=Int(floor((cueEnd-cueStart+2.48)/0.04+1e-6))
        guard steps>=0,steps<=1400 else { return }
        for step in 0...steps {
            let offset = -2.48+Double(step)*0.04
            if offset > -0.5 && offset < -1e-7 { continue }
            let t=cueStart+offset,isCue=offset >= -1e-7
            guard let x=motion.context(at:t) else { continue }
            if isCue { times.append(t);body.append(x.body);invariant.append(x.invariant) }
            if let velocity=camera?.velocity(at:t) {
                learning.append(.init(body:x.body,invariant:x.invariant,target:isCue ? velocity.mouth : [0,0,0],head:velocity.head,baseline:!isCue))
                if isCue { cameraTimes.append(t);self.camera.append(velocity.mouth);head.append(velocity.head) }
            }
        }
    }
    func sequence(_ arm:TeacherArm,model:MouthTeacherModel?)->TeacherSequence? {
        let result:TeacherSequence
        switch arm {
        case .raw,.rawClassifier:result = .init(times:times,values:body.map { Array($0.suffix(6)) })
        case .context:
            guard let model else { return nil };result = .init(times:times,values:model.mode == .body ? body : invariant)
        case .student,.studentClassifier:
            guard let model else { return nil }
            let input=model.mode == .body ? body : invariant
            let values=input.compactMap { model.predict($0) }
            guard values.count==times.count else { return nil };result = .init(times:times,values:values)
        case .camera:result = .init(times:cameraTimes,values:camera)
        case .head:result = .init(times:cameraTimes,values:head)
        }
        guard result.valid,let first=result.times.first,let last=result.times.last,
              first-cueStart<=0.12+1e-6,cueEnd-last<=0.12+1e-6,last-first>=(cueEnd-cueStart)*0.9 else { return nil }
        return result
    }
}

enum TeacherArm:String,Codable,CaseIterable,Sendable {
    case raw,rawClassifier,context,student,studentClassifier,camera,head
    var label:String {
        switch self {
        case .raw:"Raw AirPods · acceleration and rotation"
        case .rawClassifier:"Raw AirPods · linear decoder (primary baseline)"
        case .context:"AirPods context · no camera teaching"
        case .student:"Camera-taught AirPods · camera-free prediction"
        case .studentClassifier:"Camera-taught AirPods · linear decoder (primary)"
        case .camera:"Camera mouth only · visual benchmark"
        case .head:"Camera head only · confound check"
        }
    }
}
struct TeacherArmScore:Codable,Sendable {
    let arm:TeacherArm
    let correct:Int
    let evaluated:Int
    let shiftedCorrect:Int
    let confusion:[[Int]]
}
struct TeacherConditionScores:Codable,Sendable {
    let condition:SpeechCondition
    let collected:Int
    let arms:[TeacherArmScore]
    let sensorLocations:[Int]
    let sensorSourceChanged:Bool
}

struct CameraTeacherReference:Codable,Sendable {
    struct Template:Codable,Sendable { let takeID:UUID;let sentence:Int;let condition:SpeechCondition;let arm:TeacherArm;let sequence:TeacherSequence }
    struct Group:Codable,Sendable {
        let condition:SpeechCondition
        let model:MouthTeacherModel?
        let note:String?
        let scales:[String:[Double]]
        let validationScope:String
        let classifier:TeacherLinearClassifier?
        let rawClassifier:TeacherLinearClassifier?
        let currentSensorLocations:[Int]
    }
    struct TrainingSource:Codable,Sendable {
        let runID:UUID
        let sessionID:UUID
        let startedAt:Date
        let sha256:String
    }
    let id:UUID
    let sourceRunID:UUID
    let sourceSessionID:UUID
    let testID:UUID
    let targetID:UUID
    let cameraID:String
    let estimator:String?
    let seed:UInt64
    let frozenAt:Date
    let version:Int
    let sensorLocations:[Int]
    let trainingSources:[TrainingSource]
    let primaryArm:TeacherArm
    let baselineArm:TeacherArm
    let sentences:[String]
    let groups:[Group]
    let templates:[Template]

    func fingerprint() throws -> String {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return SHA256.hash(data:try encoder.encode(self)).map { String(format:"%02x",$0) }.joined()
    }
    static func trainingHash(_ source:SignalCheckRun) throws -> String {
        // Reports/transcripts are derived later and are not learning inputs. Removing them
        // makes the digest reproducible from the final archived run as well as its snapshot.
        var canonical=source;canonical.report=nil
        for i in canonical.takes.indices {canonical.takes[i].transcript=nil;canonical.takes[i].audioError=nil}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return SHA256.hash(data:try encoder.encode(canonical)).map {String(format:"%02x",$0)}.joined()
    }
    func validate() throws {
        guard version==1,sentences==SignalCheckRun.sentences,primaryArm == .studentClassifier,baselineArm == .rawClassifier,groups.count==2,Set(groups.map(\.condition))==Set([.normal,.silent]),
              (1...3).contains(trainingSources.count),Set(trainingSources.map(\.runID)).count==trainingSources.count,
              Set(trainingSources.map(\.sessionID)).count==trainingSources.count,
              trainingSources.contains(where:{$0.runID==sourceRunID && $0.sessionID==sourceSessionID}),
              templates.count<=90*trainingSources.count,templates.allSatisfy({ (0..<3).contains($0.sentence) && $0.sequence.valid }) else { throw MouthTeacherModel.Failure.model }
        for group in groups {
            try group.model?.validate()
            try group.classifier?.validate()
            try group.rawClassifier?.validate()
            for scales in group.scales.values { guard !scales.isEmpty,scales.count<=24,scales.allSatisfy({ $0.isFinite && $0>=1e-4 }) else { throw MouthTeacherModel.Failure.model } }
        }
    }

    static func freeze(_ run:SignalCheckRun,history:[SignalCheckRun]=[]) throws -> Self {
        guard run.endedAt != nil,let control=run.control,control.allowsMovement,control.block == .unchanged,
              let session=run.session else { throw CameraHeadCorrection.Failure.incompatible }
        let takes=run.takes.filter { !$0.discarded }
        guard takes.count==18,takes.allSatisfy({ SignalCheckAnalysis.metrics($0).usable }) else { throw CameraHeadCorrection.Failure.incomplete }
        if control.motionPolicy == .cameraTeacher {
            guard takes.allSatisfy({ take in
                let pacing=SignalPacing.teacher(prompt:take.prompt,block:control.block)
                return take.endReason == .scheduled && take.pacing == pacing && abs(take.recording.cueEnd-take.recording.cueStart-pacing.duration(words:4))<=0.1
            }) else { throw CameraHeadCorrection.Failure.incomplete }
        }
        for condition in [SpeechCondition.normal,.silent] {
            for sentence in 0..<3 {
                guard Set(takes.filter { $0.prompt.condition==condition && $0.prompt.sentence==sentence }.map { $0.prompt.repetition })==Set(0..<3) else { throw CameraHeadCorrection.Failure.incomplete }
            }
        }
        // Previously examined tests become training data explicitly. The prospective B run
        // must have a new identity/session and is never included in this source pool.
        var sources:[SignalCheckRun]=[]
        for source in history.prefix(2) {
            guard source.id != run.id,source.endedAt != nil,source.session?.id != session.id,
                  source.control?.target.cameraID==control.target.cameraID,
                  source.control?.target.estimator==control.target.estimator,
                  !sources.contains(where:{$0.id==source.id || $0.session?.id==source.session?.id}) else { continue }
            let accepted=source.takes.filter { !$0.discarded }
            guard accepted.count==18,accepted.allSatisfy({SignalCheckAnalysis.metrics($0).usable}) else { continue }
            let keys=accepted.map { "\($0.prompt.condition.rawValue):\($0.prompt.sentence):\($0.prompt.repetition)" }
            guard Set(keys).count==18,accepted.allSatisfy({(0..<3).contains($0.prompt.sentence) && (0..<3).contains($0.prompt.repetition)}) else {continue}
            sources.append(source)
        }
        sources.append(run)
        let sourceTakeIDs=sources.flatMap { $0.takes.filter({!$0.discarded}).map(\.id) }
        guard Set(sourceTakeIDs).count==sourceTakeIDs.count else { throw FrozenSignalReference.Failure.contaminated }
        var frames:[TeacherTakeFrames]=[],manifest:[TrainingSource]=[]
        for (index,source) in sources.enumerated() {
            guard let sourceControl=source.control,let sourceSession=source.session else {continue}
            let camera=TeacherCameraTimeline(sourceControl.readings,cameraID:control.target.cameraID,estimator:control.target.estimator)
            for take in source.takes where !take.discarded {
                var frame=TeacherTakeFrames(take,camera:camera)
                frame.validationGroup=sources.count>1 ? index:nil;frames.append(frame)
            }
            let hash=try trainingHash(source)
            manifest.append(.init(runID:source.id,sessionID:sourceSession.id,startedAt:source.startedAt,sha256:hash))
        }
        var groups:[Group]=[],templates:[Template]=[]
        for condition in [SpeechCondition.normal,.silent] {
            try Task.checkCancellation()
            let source=frames.filter { $0.prompt.condition==condition }
            var model:MouthTeacherModel?,note:String?
            do { model=try MouthTeacherModel.learn(source.map { .init(repetition:$0.prompt.repetition,rows:$0.learning,validationGroup:$0.validationGroup) }) }
            catch is CancellationError { throw CancellationError() }
            catch { note=error.localizedDescription }
            var scales:[String:[Double]]=[:]
            for arm in TeacherArm.allCases {
                if arm == .studentClassifier || arm == .rawClassifier {continue}
                let armTemplates=source.compactMap { frame -> Template? in
                    guard let sequence=frame.sequence(arm,model:model) else { return nil }
                    return .init(takeID:frame.takeID,sentence:frame.prompt.sentence,condition:condition,arm:arm,sequence:sequence)
                }
                if armTemplates.count==source.count {
                    scales[arm.rawValue]=TeacherSequence.scales(armTemplates.map(\.sequence));templates += armTemplates
                } else if arm == .student { note="Some source takes lack complete paired data; the student reference is unavailable." }
            }
            let studentTemplates=templates.filter {$0.condition==condition && $0.arm == .student}
            let classifier=studentTemplates.count==source.count ? try? TeacherLinearClassifier.fit(studentTemplates.map {($0.sentence,$0.sequence)}) : nil
            let rawTemplates=templates.filter {$0.condition==condition && $0.arm == .raw}
            let rawClassifier=rawTemplates.count==source.count ? try? TeacherLinearClassifier.fit(rawTemplates.map {($0.sentence,$0.sequence)}) : nil
            groups.append(.init(condition:condition,model:model,note:note,scales:scales,
                validationScope:sources.count>1 ? "whole recording sessions held out" : "whole repetitions held out · one fit only",classifier:classifier,rawClassifier:rawClassifier,
                currentSensorLocations:Array(Set(takes.filter {$0.prompt.condition==condition}.flatMap {$0.recording.samples.compactMap(\.sensorLocation).filter({$0>0})})).sorted()))
        }
        let reference=Self(id:UUID(),sourceRunID:run.id,sourceSessionID:session.id,testID:control.testID,targetID:control.target.id,
            cameraID:control.target.cameraID,estimator:control.target.estimator,seed:run.seed,frozenAt:Date(),version:1,
            sensorLocations:Array(Set(sources.flatMap { $0.takes.flatMap { $0.recording.samples.compactMap(\.sensorLocation).filter({$0>0}) } })).sorted(),
            trainingSources:manifest,primaryArm:.studentClassifier,baselineArm:.rawClassifier,sentences:SignalCheckRun.sentences,groups:groups,templates:templates)
        try reference.validate();return reference
    }

    func evaluate(_ run:SignalCheckRun) throws -> [TeacherConditionScores] {
        try validate()
        guard let control=run.control,control.block == .reseated,control.allowsMovement,control.testID==testID,
              control.target.id==targetID,control.target.cameraID==cameraID,control.target.estimator==estimator,
              run.seed==seed,run.id != sourceRunID,let session=run.session,session.id != sourceSessionID,
              !trainingSources.contains(where:{$0.runID==run.id || $0.sessionID==session.id}),
              run.startedAt>=frozenAt,control.blockConfirmedAt>=frozenAt,run.reseatedAt.map({$0>=frozenAt})==true else { throw CameraHeadCorrection.Failure.incompatible }
        let takes=run.takes.filter { !$0.discarded }
        let keys=takes.map { "\($0.prompt.condition.rawValue):\($0.prompt.sentence):\($0.prompt.repetition)" }
        guard Set(keys).count==keys.count,Set(takes.map(\.id)).count==takes.count,
              takes.allSatisfy({(0..<3).contains($0.prompt.sentence) && (0..<3).contains($0.prompt.repetition) && $0.recording.condition==$0.prompt.condition}) else { throw FrozenSignalReference.Failure.contaminated }
        // Camera is used only by the separately labelled visual/head controls, not student inference.
        let camera=TeacherCameraTimeline(control.readings,cameraID:cameraID,estimator:estimator)
        var result:[TeacherConditionScores]=[]
        for group in groups {
            let conditionTakes=takes.filter { $0.prompt.condition==group.condition }
            let frames=conditionTakes.map { TeacherTakeFrames($0,camera:camera) }
            var arms:[TeacherArmScore]=[]
            for arm in TeacherArm.allCases {
                let scales:[Double]
                let classifier=arm == .rawClassifier ? group.rawClassifier : arm == .studentClassifier ? group.classifier : nil
                if arm == .studentClassifier || arm == .rawClassifier {guard classifier != nil else {continue};scales=[]}
                else {guard let fitted=group.scales[arm.rawValue] else {continue};scales=fitted}
                var correct=0,evaluated=0,shifted=0,confusion=Array(repeating:Array(repeating:0,count:4),count:3)
                for (take,frame) in zip(conditionTakes,frames) {
                    try Task.checkCancellation()
                    let pacing=control.motionPolicy == .cameraTeacher ? SignalPacing.teacher(prompt:take.prompt,block:control.block) : control.pacing
                    guard SignalCheckAnalysis.metrics(take).usable,take.endReason == .scheduled,
                          control.motionPolicy != .cameraTeacher || take.pacing == pacing,
                          abs(take.recording.cueEnd-take.recording.cueStart-pacing.duration(words:4))<=0.1,
                          let sequence=frame.sequence(arm,model:group.model) else { continue }
                    evaluated += 1
                    let prediction=classifier != nil ? classifier?.predict(sequence) : match(sequence,arm:arm,condition:group.condition,scales:scales)
                    if prediction==take.prompt.sentence { correct += 1 }
                    confusion[take.prompt.sentence][prediction ?? 3] += 1
                    let shiftedPrediction=classifier != nil ? classifier?.predict(sequence.shifted()) : match(sequence.shifted(),arm:arm,condition:group.condition,scales:scales)
                    if shiftedPrediction==take.prompt.sentence {shifted += 1}
                }
                arms.append(.init(arm:arm,correct:correct,evaluated:evaluated,shiftedCorrect:shifted,confusion:confusion))
            }
            let sensors=Array(Set(conditionTakes.flatMap { $0.recording.samples.compactMap(\.sensorLocation).filter({$0>0}) })).sorted()
            result.append(.init(condition:group.condition,collected:conditionTakes.count,arms:arms,sensorLocations:sensors,
                sensorSourceChanged:!group.currentSensorLocations.isEmpty && !sensors.isEmpty && !Set(sensors).isSubset(of:Set(group.currentSensorLocations))))
        }
        return result
    }

    private func match(_ sequence:TeacherSequence,arm:TeacherArm,condition:SpeechCondition,scales:[Double])->Int? {
        let distances=templates.filter { $0.arm==arm && $0.condition==condition }.compactMap { template -> (Int,Double)? in
            guard let cost=sequence.distance(to:template.sequence,scales:scales) else { return nil };return (template.sentence,cost)
        }
        guard Set(distances.map({$0.0}))==Set(0..<3),let best=distances.min(by:{$0.1<$1.1}),
              !distances.contains(where:{$0.0 != best.0 && abs($0.1-best.1)<1e-9}) else { return nil }
        return best.0
    }
}
