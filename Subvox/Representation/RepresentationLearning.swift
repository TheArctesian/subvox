import CryptoKit
import Foundation

/// All normalisation and coefficients are source-only. Fixed ridge=1; this pilot has no
/// hyperparameter search, final-test fitting, or online adaptation after reseating.
struct RepresentationRidge: Codable, Sendable {
    var mean:[Double]
    var scale:[Double]
    var weights:[[Double]]
    static func fit(_ x:[[Double]],_ y:[[Double]],baseline:[Bool]?=nil,headMotion:[[Double]]?=nil) throws -> Self {
        guard x.count==y.count,x.count>=4,let p=x.first?.count,let q=y.first?.count,
              (baseline==nil)==(headMotion==nil),(baseline?.count ?? x.count)==x.count,(headMotion?.count ?? x.count)==x.count,
              (1...220).contains(p),(1...100).contains(q),
              x.allSatisfy({$0.count==p && $0.allSatisfy(\.isFinite)}),
              y.allSatisfy({$0.count==q && $0.allSatisfy(\.isFinite)}) else {throw LinguisticFailure.insufficient}
        let n=Double(x.count)
        let mean=(0..<p).map {k in x.reduce(0){$0+$1[k]}/n}
        let scale=(0..<p).map {k in max(1e-4,sqrt(x.reduce(0){$0+pow($1[k]-mean[k],2)}/n))}
        var gram=Array(repeating:Array(repeating:0.0,count:p+1),count:p+1)
        var rhs=Array(repeating:Array(repeating:0.0,count:q),count:p+1)
        var nuisance=Array(repeating:Array(repeating:0.0,count:4),count:p+1)
        var headMean=Array(repeating:0.0,count:4),headScale=Array(repeating:1.0,count:4)
        let baselineIndices=baseline?.indices.filter {baseline![$0]} ?? []
        if let baseline,let headMotion {
            guard baseline.count==x.count,headMotion.count==x.count,baselineIndices.count>=35,
                  headMotion.allSatisfy({$0.count==4 && $0.allSatisfy(\.isFinite)}) else {throw LinguisticFailure.insufficient}
            let bn=Double(baselineIndices.count)
            headMean=(0..<4).map {k in baselineIndices.reduce(0){$0+headMotion[$1][k]}/bn}
            headScale=(0..<4).map {k in max(0.01,sqrt(baselineIndices.reduce(0){$0+pow(headMotion[$1][k]-headMean[k],2)}/bn))}
        }
        for r in x.indices {
            try Task.checkCancellation()
            let z=[1.0]+(0..<p).map {(x[r][$0]-mean[$0])/scale[$0]}
            for i in 0...p {
                for j in 0...p {gram[i][j]+=z[i]*z[j]/n}
                for k in 0..<q {rhs[i][k]+=z[i]*y[r][k]/n}
                if i>0,baseline?[r]==true,let headMotion {
                    for k in 0..<4 {nuisance[i][k]+=z[i]*(headMotion[r][k]-headMean[k])/headScale[k]/Double(baselineIndices.count)}
                }
            }
        }
        for i in 1...p {gram[i][i]+=1}
        for i in 0...p {for j in 0...p {gram[i][j]+=0.5*zip(nuisance[i],nuisance[j]).reduce(0){$0+$1.0*$1.1}}}
        var a=(0...p).map {gram[$0]+rhs[$0]}
        for i in 0...p {
            try Task.checkCancellation()
            let pivot=(i...p).max {abs(a[$0][i])<abs(a[$1][i])}!
            a.swapAt(i,pivot)
            guard abs(a[i][i])>1e-10 else {throw LinguisticFailure.incompatible}
            let value=a[i][i]
            for j in i..<(p+1+q) {a[i][j]/=value}
            for r in 0...p where r != i {
                let factor=a[r][i]
                for j in i..<(p+1+q) {a[r][j]-=factor*a[i][j]}
            }
        }
        return .init(mean:mean,scale:scale,weights:a.map {Array($0.suffix(q))})
    }
    var valid:Bool {
        !mean.isEmpty && mean.count<=220 && scale.count==mean.count && weights.count==mean.count+1 &&
        mean.allSatisfy(\.isFinite) && scale.allSatisfy {$0.isFinite && $0>0} &&
        (1...100).contains(weights.first?.count ?? 0) && weights.allSatisfy {$0.count==weights[0].count && $0.allSatisfy(\.isFinite)}
    }
    func predict(_ row:[Double])->[Double]? {
        guard valid,row.count==mean.count,row.allSatisfy(\.isFinite) else {return nil}
        let x=[1.0]+row.indices.map {(row[$0]-mean[$0])/scale[$0]}
        let result=weights[0].indices.map {k in x.indices.reduce(0){$0+x[$1]*weights[$1][k]}}
        return result.allSatisfy(\.isFinite) ? result:nil
    }
}

enum RepresentationArm:String,Codable,CaseIterable,Sendable {
    case raw, cameraAugmented, linguisticAugmented, shiftedTeacher
    var label:String {
        switch self {
        case .raw:"Raw AirPods"
        case .cameraAugmented:"Raw + camera-taught features"
        case .linguisticAugmented:"Raw + camera + linguistic features"
        case .shiftedTeacher:"Raw + shifted-camera control"
        }
    }
}

/// A switch is a discontinuity, not an axis rotation to guess. Raw rows remain on disk.
enum MotionSourceSafety {
    static func source(_ recording:SentenceRecording)->Int? {
        let samples=recording.samples
        guard let source=samples.first?.sensorLocation,[1,2].contains(source),
              samples.allSatisfy({$0.sensorLocation==source}),
              zip(samples,samples.dropFirst()).allSatisfy({$1.t>$0.t && $1.t-$0.t<=0.12}),
              let first=samples.first,let last=samples.last,
              first.t<=recording.cueStart-2.3,last.t>=recording.cueEnd-0.1,
              recording.cueEnd-recording.cueStart>=0.5,recording.cueEnd-recording.cueStart<=45 else {return nil}
        return source
    }
    static func stable(_ samples:[MotionSample],now:Double,seconds:Double=1.5)->Bool {
        let recent=samples.filter {$0.t>=now-seconds && $0.t<=now}
        guard recent.count>=15,let first=recent.first,let last=recent.last,
              now-last.t<=0.15,last.t-first.t>=seconds-0.12,
              let source=last.sensorLocation,[1,2].contains(source) else {return false}
        return recent.allSatisfy {$0.sensorLocation==source} && zip(recent,recent.dropFirst()).allSatisfy {$1.t>$0.t && $1.t-$0.t<=0.12}
    }
}

struct RepresentationTake:Codable,Identifiable,Sendable {
    var id=UUID()
    var sentenceID:String
    var text:String
    var sessionID:UUID
    var repetition:Int
    var recording:SentenceRecording
    var camera:[CameraPoseReading]
    var discardedReason:String?
    var phoneAlignment:PhoneAlignment?
    /// Optional so historical source digests and frozen references retain their encoding.
    var wordFocusGuide:WordFocusGuide?
}

enum RepresentationEdgePolicy:String,Codable,Sendable {
    case strictV1, observedTailV2
}

struct RepresentationFrames {
    var source:Int
    var raw:[[Double]]=[]
    var invariant:[[Double]]=[]
    var teacherRaw:[[Double]]=[]
    var teacherInvariant:[[Double]]=[]
    var teacher:[[Double]]=[]
    var teacherBaseline:[Bool]=[]
    var teacherHead:[[Double]]=[]
    var cameraCueCount=0
    var phoneRaw:[[Double]]=[]
    var phoneInvariant:[[Double]]=[]
    var phoneTargets:[[Double]]=[]
    init?(_ take:RepresentationTake,cameraID:String,estimator:String?,edgePolicy:RepresentationEdgePolicy = .observedTailV2) {
        guard take.discardedReason==nil,let source=MotionSourceSafety.source(take.recording) else {return nil}
        self.source=source
        let timeline=TeacherMotionTimeline(take.recording.samples)
        let camera=TeacherCameraTimeline(take.camera,cameraID:cameraID,estimator:estimator)
        let start=take.recording.cueStart,end=take.recording.cueEnd
        guard start.isFinite,end.isFinite,end>start,end-start<=45 else {return nil}
        // Motion delivery trails the button press by a few samples. The capture check
        // already bounds missing END coverage to 100 ms. Use only observed grid points;
        // never extrapolate through that short tail or relax internal gap/source checks.
        let supportedEnd=edgePolicy == .observedTailV2 ? min(end,take.recording.samples.last!.t):end
        let steps=Int(floor((supportedEnd-start)/0.04))
        guard steps>=12,Double(steps)*0.04 >= (end-start)*0.9 else {return nil}
        // Relaxed mouth with comfortable head motion supplies a nuisance-only baseline.
        // This penalty changes ONLY the auxiliary branch; raw IMU channels are retained.
        for i in 0..<50 {
            let t=start-2.48+Double(i)*0.04
            if let x=timeline.context(at:t),let v=camera.velocity(at:t) {
                teacherRaw.append(x.body+x.invariant);teacherInvariant.append(x.invariant);teacher.append([0,0,0])
                teacherBaseline.append(true);teacherHead.append(v.head)
            }
        }
        for i in 0...steps {
            let t=start+Double(i)*0.04
            guard let x=timeline.context(at:t) else {return nil}
            let full=x.body+x.invariant
            raw.append(full);invariant.append(x.invariant)
            if let alignment=take.phoneAlignment,alignment.reviewed,alignment.condition==take.recording.condition,
               alignment.provenance.method != .cameraWarpFromAcoustic || alignment.provenance.sourceAlignmentID != nil,
               (try? alignment.validate(in:start...end)) != nil,
               let phone=alignment.phones.first(where:{$0.start<=t && $0.end>t}),
               let gesture=ArticulatoryInventory.byIPA[phone.ipa]?.gesturePrior {
                phoneRaw.append(full);phoneInvariant.append(x.invariant);phoneTargets.append(gesture)
            }
            if let velocity=camera.velocity(at:t) {
                teacherRaw.append(full);teacherInvariant.append(x.invariant);teacher.append(velocity.mouth)
                teacherBaseline.append(false);teacherHead.append(velocity.head);cameraCueCount+=1
            }
        }
    }
    static func pool(_ rows:[[Double]])->[Double]? {
        guard rows.count>=12,let p=rows.first?.count,rows.allSatisfy({$0.count==p && $0.allSatisfy(\.isFinite)}) else {return nil}
        var result:[Double]=[]
        for half in 0..<2 {
            let group=Array(rows[(half*rows.count/2)..<((half+1)*rows.count/2)])
            let mean=(0..<p).map {k in group.reduce(0){$0+$1[k]}/Double(group.count)}
            let sd=(0..<p).map {k in sqrt(group.reduce(0){$0+pow($1[k]-mean[k],2)}/Double(group.count))}
            result+=mean+sd
        }
        return result
    }
}

struct RepresentationReference:Codable,Sendable {
    struct Candidate:Codable,Sendable {var sentenceID:String;var text:String;var graph:LinguisticGraph;var target:[Double]}
    struct Head:Codable,Sendable {
        var condition:SpeechCondition
        /// Nil is the predeclared gravity-invariant fallback, not an inferred sensor side.
        var source:Int?
        var camera:RepresentationRidge?
        var shiftedCamera:RepresentationRidge?
        var linguistic:RepresentationRidge
        var frameLinguistic:RepresentationRidge?
        var decoders:[String:RepresentationRidge]
    }
    var id=UUID()
    var version=1
    var studyID:UUID
    var frozenAt=Date()
    var profile:PronunciationProfile
    var candidates:[Candidate]
    var trainingSentences:Set<String>
    var evaluationSentences:Set<String>
    var forbiddenFinalSentences:Set<String>
    var trainingSessions:Set<UUID>
    var trainingTakeIDs:Set<UUID>
    var trainingDigest:String
    var cameraID:String
    var estimator:String?
    var heads:[Head]
    var sensorOverlaps:[SensorOverlap]
    /// Missing retains the original evaluator for an already frozen v1 model.
    var featurePolicy:RepresentationEdgePolicy?

    func validate() throws {
        let expected=trainingSentences.union(evaluationSentences)
        guard version==1,trainingSentences.count>=4,evaluationSentences.count>=2,
              trainingSentences.isDisjoint(with:evaluationSentences),expected.isDisjoint(with:forbiddenFinalSentences),
              Set(candidates.map(\.sentenceID))==expected,candidates.count==expected.count,
              !trainingSessions.isEmpty,!trainingTakeIDs.isEmpty,trainingDigest.count==64,
              candidates.allSatisfy({$0.sentenceID==SentencePrompt.identity($0.text) && $0.target.count==72 &&
                  $0.target.allSatisfy(\.isFinite) && $0.graph.gestureSignature==$0.target}),
              Set(heads.map {"\($0.condition.rawValue):\($0.source ?? 0)"}).count==heads.count,
              [SpeechCondition.normal,.silent].allSatisfy({c in heads.contains {$0.condition==c && $0.source==nil}}),
              heads.allSatisfy({h in
                  let dims=h.source==nil ? 21:39
                  return (h.source==nil || [1,2].contains(h.source!)) && h.linguistic.valid && h.linguistic.mean.count==dims*4 &&
                      h.linguistic.weights[0].count==8 && (h.frameLinguistic.map {$0.valid && $0.mean.count==dims && $0.weights[0].count==8} ?? true) &&
                      [h.camera,h.shiftedCamera].allSatisfy {$0.map {$0.valid && $0.mean.count==dims && $0.weights[0].count==3} ?? true} &&
                      h.decoders[RepresentationArm.raw.rawValue] != nil && h.decoders.allSatisfy {key,model in
                          guard let arm=RepresentationArm(rawValue:key) else {return false}
                          let p=arm == .raw ? dims*4:(dims+3)*4+(arm == .linguisticAugmented ? (h.frameLinguistic==nil ? 8:32):0)
                          return model.valid && model.mean.count==p && model.weights[0].count==72
                      }
              }) else {throw LinguisticFailure.incompatible}
    }
    func verifyTraining(_ takes:[RepresentationTake]) throws {
        let source=takes.filter {trainingSentences.contains($0.sentenceID)}
        let e=JSONEncoder();e.outputFormatting=[.sortedKeys]
        let digest=SHA256.hash(data:try e.encode(source)).map {String(format:"%02x",$0)}.joined()
        guard digest==trainingDigest,Set(source.map(\.id))==trainingTakeIDs else {throw LinguisticFailure.incompatible}
    }

    func fingerprint() throws -> String {
        // Sets encode in unspecified order; canonicalise the complete JSON before hashing.
        let e=JSONEncoder();e.outputFormatting=[.sortedKeys]
        var object=try JSONSerialization.jsonObject(with:e.encode(self)) as! [String:Any]
        for key in ["trainingSentences","evaluationSentences","forbiddenFinalSentences"] {
            object[key]=(object[key] as? [String])?.sorted()
        }
        for key in ["trainingSessions","trainingTakeIDs"] {object[key]=(object[key] as? [String])?.sorted()}
        return SHA256.hash(data:try JSONSerialization.data(withJSONObject:object,options:.sortedKeys)).map {String(format:"%02x",$0)}.joined()
    }
    static func freeze(studyID:UUID,takes:[RepresentationTake],training:[SentencePrompt],evaluation:[SentencePrompt],
                       forbidden:Set<String>,profile:PronunciationProfile,lexicon:Lexicon,cameraID:String,estimator:String?) throws -> Self {
        let profileHash=try profile.fingerprint()
        let train=Set(training.map(\.id)),test=Set(evaluation.map(\.id))
        guard train.count>=4,test.count>=2,train.isDisjoint(with:test),train.isDisjoint(with:forbidden),test.isDisjoint(with:forbidden),
              Set(takes.map(\.id)).count==takes.count,takes.allSatisfy({train.contains($0.sentenceID) && SentencePrompt.identity($0.text)==$0.sentenceID &&
                  ($0.phoneAlignment.map {$0.provenance.pronunciationSHA256==profileHash} ?? true)}) else {throw LinguisticFailure.split}
        let candidates=try (training+evaluation).map {prompt -> Candidate in
            let graph=LinguisticGraph(text:prompt.text,profile:profile,lexicon:lexicon)
            guard let target=graph.gestureSignature else {throw LinguisticFailure.pronunciation}
            return .init(sentenceID:prompt.id,text:prompt.text,graph:graph,target:target)
        }
        var heads:[Head]=[]
        for condition in [SpeechCondition.normal,.silent] {
            let source=takes.filter {$0.recording.condition==condition && $0.discardedReason==nil}
            let frames=source.compactMap {t in RepresentationFrames(t,cameraID:cameraID,estimator:estimator).map {(t,$0)}}
            guard frames.count==source.count,Set(source.map(\.sentenceID))==train,
                  train.allSatisfy({id in source.filter {$0.sentenceID==id}.count>=2}) else {throw LinguisticFailure.insufficient}
            let sides=Set(frames.map {$0.1.source}).sorted()
            for side in [Int?](arrayLiteral:nil)+sides.map(Optional.some) {
                let selected=frames.filter {side==nil || $0.1.source==side}
                guard Set(selected.map {$0.0.sentenceID})==train,
                      train.allSatisfy({id in selected.filter {$0.0.sentenceID==id}.count>=2}) else {continue}
                let x=selected.map {side==nil ? $0.1.invariant:$0.1.raw}
                let y=selected.map {t,_ in candidates.first {$0.sentenceID==t.sentenceID}!.target}
                let pooled=try x.map {rows -> [Double] in guard let z=RepresentationFrames.pool(rows) else {throw LinguisticFailure.insufficient};return z}
                let linguistic=try RepresentationRidge.fit(pooled,y.map {Array($0.prefix(8))})
                var cx:[[Double]]=[],cy:[[Double]]=[],shifted:[[Double]]=[],headMotion:[[Double]]=[],baseline:[Bool]=[]
                for (_,f) in selected where f.teacherBaseline.filter({$0}).count>=35 && Double(f.cameraCueCount)/Double(f.raw.count)>=0.9 {
                    cx += side==nil ? f.teacherInvariant:f.teacherRaw;cy += f.teacher
                    headMotion += f.teacherHead;baseline += f.teacherBaseline
                    var null=f.teacher
                    let active=f.teacher.indices.filter {!f.teacherBaseline[$0]}
                    for (i,index) in active.enumerated() {null[index]=f.teacher[active[(i+active.count/2)%active.count]]}
                    shifted += null
                }
                let teacher=cx.count>=100 ? try RepresentationRidge.fit(cx,cy,baseline:baseline,headMotion:headMotion):nil
                let nullTeacher=cx.count>=100 ? try RepresentationRidge.fit(cx,shifted,baseline:baseline,headMotion:headMotion):nil
                let px=selected.flatMap {side==nil ? $0.1.phoneInvariant:$0.1.phoneRaw},py=selected.flatMap {$0.1.phoneTargets}
                let frameLinguistic=px.count>=100 ? try RepresentationRidge.fit(px,py):nil
                var head=Head(condition:condition,source:side,camera:teacher,shiftedCamera:nullTeacher,linguistic:linguistic,frameLinguistic:frameLinguistic,decoders:[:])
                for arm in RepresentationArm.allCases {
                    let inputs=x.compactMap {features($0,head:head,arm:arm)}
                    if inputs.count==x.count {head.decoders[arm.rawValue]=try RepresentationRidge.fit(inputs,y)}
                }
                heads.append(head)
            }
        }
        let e=JSONEncoder();e.outputFormatting=[.sortedKeys]
        let digest=SHA256.hash(data:try e.encode(takes)).map {String(format:"%02x",$0)}.joined()
        let reference=Self(studyID:studyID,profile:profile,candidates:candidates,trainingSentences:train,evaluationSentences:test,
            forbiddenFinalSentences:forbidden,trainingSessions:Set(takes.map(\.sessionID)),trainingTakeIDs:Set(takes.map(\.id)),
            trainingDigest:digest,cameraID:cameraID,estimator:estimator,heads:heads,
            sensorOverlaps:SensorObservability.learn(takes.flatMap {SensorObservability.examples($0)}),featurePolicy:.observedTailV2)
        try reference.validate();return reference
    }
    static func features(_ raw:[[Double]],head:Head,arm:RepresentationArm)->[Double]? {
        guard let base=RepresentationFrames.pool(raw) else {return nil}
        if arm == .raw {return base}
        let model=arm == .shiftedTeacher ? head.shiftedCamera:head.camera
        guard let model else {return nil}
        let predicted=raw.compactMap {model.predict($0)}
        guard predicted.count==raw.count else {return nil}
        // The original 21/39 features remain intact in EVERY frame.
        let augmented=zip(raw,predicted).map {$0+$1}
        guard let camera=RepresentationFrames.pool(augmented) else {return nil}
        if arm == .linguisticAugmented {
            if let frameModel=head.frameLinguistic {
                let predictions=raw.compactMap {frameModel.predict($0)}
                guard predictions.count==raw.count,let gesture=RepresentationFrames.pool(predictions) else {return nil}
                return camera+gesture
            }
            guard let gesture=head.linguistic.predict(base) else {return nil};return camera+gesture
        }
        return camera
    }

    func evaluate(_ takes:[RepresentationTake],startedAt:Date,reseatedAt:Date,studyID:UUID) throws -> [RepresentationPrediction] {
        try validate()
        let accepted=takes.filter {$0.discardedReason==nil}
        let keys=accepted.map {"\($0.sentenceID):\($0.recording.condition.rawValue):\($0.repetition)"}
        guard version==1,self.studyID==studyID,startedAt>=frozenAt,reseatedAt>=frozenAt,
              Set(keys).count==keys.count,Set(accepted.map(\.sessionID)).count<=1,accepted.allSatisfy({(0..<2).contains($0.repetition)}),
              trainingSentences.isDisjoint(with:evaluationSentences),evaluationSentences.isDisjoint(with:forbiddenFinalSentences),
              Set(takes.map(\.id)).count==takes.count,
              takes.allSatisfy({evaluationSentences.contains($0.sentenceID) && !trainingSessions.contains($0.sessionID) && !trainingTakeIDs.contains($0.id) && SentencePrompt.identity($0.text)==$0.sentenceID}) else {throw LinguisticFailure.split}
        return takes.filter {$0.discardedReason==nil}.flatMap {take -> [RepresentationPrediction] in
            let frame=RepresentationFrames(take,cameraID:cameraID,estimator:estimator,edgePolicy:featurePolicy ?? .strictV1)
            let head=frame.flatMap {f in heads.first {$0.condition==take.recording.condition && $0.source==f.source} ?? heads.first {$0.condition==take.recording.condition && $0.source==nil}}
            return RepresentationArm.allCases.map {arm in
                var ranks:[RepresentationRank]=[]
                if let frame,let head,let x=Self.features(head.source==nil ? frame.invariant:frame.raw,head:head,arm:arm),
                   let predicted=head.decoders[arm.rawValue]?.predict(x) {
                    // Full predeclared candidate bank, including training and unseen sentences.
                    ranks=candidates.map {c in .init(sentenceID:c.sentenceID,text:c.text,cost:sqrt(zip(c.target,predicted).reduce(0){$0+pow($1.0-$1.1,2)}/Double(predicted.count)))}.sorted {$0.cost<$1.cost}
                }
                let margin=ranks.count>=2 ? max(0,(ranks[1].cost-ranks[0].cost)/max(1e-9,ranks[1].cost)):0
                let ambiguous=ranks.count<2 || abs(ranks[0].cost-ranks[1].cost)<1e-8
                return .init(takeID:take.id,arm:arm,expected:take.sentenceID,condition:take.recording.condition,source:frame?.source,
                    invariantFallback:head?.source==nil,ranked:Array(ranks.prefix(3)),predicted:ambiguous ? nil:ranks.first?.sentenceID,
                    margin:margin,reason:frame==nil ? "Invalid motion/source continuity":head==nil ? "No frozen source/fallback head":ranks.isEmpty ? "Auxiliary feature model unavailable":ambiguous ? "Indistinguishable candidate gestures; abstained":nil)
            }
        }
    }
}

struct RepresentationRank:Codable,Sendable {var sentenceID:String;var text:String;var cost:Double}
struct RepresentationPrediction:Codable,Sendable {
    var takeID:UUID
    var arm:RepresentationArm
    var expected:String
    var condition:SpeechCondition
    var source:Int?
    var invariantFallback:Bool
    var ranked:[RepresentationRank]
    var predicted:String?
    /// Relative score margin, NOT a calibrated probability.
    var margin:Double
    var reason:String?
}
