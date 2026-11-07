import Foundation
import Observation

struct RepresentationSourceEvent:Codable,Sendable {
    var sessionID:UUID
    var time:Double
    var previousSource:Int?
    var nextSource:Int?
    var gapSeconds:Double
}

struct RepresentationStudy:Codable,Identifiable,Sendable {
    var id=UUID()
    var createdAt=Date()
    var training:[SentencePrompt]
    var evaluation:[SentencePrompt]
    var forbiddenFinalSentences:Set<String>
    var profile:PronunciationProfile
    var takes:[RepresentationTake]=[]
    var sessions:[RecordingSession]=[]
    var evaluationStartedAt:Date?
    var reseatedAt:Date?
    var reference:RepresentationReference?
    var fingerprint:String?
    var predictions:[RepresentationPrediction]?
    var complete=false
    var repetitions=2
    var sourceEvents:[RepresentationSourceEvent]?
    var trainingWords:Set<String> {Set(training.flatMap {Lexicon.words(in:$0.text)})}
    var novelWords:Set<String> {Set(evaluation.flatMap {Lexicon.words(in:$0.text)}).subtracting(trainingWords)}

    static func plan(data:SentenceTrainingData,profile:PronunciationProfile,lexicon:Lexicon)->Self? {
        let forbidden=Set(data.prompts.filter {data.roles[$0.id] == .finalTest}.map(\.id))
        func choose(_ role:DataRole,avoiding:Set<String>)->[SentencePrompt] {
            var pool=data.prompts.filter {$0.kind == .sentence && data.roles[$0.id]==role &&
                LinguisticGraph(text:$0.text,profile:profile,lexicon:lexicon).complete}
            var chosen:[SentencePrompt]=[],covered=Set<String>()
            while chosen.count<4,!pool.isEmpty {
                let ranked=pool.sorted {a,b in
                    func gain(_ p:SentencePrompt)->Int {
                        let graph=LinguisticGraph(text:p.text,profile:profile,lexicon:lexicon)
                        return Set(graph.diphoneCoverage.keys.map(\.key)).subtracting(covered).count +
                            3*Set(Lexicon.words(in:p.text)).subtracting(avoiding).count
                    }
                    let x=gain(a),y=gain(b);return x==y ? a.id<b.id:x>y
                }
                let pick=ranked[0];chosen.append(pick);pool.removeAll {$0.id==pick.id}
                covered.formUnion(LinguisticGraph(text:pick.text,profile:profile,lexicon:lexicon).diphoneCoverage.keys.map(\.key))
            }
            return chosen
        }
        let train=choose(.training,avoiding:[])
        let test=choose(.validation,avoiding:Set(train.flatMap {Lexicon.words(in:$0.text)}))
        guard train.count==4,test.count==4 else {return nil}
        return .init(training:train,evaluation:test,forbiddenFinalSentences:forbidden,profile:profile)
    }
    /// Development errors only. Reserved final-test results are not accepted as an input.
    func requests(lexicon:Lexicon)->[(text:String,reason:String)] {
        var coverage:[String:Double]=[:],phones:[String:Double]=[:]
        for take in takes where reference?.trainingSentences.contains(take.sentenceID)==true && take.discardedReason==nil && take.recording.condition == .silent {
            for (key,n) in LinguisticGraph(text:take.text,profile:profile,lexicon:lexicon).diphoneCoverage {
                coverage[key.key,default:0]+=n
            }
            for (key,n) in LinguisticGraph(text:take.text,profile:profile,lexicon:lexicon).phonemeCoverage {phones[key,default:0]+=n}
        }
        let errors=Set((predictions ?? []).filter {$0.arm == .linguisticAugmented && $0.condition == .silent && !$0.ranked.isEmpty && $0.predicted != $0.expected}.map(\.expected))
        return (training+evaluation).compactMap {prompt in
            let graph=LinguisticGraph(text:prompt.text,profile:profile,lexicon:lexicon)
            let missing=graph.diphoneCoverage.keys.filter {coverage[$0.key,default:0]<2}.map(\.key).sorted()
            let missingPhones=graph.phonemeCoverage.keys.filter {phones[$0,default:0]<2}.sorted()
            if errors.contains(prompt.id) {return (prompt.text,"Development transfer error; collect in a NEW training study, then use different held-out sentences")}
            if !missing.isEmpty {return (prompt.text,"Sparse annotation coverage: "+missing.prefix(3).joined(separator:", "))}
            if !missingPhones.isEmpty {return (prompt.text,"Sparse IPA annotation coverage: "+missingPhones.joined(separator:", "))}
            return nil
        }
    }
}

/// Separate append-only experiments. Never loads final recordings, edits sentence splits,
/// or replaces the earlier camera-teacher reference. Diagnostics use no microphone/video.
@MainActor @Observable
final class RepresentationWorkflow {
    enum Phase {case setup,ready,relax,recording,ended,processing,reseat,finished}
    let camera=CameraPoseService()
    private(set) var profile=PronunciationProfile()
    private(set) var study:RepresentationStudy?
    private(set) var phase:Phase = .setup
    private(set) var error:String?
    private(set) var secondsRemaining=0.0
    private(set) var index=0
    private(set) var block=0
    private(set) var sourceMessage:String?
    private(set) var storageReady=true
    private var activeSession:RecordingSession?
    private var buffer:[MotionSample]=[]
    private var recent:[MotionSample]=[]
    private var readings:[CameraPoseReading]=[]
    private var cueStart:Double?
    private var takeWordFocus:WordFocusGuide?
    private var captureSource:Int?
    private var task:Task<Void,Never>?
    private weak var recorder:SessionRecorder?
    private let root:URL
    private let profileURL:URL
    private let clock:()->Double
    var isAcquiring:Bool {[.ready,.relax,.recording,.ended].contains(phase)}
    var isBusy:Bool {isAcquiring || phase == .processing}
    var canRedo:Bool {phase == .recording || phase == .ended}
    var sourceStable:Bool {MotionSourceSafety.stable(recent,now:clock())}
    var plan:[(SentencePrompt,Int,SpeechCondition)] {
        guard let study else {return []}
        let prompts=block==0 ? study.training:study.evaluation
        var rng=SplitMix64(seed:block==0 ? 0xF0510:0xF0511)
        return (0..<study.repetitions).flatMap {rep in
            prompts.shuffled(using:&rng).flatMap {prompt in [SpeechCondition.normal,.silent].map {(prompt,rep,$0)}}
        }
    }
    var currentText:String {plan.indices.contains(index) ? plan[index].0.text:""}
    var condition:SpeechCondition {plan.indices.contains(index) ? plan[index].2:.normal}
    var activeWordIndex:Int? {
        guard phase == .recording,let cueStart,let takeWordFocus else {return nil}
        return takeWordFocus.wordIndex(elapsed:clock()-cueStart,count:WordFocusGuide.words(in:currentText).count)
    }
    var currentSource:String {
        switch recent.last?.sensorLocation {case 1:"Left AirPod";case 2:"Right AirPod";default:"Source unknown"}
    }
    var canFitSavedTraining:Bool {
        guard !isBusy,storageReady,activeSession==nil,let study,study.reference==nil,study.sessions.count>0 else {return false}
        let takes=study.takes.filter {$0.discardedReason==nil}
        let expected=study.training.flatMap {p in (0..<study.repetitions).flatMap {r in
            [SpeechCondition.normal,.silent].map {"\(p.id):\(r):\($0.rawValue)"}
        }}
        let actual=takes.map {"\($0.sentenceID):\($0.repetition):\($0.recording.condition.rawValue)"}
        return actual.count==expected.count && Set(actual)==Set(expected) && takes.allSatisfy {!$0.camera.isEmpty}
    }
    func fitSavedTraining() {
        guard canFitSavedTraining,let study else {return}
        do {
            // Preserve the exact pre-recovery state as well as every original recording.
            let dir=root.appending(path:study.id.uuidString),url=dir.appending(path:"study.json")
            let bytes=try Data(contentsOf:url)
            try bytes.write(to:dir.appending(path:"before-refit-\(UUID()).json"),options:.withoutOverwriting)
            block=0;error=nil;finishBlock()
        } catch {self.error=error.localizedDescription}
    }
    init(root:URL?=nil,clock:@escaping ()->Double={ProcessInfo.processInfo.systemUptime}) {
        self.root=root ?? SentenceStore.directory.appending(path:"RepresentationStudies")
        profileURL=root?.appending(path:"pronunciation-profile.json") ?? PronunciationProfile.url
        self.clock=clock
        do {
            profile=try PronunciationProfile.load(from:profileURL)
            let latest=self.root.appending(path:"latest.json")
            if FileManager.default.fileExists(atPath:latest.path) {
                study=try JSONDecoder().decode(RepresentationStudy.self,from:Data(contentsOf:latest))
                if var reference=study?.reference {
                    try reference.validate()
                    let frozen=self.root.appending(path:study!.id.uuidString).appending(path:"frozen-reference.json")
                    let immutable=try JSONDecoder().decode(RepresentationReference.self,from:Data(contentsOf:frozen))
                    try immutable.validate()
                    guard try immutable.fingerprint()==study?.fingerprint else {throw LinguisticFailure.incompatible}
                    if try reference.fingerprint() != study?.fingerprint {
                        // An older app can round-trip the embedded reference while
                        // dropping this newer optional field. Recover ONLY that
                        // omission, with the complete frozen fingerprint intact.
                        guard reference.featurePolicy==nil,immutable.featurePolicy != nil else {throw LinguisticFailure.incompatible}
                        reference.featurePolicy=immutable.featurePolicy
                        guard try reference.fingerprint()==study?.fingerprint else {throw LinguisticFailure.incompatible}
                    }
                    try reference.verifyTraining(study!.takes)
                    study?.reference=reference
                }
                if study?.reference != nil && study?.complete==false {phase = .reseat;block=1}
            }
        } catch {self.error=error.localizedDescription;storageReady=false}
        camera.onReading={ [weak self] reading in self?.ingestCamera(reading) }
    }
    func editPronunciation(word:String,variants:String) {
        guard !isBusy,phase != .reseat else {return}
        do {try profile.set(word:word,ipaLines:variants);try profile.save(to:profileURL);error=nil}
        catch {self.error=error.localizedDescription}
    }
    func prepare(data:SentenceTrainingData) {
        guard !isBusy,phase != .reseat,storageReady else {return}
        guard let planned=RepresentationStudy.plan(data:data,profile:profile,lexicon:.bundled) else {
            error="Need four training and four validation sentences with pronunciations. Reserved final sentences are excluded.";return
        }
        var next=planned
        do {
            next.takes=try data.pairs.filter {pair in
                next.training.contains {$0.id==SentencePrompt.identity(pair.text)} &&
                pair.promptIDs.allSatisfy {data.roles[$0] == .training}
            }.flatMap {try $0.representationTakes(profile:profile)}
        } catch {self.error=error.localizedDescription;return}
        study=next;block=0;index=0;phase = .setup;error=nil;save()
    }
    func ingest(_ sample:MotionSample) {
        if let previous=recent.last,let session=activeSession,previous.sensorLocation != sample.sensorLocation {
            if study?.sourceEvents==nil {study?.sourceEvents=[]}
            study?.sourceEvents?.append(.init(sessionID:session.id,time:sample.t,previousSource:previous.sensorLocation,
                nextSource:sample.sensorLocation,gapSeconds:sample.t-previous.t))
            recorder?.logMarker(uptime:sample.t,kind:"airpod-source-change",label:"\(previous.sensorLocation ?? 0) → \(sample.sensorLocation ?? 0)")
        }
        recent.append(sample);recent.removeAll {$0.t<sample.t-3}
        if phase == .relax || phase == .recording {
            let gap=buffer.last.map {sample.t <= $0.t || sample.t-$0.t>0.12} ?? false
            buffer.append(sample)
            if gap || captureSource.map({sample.sensorLocation != $0})==true {
                sourceMessage=gap ? "Motion has a capture gap. Wait for stable motion, then redo this take.":"AirPod source switched. Wait for stable motion, then redo this take."
                if phase == .recording {finish(reason:LinguisticFailure.source.localizedDescription)}
                else {task?.cancel();buffer=[];phase = .ready;error=LinguisticFailure.source.localizedDescription}
            }
        }
    }
    func ingestCamera(_ reading:CameraPoseReading) {
        guard activeSession != nil else {return}
        readings.append(reading)
        if let activeSession { _=recorder?.recordCamera(reading,sessionID:activeSession.id) }
    }
    var cameraReady:Bool {camera.mouthReady(now:clock())}
    func startBlock(audio:AudioService,recorder:SessionRecorder) {
        guard study != nil,!isBusy,storageReady,!recorder.isRecording,sourceStable,cameraReady,
              phase == .setup || phase == .reseat else {return}
        self.recorder=recorder
        if block==1 {
            do {
                guard let reference=study?.reference,try reference.fingerprint()==study?.fingerprint else {throw LinguisticFailure.incompatible}
                try reference.verifyTraining(study!.takes)
            } catch {self.error=error.localizedDescription;return}
        }
        audio.stop()
        recorder.condition = .signalCheck
        recorder.prompt="Raw + camera + linguistic representation pilot; block \(block); no audio/video; sentence/session-disjoint development transfer; final sentence test excluded"
        guard let session=recorder.start(audioSink:nil,cameraPoints:true) else {error=recorder.error;return}
        activeSession=session;study?.sessions.append(session);readings=[];index=0;error=nil;sourceMessage=nil
        if block==1 {
            // A stopped B is replaced as a WHOLE session. Preserve earlier numeric data,
            // but never combine partial sessions or count duplicate repetition cells.
            for i in study!.takes.indices where study!.evaluation.contains(where:{$0.id==study!.takes[i].sentenceID}) {
                study?.takes[i].discardedReason="Earlier interrupted B session; retained but excluded from the replacement session"
            }
            study?.reseatedAt=Date();study?.evaluationStartedAt=Date()
        }
        phase = .ready;save()
    }
    func next() {
        switch phase {
        case .ready:startTake()
        case .recording:finish()
        case .ended:
            guard let study,let last=study.takes.last,last.discardedReason==nil else {error="Redo the interrupted take before continuing.";return}
            if index+1<plan.count {index+=1;phase = .ready}
            else {finishBlock()}
        default:break
        }
    }
    func prepareReplacementB() {
        guard !isBusy,study?.reference != nil,study?.complete==false else {return}
        block=1;phase = .reseat;error=nil
    }
    func startTake() {
        guard phase == .ready,sourceStable,cameraReady else {error="Wait for stable AirPod motion and visible lip markers.";return}
        task?.cancel();buffer=[];captureSource=recent.last?.sensorLocation;error=nil;sourceMessage=nil
        phase = .relax
        task=Task {
            do {
                let end=clock()+3
                while clock()<end {try Task.checkCancellation();secondsRemaining=max(0,end-clock());try await Task.sleep(for:.milliseconds(40))}
                guard phase == .relax,sourceStable,cameraReady else {phase = .ready;error="Tracking changed during the baseline. Start again.";return}
                beginCapture(at:clock())
                guard let cueStart else {return}
                let deadline=cueStart+min(35,max(12,Double(Lexicon.words(in:currentText).count)*1.3+4))
                while phase == .recording,clock()<deadline {
                    try Task.checkCancellation();secondsRemaining=max(0,deadline-clock())
                    if let last=recent.last,clock()-last.t>0.3 {finish(reason:"Motion stopped; redo this take.");return}
                    try await Task.sleep(for:.milliseconds(40))
                }
                if phase == .recording {finish(reason:"Timed out without an explicit END; redo this take.")}
            } catch is CancellationError {} catch {self.error=error.localizedDescription;stop()}
        }
    }
    /// Shared by the timed baseline and deterministic recorder lifecycle tests.
    func beginCapture(at time:Double) {
        guard phase == .relax || phase == .ready,activeSession != nil,time.isFinite else {return}
        captureSource=recent.last?.sensorLocation
        takeWordFocus = .comfortable
        cueStart=time;phase = .recording
        recorder?.logMarker(uptime:time,kind:"representation-start",label:currentText)
    }
    func finish(reason:String?=nil) {
        guard phase == .recording,let cueStart,let session=activeSession,plan.indices.contains(index) else {return}
        task?.cancel()
        let end=clock(),item=plan[index]
        let recording=SentenceRecording(condition:item.2,samples:buffer.filter {$0.t<=end},cueStart:cueStart,cueEnd:end)
        let invalid=reason ?? (MotionSourceSafety.source(recording)==nil ? LinguisticFailure.source.localizedDescription:nil)
        study?.takes.append(.init(sentenceID:item.0.id,text:item.0.text,sessionID:session.id,repetition:item.1,
            recording:recording,camera:[],discardedReason:invalid,wordFocusGuide:takeWordFocus))
        recorder?.logMarker(uptime:end,kind:invalid==nil ? "representation-end":"representation-interrupted",label:currentText)
        phase = .ended;self.cueStart=nil;takeWordFocus=nil;buffer=[];error=invalid;secondsRemaining=0;save()
    }
    func redo() {
        guard canRedo else {return}
        if phase == .recording {finish(reason:"Discarded by Redo")}
        else if let i=study?.takes.indices.last {study?.takes[i].discardedReason="Discarded by Redo"}
        phase = .ready;error=nil;save()
    }
    func stop() {
        guard isBusy || phase == .reseat else {return}
        let ownedSession=activeSession != nil
        task?.cancel()
        if phase == .recording {finish(reason:"Stopped before completing take")}
        closeSession();phase = .finished
        // A window that only loaded an existing study must not overwrite a recording
        // being acquired by another running instance (including app-hosted tests).
        if ownedSession {save()}
        camera.stop()
    }
    private func closeSession() {
        guard let session=activeSession else {return}
        if let closed=recorder?.stop(sessionID:session.id),let i=study?.sessions.firstIndex(where:{$0.id==session.id}) {study?.sessions[i]=closed}
        if let ids=study?.takes.indices {
            for i in ids where study?.takes[i].sessionID==session.id {
                let take=study!.takes[i]
                study?.takes[i].camera=readings.filter {$0.t>=take.recording.cueStart-3 && $0.t<=take.recording.cueEnd+0.3}
            }
        }
        activeSession=nil;readings=[]
    }
    private func finishBlock() {
        phase = .processing
        task=Task {
            do {
                // Retain late numeric landmarks for centred derivatives without extending END.
                try await Task.sleep(for:.milliseconds(500));try Task.checkCancellation()
                closeSession();save()
                guard let snapshot=study,let first=snapshot.takes.first?.camera.first else {throw LinguisticFailure.insufficient}
                if block==0 {
                    let fit=Task.detached(priority:.userInitiated) {
                        try RepresentationReference.freeze(studyID:snapshot.id,takes:snapshot.takes,training:snapshot.training,evaluation:snapshot.evaluation,
                            forbidden:snapshot.forbiddenFinalSentences,profile:snapshot.profile,lexicon:.bundled,cameraID:first.cameraID,estimator:first.estimator)
                    }
                    let reference=try await withTaskCancellationHandler(operation:{try await fit.value},onCancel:{fit.cancel()})
                    try Task.checkCancellation()
                    let fingerprint=try reference.fingerprint()
                    let dir=root.appending(path:snapshot.id.uuidString)
                    try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
                    try JSONEncoder().encode(reference).write(to:dir.appending(path:"frozen-reference.json"),options:.withoutOverwriting)
                    study?.reference=reference;study?.fingerprint=fingerprint;block=1;phase = .reseat
                } else {
                    guard let reference=snapshot.reference,try reference.fingerprint()==snapshot.fingerprint,
                          let started=snapshot.evaluationStartedAt,let reseated=snapshot.reseatedAt else {throw LinguisticFailure.incompatible}
                    let evaluation=snapshot.takes.filter {snapshot.evaluation.map(\.id).contains($0.sentenceID)}
                    let score=Task.detached(priority:.userInitiated) {try reference.evaluate(evaluation,startedAt:started,reseatedAt:reseated,studyID:snapshot.id)}
                    let predictions=try await withTaskCancellationHandler(operation:{try await score.value},onCancel:{score.cancel()})
                    try Task.checkCancellation()
                    study?.predictions=predictions
                    study?.complete=true;phase = .finished;camera.stop()
                }
                save()
            } catch {self.error=error.localizedDescription;phase = .finished;save()}
        }
    }
    private func save() {
        guard let study else {return}
        do {
            let dir=root.appending(path:study.id.uuidString)
            try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
            let e=JSONEncoder();e.outputFormatting=[.sortedKeys]
            let bytes=try e.encode(study)
            try bytes.write(to:dir.appending(path:"study.json"),options:.atomic)
            try bytes.write(to:root.appending(path:"latest.json"),options:.atomic)
        } catch {self.error=error.localizedDescription;storageReady=false}
    }
}
