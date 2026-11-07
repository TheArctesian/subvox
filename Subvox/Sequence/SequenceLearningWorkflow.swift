import CryptoKit
import Foundation
import Observation

struct SequenceLanguage:Codable,Sendable {
    var version=1
    var inventory:[ArticulatoryPhone]
    var profile:PronunciationProfile
    var entries:[String:[PronunciationVariant]]
    var diagnosticSentences:[String]
    static func export(to url:URL) throws {
        let profile=try PronunciationProfile.load(),lexicon=Lexicon.bundled
        var entries:[String:[PronunciationVariant]]=[:]
        for word in lexicon.words {
            let values=profile.alternatives(word,lexicon:lexicon)
            if !values.isEmpty {entries[word]=values}
        }
        let value=Self(inventory:ArticulatoryInventory.all,profile:profile,entries:entries,diagnosticSentences:SignalCheckRun.sentences)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        try encoder.encode(value).write(to:url,options:.withoutOverwriting)
    }
}

struct NeuralComparison:Decodable {
    struct Score:Decodable {
        var condition:String;var experiment:String;var utterances:Int;var negativeExamples:Int
        var phonemeErrorRate:Double?;var wordErrorRateNoLm:Double?;var wordErrorRateLm:Double?
        var rejectionRate:Double?;var falseAcceptRate:Double?
        var folds:Int?;var refitEpochs:Int?;var wordErrorRateNoLmHomophonesMerged:Double?;var wordErrorRateTransitions:Double?
    }
    var models:[String]
    var comparisons:[Score]
    var finalTestConsumed:Bool
    var schema:Int?
}
struct NeuralPrediction:Decodable {
    var text:String;var textWithLm:String;var phonemes:[String];var accepted:Bool
    var margin:Double;var activityScore:Double;var invariantFallback:Bool;var rejectionReason:String?
    var homophones:[String]?;var textWithTransitions:String?
}
struct NeuralVisit:Codable,Identifiable,Sendable {
    struct Planned:Codable,Sendable {var text:String;var condition:SpeechCondition;var repetition:Int}
    var id=UUID();var startedUnix=Date().timeIntervalSince1970;var endedUnix:Double?
    var role:String
    var session:RecordingSession
    var planned:[Planned]
    var takes:[RepresentationTake]=[]
    var modelPaths:[String:String]
    var modelFingerprints:[String:String]
    var complete=false
    var modalities=["AirPods motion"]
    var continuedFrom:UUID?
    var movementGuideVersion:String?
    var carriedCompletedCount:Int?
}

/// Owns local Python processes and independent motion-only acquisition. Original experiments
/// and their frozen weights are read-only inputs to new versioned ML datasets/runs.
@MainActor @Observable final class SequenceLearningWorkflow {
    enum Phase {case idle,working,ready,relax,recording,ended}
    enum Mode:String,CaseIterable,Identifiable {
        case training,development,sameFit,reseated,inference,nonSpeech
        var id:String {rawValue}
        var title:String {switch self {
        case .training:"Collect training sentences";case .development:"Collect development sentences"
        case .sameFit:"Fresh test · AirPods unchanged";case .reseated:"Fresh test · I’ve reseated"
        case .inference:"Try frozen neural model"
        case .nonSpeech:"Record non-speech motion"
        }}
    }
    private(set) var phase:Phase = .idle
    private(set) var error:String?
    private(set) var status="Ready for local neural training"
    private(set) var datasetURL:URL?
    private(set) var datasetSchema:Int?
    /// Mirrors `load_dataset` in ml/subvox_ml/data.py, which accepts any snapshot with cross-validation folds (schema 2+).
    static let minimumDatasetSchema=2
    var datasetCompatible:Bool {(datasetSchema ?? 0)>=Self.minimumDatasetSchema}
    private(set) var runURL:URL?
    private(set) var comparison:NeuralComparison?
    private(set) var prediction:NeuralPrediction?
    private(set) var freshEvaluationSummary=""
    private(set) var visit:NeuralVisit?
    private(set) var resumableNonSpeech:NeuralVisit?
    var resumableNonSpeechCount:Int {resumableNonSpeech.map {remainingNonSpeech($0).count} ?? 0}
    private(set) var index=0
    var displayIndex:Int {index+1+(visit?.carriedCompletedCount ?? 0)}
    var displayTotal:Int {(visit?.planned.count ?? 0)+(visit?.carriedCompletedCount ?? 0)}
    private(set) var secondsRemaining=0.0
    var epochs=60
    var experiment="plain"
    var mode:Mode = .training
    var inferenceCondition:SpeechCondition = .silent
    var expectedText=""
    var correctionID=""
    var correctionText=""
    var negativeRole:DataRole = .training
    static let negativePrompts=NonSpeechMovement.allCases.map(\.instruction)
    var recordingNonSpeech:Bool {visit?.role=="negativeTraining" || visit?.role=="negativeDevelopment"}
    var currentMovement:NonSpeechMovement? {recordingNonSpeech ? NonSpeechMovement.allCases.first {$0.instruction==currentText}:nil}
    /// Started only for models that take camera landmarks as input (`modalities` camera/fused).
    let camera=CameraPoseService()
    private(set) var faceTracked=false
    private var cameraReadings:[CameraPoseReading]=[]
    private var capturingCamera=false
    private let root:URL
    private var task:Task<Void,Never>?
    private var process:Process?
    private var recorder:SessionRecorder?
    private var buffer:[MotionSample]=[]
    private var latest:MotionSample?
    private var cueStart:Double?
    private var selectedSource:Int?
    private let clock:()->Double
    var isAcquiring:Bool {[.ready,.relax,.recording,.ended].contains(phase)}
    var isBusy:Bool {phase != .idle}
    var currentText:String {visit.flatMap {$0.planned.indices.contains(index) ? $0.planned[index].text:nil} ?? ""}
    var condition:SpeechCondition {visit.flatMap {$0.planned.indices.contains(index) ? $0.planned[index].condition:nil} ?? inferenceCondition}
    var activeWordIndex:Int? {
        guard phase == .recording,let cueStart else {return nil}
        return WordFocusGuide.comfortable.wordIndex(elapsed:clock()-cueStart,count:WordFocusGuide.words(in:currentText).count)
    }
    var canRedo:Bool {phase == .recording || phase == .ended}
    var takeIssue:String? {phase == .ended ? visit?.takes.last?.discardedReason:nil}
    var requiresRedo:Bool {takeIssue != nil}
    var canAdvance:Bool {
        phase == .ready || phase == .recording || (phase == .ended && visit?.takes.last != nil && !requiresRedo)
    }
    var captureDuration:Double {recordingNonSpeech ? 8:35}
    var hasModel:Bool {modelURL(condition:.silent) != nil && modelURL(condition:.normal) != nil}
    init(root:URL?=nil,clock:@escaping ()->Double={ProcessInfo.processInfo.systemUptime}) {
        self.root=root ?? SentenceStore.directory.appending(path:"SequenceLearning")
        self.clock=clock
        do {
            let state=self.root.appending(path:"latest-ui.json")
            if FileManager.default.fileExists(atPath:state.path) {
                let value=try JSONDecoder().decode([String:String].self,from:Data(contentsOf:state))
                datasetURL=value["dataset"].flatMap {$0.isEmpty ? nil:URL(fileURLWithPath:$0)}
                runURL=value["run"].flatMap {$0.isEmpty ? nil:URL(fileURLWithPath:$0)}
                loadDatasetSchema()
                try loadComparison()
            }
        } catch {self.error=error.localizedDescription}
        loadResumableNonSpeech()
        camera.onReading={[weak self] reading in
            guard let self else {return}
            faceTracked=reading.mouth != nil
            guard capturingCamera,phase == .relax || phase == .recording else {return}
            cameraReadings.append(reading)
            if let id=visit?.session.id {_=recorder?.recordCamera(reading,sessionID:id)}
        }
    }
    /// The frozen model's input set, from its manifest: "imu", "camera" or "fused".
    func modalities(condition:SpeechCondition)->String {
        guard let url=modelURL(condition:condition),let bytes=try? Data(contentsOf:url.appending(path:"model.json")),
              let manifest=(try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any] else {return "imu"}
        return manifest["modalities"] as? String ?? "imu"
    }
    /// Camera is captured for a visit whose frozen model reads it; training collection stays motion-only.
    var needsCamera:Bool {
        switch mode {
        case .inference:modalities(condition:inferenceCondition) != "imu"
        case .sameFit,.reseated:[SpeechCondition.normal,.silent].contains {modalities(condition:$0) != "imu"}
        default:false
        }
    }
    private func remainingNonSpeech(_ value:NeuralVisit)->[NeuralVisit.Planned] {
        value.planned.filter {item in !value.takes.contains {take in
            take.discardedReason==nil && take.sentenceID==SentencePrompt.identity(item.text) && take.repetition==item.repetition && take.recording.condition==item.condition
        }}
    }
    private func loadResumableNonSpeech() {
        let folders=(try? FileManager.default.contentsOfDirectory(at:root.appending(path:"Collections"),includingPropertiesForKeys:nil)) ?? []
        let visits=folders.compactMap {url -> NeuralVisit? in
            guard let bytes=try? Data(contentsOf:url.appending(path:"session.json")) else {return nil}
            return try? JSONDecoder().decode(NeuralVisit.self,from:bytes)
        }
        let continued=Set(visits.compactMap(\.continuedFrom))
        resumableNonSpeech=visits.filter {$0.complete && $0.endedUnix != nil && ($0.role=="negativeTraining" || $0.role=="negativeDevelopment") && !continued.contains($0.id) && !remainingNonSpeech($0).isEmpty}
            .max {$0.startedUnix<$1.startedUnix}
    }
    func resumeNonSpeech(audio:AudioService,recorder:SessionRecorder) {
        // Ready is also a preview/waiting state. Actual capture still requires
        // fresh sensor samples in startTake; no cue is recorded while offline.
        guard !isBusy,!recorder.isRecording,let previous=resumableNonSpeech else {return}
        let remaining=remainingNonSpeech(previous)
        guard !remaining.isEmpty else {return}
        audio.stop();recorder.condition = .signalCheck;recorder.prompt="Continue non-speech motion; microphone/camera/video off"
        guard let session=recorder.start(audioSink:nil,cameraPoints:false) else {error=recorder.error;return}
        self.recorder=recorder;mode = .nonSpeech
        negativeRole=previous.role=="negativeTraining" ? .training:.validation
        let carried=(previous.carriedCompletedCount ?? 0)+previous.planned.count-remaining.count
        visit=NeuralVisit(role:previous.role,session:session,planned:remaining,modelPaths:[:],modelFingerprints:[:],continuedFrom:previous.id,movementGuideVersion:"illustrated-v1",carriedCompletedCount:carried)
        resumableNonSpeech=nil;index=0;error=nil;secondsRemaining=0;phase = .ready;saveVisit()
    }
    private func loadDatasetSchema() {
        guard let datasetURL,let bytes=try? Data(contentsOf:datasetURL.appending(path:"manifest.json")),
              let value=(try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any] else {datasetSchema=nil;return}
        datasetSchema=value["schema"] as? Int ?? 1
    }
    private func loadComparison() throws {
        guard let runURL else {return}
        let decoder=JSONDecoder();decoder.keyDecodingStrategy = .convertFromSnakeCase
        comparison=try decoder.decode(NeuralComparison.self,from:Data(contentsOf:runURL.appending(path:"comparison.json")))
    }
    func modelURL(condition:SpeechCondition)->URL? {
        guard let runURL else {return nil}
        let name=experiment+(condition == .silent ? "-mouthed":"-spoken")
        let url=comparison?.models.first(where:{URL(fileURLWithPath:$0).lastPathComponent==name}).map {URL(fileURLWithPath:$0)} ?? runURL.appending(path:name)
        return FileManager.default.fileExists(atPath:url.appending(path:"weights.pt").path) ? url:nil
    }
    private func persistUI() throws {
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let value=["dataset":datasetURL?.path ?? "","run":runURL?.path ?? ""]
        try JSONEncoder().encode(value).write(to:root.appending(path:"latest-ui.json"),options:.atomic)
    }
    func prepareDataset() {
        guard !isBusy else {return}
        phase = .working;error=nil
        task=Task {
            do {
                try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
                let language=root.appending(path:"language-\(UUID()).json"),destination=root.appending(path:"datasets/\(UUID())")
                try SequenceLanguage.export(to:language)
                var args=["prepare","--root",SentenceStore.directory.path,"--language",language.path,"--output",destination.path]
                let corrections=root.appending(path:"label-corrections.json")
                if FileManager.default.fileExists(atPath:corrections.path) {args += ["--corrections",corrections.path]}
                try await execute(args)
                datasetURL=destination;loadDatasetSchema();try persistUI();status="Cross-validation dataset prepared with disjoint sentence groups and sessions within each fold."
            } catch {self.error=error.localizedDescription}
            phase = .idle
        }
    }
    func train() {
        guard !isBusy,let datasetURL else {return}
        guard datasetCompatible else {error="This saved dataset predates cross-validation. Prepare a new snapshot; the old dataset and models will stay intact.";return}
        phase = .working;error=nil
        let destination=root.appending(path:"runs/\(UUID())"),epochs=epochs
        task=Task {
            do {
                try await execute(["train","--dataset",datasetURL.path,"--output",destination.path,"--epochs",String(epochs)])
                runURL=destination;try loadComparison();try persistUI();status="Neural comparisons trained and frozen. See development errors below."
            } catch {self.error=error.localizedDescription}
            phase = .idle
        }
    }
    /// IMU only, camera only and fused, on the same camera-covered recordings, folds and settings.
    func trainFusion() {
        guard !isBusy,let datasetURL else {return}
        guard datasetCompatible else {error="Prepare a new dataset snapshot first; older snapshots have no camera inputs.";return}
        phase = .working;error=nil
        let destination=root.appending(path:"runs/fusion-\(UUID())"),epochs=epochs
        task=Task {
            do {
                try await execute(["train","--dataset",datasetURL.path,"--output",destination.path,"--epochs",String(epochs),"--experiments","imu,camera,fused"])
                runURL=destination;experiment="fused";try loadComparison();try persistUI()
                status="Fusion comparison trained: AirPods only, camera only and fused, on the same recordings."
            } catch {self.error=error.localizedDescription}
            phase = .idle
        }
    }
    func trainWithReviewedTeachers() {
        guard !isBusy,let datasetURL,let base=runURL,comparison != nil else {return}
        guard datasetCompatible else {error="Prepare a new cross-validation dataset and train its base comparisons first.";return}
        phase = .working;error=nil
        let dataset=root.appending(path:"datasets/teachers-\(UUID())"),destination=root.appending(path:"runs/teachers-\(UUID())")
        task=Task {
            do {
                try await execute(["prepare-teachers","--dataset",datasetURL.path,"--base-run",base.path,"--output",dataset.path])
                try await execute(["train","--dataset",dataset.path,"--output",destination.path,"--epochs",String(epochs),"--experiments","teacher"])
                runURL=destination;try loadComparison();try persistUI();status="Separate teacher-supervised comparison saved. Camera/audio remain absent at inference."
            } catch {self.error=error.localizedDescription}
            phase = .idle
        }
    }
    func correctLabel() {
        guard !isBusy,!correctionID.isEmpty else {return}
        do {
            let url=root.appending(path:"label-corrections.json")
            var values=FileManager.default.fileExists(atPath:url.path) ? try JSONDecoder().decode([String:String].self,from:Data(contentsOf:url)):[:]
            values[correctionID]=correctionText
            try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
            try JSONEncoder().encode(values).write(to:url,options:.atomic)
            status="Correction saved as a sidecar. Prepare a NEW dataset/run to use it; existing recordings and models are unchanged."
        } catch {self.error=error.localizedDescription}
    }

    private func runtime() throws -> URL {
        let bundled=Bundle.main.resourceURL!.appending(path:"SequenceRuntime")
        let development=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appending(path:"ml")
        let isBundled=FileManager.default.fileExists(atPath:bundled.appending(path:"uv.lock").path)
        let source=isBundled ? bundled:development
        // evaluation.py reads the distractor bank from <runtime>/Corpus, so it travels with the runtime.
        let corpus=isBundled ? Bundle.main.resourceURL!.appending(path:"Corpus"):development.deletingLastPathComponent().appending(path:"Corpus")
        let python=["pyproject.toml","uv.lock"]+(try FileManager.default.contentsOfDirectory(atPath:source.appending(path:"subvox_ml").path)).filter {$0.hasSuffix(".py")}.sorted().map {"subvox_ml/"+$0}
        let files=python.map {($0,source.appending(path:$0))}+[("Corpus/diphones.txt",corpus.appending(path:"diphones.txt"))]
        var hash=SHA256()
        for (file,url) in files {hash.update(data:Data(file.utf8));hash.update(data:try Data(contentsOf:url))}
        let identity=hash.finalize().map {String(format:"%02x",$0)}.joined()
        let destination=root.appending(path:"Runtime/\(identity)")
        if !FileManager.default.fileExists(atPath:destination.path) {
            for directory in ["subvox_ml","Corpus"] {try FileManager.default.createDirectory(at:destination.appending(path:directory),withIntermediateDirectories:true)}
            for (file,url) in files {try FileManager.default.copyItem(at:url,to:destination.appending(path:file))}
        }
        return destination
    }
    private func execute(_ arguments:[String]) async throws {
        let uv=["/opt/homebrew/bin/uv","/usr/local/bin/uv"].first {FileManager.default.isExecutableFile(atPath:$0)}
        guard let uv else {throw Failure.runtime}
        let project=try runtime(),child=Process(),log=root.appending(path:"process-\(UUID()).log")
        FileManager.default.createFile(atPath:log.path,contents:nil)
        let handle=try FileHandle(forWritingTo:log);defer {try? handle.close()}
        child.executableURL=URL(fileURLWithPath:uv);child.arguments=["run","--frozen","--project",project.path,"subvox-ml"]+arguments
        child.standardOutput=handle;child.standardError=handle
        try child.run();process=child
        defer {if child.isRunning {child.terminate()};process=nil}
        while child.isRunning {
            try Task.checkCancellation()
            if let data=try? Data(contentsOf:log),let line=String(data:data,encoding:.utf8)?.split(separator:"\n").last {
                status=line.hasPrefix("{") ? "Finishing local operation…":String(line.prefix(200))
            }
            try await Task.sleep(for:.milliseconds(300))
        }
        try Task.checkCancellation()
        guard child.terminationStatus==0 else {
            let detail=(try? String(contentsOf:log,encoding:.utf8)).map {String($0.suffix(1600))} ?? "See \(log.path)"
            throw Failure.backend(detail)
        }
    }

    func ingest(_ sample:MotionSample) {
        latest=sample
        if phase == .relax || phase == .recording {
            let changed=selectedSource.map {$0 != sample.sensorLocation} ?? false
            let gap=buffer.last.map {sample.t-$0.t>0.12 || sample.t<=$0.t} ?? false
            buffer.append(sample)
            if changed || gap {
                if phase == .recording {finishTake(reason:"Source changed or motion has a gap; redo this take")}
                else {task?.cancel();buffer=[];phase = .ready;error="Motion source changed; start the baseline again."}
            }
        }
    }
    var collectionDirectory:URL? {visit.map {root.appending(path:"Collections/\($0.id)")}}
    func trainingPrompts(data:SentenceTrainingData)->[SentencePrompt] {
        var coverage:[String:Double]=[:],diphones:[String:Double]=[:],errors:[String:Double]=[:],used=Set<String>()
        if let model=modelURL(condition:.silent),let bytes=try? Data(contentsOf:model.appending(path:"model.json")),
           let json=(try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any] {
            coverage=json["coverage"] as? [String:Double] ?? [:];diphones=json["diphone_coverage"] as? [String:Double] ?? [:]
            used=Set(json["training_groups"] as? [String] ?? [])
            if let bytes=try? Data(contentsOf:model.appending(path:"development-report.json")),let report=(try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any],
               let summary=report["summary"] as? [String:Any],let confusions=summary["phone_confusions"] as? [String:Double] {
                for (key,count) in confusions {if let phone=key.components(separatedBy:" → ").first {errors[phone,default:0]+=count}}
            }
        }
        let profile=(try? PronunciationProfile.load()) ?? PronunciationProfile()
        var pool=data.prompts.filter {$0.kind == .sentence && data.roles[$0.id] == .training}
        if pool.filter({!used.contains($0.id)}).count>=4 {pool.removeAll {used.contains($0.id)}}
        var selected:[SentencePrompt]=[]
        while selected.count<4,!pool.isEmpty {
            func score(_ prompt:SentencePrompt)->Double {
                let graph=LinguisticGraph(text:prompt.text,profile:profile,lexicon:.bundled)
                return graph.phonemeCoverage.reduce(0) {$0+min($1.value,max(0,4-coverage[$1.key,default:0]))*(1+min(3,errors[$1.key,default:0]))} +
                    graph.diphoneCoverage.reduce(0) {$0+min($1.value,max(0,2-diphones[$1.key.key,default:0]))}
            }
            let choice=pool.sorted {a,b in let x=score(a),y=score(b);return x==y ? a.id<b.id:x>y}[0]
            selected.append(choice);pool.removeAll {$0.id==choice.id}
            let graph=LinguisticGraph(text:choice.text,profile:profile,lexicon:.bundled)
            for (p,n) in graph.phonemeCoverage {coverage[p,default:0]+=n}
            for (p,n) in graph.diphoneCoverage {diphones[p.key,default:0]+=n}
        }
        return selected
    }
    func startVisit(data:SentenceTrainingData,audio:AudioService,recorder:SessionRecorder) {
        guard !isBusy,!recorder.isRecording,let latest,clock()-latest.t>=0,clock()-latest.t<0.3 else {return}
        let role=mode
        var prompts:[SentencePrompt]
        if role == .nonSpeech {
            prompts=Self.negativePrompts.map {SentencePrompt(text:$0)}
        } else if role == .inference {
            guard modelURL(condition:inferenceCondition) != nil else {error="Train or load a frozen neural model first.";return}
            guard !data.prompts.contains(where:{data.roles[$0.id] == .finalTest && $0.id==SentencePrompt.identity(expectedText)}) else {error="That sentence remains reserved for the final test.";return}
            prompts=[SentencePrompt(text:expectedText)]
        } else {
            let pool=data.prompts.filter {$0.kind == .sentence && data.roles[$0.id] == (role == .training ? .training:.validation)}.sorted {$0.id<$1.id}
            var available=pool
            if role == .sameFit || role == .reseated {
                var used=Set<String>()
                for condition in [SpeechCondition.normal,.silent] {
                    if let url=modelURL(condition:condition),let bytes=try? Data(contentsOf:url.appending(path:"model.json")),let m=(try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any] {
                        used.formUnion((m["training_groups"] as? [String] ?? [])+(m["development_groups"] as? [String] ?? []))
                    }
                }
                available.removeAll {used.contains($0.id)}
            }
            let offset=role == .reseated ? 2:0
            prompts=role == .training ? trainingPrompts(data:data):Array(available.dropFirst(offset).prefix(2))
            guard !prompts.isEmpty else {error="No unreserved sentence groups available.";return}
            if role == .sameFit || role == .reseated {guard hasModel else {error="Freeze a model before prospective tests.";return}}
        }
        var paths:[String:String]=[:],fingerprints:[String:String]=[:]
        if role == .sameFit || role == .reseated || role == .inference {
            for condition in [SpeechCondition.normal,.silent] {
                if let model=modelURL(condition:condition),let bytes=try? Data(contentsOf:model.appending(path:"model.json")),
                   let manifest=(try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any],let fingerprint=manifest["weights_sha256"] as? String {
                    let used=Set((manifest["training_groups"] as? [String] ?? [])+(manifest["development_groups"] as? [String] ?? []))
                    if role != .inference,prompts.contains(where:{used.contains($0.id)}) {error="These groups were used in training/development. Choose different held-out groups.";return}
                    paths[condition.rawValue]=model.path;fingerprints[condition.rawValue]=fingerprint
                }
            }
            let required=role == .inference ? [inferenceCondition]:[.normal,.silent]
            for condition in required {
                guard let path=paths[condition.rawValue],let fingerprint=fingerprints[condition.rawValue],fingerprint.count==64,
                      let bytes=try? Data(contentsOf:URL(fileURLWithPath:path).appending(path:"weights.pt")),
                      SHA256.hash(data:bytes).map({String(format:"%02x",$0)}).joined()==fingerprint else {
                    error="Frozen model is missing or changed. No prospective recording was started.";return
                }
            }
        }
        let withCamera=needsCamera
        guard !withCamera || camera.isRunning else {error=camera.error ?? "This model reads camera landmarks. Allow camera access and try again.";return}
        audio.stop();recorder.condition = .signalCheck
        recorder.prompt="Neural CTC \(role.rawValue); motion\(withCamera ? " + numeric camera points":""); no audio/video"
        guard let session=recorder.start(audioSink:nil,cameraPoints:withCamera) else {error=recorder.error;return}
        capturingCamera=withCamera;cameraReadings=[]
        self.recorder=recorder
        let plan: [NeuralVisit.Planned] = role == .inference ? [.init(text:expectedText,condition:inferenceCondition,repetition:0)] :
            (0..<2).flatMap {rep in prompts.flatMap {p in [SpeechCondition.normal,.silent].map {.init(text:p.text,condition:$0,repetition:rep)}}}
        let visitRole=role == .nonSpeech ? (negativeRole == .training ? "negativeTraining":"negativeDevelopment"):role.rawValue
        visit=NeuralVisit(role:visitRole,session:session,planned:plan,modelPaths:paths,modelFingerprints:fingerprints)
        if withCamera {visit?.modalities=["AirPods motion","Camera landmarks"]}
        if role == .nonSpeech {visit?.movementGuideVersion="illustrated-v1"}
        index=0;error=nil;prediction=nil;phase = .ready;saveVisit()
    }
    func next() {
        switch phase {
        case .ready:startTake()
        case .recording:finishTake()
        case .ended:
            guard canAdvance else {error=takeIssue ?? "Redo the interrupted take before continuing.";return}
            if index+1<(visit?.planned.count ?? 0) {index+=1;phase = .ready}
            else {endVisit()}
        default:break
        }
    }
    private func startTake() {
        guard phase == .ready,let latest,clock()-latest.t>=0,clock()-latest.t<0.3 else {error="Waiting for fresh AirPods motion.";return}
        buffer=[];cameraReadings=[];selectedSource=latest.sensorLocation;phase = .relax;error=nil
        task=Task {
            do {
                let rest=clock()+3
                while clock()<rest {try Task.checkCancellation();secondsRemaining=max(0,rest-clock());try await Task.sleep(for:.milliseconds(40))}
                beginCapture(at:clock())
                let end=cueStart!+captureDuration
                while phase == .recording,clock()<end {
                    try Task.checkCancellation();secondsRemaining=max(0,end-clock())
                    if self.latest.map({clock()-$0.t>0.3})==true {finishTake(reason:"Motion stopped; redo this take.");return}
                    try await Task.sleep(for:.milliseconds(40))
                }
                if phase == .recording {finishAtTimeLimit()}
            } catch is CancellationError {} catch {self.error=error.localizedDescription;stop()}
        }
    }
    func beginCapture(at time:Double) {
        guard phase == .relax || phase == .ready,visit != nil,time.isFinite else {return}
        cueStart=time;selectedSource=latest?.sensorLocation;phase = .recording
        recorder?.logMarker(uptime:time,kind:"neural-start",label:currentText)
    }
    func finishAtTimeLimit() {
        // Non-speech examples have no word boundary to mark manually. Their
        // measured timer boundary is valid; sentence takes still require Done.
        guard phase == .recording else {return}
        finishTake(reason:recordingNonSpeech ? nil:"Timed out without Done. Redo this sentence and choose Done when finished.")
    }
    private func finishTake(reason:String?=nil) {
        guard phase == .recording,let cueStart,let visit else {return}
        task?.cancel();let end=clock(),item=visit.planned[index]
        let recording=SentenceRecording(condition:item.condition,samples:buffer.filter {$0.t<=end},cueStart:cueStart,cueEnd:end)
        let label=(visit.role=="negativeTraining" || visit.role=="negativeDevelopment") ? "":item.text
        self.visit?.takes.append(.init(sentenceID:SentencePrompt.identity(item.text),text:label,sessionID:visit.session.id,repetition:item.repetition,
            recording:recording,camera:capturingCamera ? cameraReadings.filter {$0.t>=cueStart-2.6 && $0.t<=end}:[],discardedReason:reason,wordFocusGuide:.comfortable))
        cameraReadings=[]
        recorder?.logMarker(uptime:end,kind:reason==nil ? "neural-end":"neural-interrupted",label:item.text)
        self.cueStart=nil;buffer=[];secondsRemaining=0;phase = .ended;error=reason;saveVisit()
    }
    func redo() {
        guard canRedo else {return}
        if phase == .recording {finishTake(reason:"Discarded by Redo")}
        else if let i=visit?.takes.indices.last {visit?.takes[i].discardedReason="Discarded by Redo"}
        phase = .ready;error=nil;saveVisit()
    }
    private func endVisit(evaluate:Bool=true) {
        guard let visit else {return}
        _=recorder?.stop(sessionID:visit.session.id);self.visit?.endedUnix=Date().timeIntervalSince1970;self.visit?.complete=true
        if capturingCamera {camera.stop();capturingCamera=false;faceTracked=false}
        saveVisit();phase = .idle
        loadResumableNonSpeech()
        guard evaluate else {return}
        if visit.role == Mode.inference.rawValue,let take=visit.takes.last(where:{$0.discardedReason==nil}),let path=visit.modelPaths[take.recording.condition.rawValue] {
            phase = .working;task=Task {
                do {
                    let input=root.appending(path:"input-\(UUID()).json"),output=root.appending(path:"prediction-\(UUID()).json")
                    struct Input:Encodable {let recording:SentenceRecording;let expectedText:String?;let camera:[CameraPoseReading]}
                    try JSONEncoder().encode(Input(recording:take.recording,expectedText:take.text.isEmpty ? nil:take.text,camera:take.camera)).write(to:input)
                    try await execute(["infer","--model",path,"--input",input.path,"--output",output.path])
                    let decoder=JSONDecoder();decoder.keyDecodingStrategy = .convertFromSnakeCase
                    prediction=try decoder.decode(NeuralPrediction.self,from:Data(contentsOf:output));status="Decoding complete. Scores are uncalibrated hypotheses."
                } catch {self.error=error.localizedDescription}
                phase = .idle
            }
        } else if visit.role == Mode.sameFit.rawValue || visit.role == Mode.reseated.rawValue {
            phase = .working;task=Task {
                do {
                    let file=root.appending(path:"Collections/\(visit.id)/session.json")
                    try await execute(["evaluate-fresh","--session",file.path]);status="Frozen prospective evaluation saved beside this recording."
                    if let bytes=try? Data(contentsOf:file.deletingLastPathComponent().appending(path:"frozen-evaluation.json")),let report=(try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any],
                       let scores=report["reports"] as? [[String:Any]] {
                        freshEvaluationSummary=scores.compactMap {row in
                            guard let condition=row["condition"] as? String,let s=row["summary"] as? [String:Any],let wer=s["word_error_rate_no_lm"] as? Double,let per=s["phoneme_error_rate"] as? Double else {return nil}
                            return String(format:"%@: phone error %.1f%% · word error %.1f%% · %d planned utterances",condition,per*100,wer*100,s["utterances"] as? Int ?? 0)
                        }.joined(separator:"\n")
                    }
                } catch {self.error=error.localizedDescription}
                phase = .idle
            }
        }
    }
    private func saveVisit() {
        guard let visit else {return}
        do {
            let dir=root.appending(path:"Collections/\(visit.id)")
            try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
            let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
            try encoder.encode(visit).write(to:dir.appending(path:"session.json"),options:.atomic)
        } catch {self.error=error.localizedDescription}
    }
    func stop() {
        task?.cancel();if let process,process.isRunning {process.terminate()}
        if phase == .recording {finishTake(reason:"Stopped during capture")}
        if isAcquiring {endVisit(evaluate:false)}
        camera.stop();capturingCamera=false;faceTracked=false
        phase = .idle
    }
    enum Failure:LocalizedError {
        case runtime,backend(String)
        var errorDescription:String? {switch self {
        case .runtime:"Install uv locally to run the pinned Python/PyTorch runtime."
        case .backend(let message):"Local neural training/decoding failed: "+message
        }}
    }
}
