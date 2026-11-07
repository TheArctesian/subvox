import AppKit
import Observation

struct DecodedWord: Identifiable, Hashable {
    let id = UUID()
    var prediction: Prediction
    var text: String { prediction.best?.label ?? "?" }
}

@MainActor @Observable
final class AppModel {
    enum Screen: String, CaseIterable, Identifiable {
        case sensors = "Sensors"
        case sentences = "Train"
        case collect = "Signal check"
        case train = "Word diagnostics"
        case read = "Read"
        case decode = "Decode"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .sensors: "waveform.path.ecg"
            case .sentences: "text.book.closed"
            case .collect: "waveform.and.mic"
            case .train: "square.stack.3d.up"
            case .read: "text.book.closed"
            case .decode: "text.bubble"
            }
        }
    }

    let motion = HeadphoneMotionService()
    let audio = AudioService()
    let transcriber = MicTranscriber()
    let capture = CaptureController()
    let recorder = SessionRecorder()
    let reading = ReadingController()
    let signalCheck = GuidedSignalCheck()
    let localWhisper = LocalWhisper()
    let sentenceTraining = SentenceTrainingWorkflow()
    private(set) var usesSentenceModel = UserDefaults.standard.bool(forKey: "useReviewedSentenceModel")
    var decoderSourceLabel: String { usesSentenceModel ? "Reviewed sentence-derived words" : "Legacy word training" }

    var screen: Screen = .sentences {
        didSet {
            if oldValue == .sentences && screen != .sentences { sentenceTraining.stop() }
            if oldValue == .collect && screen != .collect { signalCheck.stop() }
            syncCaptureMode()
        }
    }

    // MARK: Training

    private(set) var dataset = TemplateStore.load()
    let lexicon = Lexicon.bundled
    var featureSet: FeatureSet {
        get { settings.features }
        set { changeSettings { $0.features = newValue } }
    }
    var settings: DecoderSettings { dataset.experiment?.settings ?? DecoderSettings() }
    var experimentLocked: Bool { dataset.experiment?.frozenAt != nil }
    var sessionCondition: SpeechCondition = .silent
    var sessionNotes = ""
    private(set) var activeVisitID: UUID?
    private(set) var evaluationProgress = ""
    private(set) var trainingError: String?
    private(set) var storageReady = true
    private var revision = 0

    var learningMethod: LearningMethod {
        get { settings.method }
        set { changeSettings { $0.method = newValue } }
    }
    var vocabularyScope: VocabularyScope {
        get { settings.vocabularyScope }
        set { changeSettings { $0.vocabularyScope = newValue } }
    }
    var hmmMaxScore: Double {
        get { settings.hmmMaxScore }
        set { changeSettings { $0.hmmMaxScore = newValue } }
    }
    var hmmMinGap: Double {
        get { settings.hmmMinGap }
        set { changeSettings { $0.hmmMinGap = newValue } }
    }
    var comparisonReports: [DecoderReport] { dataset.experiment?.comparisonReports ?? [] }
    var trainingCondition: SpeechCondition {
        get { settings.condition }
        set { changeSettings { $0.condition = newValue } }
    }
    var validationReport: DecoderReport? { dataset.experiment?.validationReport }
    var finalReport: DecoderReport? { dataset.experiment?.finalReport }
    var pendingReviewCount: Int {
        dataset.templates.filter { ($0.purpose == nil || $0.purpose == .training) && $0.reviewDecision == .pending }.count
    }
    var fittingTakes: [Template] { dataset.experiment?.fittingTakes(from: dataset.templates) ?? [] }
    func role(of take: Template) -> DataRole? { dataset.experiment?.role(of: take) }

    /// Built from the training takes in the background; nil until there's something to build from.
    private(set) var decoder: MotionWordDecoder?
    private(set) var isBuildingDecoder = false
    private(set) var isEvaluating = false
    private(set) var lastTemplateID: Template.ID?
    private(set) var session: TrainingSession?

    // MARK: Decoding

    var decodeMode: CaptureController.Mode = .hold {
        didSet { syncCaptureMode() }
    }
    var maxDistance: Double {
        get { settings.maxDistance }
        set { changeSettings { $0.maxDistance = newValue } }
    }
    var minConfidence: Double {
        get { settings.minConfidence }
        set { changeSettings { $0.minConfidence = newValue } }
    }
    var commonBias: Double {
        get { settings.commonBias }
        set { changeSettings { $0.commonBias = newValue } }
    }
    private(set) var isDecoding = false
    private(set) var decoded: [DecodedWord] = []
    private(set) var lastSegment: [MotionSample] = []

    @ObservationIgnored private var decoderBuild: Task<Void, Never>?
    @ObservationIgnored private var keyMonitor: Any?
    @ObservationIgnored private var started = false

    init() {
        // Keep the exact pre-migration file before adding assignments or inferring review state.
        if dataset.experiment == nil {
            do {
                if FileManager.default.fileExists(atPath: TemplateStore.url.path) {
                    let backup = TemplateStore.url.deletingLastPathComponent()
                        .appending(path: "dataset-before-experiment-\(UUID().uuidString).json")
                    try FileManager.default.copyItem(at: TemplateStore.url, to: backup)
                }
                var experiment = TrainingExperiment()
                experiment.historicalTakeCount = dataset.templates.count
                experiment.assign(dataset.templates.map(\.label))
                if experiment.wordRoles.isEmpty { experiment.assign(SignalCheckPlan.seedWords) }
                dataset.experiment = experiment
                for i in dataset.templates.indices {
                    if dataset.templates[i].review == nil {
                        dataset.templates[i].review = dataset.templates[i].quality.issues.isEmpty ? .accepted : .pending
                    }
                }
                storageReady = TemplateStore.save(dataset)
            } catch {
                storageReady = false
                trainingError = "Could not back up training data: \(error.localizedDescription)"
            }
        }
        if dataset.experiment?.wordRoles.isEmpty == true, !experimentLocked {
            dataset.experiment?.assign(SignalCheckPlan.seedWords)
            storageReady = TemplateStore.save(dataset)
        }
        if !storageReady { trainingError = trainingError ?? "Could not save experiment assignments." }
        sentenceTraining.onTrainingChanged = { [weak self] in
            guard let self, self.usesSentenceModel else { return }
            self.rebuildDecoder()
        }
        rebuildDecoder()
    }

    func start() {
        guard !started else { return }
        started = true
        motion.onSample = { [weak self] sample in self?.ingest(sample) }
        capture.onSegment = { [weak self] segment in self?.handleSegment(segment) }
        motion.start()
        audio.refreshDevices()
        installKeyMonitor()
    }

    private func ingest(_ sample: MotionSample) {
        capture.ingest(sample)
        recorder.record(sample)
        reading.ingest(sample)
        signalCheck.ingest(sample)
        sentenceTraining.ingest(sample)
    }

    private func syncCaptureMode() {
        capture.autoEnabled = screen == .decode && decodeMode == .auto
    }

    private func handleSegment(_ segment: [MotionSample]) {
        lastSegment = segment
        switch screen {
        case .train:
            guard let word = session?.current else { return }
            guard !experimentLocked, !isEvaluating, storageReady else { return }
            let quality = TakeQuality.inspect(segment)
            let purpose = session?.currentPurpose ?? .training
            let decision: ReviewDecision = purpose != .training && !quality.invalid ? .accepted
                : (quality.issues.isEmpty ? .accepted : .pending)
            let condition = dataset.visits?.first { $0.id == activeVisitID }?.condition ?? sessionCondition
            let template = Template(label: word, samples: segment, sessionID: activeVisitID,
                                    condition: condition, measuredSampleRate: quality.sampleRate,
                                    review: decision, purpose: purpose)
            addTemplate(template)
            recorder.logEvent(segment, kind: "train", label: word, confidence: nil)
            session?.record(template.id)
            sessionDidMove()
        case .decode:
            guard let decoder else { return }
            isDecoding = true
            Task {
                // Matching against ~18k word templates takes a moment; keep it off the main thread.
                let prediction = await Task.detached(priority: .userInitiated) {
                    decoder.predict(segment)
                }.value
                self.isDecoding = false
                self.decoded.append(DecodedWord(prediction: prediction))
                self.recorder.logEvent(segment, kind: prediction.accepted ? "decode" : "reject",
                                       label: prediction.best?.label ?? "", confidence: prediction.confidence)
            }
        case .sensors:
            recorder.logEvent(segment, kind: "segment", label: recorder.prompt, confidence: nil)
        case .read, .collect, .sentences:
            break
        }
    }

    // MARK: Space bar = hold to capture

    private func installKeyMonitor() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
            guard event.keyCode == 49 else { return event }
            let isDown = event.type == .keyDown
            let isRepeat = isDown && event.isARepeat
            let consumed = MainActor.assumeIsolated { () -> Bool in
                guard let self, self.spaceCapturesMotion else { return false }
                if !isDown {
                    self.capture.end()
                } else if !isRepeat {
                    self.capture.begin()
                }
                return true
            }
            return consumed ? nil : event
        }
    }

    private var spaceCapturesMotion: Bool {
        // Let text fields keep the space bar.
        if NSApp.keyWindow?.firstResponder is NSText { return false }
        switch screen {
        case .train: return session.map { !$0.isFinished } ?? false
        case .decode: return decodeMode == .hold && decoder != nil
        case .sensors: return recorder.isRecording
        // Reading uses Space to advance lines.
        case .read, .collect, .sentences: return false
        }
    }

    // MARK: Phoneme checklist

    /// Coverage targets apply only to accepted fitting takes, never validation or final test.
    static let startingTarget = 4
    static let roundSize = 4

    /// Everyday words of one to three syllables: short enough for a single clean take.
    private static let promptCandidates: [String] = Lexicon.vocabulary.prefix(4000).compactMap { entry in
        let vowels = Lexicon.bundled.phones(entry.word)?.filter(Lexicon.vowels.contains).count ?? 0
        return (1...3).contains(vowels) ? entry.word : nil
    }

    enum PhonemeStatus {
        case collecting, covered
    }

    /// Examples recorded so far of each phoneme, across every take with a known pronunciation.
    var phonemeCounts: [String: Int] {
        var counts: [String: Int] = [:]
        for take in fittingTakes {
            for phone in lexicon.phones(take.label) ?? [] { counts[phone, default: 0] += 1 }
        }
        return counts
    }

    func target(for phoneme: String) -> Int {
        dataset.phoneTargets?[phoneme] ?? Self.startingTarget
    }

    func status(of phoneme: String, counts: [String: Int]) -> PhonemeStatus {
        counts[phoneme, default: 0] < target(for: phoneme) ? .collecting : .covered
    }

    /// Prompts left to record to reach every phoneme's current target.
    var plannedPrompts: [String] {
        var generator = SystemRandomNumberGenerator()
        return plan(using: &generator)
    }

    private func plan<G: RandomNumberGenerator>(using generator: inout G) -> [String] {
        let counts = phonemeCounts
        let needs = Dictionary(uniqueKeysWithValues: Phoneme.all.map {
            ($0.symbol, target(for: $0.symbol) - counts[$0.symbol, default: 0])
        })
        return CoveragePlanner.plan(needs: needs, candidates: Self.promptCandidates, lexicon: lexicon,
                                    avoiding: Set(dataset.templates.map(\.label)), using: &generator)
    }

    /// Requests new contexts suggested by validation word errors, without consulting final data.
    func requestMoreForFailing() {
        guard !experimentLocked, !isEvaluating, let report = validationReport else { return }
        // Word errors identify contexts worth revisiting, not proof that a particular phone failed.
        let phones = Set(report.rows.filter { !$0.top1 || !$0.accepted }.flatMap { lexicon.phones($0.expected) ?? [] })
        let counts = phonemeCounts
        var targets = dataset.phoneTargets ?? [:]
        for phone in phones {
            targets[phone] = max(target(for: phone), counts[phone, default: 0] + Self.roundSize)
        }
        dataset.phoneTargets = targets
        persist()
        startSession()
    }

    func resetTraining() {
        guard !isEvaluating, !signalCheck.isBusy, !sentenceTraining.isBusy else { return }
        endSession()
        // Archive the previous experiment, including its assignments, before a destructive reset.
        do {
            let archive = TemplateStore.url.deletingLastPathComponent().appending(path: "experiment-\(UUID().uuidString).json")
            try JSONEncoder().encode(dataset).write(to: archive, options: .atomic)
        } catch { trainingError = error.localizedDescription; return }
        dataset = .starter
        var fresh = TrainingExperiment()
        fresh.assign(SignalCheckPlan.seedWords)
        dataset.experiment = fresh
        persist()
    }

    func reviewTake(_ id: UUID, decision: ReviewDecision) {
        guard !experimentLocked, !isEvaluating, !signalCheck.isBusy, !sentenceTraining.isBusy, session?.isFinished != false,
              let index = dataset.templates.firstIndex(where: { $0.id == id }) else { return }
        guard decision != .accepted || !dataset.templates[index].quality.invalid else { return }
        dataset.templates[index].review = decision
        persist()
    }

    private func changeSettings(_ change: (inout DecoderSettings) -> Void) {
        guard !experimentLocked, !isEvaluating, !signalCheck.isBusy, !sentenceTraining.isBusy, session?.isFinished != false else { return }
        var next = settings
        change(&next)
        guard next != settings else { return }
        dataset.experiment?.settings = next
        persist()
    }

    // MARK: Dataset editing

    func addTemplate(_ template: Template) {
        guard !experimentLocked, !isEvaluating, storageReady else { return }
        dataset.experiment?.assign([template.label])
        dataset.templates.append(template)
        lastTemplateID = template.id
        persist(rebuild: template.purpose == nil || template.purpose == .training)
    }

    func deleteTemplate(_ id: Template.ID) {
        guard !experimentLocked, !isEvaluating else { return }
        // Keep the word's assignment even after deletion: re-recording cannot move its split.
        dataset.templates.removeAll { $0.id == id }
        persist()
    }

    private func persist(rebuild: Bool = true) {
        revision += 1
        if rebuild {
            dataset.experiment?.validationReport = nil
            dataset.experiment?.comparisonReports = nil
        }
        storageReady = TemplateStore.save(dataset)
        if !storageReady { trainingError = "Could not save training data. Recording and evaluation are disabled." }
        if rebuild { rebuildDecoder() }
    }

    private func rebuildDecoder() {
        decoder = nil
        isBuildingDecoder = true
        let (takes, lexicon, config) = (usesSentenceModel ? sentenceTraining.fittingWords : fittingTakes, lexicon, settings)
        decoderBuild?.cancel()
        decoderBuild = Task {
            let decoder = await Task.detached(priority: .utility) {
                MotionWordDecoder(takes: takes, lexicon: lexicon, vocabulary: Lexicon.vocabulary, settings: config)
            }.value
            guard !Task.isCancelled else { return }
            self.decoder = decoder
            self.isBuildingDecoder = false
        }
    }

    // MARK: Guided session control

    func startSession() {
        var generator = SystemRandomNumberGenerator()
        let prompts = plan(using: &generator)
        beginSession(prompts: prompts)
    }

    func rerecordTake(_ id: UUID) {
        guard let take = dataset.templates.first(where: { $0.id == id }) else { return }
        sessionCondition = take.condition ?? sessionCondition
        beginSession(prompts: [take.label], purposes: take.purpose.map { [$0] })
    }

    func startSignalCheck() {
        guard let experiment = dataset.experiment, !signalCheck.isBusy, !sentenceTraining.isBusy, !isEvaluating,
              !experimentLocked, storageReady, motion.status == .streaming,
              session?.isFinished != false, !recorder.isRecording, !reading.isReading else { return }
        guard localWhisper.isConfigured else {
            trainingError = "Choose a local Whisper model and runtime before starting."
            return
        }
        let words = SignalCheckPlan.words(in: experiment)
        guard words.count >= 2 else {
            trainingError = "Collect at least two training word groups before a signal check."
            return
        }
        var generator = SystemRandomNumberGenerator()
        let trials = SignalCheckPlan.trials(words: words, rounds: signalCheck.rounds, using: &generator)
        let visit = RecordingVisit(condition: .silent, notes: "Mixed voiced/silent/cue-only signal check. Pace: \(signalCheck.pace.rawValue), \(signalCheck.pace.baseline)s baseline, \(signalCheck.pace.cue)s cue, \(signalCheck.pace.recovery)s recovery. " + signalCheck.notes,
                                   sampleRate: motion.sampleRate, prompts: trials.map(\.word))
        dataset.visits = (dataset.visits ?? []) + [visit]
        persist(rebuild: false)
        guard storageReady else { return }
        signalCheck.start(trials: trials, visitID: visit.id, audio: audio, recorder: recorder, whisper: localWhisper,
            canRecord: { [weak self] in self?.motion.status == .streaming },
            onTake: { [weak self] take in
                guard let self else { return }
                self.addTemplate(take)
                self.recorder.logEvent(take.samples, kind: take.purpose?.rawValue ?? "diagnostic", label: take.label, confidence: nil)
            }, onFinished: { [weak self] directory in
                guard let self, let i = self.dataset.visits?.firstIndex(where: { $0.id == visit.id }) else { return }
                self.dataset.visits?[i].endedAt = Date()
                self.dataset.visits?[i].recordingDirectory = directory?.path
                self.persist(rebuild: false)
            })
    }

    private func beginSession(prompts: [String], purposes: [RecordingPurpose]? = nil) {
        guard motion.status == .streaming, !signalCheck.isBusy, !sentenceTraining.isBusy else { return }
        guard !prompts.isEmpty, !experimentLocked, !isEvaluating, storageReady,
              session?.isFinished != false else { return }
        // Allocate before recording: review, exclusion and retry cannot influence assignments.
        dataset.experiment?.assign(prompts)
        let visit = RecordingVisit(condition: sessionCondition,
                                   notes: purposes == nil ? sessionNotes : "Interleaved signal check. " + sessionNotes,
                                   sampleRate: motion.sampleRate, prompts: prompts)
        dataset.visits = (dataset.visits ?? []) + [visit]
        activeVisitID = visit.id
        persist()
        guard storageReady else { activeVisitID = nil; return }
        session = TrainingSession(prompts: prompts, purposes: purposes)
    }

    func skipSessionPrompt() {
        session?.skip()
        sessionDidMove()
    }

    func undoSessionTake() {
        if let id = session?.undo() { deleteTemplate(id) }
        sessionDidMove()
    }

    func endSession() {
        closeVisit()
        session = nil
    }

    private func closeVisit() {
        if let id = activeVisitID, let index = dataset.visits?.firstIndex(where: { $0.id == id }) {
            dataset.visits?[index].endedAt = Date()
            storageReady = TemplateStore.save(dataset)
            if !storageReady { trainingError = "Could not save the session end time." }
        }
        activeVisitID = nil
    }

    private func sessionDidMove() {
        if session?.isFinished == true { closeVisit() }
    }

    // MARK: Evaluation

    func evaluate() { runDecoderEvaluation(final: false) }
    func runFinalTest() { runDecoderEvaluation(final: true) }

    private func runDecoderEvaluation(final: Bool) {
        guard !isEvaluating, !signalCheck.isBusy, !sentenceTraining.isBusy, session?.isFinished != false, storageReady,
              var experiment = dataset.experiment else { return }
        if final {
            guard experiment.finalReport == nil else { return }
            guard pendingReviewCount == 0 else {
                trainingError = "Review or exclude every flagged take before freezing the final test."
                return
            }
        } else if experiment.frozenAt != nil { return }
        let role: DataRole = final ? .finalTest : .validation
        if final { experiment.frozenAt = experiment.frozenAt ?? Date() }
        let training = experiment.fittingTakes(from: dataset.templates)
        let testing = dataset.templates.filter { $0.isUsable && experiment.role(of: $0) == role && experiment.matchesCondition($0) }
        guard !training.isEmpty, !testing.isEmpty else {
            trainingError = "This split needs accepted training and evaluation recordings. Review takes or record more words."
            return
        }
        if final {
            let manifest = MotionWordDecoder.manifest(takes: training, lexicon: lexicon,
                                                       vocabulary: Lexicon.vocabulary, settings: experiment.settings)
            if let frozen = experiment.finalManifest, frozen != manifest {
                trainingError = "The model implementation, vocabulary or data changed since freezing. Resume requires the original frozen configuration."
                return
            }
            experiment.finalManifest = manifest
            dataset.experiment = experiment
            do {
                let snapshot = TemplateStore.url.deletingLastPathComponent()
                    .appending(path: "experiment-\(experiment.id.uuidString)-frozen.json")
                if !FileManager.default.fileExists(atPath: snapshot.path) {
                    try JSONEncoder().encode(dataset).write(to: snapshot, options: .atomic)
                }
            } catch {
                trainingError = "Could not archive frozen experiment: \(error.localizedDescription)"
                return
            }
            guard TemplateStore.save(dataset) else {
                storageReady = false
                trainingError = "Could not save the frozen experiment. Final test was not run."
                return
            }
            rebuildDecoder()
        }
        trainingError = nil
        isEvaluating = true
        evaluationProgress = "Building decoder…"
        let currentRevision = revision
        let config = experiment.settings
        let lexicon = lexicon
        Task {
            do {
                let report = try await Task.detached(priority: .userInitiated) {
                    try await DecoderEvaluation.run(training: training, testing: testing, role: role,
                        lexicon: lexicon, vocabulary: Lexicon.vocabulary, settings: config) { done, total in
                            await MainActor.run { self.evaluationProgress = "Decoded \(done) of \(total) words" }
                        }
                }.value
                guard self.revision == currentRevision else {
                    self.isEvaluating = false
                    return
                }
                if final { self.dataset.experiment?.finalReport = report }
                else { self.dataset.experiment?.validationReport = report }
                if !TemplateStore.save(self.dataset) {
                    self.storageReady = false
                    self.trainingError = "Results could not be saved."
                }
            } catch { self.trainingError = error.localizedDescription }
            self.isEvaluating = false
            self.evaluationProgress = ""
        }
    }

    /// The council's ablation matrix uses only development data and disables frequency bias.
    /// Candidate prefixes are fixed independently of test labels; all test words stay counted.
    func compareLearningMethods() {
        guard !isEvaluating, !signalCheck.isBusy, !sentenceTraining.isBusy, !experimentLocked, storageReady, session?.isFinished != false,
              let experiment = dataset.experiment else { return }
        let training = fittingTakes
        let testing = dataset.templates.filter {
            $0.isUsable && experiment.role(of: $0) == .validation && experiment.matchesCondition($0)
        }
        guard !training.isEmpty, !testing.isEmpty else {
            trainingError = "Accepted training and validation recordings are needed for comparison."
            return
        }
        let baseline = settings
        let lexicon = lexicon
        let currentRevision = revision
        isEvaluating = true
        trainingError = nil
        evaluationProgress = "Comparing learning methods…"
        Task {
            do {
                let reports = try await Task.detached(priority: .userInitiated) {
                    var reports: [DecoderReport] = []
                    for scope in VocabularyScope.allCases {
                        for method in LearningMethod.allCases {
                            var config = baseline
                            config.method = method
                            config.vocabularyScope = scope
                            config.commonBias = 0
                            let report = try await DecoderEvaluation.run(training: training, testing: testing,
                                role: .validation, lexicon: lexicon, vocabulary: Lexicon.vocabulary, settings: config) { done, total in
                                    await MainActor.run {
                                        self.evaluationProgress = "\(scope.title) · \(method.title): \(done)/\(total)"
                                    }
                                }
                            reports.append(report)
                        }
                    }
                    return reports
                }.value
                if self.revision == currentRevision {
                    self.dataset.experiment?.comparisonReports = reports
                    if !TemplateStore.save(self.dataset) {
                        self.storageReady = false
                        self.trainingError = "Could not save comparison results."
                    }
                }
            } catch { self.trainingError = error.localizedDescription }
            self.isEvaluating = false
            self.evaluationProgress = ""
        }
    }

    func startSentenceTraining(repairs: Bool = false, retry: UUID? = nil) {
        guard !isEvaluating, !signalCheck.isBusy, session?.isFinished != false, !reading.isReading,
              motion.status == .streaming, localWhisper.isConfigured else { return }
        sentenceTraining.start(audio: audio, recorder: recorder, whisper: localWhisper,
            canRecord: { [weak self] in self?.motion.status == .streaming }, repairs: repairs, retry: retry)
    }

    func useReviewedSentenceModel() {
        guard !sentenceTraining.isBusy, !isEvaluating, !sentenceTraining.fittingWords.isEmpty else { return }
        usesSentenceModel = true
        UserDefaults.standard.set(true, forKey: "useReviewedSentenceModel")
        rebuildDecoder()
    }

    // MARK: Decode transcript

    var transcript: String {
        decoded.filter(\.prediction.accepted).map(\.text).joined(separator: " ")
    }

    func clearDecoded() {
        decoded.removeAll()
        lastSegment = []
    }

    // MARK: Mic

    func toggleMic() async {
        if audio.isRunning {
            transcriber.stop()
            audio.stop()
        } else {
            await audio.start()
            if audio.isRunning { await transcriber.start(sink: audio.sink) }
        }
    }
}
