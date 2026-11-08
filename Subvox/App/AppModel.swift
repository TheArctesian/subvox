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
        case train = "Train", decode = "Try model", devices = "Devices"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .train: "text.book.closed"
            case .decode: "text.bubble"
            case .devices: "waveform.path.ecg"
            }
        }
    }

    let motion = HeadphoneMotionService()
    let audio = AudioService()
    let capture = CaptureController()
    let recorder = SessionRecorder()
    let localWhisper = LocalWhisper()
    let signalCheck: SignalCheckWorkflow
    var showSignalCheck = false
    let representation = RepresentationWorkflow()
    var showRepresentation = false
    let sequenceLearning = SequenceLearningWorkflow()
    var showSequenceLearning = false
    var showNonSpeechCollection = false
    let observability = ObservabilityWorkflow()
    var showObservability = false
    let huggingFace = HuggingFaceAccount()
    let contribution = ContributionSettings()
    let uploader = CommunityUploader()
    var showContribute = false
    /// Off for new installs: the Train page shows one recording path. Research tools (sentence
    /// banks, neural training, signal checks, Try model) appear when this is on.
    var researchTools = UserDefaults.standard.bool(forKey: "ui.researchTools") {
        didSet {
            UserDefaults.standard.set(researchTools, forKey: "ui.researchTools")
            if !researchTools { sentenceTraining.readingSource = .community; if screen == .decode { screen = .train } }
        }
    }
    /// False until the first-run welcome (sharing choice) has been completed.
    var onboarded = UserDefaults.standard.bool(forKey: "ui.onboarded") {
        didSet { UserDefaults.standard.set(onboarded, forKey: "ui.onboarded") }
    }
    var visibleScreens: [Screen] { researchTools ? Screen.allCases : [.train, .devices] }
    let sentenceTraining: SentenceTrainingWorkflow
    var screen: Screen = .train {
        didSet {
            if oldValue == .train && screen != .train { sentenceTraining.stop(); signalCheck.stop(); representation.stop(); sequenceLearning.stop(); observability.stop() }
            syncCaptureMode()
        }
    }
    var decodeSettings: DecoderSettings { sentenceTraining.settings }
    var decodeSettingsLocked: Bool { sentenceTraining.settingsLocked }
    var hmmMaxScore: Double {
        get { decodeSettings.hmmMaxScore }
        set { sentenceTraining.changeSettings { $0.hmmMaxScore = newValue } }
    }
    var hmmMinGap: Double {
        get { decodeSettings.hmmMinGap }
        set { sentenceTraining.changeSettings { $0.hmmMinGap = newValue } }
    }
    var maxDistance: Double {
        get { decodeSettings.maxDistance }
        set { sentenceTraining.changeSettings { $0.maxDistance = newValue } }
    }
    var minConfidence: Double {
        get { decodeSettings.minConfidence }
        set { sentenceTraining.changeSettings { $0.minConfidence = newValue } }
    }
    var commonBias: Double {
        get { decodeSettings.commonBias }
        set { sentenceTraining.changeSettings { $0.commonBias = newValue } }
    }
    var decodeMode: CaptureController.Mode = .hold {
        didSet { syncCaptureMode() }
    }
    private(set) var decoder: MotionWordDecoder?
    private(set) var isBuildingDecoder = false
    private(set) var isDecoding = false
    private(set) var decoded: [DecodedWord] = []
    private(set) var lastSegment: [MotionSample] = []
    @ObservationIgnored private var decoderBuild: Task<Void, Never>?
    @ObservationIgnored private var keyMonitor: Any?
    @ObservationIgnored private var terminationObserver: NSObjectProtocol?
    @ObservationIgnored private var started = false

    init(sentenceTraining: SentenceTrainingWorkflow? = nil, signalCheck: SignalCheckWorkflow? = nil) {
        self.signalCheck = signalCheck ?? SignalCheckWorkflow()
        self.sentenceTraining = sentenceTraining ?? SentenceTrainingWorkflow()
        if sentenceTraining == nil {
            self.sentenceTraining.guidedRecording=true
            self.sentenceTraining.variedSentences=true
            self.sentenceTraining.collectionRole = .training
            self.sentenceTraining.presentation = .sentences
            self.sentenceTraining.captureCameraPoints=true
            self.sentenceTraining.excludedCollectionGroups=Set((representation.study?.training ?? []).map(\.id)+(representation.study?.evaluation ?? []).map(\.id))
        }
        if !researchTools { self.sentenceTraining.readingSource = .community }
        self.sentenceTraining.onTrainingChanged = { [weak self] in self?.rebuildDecoder() }
        rebuildDecoder()
    }

    func start() {
        guard !started else { return }
        started = true
        terminationObserver = NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification,
            object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.prepareForTermination() }
            }
        motion.onSample = { [weak self] sample in
            guard let self else { return }
            self.syncCaptureMode()
            self.capture.ingest(sample)
            self.recorder.record(sample)
            self.sentenceTraining.ingest(sample)
            self.signalCheck.ingest(sample)
            self.representation.ingest(sample)
            self.sequenceLearning.ingest(sample)
            self.observability.ingest(sample)
        }
        capture.onSegment = { [weak self] in self?.handleSegment($0) }
        motion.start()
        audio.refreshDevices()
        installKeyMonitor()
    }

    func prepareForTermination() {
        uploader.stop()
        observability.stop()
        sequenceLearning.stop()
        representation.stop()
        signalCheck.prepareForTermination()
        sentenceTraining.prepareForTermination(recorder: recorder)
        recorder.stop()
        audio.stop()
        localWhisper.stopOwnedProcess()
    }

    private func rebuildDecoder() {
        decoderBuild?.cancel()
        decoder = nil
        syncCaptureMode()
        let takes = sentenceTraining.fittingWords
        guard !takes.isEmpty else { isBuildingDecoder = false; return }
        isBuildingDecoder = true
        let settings = decodeSettings
        decoderBuild = Task {
            let build = Task.detached(priority: .utility) {
                MotionWordDecoder(takes: takes, lexicon: .bundled, vocabulary: Lexicon.vocabulary, settings: settings)
            }
            let result = await withTaskCancellationHandler(operation: { await build.value }, onCancel: { build.cancel() })
            guard !Task.isCancelled else { return }
            decoder = result
            isBuildingDecoder = false
            syncCaptureMode()
        }
    }

    private func syncCaptureMode() {
        let enabled = screen == .decode && decodeMode == .auto && decoder != nil && !sentenceTraining.isBusy && !signalCheck.isBusy && !representation.isBusy && !sequenceLearning.isBusy && !observability.isBusy
        if capture.autoEnabled != enabled { capture.autoEnabled = enabled }
    }

    func startSentenceTraining(repairs: Bool = false, retry: UUID? = nil) {
        guard motion.status == .streaming, localWhisper.isConfigured || sentenceTraining.readingSource == .community, !signalCheck.isBusy, !representation.isBusy, !sequenceLearning.isBusy, !observability.isBusy else { return }
        sentenceTraining.start(audio: audio, recorder: recorder, whisper: localWhisper,
            canRecord: { [weak self] in self?.motion.status == .streaming }, repairs: repairs, retry: retry)
    }

    func startSignalCheck(againstReference: Bool = false, controlled: Bool = false) {
        guard !sentenceTraining.isBusy, !representation.isBusy, !sequenceLearning.isBusy, !observability.isBusy, motion.status == .streaming else { return }
        signalCheck.start(audio: audio, recorder: recorder, againstReference: againstReference, controlled: controlled,
            canRecord: { [weak self] in self?.motion.status == .streaming })
    }

    private func handleSegment(_ segment: [MotionSample]) {
        guard screen == .decode, !sentenceTraining.isBusy, !signalCheck.isBusy, !representation.isBusy, !sequenceLearning.isBusy, !observability.isBusy, let decoder else { return }
        lastSegment = segment
        isDecoding = true
        Task {
            let prediction = await Task.detached(priority: .userInitiated) { decoder.predict(segment) }.value
            isDecoding = false
            decoded.append(DecodedWord(prediction: prediction))
        }
    }

    private func installKeyMonitor() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
            let consumed = MainActor.assumeIsolated { () -> Bool in
                guard let self else { return false }
                if event.type == .keyDown, self.handleTrainingArrowKey(keyCode: event.keyCode,
                    isRepeat: event.isARepeat, modifiers: event.modifierFlags,
                    isEditingText: NSApp.keyWindow?.firstResponder is NSText) { return true }
                guard event.keyCode == 49, self.screen == .decode, self.decodeMode == .hold,
                      self.decoder != nil, !self.sentenceTraining.isBusy, !self.signalCheck.isBusy, !self.representation.isBusy, !self.sequenceLearning.isBusy, !self.observability.isBusy,
                      !(NSApp.keyWindow?.firstResponder is NSText) else { return false }
                if event.type == .keyUp { self.capture.end() }
                else if !event.isARepeat { self.capture.begin() }
                return true
            }
            return consumed ? nil : event
        }
    }

    /// Space and the forward arrows start, finish or continue; backward arrows redo. Held keys never carry into later takes.
    func handleTrainingArrowKey(keyCode: UInt16, isRepeat: Bool, modifiers: NSEvent.ModifierFlags = [],
                                  isEditingText: Bool = false) -> Bool {
        guard screen == .train, ((showObservability && observability.isAcquiring) || sentenceTraining.isAcquiring || (showSignalCheck && signalCheck.isAcquiring) || (showRepresentation && representation.isAcquiring) || ((showSequenceLearning || showNonSpeechCollection) && sequenceLearning.isAcquiring)), !isEditingText,
              [123, 124, 125, 126, 49].contains(keyCode),
              modifiers.intersection([.command, .control, .option, .shift]).isEmpty else { return false }
        let forward = keyCode == 124 || keyCode == 125 || keyCode == 49
        if !isRepeat {
            if showObservability && observability.isAcquiring {
                if forward { observability.next() } else { observability.redo() }
                return true
            }
            if (showSequenceLearning || showNonSpeechCollection) && sequenceLearning.isAcquiring {
                if forward {sequenceLearning.next()} else {sequenceLearning.redo()}
                return true
            }
            if showRepresentation && representation.isAcquiring {
                if forward { representation.next() }
                else { representation.redo() }
                return true
            }
            if showSignalCheck && signalCheck.isAcquiring {
                if forward { signalCheck.next() }
                else { signalCheck.redo() }
                return true
            }
            if forward { sentenceTraining.advance() }
            else { sentenceTraining.redoCurrentTake() }
        }
        return true
    }

    func clearDecoded() { decoded.removeAll(); lastSegment = [] }
    func toggleMic() async {
        guard !sentenceTraining.isBusy, !signalCheck.isBusy, !representation.isBusy, !sequenceLearning.isBusy, !observability.isBusy else { return }
        if audio.isRunning { audio.stop() }
        else { await audio.start() }
    }
}
