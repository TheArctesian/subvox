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
    let sentenceTraining: SentenceTrainingWorkflow
    var screen: Screen = .train {
        didSet {
            if oldValue == .train && screen != .train { sentenceTraining.stop() }
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

    init(sentenceTraining: SentenceTrainingWorkflow? = nil) {
        self.sentenceTraining = sentenceTraining ?? SentenceTrainingWorkflow()
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
        }
        capture.onSegment = { [weak self] in self?.handleSegment($0) }
        motion.start()
        audio.refreshDevices()
        installKeyMonitor()
    }

    func prepareForTermination() {
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
        let enabled = screen == .decode && decodeMode == .auto && decoder != nil && !sentenceTraining.isBusy
        if capture.autoEnabled != enabled { capture.autoEnabled = enabled }
    }

    func startSentenceTraining(repairs: Bool = false, retry: UUID? = nil) {
        guard motion.status == .streaming, localWhisper.isConfigured else { return }
        sentenceTraining.start(audio: audio, recorder: recorder, whisper: localWhisper,
            canRecord: { [weak self] in self?.motion.status == .streaming }, repairs: repairs, retry: retry)
    }

    private func handleSegment(_ segment: [MotionSample]) {
        guard screen == .decode, !sentenceTraining.isBusy, let decoder else { return }
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
            guard event.keyCode == 49 else { return event }
            let consumed = MainActor.assumeIsolated { () -> Bool in
                guard let self, self.screen == .decode, self.decodeMode == .hold,
                      self.decoder != nil, !self.sentenceTraining.isBusy,
                      !(NSApp.keyWindow?.firstResponder is NSText) else { return false }
                if event.type == .keyUp { self.capture.end() }
                else if !event.isARepeat { self.capture.begin() }
                return true
            }
            return consumed ? nil : event
        }
    }

    func clearDecoded() { decoded.removeAll(); lastSegment = [] }
    func toggleMic() async {
        guard !sentenceTraining.isBusy else { return }
        if audio.isRunning { audio.stop() }
        else { await audio.start() }
    }
}
