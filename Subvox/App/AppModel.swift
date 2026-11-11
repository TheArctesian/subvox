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
        case home = "Home", train = "Train", decode = "Try model", devices = "Devices"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .home: "house"
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
    // As a unit-test host the app never reads the Keychain: a rebuilt (re-signed) app would get a
    // Keychain password prompt there, and the test runner would wait on it forever.
    let huggingFace = HuggingFaceAccount(store: SubvoxApp.isTestHost ? EphemeralTokenStore() : KeychainTokenStore())
    let contribution = ContributionSettings()
    let uploader = CommunityUploader()
    @ObservationIgnored private var automaticSharingTimer: Task<Void, Never>?
    @ObservationIgnored private var automaticSharingRunning = false

    var automaticSharingGrant: ContributionSettings.AutomaticGrant? {
        contribution.activeAutomaticGrant(for: huggingFace.username)
    }

    var librarySharingGrant: ContributionSettings.AutomaticGrant? {
        contribution.activeLibraryGrant(for: huggingFace.username)
    }

    func enableLibrarySharing() {
        if let username = huggingFace.username { contribution.enableLibrarySharing(for: username) }
    }

    func stopLibrarySharing() { contribution.stopLibrarySharing() }

    /// Subvox is a data-collection app first: the main window opens only while the learner is
    /// signed in to Hugging Face and automatic sharing is on. Otherwise Welcome asks for both.
    var canUseApp: Bool { onboarded && huggingFace.username != nil && automaticSharingGrant != nil }

    func stopAutomaticSharing() {
        contribution.stopAutomaticSharing()
        if automaticSharingRunning { uploader.stop() }
    }

    func signOutOfHuggingFace() {
        stopAutomaticSharing()
        huggingFace.signOut()
    }

    func sendAutomaticContributions() async {
        guard started, !automaticSharingRunning, !uploader.busy, let grant = automaticSharingGrant else { return }
        let library = librarySharingGrant
        let visits = Set(sentenceTraining.data.visits.lazy.filter {
            $0.automaticContributionID == grant.id && $0.endedAt != nil
        }.map(\.id))
        let checkingVisits = Set(sentenceTraining.data.pairs.lazy.filter { $0.needsBackgroundCheck == true }.map(\.visitID))
        let libraryVisits = Set(sentenceTraining.data.visits.lazy.filter { visit in
            library.map { visit.libraryContributionID == $0.id } == true
                && visit.endedAt != nil && !checkingVisits.contains(visit.id)
        }.map(\.id))
        guard !visits.isEmpty || !libraryVisits.isEmpty else { return }
        let handled = Set(uploader.items.filter { $0.status == .uploaded || ($0.takes > 0 && $0.dropped.count >= $0.takes) }.map(\.id))
        let hasPending = sentenceTraining.data.pairs.contains { pair in
            guard pair.role != .finalTest, pair.review != .excluded, let session = pair.recordingSessionID else { return false }
            if libraryVisits.contains(pair.visitID) {
                guard pair.canAccept, pair.hasReliableTranscript, LibraryPilotCatalog.bundled.prompt(for: pair) != nil else { return false }
            } else {
                guard visits.contains(pair.visitID), pair.promptIDs.count == 1,
                      sentenceTraining.community.prompt(pair.promptIDs[0]) != nil,
                      pair.spoken?.samples.isEmpty == false || pair.silent?.samples.isEmpty == false else { return false }
            }
            return !handled.contains(CommunityUploader.publicSessionID(session))
        }
        guard hasPending else { return }
        automaticSharingRunning = true
        defer { automaticSharingRunning = false }
        guard let token = try? await huggingFace.accessToken(),
              automaticSharingGrant == grant, librarySharingGrant == library, !Task.isCancelled,
              let sessions = await uploader.prepare(contributor: grant.username, automatic: grant, library: library),
              automaticSharingGrant == grant, librarySharingGrant == library, !Task.isCancelled else { return }
        let pending = Set(uploader.pending.map(\.id))
        let ids = sessions.map(\.id).filter { pending.contains($0) }
        guard !ids.isEmpty else { return }
        await uploader.sendReviewed(ids, repo: grant.repo, token: token, automatic: true, library: library) {
            self.started && self.automaticSharingGrant == grant && self.librarySharingGrant == library
        }
    }
    var trainMode: TrainMode = .reading
    /// Bumped by Practice › Start (⌘↩); the visible practice page starts its round.
    var startRequest = 0
    /// The active practice language for every mode, always one of `practiceLanguages`.
    var language = AppLanguage(rawValue: UserDefaults.standard.string(forKey: "ui.language") ?? "") ?? .en {
        didSet {
            UserDefaults.standard.set(language.rawValue, forKey: "ui.language")
            if !practiceLanguages.contains(language) { practiceLanguages.append(language) }
        }
    }
    /// Every language the learner practises; the toolbar switches between them.
    var practiceLanguages: [AppLanguage] = AppModel.storedPracticeLanguages() {
        didSet {
            if practiceLanguages.isEmpty { practiceLanguages = [language]; return }
            UserDefaults.standard.set(practiceLanguages.map(\.rawValue), forKey: "ui.practiceLanguages")
            if !practiceLanguages.contains(language) { language = practiceLanguages[0] }
        }
    }
    /// Spoken-language codes the learner offers ("I speak"), as used by the corpus and hints.
    static let spokenLanguageCodes = ["en", "fr", "es", "zh"]
    /// Every language the learner speaks. `nativeLanguage` is the one used for hints and translations.
    var spokenLanguages: [String] = AppModel.storedSpokenLanguages() {
        didSet {
            if spokenLanguages.isEmpty { spokenLanguages = [nativeLanguage]; return }
            UserDefaults.standard.set(spokenLanguages, forKey: "ui.spokenLanguages")
            if !spokenLanguages.contains(nativeLanguage) { nativeLanguage = spokenLanguages[0] }
        }
    }
    /// The spoken language for "tricky for you" hints and translations. Defaults to the Mac's language.
    var nativeLanguage = UserDefaults.standard.string(forKey: "ui.nativeLanguage")
        ?? (AppModel.spokenLanguageCodes.first { $0 == Locale.current.language.languageCode?.identifier } ?? "en") {
        didSet {
            UserDefaults.standard.set(nativeLanguage, forKey: "ui.nativeLanguage")
            if !spokenLanguages.contains(nativeLanguage) { spokenLanguages.append(nativeLanguage) }
        }
    }
    func togglePracticeLanguage(_ item: AppLanguage) {
        if let index = practiceLanguages.firstIndex(of: item) {
            if practiceLanguages.count > 1 { practiceLanguages.remove(at: index) }
        } else {
            practiceLanguages.append(item)
        }
    }
    func toggleSpokenLanguage(_ code: String) {
        if let index = spokenLanguages.firstIndex(of: code) {
            if spokenLanguages.count > 1 { spokenLanguages.remove(at: index) }
        } else {
            spokenLanguages.append(code)
        }
    }
    private static func storedPracticeLanguages() -> [AppLanguage] {
        let stored = (UserDefaults.standard.stringArray(forKey: "ui.practiceLanguages") ?? []).compactMap(AppLanguage.init(rawValue:))
        if !stored.isEmpty { return stored }
        return [AppLanguage(rawValue: UserDefaults.standard.string(forKey: "ui.language") ?? "") ?? .en]
    }
    private static func storedSpokenLanguages() -> [String] {
        let stored = (UserDefaults.standard.stringArray(forKey: "ui.spokenLanguages") ?? []).filter(spokenLanguageCodes.contains)
        if !stored.isEmpty { return stored }
        return [UserDefaults.standard.string(forKey: "ui.nativeLanguage")
                ?? (spokenLanguageCodes.first { $0 == Locale.current.language.languageCode?.identifier } ?? "en")]
    }
    /// Leaves any Reading-only full-page view (sharing, research tools) when switching mode.
    func closeOverlays() {
        showObservability = false; showNonSpeechCollection = false
        showSequenceLearning = false; showRepresentation = false; showSignalCheck = false
    }
    let practice = PracticeCorpus.bundled
    let personalPracticePieces = PracticeCorpus.personalPieces()
    @ObservationIgnored lazy var practiceChecker = PracticeChecker()
    @ObservationIgnored let referenceSpeaker = ReferenceSpeaker()
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
    let sentenceTraining: SentenceTrainingWorkflow
    var backgroundActivity: String? {
        if sentenceTraining.phase == .preparing { return String(localized: "Preparing recording…") }
        if sentenceTraining.phase == .processing || sentenceTraining.isCheckingRecordings {
            return sentenceTraining.processingText.isEmpty ? "Saving recordings on this Mac" : sentenceTraining.processingText
        }
        if signalCheck.phase == .processing { return "Scoring the motion check on this Mac" }
        if representation.phase == .processing { return "Evaluating frozen models on this Mac" }
        if sequenceLearning.phase == .working { return "Running a local model job" }
        if observability.phase == .analysing { return "Scoring the mouth signal check on this Mac" }
        if uploader.busy { return uploader.status.isEmpty ? "Preparing the sharing queue" : uploader.status }
        return nil
    }
    var captureShortcutsSuspended = false

    var navigationLocked: Bool {
        sentenceTraining.isAcquiring || sentenceTraining.phase == .preparing ||
        signalCheck.showsCaptureCue || signalCheck.phase == .preparing ||
        representation.isAcquiring || sequenceLearning.isAcquiring || observability.isAcquiring
    }
    var canRestartInterface: Bool {
        !navigationLocked && !recorder.isRecording && !uploader.busy && !automaticSharingRunning &&
        !sentenceTraining.isBusy && !signalCheck.isBusy && !representation.isBusy &&
        !sequenceLearning.isBusy && !observability.isBusy
    }
    func openReading(showModels: Bool = false) {
        guard !navigationLocked else { return }
        closeOverlays()
        trainMode = .reading
        showSequenceLearning = showModels
        screen = .train
    }
    func openHome() {
        guard !navigationLocked else { return }
        closeOverlays()
        screen = .home
    }

    func openPractice(_ mode: TrainMode) {
        guard !navigationLocked else { return }
        if mode != trainMode {
            closeOverlays()
            referenceSpeaker.stop()
        }
        trainMode = mode
        screen = .train
    }

    var screen: Screen = .home {
        didSet {
            if oldValue != screen { referenceSpeaker.stop() }
            if oldValue == .decode && screen != .decode {
                capture.cancelCapture()
                decodeGeneration += 1
                pendingDecodes = 0
            }
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
    private var decodeGeneration = 0
    private var pendingDecodes = 0
    var isDecoding: Bool { pendingDecodes > 0 }
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
        self.sentenceTraining.onCommunitySessionSaved = { [weak self] in
            Task { await self?.sendAutomaticContributions() }
        }
        self.sentenceTraining.onTrainingChanged = { [weak self] in
            self?.rebuildDecoder()
            if self?.librarySharingGrant != nil { Task { await self?.sendAutomaticContributions() } }
        }
        self.sentenceTraining.onPracticeProcessed = { [weak self] id in await self?.checkPhones(id) }
        self.sentenceTraining.playReference = { [weak self] text, language in await self?.referenceSpeaker.speak(text, language: language) }
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
        sentenceTraining.resumeBackgroundChecks(whisper: localWhisper)
        automaticSharingTimer = Task { [weak self] in
            while !Task.isCancelled {
                await self?.sendAutomaticContributions()
                do { try await Task.sleep(for: .seconds(60)) }
                catch { return }
            }
        }
    }

    func prepareForTermination() {
        started = false
        automaticSharingTimer?.cancel()
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
        let snapshot = sentenceTraining.data
        guard !snapshot.pairs.isEmpty else { isBuildingDecoder = false; return }
        isBuildingDecoder = true
        let settings = decodeSettings
        decoderBuild = Task {
            let build = Task.detached(priority: .utility) {
                let takes = snapshot.derivedWords(role: .training, lexicon: .bundled)
                guard !takes.isEmpty, !Task.isCancelled else { return nil as MotionWordDecoder? }
                return MotionWordDecoder(takes: takes, lexicon: .bundled, vocabulary: Lexicon.vocabulary, settings: settings)
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

    /// Records a practice round through the shared paired interface: each group is spoken, then mouthed.
    func startPractice(_ groups: [[SentencePrompt]], libraryContributionID: UUID? = nil) {
        guard !groups.isEmpty, localWhisper.isConfigured else { return }
        sentenceTraining.practicePlan = groups
        startSentenceTraining(libraryContributionID: libraryContributionID)
        sentenceTraining.practicePlan = nil
    }

    var canResumeLiterature: Bool {
        canUseApp && !navigationLocked && motion.status == .streaming && localWhisper.isConfigured
            && sentenceTraining.storageReady && sentenceTraining.data.reservedUnlockedAt == nil
            && sentenceTraining.canStartRecording && !recorder.isRecording
            && !signalCheck.isBusy && !representation.isBusy && !sequenceLearning.isBusy && !observability.isBusy
    }

    func startLiterature(_ groups: [[SentencePrompt]], pieceID: String, title: String) {
        guard canResumeLiterature else { return }
        sentenceTraining.literatureSelection = .new(pieceID: pieceID, title: title)
        defer { sentenceTraining.literatureSelection = nil }
        startPractice(groups, libraryContributionID: librarySharingGrant?.id)
    }

    func resumeLiterature(_ round: LiteraturePracticeRound) {
        guard canResumeLiterature,
              sentenceTraining.resumableLiterature.contains(where: { $0.id == round.id }),
              let language = AppLanguage(rawValue: round.language) else { return }
        self.language = language
        openPractice(round.kind == .poetry ? .poetry : .speech)
        // Use the same selection key as the library so the piece is visible after the round.
        SavedSelection.set(round.kind.rawValue, language, round.pieceID)
        sentenceTraining.literatureSelection = .resume(round.id)
        defer { sentenceTraining.literatureSelection = nil }
        startPractice(round.groups, libraryContributionID: librarySharingGrant?.id)
    }

    private func checkPhones(_ id: UUID) async {
        guard let pair = sentenceTraining.data.pairs.first(where: { $0.id == id }), let heard = pair.transcript?.text else { return }
        let result = await practiceChecker.check(text: pair.text, language: pair.language ?? "en", heard: heard)
        await sentenceTraining.setPhoneCheck(result, for: id)
    }

    func startSentenceTraining(repairs: Bool = false, retry: UUID? = nil, libraryContributionID: UUID? = nil) {
        guard motion.status == .streaming, localWhisper.isConfigured || sentenceTraining.readingSource == .community, !signalCheck.isBusy, !representation.isBusy, !sequenceLearning.isBusy, !observability.isBusy else { return }
        referenceSpeaker.stop()
        sentenceTraining.start(audio: audio, recorder: recorder, whisper: localWhisper,
            canRecord: { [weak self] in self?.motion.status == .streaming }, repairs: repairs, retry: retry,
            automaticContributionID: automaticSharingGrant?.id, libraryContributionID: libraryContributionID)
    }

    func startSignalCheck(againstReference: Bool = false, controlled: Bool = false) {
        guard !sentenceTraining.isBusy, !representation.isBusy, !sequenceLearning.isBusy, !observability.isBusy, motion.status == .streaming else { return }
        signalCheck.start(audio: audio, recorder: recorder, againstReference: againstReference, controlled: controlled,
            canRecord: { [weak self] in self?.motion.status == .streaming })
    }

    private func handleSegment(_ segment: [MotionSample]) {
        guard screen == .decode, !sentenceTraining.isBusy, !signalCheck.isBusy, !representation.isBusy, !sequenceLearning.isBusy, !observability.isBusy, let decoder else { return }
        lastSegment = segment
        let generation = decodeGeneration
        pendingDecodes += 1
        Task {
            let prediction = await Task.detached(priority: .userInitiated) { decoder.predict(segment) }.value
            guard generation == decodeGeneration else { return }
            pendingDecodes -= 1
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
        guard screen == .train, ((showObservability && observability.isAcquiring) || sentenceTraining.isAcquiring || (showSignalCheck && signalCheck.showsCaptureCue) || (showRepresentation && representation.isAcquiring) || ((showSequenceLearning || showNonSpeechCollection) && sequenceLearning.isAcquiring)), !isEditingText, !captureShortcutsSuspended,
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
            if showSignalCheck && signalCheck.showsCaptureCue {
                if forward { signalCheck.next() }
                else { signalCheck.redo() }
                return true
            }
            if forward { sentenceTraining.advance() }
            else { sentenceTraining.redoCurrentTake() }
        }
        return true
    }

    func clearDecoded() {
        decodeGeneration += 1
        pendingDecodes = 0
        capture.clearFeedback()
        decoded.removeAll()
        lastSegment = []
    }
    func toggleMic() async {
        guard !sentenceTraining.isBusy, !signalCheck.isBusy, !representation.isBusy, !sequenceLearning.isBusy, !observability.isBusy else { return }
        if audio.isRunning { audio.stop() }
        else { await audio.start() }
    }
}
