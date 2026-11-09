import AppKit
import Foundation
import Observation
import SwiftUI

/// Paired acquisition, annotation review, coverage repairs and word validation.
@MainActor @Observable
final class SentenceTrainingWorkflow {
    enum Phase: Equatable { case setup, preparing, ready, relax, speak, mouth, ended, recovery, processing, finished }
    enum Presentation: String, CaseIterable, Identifiable {
        case sentences = "Sentences", passage = "Short passages"
        var id: String { rawValue }
    }
    private(set) var data = SentenceTrainingData()
    private(set) var pairUses: [UUID: SentencePairUse] = [:]
    private(set) var phase: Phase = .setup { didSet { if phase != oldValue { announce(phase) } } }

    /// VoiceOver cues, only at moments outside the recorded cue window: before a take starts
    /// (ready), after it is saved (ended), and when the round has been processed (finished).
    private func announce(_ phase: Phase) {
        var text: AttributedString
        switch phase {
        case .ready:
            let instruction = currentCondition == .normal ? String(localized: "Say it aloud") : String(localized: "Mouth it silently, lips only")
            text = AttributedString(String(localized: "Take \(min(index + 1, total)) of \(total). \(instruction): "))
            var prompt = AttributedString(currentText)
            prompt.languageIdentifier = currentLanguage  // read in its own language
            text += prompt + AttributedString(String(localized: ". Press Space to start; a tone marks the start."))
        case .ended: text = AttributedString(String(localized: "Take saved. Space for the next one, left arrow to redo."))
        case .processing: text = AttributedString(String(localized: "Saving and checking your takes."))
        case .finished: text = AttributedString(lastPracticePairIDs.isEmpty ? String(localized: "Recording finished.") : String(localized: "Round finished. Your results are at the top of the page."))
        default: return
        }
        AccessibilityNotification.Announcement(text).post()
    }
    private(set) var error: String?
    private(set) var storageReady = true
    private(set) var currentText = ""
    private(set) var currentEnglishTranslation: String?
    private(set) var currentPinyin: [String?] = []
    private(set) var currentVerseLines: [String] = []
    private(set) var isSavingTake = false
    private(set) var currentCondition: SpeechCondition = .normal
    /// The practice mode of the take being recorded (nil for Reading), for the recording screen's title.
    private(set) var currentKind: SentencePromptKind?
    /// The language tag of the take being recorded, for VoiceOver.
    private(set) var currentLanguage: String?
    private(set) var index = 0
    private(set) var total = 0
    private(set) var secondsRemaining = 0.0
    private(set) var processingText = ""
    var presentation: Presentation = .passage
    var pace: TrainingPace = .comfortable
    var sentenceCount = 6
    var notes = ""
    var guidedRecording = false
    var variedSentences = false
    var excludedCollectionGroups:Set<String> = []
    private var advanceRequested = false
    let phoneAligner = LocalPhonemeAligner()
    let forcedAligner = ForcedAligner()
    private(set) var realignmentReport: (recovered: Int, total: Int)?
    let camera = CameraPoseService()
    var captureCameraPoints = false
    /// Optional explicit teacher collection keeps whole train/development visits apart.
    var collectionRole:DataRole?
    @ObservationIgnored private var cameraReadings: [CameraPoseReading] = []
    @ObservationIgnored private weak var cameraRecorder: SessionRecorder?
    var onTrainingChanged: (() -> Void)?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var buffer: [MotionSample] = []
    @ObservationIgnored private let storeURL: URL
    @ObservationIgnored private var changedPairUses: Set<UUID> = []
    @ObservationIgnored private var needsAllPairUses = true
    @ObservationIgnored private let storeWrites = DispatchQueue(label: "Subvox.sentence-store", qos: .userInitiated)
    @ObservationIgnored private var activeVisitID: UUID?
    @ObservationIgnored private var activeRecordingID: UUID?
    @ObservationIgnored private var isTerminating = false
    private var activeCueStart: Double?
    private var requestedCueEnd: Double?
    private var redoRequested = false
    @ObservationIgnored private var retryableTake: SentenceRecording?
    @ObservationIgnored private var pendingRedoCleanup: (() throws -> Void)?
    var canFinishTake: Bool {
        (phase == .speak || phase == .mouth) && activeCueStart != nil && requestedCueEnd == nil && !redoRequested
    }
    var canRedoTake: Bool { !redoRequested && (activeCueStart != nil || ([.ended,.recovery].contains(phase) && retryableTake != nil)) }
    var canAdvance:Bool {canFinishTake || (guidedRecording && [.ready,.ended].contains(phase) && !advanceRequested && !redoRequested)}
    /// Seconds left in the word highlighting, or nil once every word has been shown (the take
    /// then waits for Done/Space; `secondsRemaining` is only the safety timeout).
    var guideSecondsRemaining: Double? {
        // Reading the countdown (updated every 50 ms) makes views that show this redraw as time passes.
        _ = secondsRemaining
        guard let activeCueStart, phase == .speak || phase == .mouth else { return nil }
        let left = WordFocusGuide.comfortable.duration(count: WordFocusGuide.words(in: currentText).count) - (ProcessInfo.processInfo.systemUptime - activeCueStart)
        return left > 0 ? left : nil
    }
    var activeWordIndex:Int? {
        _ = secondsRemaining  // ticks every 50 ms while recording, so the highlight advances
        guard let activeCueStart,phase == .speak || phase == .mouth else {return nil}
        return WordFocusGuide.comfortable.wordIndex(elapsed:ProcessInfo.processInfo.systemUptime-activeCueStart,count:WordFocusGuide.words(in:currentText).count)
    }
    var settings: DecoderSettings { data.decoderSettings ?? DecoderSettings() }
    var settingsLocked: Bool { isBusy || !storageReady || data.reservedUnlockedAt != nil }
    var learningMethod: LearningMethod {
        get { settings.method }
        set { changeSettings { $0.method = newValue } }
    }
    var isAcquiring: Bool { [.ready,.relax, .speak, .mouth,.ended, .recovery].contains(phase) }
    var isBusy: Bool { isAcquiring || phase == .preparing || phase == .processing }
    var coverage: [String: Int] { data.coverage(lexicon: .bundled) }
    var fittingWords: [Template] { data.derivedWords(role: .training, lexicon: .bundled) }
    var repairWords: [String] { data.repairWords(lexicon: .bundled) }
    enum ReadingSource: String, CaseIterable, Identifiable { case banks = "Sentence banks", community = "Community passages"; var id: String { rawValue } }
    var readingSource: ReadingSource = .banks
    /// Community mode only: skip the spoken half (pairs still make phone alignment possible, so off by default).
    var mouthedOnly = false
    var communityPassageCount = 3
    let community: CommunityCorpus
    private(set) var currentPromptIDs: [String] = []
    var currentPassageLabel: String? {
        guard let first = currentPromptIDs.first, let prompt = community.prompt(first) else { return nil }
        return prompt.passageID == nil ? "Calibration · read by every contributor" : "\(prompt.title) — \(prompt.author)"
    }
    var calibrationRemaining: [CommunityPrompt] {
        let done = data.mouthedPromptIDs
        return community.calibration.filter { !done.contains($0.promptID) }
    }
    /// Calibration first; then passages that fill this person's diphone gaps, each in source order.
    var communityPlan: [[SentencePrompt]] {
        let done = data.mouthedPromptIDs
        if !calibrationRemaining.isEmpty { return calibrationRemaining.map { [$0.sentencePrompt] } }
        let finished = Set(community.passages.filter { $0.prompts.allSatisfy { done.contains($0.promptID) } }.map(\.id))
        return community.assign(counts: community.counts(recordedPromptIDs: done), recordedPassages: finished, count: communityPassageCount)
            .flatMap { $0.prompts.filter { !done.contains($0.promptID) }.map { [$0.sentencePrompt] } }
    }
    /// Set by a practice tab just before `start`; recording then uses these prompts (spoken, then mouthed).
    var practicePlan: [[SentencePrompt]]?
    /// Pairs recorded by the most recent practice round, for the tab's feedback view.
    private(set) var lastPracticePairIDs: [UUID] = []
    /// Called after a practice pair is transcribed, to run the audio phone check.
    @ObservationIgnored var onPracticeProcessed: ((UUID) async -> Void)?
    /// Shadowing drills: plays the phrase in its language before the spoken take (text, language).
    @ObservationIgnored var playReference: ((String, String) async -> Void)?
    /// Free practice trains only when Whisper agrees with this share of the typed words.
    static let freeTextAgreement = 0.85

    func clearPracticeResults() { lastPracticePairIDs = [] }

    func setPhoneCheck(_ result: PhoneCheckResult?, for id: UUID) async {
        guard let i = data.pairs.firstIndex(where: { $0.id == id }) else { return }
        data.pairs[i].phoneCheck = result
        await saveInBackground()
    }

    var plan: [[SentencePrompt]] {
        if let practicePlan { return practicePlan }
        if readingSource == .community { return communityPlan }
        if variedSentences {
            return data.variedSentences(role:collectionRole ?? .training,limit:sentenceCount,excluding:excludedCollectionGroups).map {[$0]}
        }
        return data.plannedSentences(limit: sentenceCount, passage: presentation == .passage, lexicon: .bundled).filter {group in
            collectionRole.map {role in group.allSatisfy {data.roles[$0.id]==role}} ?? true
        }
    }
    func addSentenceBank(_ prompts:[SentencePrompt] = SentencePrompt.diphoneBank) {
        guard !isBusy,storageReady,data.reservedUnlockedAt==nil else {return}
        var generator = SystemRandomNumberGenerator()
        data.add(prompts, using: &generator)
        needsAllPairUses = true
        save()
    }
    var estimatedSeconds: Double { plan.reduce(0) { $0 + 2 * (pace.baseline + cueDuration($1.map(\.text).joined(separator: " ")) + pace.recovery) } }

    init(storeURL: URL? = nil, corpus: [SentencePrompt]? = nil, community: CommunityCorpus = .bundled) {
        self.community = community
        self.storeURL = storeURL ?? SentenceStore.url
        do {
            data = try SentenceStore.load(from: self.storeURL)
            if data.prompts.isEmpty {
                var generator = SystemRandomNumberGenerator()
                data.add(corpus ?? SentencePrompt.bundled, using: &generator)
                guard !data.prompts.isEmpty else { throw Failure.noCorpus }
                try persist()
            }
        } catch { self.error = error.localizedDescription; storageReady = false }
        refreshPairUses()
        camera.onReading = { [weak self] reading in
            guard let self, let id = self.activeRecordingID, self.captureCameraPoints else { return }
            self.cameraReadings.append(reading)
            _ = self.cameraRecorder?.recordCamera(reading, sessionID: id)
        }
    }

    func changeSettings(_ change: (inout DecoderSettings) -> Void) {
        guard !settingsLocked else { return }
        var next = settings
        change(&next)
        guard next != settings else { return }
        data.decoderSettings = next
        data.validationReport = nil
        save(notify: true)
    }

    @discardableResult
    func beginVisit(session: RecordingSession, audioDevice: String?, timing: TrainingPace,
                    presentation: Presentation, persistImmediately: Bool = true) throws -> SentenceVisit {
        let visit = SentenceVisit(notes: notes, directory: session.directory.path, audioDevice: audioDevice,
                                  pace: timing.rawValue, presentation: presentation.rawValue, recordingSessionID: session.id)
        activeVisitID = visit.id; activeRecordingID = session.id
        data.visits.append(visit)
        if persistImmediately { try persist() }
        return visit
    }

    func beginPair(prompts: [SentencePrompt], visit: SentenceVisit, session: RecordingSession, persistImmediately: Bool = true) throws -> SentencePair {
        guard session.id == activeRecordingID, visit.id == activeVisitID,
              let first = prompts.first, let role = data.roles[first.id], role != .finalTest,
              prompts.allSatisfy({ data.roles[$0.id] == role }) else { throw Failure.split }
        let pair = SentencePair(visitID: visit.id, promptIDs: prompts.map(\.id),
            text: prompts.map(\.text).joined(separator: " "), role: role, directory: session.directory.path,
            sessionStartUptime: session.startUptime, audioOffset: session.audioOffset, recordingSessionID: session.id)
        data.pairs.append(pair)
        changedPairUses.insert(pair.id)
        data.pairs[data.pairs.count - 1].language = first.language
        data.pairs[data.pairs.count - 1].practiceMode = first.kind.isPractice ? first.kind : nil
        data.pairs[data.pairs.count - 1].pronunciationProfile = try PronunciationProfile.load()
        if let v = data.visits.firstIndex(where: { $0.id == visit.id }) { data.visits[v].pairIDs.append(pair.id) }
        if persistImmediately { try persist() }
        return pair
    }

    func saveCompletedHalf(_ recording: SentenceRecording, pairID: UUID, session: RecordingSession, persistImmediately: Bool = true) throws {
        guard let i = data.pairs.firstIndex(where: { $0.id == pairID }),
              data.pairs[i].recordingSessionID == session.id, session.id == activeRecordingID else { throw Failure.ownership }
        if recording.condition == .normal { data.pairs[i].spoken = recording }
        else if recording.condition == .silent { data.pairs[i].silent = recording }
        else { throw Failure.split }
        data.pairs[i].directory = session.directory.path
        data.pairs[i].sessionStartUptime = session.startUptime
        data.pairs[i].audioOffset = session.audioOffset
        changedPairUses.insert(pairID)
        if persistImmediately { try persist() }
    }

    private func finishVisit(session: RecordingSession?) {
        guard let id = activeVisitID, let v = data.visits.firstIndex(where: { $0.id == id }) else { return }
        data.visits[v].endedAt = Date()
        let rates = data.pairs.filter { $0.visitID == id }.compactMap { $0.spoken?.quality.sampleRate }
        if !rates.isEmpty { data.visits[v].motionRate = rates.reduce(0, +) / Double(rates.count) }
        if let session, session.id == activeRecordingID {
            for i in data.pairs.indices where data.pairs[i].visitID == id {
                data.pairs[i].directory = session.directory.path
                data.pairs[i].sessionStartUptime = session.startUptime
                data.pairs[i].audioOffset = session.audioOffset
                if captureCameraPoints, let a = data.pairs[i].spoken, let b = data.pairs[i].silent {
                    data.pairs[i].cameraReadings = cameraReadings.filter {
                        $0.t >= a.cueStart - 3 && $0.t <= b.cueEnd + 0.3
                    }
                }
            }
        }
        activeVisitID = nil; activeRecordingID = nil
        cameraReadings = []; cameraRecorder = nil; camera.stop()
    }

    /// Synchronous: willTerminate cannot wait for the acquisition task to unwind.
    func prepareForTermination(recorder: SessionRecorder) {
        isTerminating = true
        task?.cancel()
        if redoRequested {
            do { try discardRedoAttempt() }
            catch { self.error = error.localizedDescription }
        }
        if let id = activeRecordingID {
            finishVisit(session: recorder.stop(sessionID: id))
            save()
        }
        phase = .finished
    }

    func ingest(_ sample: MotionSample) {
        if !isSavingTake && (phase == .relax || phase == .speak || phase == .mouth) {
            if guidedRecording,(phase == .speak || phase == .mouth),let previous=buffer.last,
               sample.t<=previous.t || sample.t-previous.t>0.12 || sample.sensorLocation != previous.sensorLocation {
                error="Motion source changed or capture has a gap. The interrupted attempt is discarded; start again when motion is stable."
                redoRequested=true
            }
            buffer.append(sample)
        }
    }
    func reset() { guard !isBusy else { return }; phase = .setup }
    func stop() {
        guard isAcquiring || phase == .preparing else { return }
        task?.cancel()
    }

    /// Finish this take at the button/key press, without cancelling the recording visit.
    @discardableResult
    func finishCurrentTake() -> Bool {
        guard canFinishTake else { return false }
        requestedCueEnd = ProcessInfo.processInfo.systemUptime
        return true
    }

    @discardableResult func advance() -> Bool {
        if canFinishTake {return finishCurrentTake()}
        guard canAdvance else {return false}
        advanceRequested=true;return true
    }
    private func waitForAdvance(_ phase:Phase) async throws {
        self.phase=phase;advanceRequested=false;secondsRemaining=0
        defer {advanceRequested=false}
        while !advanceRequested {
            try Task.checkCancellation()
            if redoRequested {throw CaptureControl.redo}
            try await Task.sleep(for:.milliseconds(30))
        }
        if redoRequested {throw CaptureControl.redo}
    }

    @discardableResult
    func redoCurrentTake() -> Bool {
        guard canRedoTake else { return false }
        redoRequested = true
        return true
    }

    private enum CaptureControl: Error { case redo }

    /// Retries only this half. A recovery-time redo first removes the saved, unwanted half.
    func collectTake(condition: SpeechCondition, maximumDuration: Double, baselineSeconds: Double,
                     recoverySeconds: Double,text:String?=nil, canRecord: () -> Bool = { true }, onCompleted: (SentenceRecording) async throws -> Void,
                     onRedo: @escaping (SentenceRecording?) throws -> Void) async throws {
        if let text {
            currentText = text
            currentEnglishTranslation = nil
            currentPinyin = []
            currentVerseLines = []
        }
        currentCondition=condition
        defer { retryableTake = nil; redoRequested = false; pendingRedoCleanup = nil }
        while true {
            try Task.checkCancellation()
            buffer = []; retryableTake = nil; redoRequested = false
            pendingRedoCleanup = { try onRedo(self.retryableTake) }
            do {
                if guidedRecording {try await waitForAdvance(.ready)}
                try await rest(.relax, seconds: baselineSeconds)
                guard canRecord() else { throw Failure.disconnected }
                let recording = try await captureTake(condition: condition, maximumDuration: maximumDuration)
                retryableTake = recording
                do {
                    isSavingTake = true
                    defer { isSavingTake = false }
                    try await onCompleted(recording)
                }
                if guidedRecording {try await waitForAdvance(.ended)}
                else {try await rest(.recovery, seconds: recoverySeconds)}
                if redoRequested { throw CaptureControl.redo }
                return
            } catch CaptureControl.redo {
                try discardRedoAttempt()
            } catch {
                if redoRequested { try discardRedoAttempt() }
                throw error
            }
        }
    }

    private func discardRedoAttempt() throws {
        let cleanup = pendingRedoCleanup
        pendingRedoCleanup = nil
        try cleanup?()
    }

    func discardHalf(condition: SpeechCondition, pairID: UUID) throws {
        guard let id = activeRecordingID, let i = data.pairs.firstIndex(where: { $0.id == pairID }),
              data.pairs[i].recordingSessionID == id else { throw Failure.ownership }
        if condition == .normal { data.pairs[i].spoken = nil }
        else { data.pairs[i].silent = nil }
        data.pairs[i].alignment = nil; data.pairs[i].review = .pending
        data.pairs[i].wordTimingSource = nil; data.pairs[i].wordTimingModelRevision = nil
        data.pairs[i].spokenPhones = nil; data.pairs[i].mouthedPhones = nil
        changedPairUses.insert(pairID)
        try persist()
    }

    /// Short tones mark the start and end of a take, so it can be done without watching the screen.
    /// The start tone ends before the cue window opens, so it is never part of the take.
    var soundCues: Bool { !SubvoxApp.isTestHost && (UserDefaults.standard.object(forKey: "ui.soundCues") as? Bool ?? true) }

    func captureTake(condition: SpeechCondition, maximumDuration: Double) async throws -> SentenceRecording {
        if soundCues, let tone = NSSound(named: condition == .normal ? "Tink" : "Morse") {
            tone.play()
            try await Task.sleep(for: .milliseconds(350))
        }
        let start = ProcessInfo.processInfo.systemUptime
        activeCueStart = start; requestedCueEnd = nil; redoRequested = false
        phase = condition == .normal ? .speak : .mouth
        defer { activeCueStart = nil; requestedCueEnd = nil }
        try await countdown(maximumDuration, allowsAdvance: true, allowsRedo: true)
        try Task.checkCancellation()
        if redoRequested { throw CaptureControl.redo }
        let end = requestedCueEnd ?? ProcessInfo.processInfo.systemUptime
        if soundCues { NSSound(named: "Pop")?.play() }  // after the cue window has closed
        // The timer resumes on its next tick. Samples after the user's press are not speech.
        return SentenceRecording(condition: condition, samples: buffer.filter { $0.t <= end }, cueStart: start, cueEnd: end,
                                 wordFocusGuide:guidedRecording ? .comfortable:nil)
    }

    func rest(_ phase: Phase, seconds: Double) async throws {
        precondition(phase == .relax || phase == .recovery)
        self.phase = phase
        try await countdown(seconds, allowsRedo: phase == .recovery)
    }
    func cueDuration(_ text: String) -> Double {
        let words = Double(Lexicon.words(in: text).count)
        return min(45, max(pace == .comfortable ? 12 : 16, words * (pace == .comfortable ? 0.85 : 1.1) + 5))
    }

    func review(_ id: UUID, decision: ReviewDecision) {
        guard !isBusy, data.reservedUnlockedAt == nil,
              let i = data.pairs.firstIndex(where: { $0.id == id }), data.pairs[i].role != .finalTest,
              !data.pairs[i].promptIDs.contains(where: { data.roles[$0] == .finalTest }) else { return }
        guard decision != .accepted || data.pairs[i].canIncludeManually else { return }
        data.pairs[i].review = decision
        changedPairUses.insert(id)
        data.validationReport = nil
        save(notify: true)
    }

    func reviewPhoneAlignment(_ id:UUID) {
        guard !isBusy,data.reservedUnlockedAt==nil,
              let i=data.pairs.firstIndex(where:{$0.id==id}),data.pairs[i].role != .finalTest else {return}
        for condition in [SpeechCondition.normal,.silent] {
            let recording=condition == .normal ? data.pairs[i].spoken:data.pairs[i].silent
            let alignment=condition == .normal ? data.pairs[i].spokenPhones:data.pairs[i].mouthedPhones
            if let recording,let alignment,(try? alignment.validate(in:recording.cueStart...recording.cueEnd)) != nil {
                if condition == .normal {data.pairs[i].spokenPhones?.reviewed=true}
                else {data.pairs[i].mouthedPhones?.reviewed=true}
            }
        }
        save()
    }

    func start(audio: AudioService, recorder: SessionRecorder, whisper: LocalWhisper,
               canRecord: @escaping () -> Bool, repairs: Bool = false, retry: UUID? = nil) {
        let practiceMode = practicePlan != nil && retry == nil && !repairs
        let communityMode = !practiceMode && readingSource == .community && retry == nil && !repairs
        guard !isBusy, storageReady, data.reservedUnlockedAt == nil, !recorder.isRecording,
              whisper.isConfigured || communityMode, canRecord() else { return }
        var planned = plan
        let promptCount = data.prompts.count
        if practiceMode {
            data.addPractice(planned.flatMap { $0 })
        }
        if communityMode {
            data.addCommunity(planned.flatMap { $0 })
        }
        if data.prompts.count != promptCount { needsAllPairUses = true }
        if practiceMode { lastPracticePairIDs = [] }
        let conditions: [SpeechCondition] = communityMode && mouthedOnly ? [.silent] : [.normal, .silent]
        if let retry, let pair = data.pairs.first(where: { $0.id == retry }) {
            planned = [data.prompts.filter { pair.promptIDs.contains($0.id) }]
        } else if repairs {
            let prompts = repairWords.map { SentencePrompt(text: $0, kind: .wordRepair) }
            var generator = SystemRandomNumberGenerator()
            data.add(prompts, using: &generator)
            needsAllPairUses = true
            guard save() else { return }
            planned = prompts.prefix(8).map { [$0] }
        }
        guard !planned.isEmpty, planned.allSatisfy({ !$0.isEmpty }),
              planned.allSatisfy({ group in group.allSatisfy {
                  data.roles[$0.id] == .training || data.roles[$0.id] == .validation
              } }) else { return }
        // Keep visits bounded for local audio conversion. Paragraph groups never cross roles.
        var batch: [[SentencePrompt]] = []
        var duration = 0.0
        for group in planned {
            let added = Double(conditions.count) * (pace.baseline + cueDuration(group.map(\.text).joined(separator: " ")) + pace.recovery)
            guard duration + added < 840 else { break }
            batch.append(group); duration += added
        }
        guard !batch.isEmpty else { error = "This batch is too long. Choose fewer sentences."; return }
        let timing = pace
        let cueTimes = batch.map { cueDuration($0.map(\.text).joined(separator: " ")) }
        phase = .preparing; error = nil; index = 0; total = batch.count * conditions.count
        task = Task {
            defer {if activeRecordingID==nil {camera.stop();cameraRecorder=nil;cameraReadings=[]}}
            if practiceMode || communityMode {
                guard await saveInBackground(), !Task.isCancelled else { phase = .setup; return }
            }
            if captureCameraPoints {
                // Acquisition owns this camera; an early microphone/recorder
                // failure must also release the preview and numeric callback.
                await camera.start()
                guard camera.isRunning, !Task.isCancelled else {
                    error = camera.error ?? "Camera tracking could not start."; phase = .setup; return
                }
                cameraRecorder = recorder; cameraReadings = []
            }
            let ownedAudio = !audio.isRunning
            await audio.start()
            guard audio.isRunning, !Task.isCancelled else {
                if ownedAudio { audio.stop() }
                error = audio.error ?? "Recording cancelled before the microphone started."; phase = .setup; return
            }
            recorder.condition = .paired
            recorder.prompt = "Paired sentence training; spoken then silently mouthed; independent sentence split"
            guard let recordingSession = recorder.start(audio: audio, cameraPoints: captureCameraPoints) else {
                if ownedAudio { audio.stop() }
                error = recorder.error ?? "Could not start recording."; phase = .setup; return
            }
            var pairIDs: [UUID] = []
            let device = await MotionDeviceInfo.probe()
            do {
                let visit = try beginVisit(session: recordingSession, audioDevice: audio.activeDeviceName,
                                           timing: timing, presentation: presentation, persistImmediately: false)
                if let v = data.visits.firstIndex(where: { $0.id == visit.id }) { data.visits[v].motionDevice = device }
                try await persistInBackground()
                for (groupIndex, group) in batch.enumerated() {
                    try Task.checkCancellation()
                    guard canRecord() else { throw Failure.disconnected }
                    let pair = try beginPair(prompts: group, visit: visit, session: recordingSession, persistImmediately: false)
                    try await persistInBackground()
                    pairIDs.append(pair.id); currentPromptIDs = pair.promptIDs
                    if practiceMode { lastPracticePairIDs.append(pair.id) }
                    let translations = group.compactMap(\.englishTranslation)
                    currentEnglishTranslation = translations.count == group.count ? translations.joined(separator: " ") : nil
                    currentPinyin = group.flatMap { WordFocusGuide.pinyinByWord(in: $0.text, pinyin: $0.pinyin) }
                    currentText = pair.text
                    currentVerseLines = group.first?.kind.isLiterature == true
                        ? group.map(\.text) : []
                    currentKind = group.first.flatMap { $0.kind.isPractice ? $0.kind : nil }
                    currentLanguage = group.first?.language ?? "en"
                    for condition in conditions {
                        try Task.checkCancellation()
                        currentCondition = condition
                        if condition == .normal, let first = group.first, [.drill, .sound, .minimalPair].contains(first.kind), let playReference {
                            await playReference(pair.text, first.language ?? "en")
                        }
                        try await collectTake(condition: condition, maximumDuration: cueTimes[groupIndex],
                            baselineSeconds: timing.baseline, recoverySeconds: timing.recovery, canRecord: canRecord,
                            onCompleted: { recording in
                                guard canRecord() else { throw Failure.disconnected }
                                guard let checkpoint = try recorder.checkpoint(sessionID: recordingSession.id) else { throw Failure.ownership }
                                try saveCompletedHalf(recording, pairID: pair.id, session: checkpoint, persistImmediately: false)
                                try await persistInBackground()
                                try Task.checkCancellation()
                                recorder.logEvent(recording.samples, kind: condition == .normal ? "sentence-spoken" : "sentence-silent",
                                                  label: pair.text, confidence: nil)
                            }, onRedo: { discarded in
                                try self.discardHalf(condition: condition, pairID: pair.id)
                                recorder.logEvent(discarded?.samples ?? self.buffer, kind: "sentence-redo-discarded", label: pair.text, confidence: nil)
                            })
                        buffer = []; index += 1
                    }
                }
            } catch is CancellationError {
                // Never turn an interrupted cue into a completed reference or silent take.
            } catch { self.error = error.localizedDescription }
            buffer = []
            let finished = recorder.stop(sessionID: recordingSession.id)
            if ownedAudio, !recorder.isRecording { audio.stop() }
            finishVisit(session: finished)
            currentPromptIDs = []
            phase = .processing
            guard !isTerminating, await saveInBackground() else { phase = .finished; return }
            // A fresh unstructured task processes saved complete pairs even after Stop.
            // Community takes need no local Whisper; alignment can run later or elsewhere.
            await Task { @MainActor in
                for id in pairIDs where whisper.isConfigured {
                    await self.process(id, whisper: whisper)
                    if practiceMode { await self.onPracticeProcessed?(id) }
                }
            }.value
            phase = .finished; processingText = ""; task = nil
        }
    }

    func retryProcessing(_ id: UUID, whisper: LocalWhisper) {
        guard !isBusy, data.reservedUnlockedAt == nil else { return }
        phase = .processing
        task = Task { await process(id, whisper: whisper); phase = .finished; processingText = "" }
    }

    var pendingTranscriptionIDs:[UUID] {
        data.pairs.filter {$0.role != .finalTest && $0.promptIDs.allSatisfy {data.roles[$0] != .finalTest} && $0.review == .pending && $0.spoken != nil && ($0.transcript == nil || $0.transcriptMatchRate == nil)}.map(\.id)
    }
    func transcribePending(whisper:LocalWhisper) {
        guard !isBusy,storageReady,whisper.isConfigured,data.reservedUnlockedAt==nil else {return}
        let pending=pendingTranscriptionIDs
        guard !pending.isEmpty else {return}
        phase = .processing
        task=Task {
            for id in pending {if Task.isCancelled {break};await process(id,whisper:whisper)}
            phase = .finished;processingText="";task=nil
        }
    }

    var realignableIDs: [UUID] {
        guard data.reservedUnlockedAt == nil else { return [] }
        let errors = [Failure.timing.localizedDescription, Failure.transcript.localizedDescription]
        return data.pairs.filter { pair in
            pair.complete && pair.role != .finalTest && pair.review != .excluded
                && pair.promptIDs.allSatisfy { data.roles[$0] != .finalTest }
                && pair.error.map(errors.contains) == true
                && pair.directory.map {
                    FileManager.default.isReadableFile(atPath: URL(fileURLWithPath: $0).appending(path: "audio.caf").path)
                } == true
        }.map(\.id)
    }

    func realignSavedTakes(whisper: LocalWhisper) {
        guard !isBusy, storageReady, forcedAligner.isInstalled else { return }
        let ids = realignableIDs
        guard !ids.isEmpty else { return }
        realignmentReport = nil
        error = nil
        phase = .processing
        task = Task {
            var attempted = 0
            var recovered = 0
            for id in ids {
                guard !Task.isCancelled, storageReady else { break }
                await process(id, whisper: whisper, retrying: true)
                guard !Task.isCancelled, storageReady else { break }
                attempted += 1
                if data.pairs.first(where: { $0.id == id })?.usesAlignedWords == true { recovered += 1 }
            }
            realignmentReport = (recovered, attempted)
            phase = .finished; processingText = ""; task = nil
        }
    }

    private func process(_ id: UUID, whisper: LocalWhisper, retrying: Bool = false) async {
        guard let i = data.pairs.firstIndex(where: { $0.id == id }), data.pairs[i].spoken != nil,
              data.pairs[i].role != .finalTest,
              !data.pairs[i].promptIDs.contains(where: { data.roles[$0] == .finalTest }) else { return }
        var pair = data.pairs[i]
        // Reprocessing annotations does not undo the user's inclusion/exclusion override.
        pair.alignment = nil
        pair.spokenPhones = nil; pair.mouthedPhones = nil; pair.phoneAlignmentNote = nil
        let savedTranscript = pair.transcript
        let savedMatchRate = pair.transcriptMatchRate
        pair.transcriptMatchRate = nil; pair.timedWordCoverage = nil
        pair.wordTimingSource = nil; pair.wordTimingModelRevision = nil
        data.validationReport = nil
        do {
            guard let spoken = pair.spoken, let directory = pair.directory,
                  let start = pair.sessionStartUptime, let offset = pair.audioOffset else { throw Failure.clock }
            let origin = start + offset
            guard origin.isFinite, spoken.cueStart.isFinite, spoken.cueEnd.isFinite,
                  origin <= spoken.cueStart, spoken.cueStart < spoken.cueEnd else { throw Failure.clock }
            let cue = spoken.cueStart...spoken.cueEnd
            let range = (cue.lowerBound - origin)...(cue.upperBound - origin)
            let folder = URL(fileURLWithPath: directory)
            let audio = folder.appending(path: "audio.caf")
            let language = pair.language ?? "en"
            let freeText = pair.practiceMode == .freeText
            let useForced = forcedAligner.isInstalled
            processingText = "Checking the spoken sentence locally…"
            // Keep Free Practice independent of the typed text. With forced alignment,
            // Whisper is advisory for a known prompt, including when Whisper is unavailable.
            let transcript: LocalWhisperTranscript?
            if retrying, let savedTranscript, savedMatchRate != nil {
                transcript = savedTranscript
            } else {
                do {
                    transcript = try await whisper.transcribe(audioURL: audio, outputDirectory: folder, timeRange: range,
                        language: String(language.prefix(2)), initialPrompt: useForced && !freeText ? pair.text : nil)
                } catch {
                    if Task.isCancelled || !useForced || freeText { throw error }
                    transcript = nil
                }
            }
            pair.transcript = transcript
            let expected = Lexicon.words(in: pair.text).map(Lexicon.matchKey)
            if let transcript {
                let heard = Lexicon.words(in: transcript.text).map(Lexicon.matchKey)
                let textMatches = Alignment.commonSubsequence(expected, heard)
                pair.transcriptMatchRate = Double(textMatches.count) / Double(max(1, expected.count))
            }
            guard let silent = pair.silent else { throw Failure.missingSilent }
            if freeText, (pair.transcriptMatchRate ?? 0) < Self.freeTextAgreement { throw Failure.practiceOnly }
            let forced: ForcedAligner.Result?
            let anchors: [MotionWordAnchor]
            if useForced {
                pair.wordTimingSource = .forcedAlignment
                processingText = String(localized: "Matching words to the audio…")
                let result = try await forcedAligner.align(audio: audio, range: range, text: pair.text, language: language)
                pair.timedWordCoverage = result.coverage
                guard result.accepted else {
                    throw (pair.transcriptMatchRate ?? 0) >= 0.8 ? Failure.timing : Failure.transcript
                }
                anchors = try result.anchors(origin: cue.lowerBound, cue: cue)
                pair.wordTimingModelRevision = result.revision
                forced = result
            } else {
                guard let transcript, (pair.transcriptMatchRate ?? 0) >= 0.8 else { throw Failure.transcript }
                let heard = try SentenceReference.anchors(transcript, sessionStart: start, audioOffset: offset)
                let matches = Alignment.commonSubsequence(expected, heard.map { Lexicon.matchKey($0.word) })
                pair.timedWordCoverage = Double(matches.count) / Double(max(1, expected.count))
                guard (pair.timedWordCoverage ?? 0) >= 0.8 else { throw Failure.timing }
                anchors = matches.map { heard[$0.1] }
                pair.wordTimingSource = .whisperAttention
                forced = nil
            }
            processingText = "Aligning spoken and silent motion…"
            let a = spoken.cueSamples, b = silent.cueSamples
            let transfer = try await Task.detached(priority: .userInitiated) {
                try SentenceMotionAlignment.align(voiced: a, silent: b, words: anchors,
                                                  omitCollapsedWords: useForced)
            }.value
            if let forced, Double(transfer.words.count) / Double(forced.words.count) < 0.8 { throw Failure.timing }
            pair.alignment = transfer
            if forced != nil || (phoneAligner.isConfigured && language == "en") {
                do {
                    let profile = try pair.pronunciationProfile ?? PronunciationProfile.load()
                    let profileHash = try profile.fingerprint()
                    let alignment: PhoneAlignment?
                    if let forced {
                        alignment = try forced.phoneAlignment(origin: cue.lowerBound, cue: cue, profileHash: profileHash)
                    } else if let transcript {
                        let config = phoneAligner.configuration
                        processingText = "Estimating phoneme boundaries against local audio…"
                        let phoneTask = Task.detached(priority: .userInitiated) {
                            try await LocalPhonemeAligner.align(configuration: config, audio: audio, range: range,
                                sessionAudioOrigin: origin, text: transcript.text, profile: profile,
                                lexicon: .bundled, output: folder)
                        }
                        alignment = try await withTaskCancellationHandler(operation: { try await phoneTask.value }, onCancel: { phoneTask.cancel() })
                    } else {
                        alignment = nil
                    }
                    pair.spokenPhones = alignment
                    if let alignment, let points = pair.cameraReadings {
                        pair.mouthedPhones = try CameraPhoneAlignment.transfer(alignment,
                            spoken: CameraPhoneAlignment.frames(points, range: cue),
                            mouthed: CameraPhoneAlignment.frames(points, range: silent.cueStart...silent.cueEnd),
                            profileHash: profileHash, allowPartial: useForced)
                    } else if alignment != nil {
                        pair.phoneAlignmentNote = "Spoken acoustic estimates saved. No mouthed phone times: camera points were not collected."
                    }
                } catch { pair.phoneAlignmentNote = error.localizedDescription }
            } else {
                pair.phoneAlignmentNote = "No local phoneme aligner configured. Whisper word times remain word estimates; no phoneme boundaries were invented."
            }
            try Task.checkCancellation()
            pair.error = nil
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(pair).write(to: URL(fileURLWithPath: directory).appending(path: "sentence-pair-\(id).json"), options: .atomic)
        } catch {
            if Task.isCancelled { return }
            if retrying, !(error is Failure) {
                self.error = error.localizedDescription
                return
            }
            pair.error = error.localizedDescription
        }
        if let current = data.pairs.firstIndex(where: { $0.id == id }) {
            data.pairs[current] = pair
            changedPairUses.insert(id)
        }
        await saveInBackground(notify: true)
    }

    func validate() {
        guard !isBusy, storageReady else { return }
        let snapshot = data
        let settings = settings
        phase = .processing; error = nil; processingText = "Checking aligned words on held-out sentences…"
        task = Task {
            do {
                let report = try await Task.detached(priority: .userInitiated) {
                    try SentenceValidation.run(data: snapshot, settings: settings, lexicon: .bundled, vocabulary: Lexicon.vocabulary)
                }.value
                data.validationReport = report
                save()
            } catch { self.error = error.localizedDescription }
            phase = .finished; processingText = ""
        }
    }

    @discardableResult private func save(notify: Bool = false) -> Bool {
        do { try persist(); if notify { onTrainingChanged?() }; return true }
        catch { storageReady = false; self.error = error.localizedDescription; return false }
    }
    private func refreshPairUses() {
        // Role changes can affect several pairs. Ordinary checkpoints only change one take.
        if needsAllPairUses {
            pairUses = Dictionary(uniqueKeysWithValues: data.pairs.map { ($0.id, assessUse($0)) })
        } else if !changedPairUses.isEmpty {
            for pair in data.pairs where changedPairUses.contains(pair.id) {
                pairUses[pair.id] = assessUse(pair)
            }
        }
        needsAllPairUses = false
        changedPairUses.removeAll()
    }
    private func assessUse(_ pair: SentencePair) -> SentencePairUse {
        var assessed = pair
        if pair.promptIDs.contains(where: { data.roles[$0] == .finalTest }) { assessed.role = .finalTest }
        return assessed.automaticUse
    }
    func use(for pair: SentencePair) -> SentencePairUse { pairUses[pair.id] ?? pair.automaticUse }
    private func persist() throws {
        try storeWrites.sync { try SentenceStore.save(data, to: storeURL) }
        refreshPairUses()
    }

    private func persistInBackground() async throws {
        let snapshot = data
        let url = storeURL
        // Enqueue on the main actor so the synchronous termination save is ordered after this snapshot.
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            storeWrites.async {
                continuation.resume(with: Result { try SentenceStore.save(snapshot, to: url) })
            }
        }
        refreshPairUses()
    }

    @discardableResult private func saveInBackground(notify: Bool = false) async -> Bool {
        do {
            try await persistInBackground()
            if notify { onTrainingChanged?() }
            return true
        } catch {
            storageReady = false
            self.error = error.localizedDescription
            return false
        }
    }

    func tickForTesting() { secondsRemaining += 0.05 }
    private func countdown(_ seconds: Double, allowsAdvance: Bool = false, allowsRedo: Bool = false) async throws {
        let end = ProcessInfo.processInfo.systemUptime + seconds
        while ProcessInfo.processInfo.systemUptime < end {
            try Task.checkCancellation()
            if allowsRedo, redoRequested { throw CaptureControl.redo }
            if allowsAdvance, requestedCueEnd != nil { break }
            secondsRemaining = max(0, end - ProcessInfo.processInfo.systemUptime)
            try await Task.sleep(for: .milliseconds(50))
        }
        secondsRemaining = 0
    }
    enum Failure: LocalizedError {
        case practiceOnly
        case noCorpus, disconnected, split, clock, transcript, timing, ownership,missingSilent
        var errorDescription: String? {
            switch self {
            case .practiceOnly:"Practice only: Whisper agreed with less than 85% of the typed text, so this take isn't used for training."
            case .missingSilent:"Spoken transcript saved. Re-record to complete the mouthed half."
            case .noCorpus: "The paired sentence corpus is missing from the app."
            case .disconnected: "Motion stopped. Completed halves were saved; reconnect and re-record this pair."
            case .split: "This passage crosses sentence data splits. Recording stopped."
            case .ownership: "The sentence recording was stopped or replaced. Completed halves keep their original session metadata."
            case .clock: "Audio clock metadata is missing; no word timing was invented. Re-record this pair."
            case .timing: "Too few words have usable audio-aligned times. Review the audio and omitted words, or retry alignment. No word times were invented."
            case .transcript: "Whisper did not match enough of the spoken text. Review the transcript or re-record the pair; missing words were not assigned timings."
            }
        }
    }
}
