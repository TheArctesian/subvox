import Foundation
import Observation

/// One acquisition/review/coverage loop. Sentence data never enters the old word split.
@MainActor @Observable
final class SentenceTrainingWorkflow {
    enum Phase: Equatable { case setup, preparing, relax, speak, mouth, recovery, processing, finished }
    enum Presentation: String, CaseIterable, Identifiable {
        case sentences = "Sentences", passage = "Short passages"
        var id: String { rawValue }
    }
    private(set) var data = SentenceTrainingData()
    private(set) var phase: Phase = .setup
    private(set) var error: String?
    private(set) var storageReady = true
    private(set) var currentText = ""
    private(set) var currentCondition: SpeechCondition = .normal
    private(set) var index = 0
    private(set) var total = 0
    private(set) var secondsRemaining = 0.0
    private(set) var processingText = ""
    var presentation: Presentation = .passage
    var pace: SignalCheckPace = .comfortable
    var sentenceCount = 6
    var notes = ""
    var onTrainingChanged: (() -> Void)?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var buffer: [MotionSample] = []
    @ObservationIgnored private let storeURL: URL
    @ObservationIgnored private var activeVisitID: UUID?
    @ObservationIgnored private var activeRecordingID: UUID?
    @ObservationIgnored private var isTerminating = false
    var settings: DecoderSettings { data.decoderSettings ?? DecoderSettings() }
    var settingsLocked: Bool { isBusy || !storageReady || data.reservedUnlockedAt != nil }
    var learningMethod: LearningMethod {
        get { settings.method }
        set { changeSettings { $0.method = newValue } }
    }
    var isAcquiring: Bool { [.relax, .speak, .mouth, .recovery].contains(phase) }
    var isBusy: Bool { isAcquiring || phase == .preparing || phase == .processing }
    var coverage: [String: Int] { data.coverage(lexicon: .bundled) }
    var fittingWords: [Template] { data.derivedWords(role: .training, lexicon: .bundled) }
    var repairWords: [String] { data.repairWords(lexicon: .bundled) }
    var plan: [[SentencePrompt]] { data.plannedSentences(limit: sentenceCount, passage: presentation == .passage, lexicon: .bundled) }
    var estimatedSeconds: Double { plan.reduce(0) { $0 + 2 * (pace.baseline + cueDuration($1.map(\.text).joined(separator: " ")) + pace.recovery) } }

    init(storeURL: URL? = nil, corpus: [SentencePrompt]? = nil) {
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
    }

    /// One-time migration from the formerly shared word settings; later edits are independent.
    func initializeSettings(from settings: DecoderSettings) {
        guard storageReady, data.reservedUnlockedAt == nil, data.decoderSettings == nil else { return }
        var next = settings
        next.condition = .silent
        data.decoderSettings = next
        save()
    }

    func changeSettings(_ change: (inout DecoderSettings) -> Void) {
        guard !settingsLocked else { return }
        var next = settings
        change(&next)
        next.condition = .silent
        guard next != settings else { return }
        data.decoderSettings = next
        data.validationReport = nil
        save(notify: true)
    }

    @discardableResult
    func beginVisit(session: RecordingSession, audioDevice: String?, timing: SignalCheckPace,
                    presentation: Presentation) throws -> SentenceVisit {
        let visit = SentenceVisit(notes: notes, directory: session.directory.path, audioDevice: audioDevice,
                                  pace: timing.rawValue, presentation: presentation.rawValue, recordingSessionID: session.id)
        activeVisitID = visit.id; activeRecordingID = session.id
        data.visits.append(visit)
        try persist()
        return visit
    }

    func beginPair(prompts: [SentencePrompt], visit: SentenceVisit, session: RecordingSession) throws -> SentencePair {
        guard session.id == activeRecordingID, visit.id == activeVisitID,
              let first = prompts.first, let role = data.roles[first.id], role != .finalTest,
              prompts.allSatisfy({ data.roles[$0.id] == role }) else { throw Failure.split }
        let pair = SentencePair(visitID: visit.id, promptIDs: prompts.map(\.id),
            text: prompts.map(\.text).joined(separator: " "), role: role, directory: session.directory.path,
            sessionStartUptime: session.startUptime, audioOffset: session.audioOffset, recordingSessionID: session.id)
        data.pairs.append(pair)
        if let v = data.visits.firstIndex(where: { $0.id == visit.id }) { data.visits[v].pairIDs.append(pair.id) }
        try persist()
        return pair
    }

    func saveCompletedHalf(_ recording: SentenceRecording, pairID: UUID, session: RecordingSession) throws {
        guard let i = data.pairs.firstIndex(where: { $0.id == pairID }),
              data.pairs[i].recordingSessionID == session.id, session.id == activeRecordingID else { throw Failure.ownership }
        if recording.condition == .normal { data.pairs[i].spoken = recording }
        else if recording.condition == .silent { data.pairs[i].silent = recording }
        else { throw Failure.split }
        data.pairs[i].directory = session.directory.path
        data.pairs[i].sessionStartUptime = session.startUptime
        data.pairs[i].audioOffset = session.audioOffset
        try persist()
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
            }
        }
        activeVisitID = nil; activeRecordingID = nil
    }

    /// Synchronous: willTerminate cannot wait for the acquisition task to unwind.
    func prepareForTermination(recorder: SessionRecorder) {
        isTerminating = true
        task?.cancel()
        if let id = activeRecordingID {
            finishVisit(session: recorder.stop(sessionID: id))
            save()
        }
        phase = .finished
    }

    func ingest(_ sample: MotionSample) {
        if phase == .relax || phase == .speak || phase == .mouth { buffer.append(sample) }
    }
    func reset() { guard !isBusy else { return }; phase = .setup }
    func stop() {
        guard isAcquiring || phase == .preparing else { return }
        task?.cancel()
    }
    func cueDuration(_ text: String) -> Double {
        let words = Double(Lexicon.words(in: text).count)
        return min(45, max(pace == .comfortable ? 12 : 16, words * (pace == .comfortable ? 0.85 : 1.1) + 5))
    }

    func review(_ id: UUID, decision: ReviewDecision) {
        guard !isBusy, data.reservedUnlockedAt == nil,
              let i = data.pairs.firstIndex(where: { $0.id == id }), data.pairs[i].role != .finalTest else { return }
        guard decision != .accepted || data.pairs[i].canAccept else { return }
        data.pairs[i].review = decision
        data.validationReport = nil
        save(notify: true)
    }

    func start(audio: AudioService, recorder: SessionRecorder, whisper: LocalWhisper,
               canRecord: @escaping () -> Bool, repairs: Bool = false, retry: UUID? = nil) {
        guard !isBusy, storageReady, data.reservedUnlockedAt == nil, !recorder.isRecording,
              whisper.isConfigured, canRecord() else { return }
        var planned = plan
        if let retry, let pair = data.pairs.first(where: { $0.id == retry }) {
            planned = [data.prompts.filter { pair.promptIDs.contains($0.id) }]
        } else if repairs {
            let prompts = repairWords.map { SentencePrompt(text: $0, kind: .wordRepair) }
            var generator = SystemRandomNumberGenerator()
            data.add(prompts, using: &generator)
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
            let added = 2 * (pace.baseline + cueDuration(group.map(\.text).joined(separator: " ")) + pace.recovery)
            guard duration + added < 840 else { break }
            batch.append(group); duration += added
        }
        guard !batch.isEmpty else { error = "This batch is too long. Choose fewer sentences."; return }
        let timing = pace
        let cueTimes = batch.map { cueDuration($0.map(\.text).joined(separator: " ")) }
        phase = .preparing; error = nil; index = 0; total = batch.count * 2
        task = Task {
            let ownedAudio = !audio.isRunning
            await audio.start()
            guard audio.isRunning, !Task.isCancelled else {
                if ownedAudio { audio.stop() }
                error = audio.error ?? "Recording cancelled before the microphone started."; phase = .setup; return
            }
            recorder.condition = .mixed
            recorder.prompt = "Paired sentence training; spoken then silently mouthed; independent sentence split"
            guard let recordingSession = recorder.start(audio: audio) else {
                if ownedAudio { audio.stop() }
                error = recorder.error ?? "Could not start recording."; phase = .setup; return
            }
            var pairIDs: [UUID] = []
            do {
                let visit = try beginVisit(session: recordingSession, audioDevice: audio.activeDeviceName,
                                           timing: timing, presentation: presentation)
                for (groupIndex, group) in batch.enumerated() {
                    guard canRecord() else { throw Failure.disconnected }
                    let pair = try beginPair(prompts: group, visit: visit, session: recordingSession)
                    pairIDs.append(pair.id)
                    for condition in [SpeechCondition.normal, .silent] {
                        try Task.checkCancellation()
                        currentText = pair.text; currentCondition = condition; buffer = []
                        phase = .relax
                        try await countdown(timing.baseline)
                        guard canRecord() else { throw Failure.disconnected }
                        let start = ProcessInfo.processInfo.systemUptime
                        phase = condition == .normal ? .speak : .mouth
                        try await countdown(cueTimes[groupIndex])
                        let end = ProcessInfo.processInfo.systemUptime
                        guard canRecord() else { throw Failure.disconnected }
                        let recording = SentenceRecording(condition: condition, samples: buffer, cueStart: start, cueEnd: end)
                        guard let checkpoint = try recorder.checkpoint(sessionID: recordingSession.id) else { throw Failure.ownership }
                        try saveCompletedHalf(recording, pairID: pair.id, session: checkpoint)
                        recorder.logEvent(recording.samples, kind: condition == .normal ? "sentence-spoken" : "sentence-silent",
                                          label: pair.text, confidence: nil)
                        buffer = []; index += 1; phase = .recovery
                        try await countdown(timing.recovery)
                    }
                }
            } catch is CancellationError {
                // Never turn an interrupted cue into a completed reference or silent take.
            } catch { self.error = error.localizedDescription }
            buffer = []
            let finished = recorder.stop(sessionID: recordingSession.id)
            if ownedAudio, !recorder.isRecording { audio.stop() }
            finishVisit(session: finished)
            guard save(), !isTerminating else { phase = .finished; return }
            phase = .processing
            // A fresh unstructured task processes saved complete pairs even after Stop.
            await Task { @MainActor in
                for id in pairIDs {
                    await self.process(id, whisper: whisper)
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

    private func process(_ id: UUID, whisper: LocalWhisper) async {
        guard let i = data.pairs.firstIndex(where: { $0.id == id }), data.pairs[i].complete,
              data.pairs[i].role != .finalTest else { return }
        var pair = data.pairs[i]
        pair.alignment = nil; pair.review = .pending
        data.validationReport = nil
        do {
            guard let spoken = pair.spoken, let silent = pair.silent, let directory = pair.directory,
                  let start = pair.sessionStartUptime, let offset = pair.audioOffset else { throw Failure.clock }
            let range = (spoken.cueStart - start - offset)...(spoken.cueEnd - start - offset)
            processingText = "Checking the spoken sentence locally…"
            let transcript = try await whisper.transcribe(audioURL: URL(fileURLWithPath: directory).appending(path: "audio.caf"),
                outputDirectory: URL(fileURLWithPath: directory), timeRange: range)
            pair.transcript = transcript
            let heard = try SentenceReference.anchors(transcript, sessionStart: start, audioOffset: offset)
            let expected = Lexicon.words(in: pair.text)
            let matches = Alignment.commonSubsequence(expected, heard.map(\.word))
            pair.transcriptMatchRate = Double(matches.count) / Double(max(1, expected.count))
            guard matches.count >= max(1, Int(ceil(Double(expected.count) * 0.8))) else { throw Failure.transcript }
            let anchors = matches.map { heard[$0.1] }
            processingText = "Aligning spoken and silent motion…"
            let a = spoken.cueSamples, b = silent.cueSamples
            pair.alignment = try await Task.detached(priority: .userInitiated) {
                try SentenceMotionAlignment.align(voiced: a, silent: b, words: anchors)
            }.value
            pair.review = .pending; pair.error = nil
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(pair).write(to: URL(fileURLWithPath: directory).appending(path: "sentence-pair-\(id).json"), options: .atomic)
        } catch { pair.error = error.localizedDescription; pair.review = .pending }
        if let current = data.pairs.firstIndex(where: { $0.id == id }) { data.pairs[current] = pair }
        save(notify: true)
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
    private func persist() throws { try SentenceStore.save(data, to: storeURL) }
    private func countdown(_ seconds: Double) async throws {
        let end = ProcessInfo.processInfo.systemUptime + seconds
        while ProcessInfo.processInfo.systemUptime < end {
            try Task.checkCancellation()
            secondsRemaining = max(0, end - ProcessInfo.processInfo.systemUptime)
            try await Task.sleep(for: .milliseconds(50))
        }
        secondsRemaining = 0
    }
    enum Failure: LocalizedError {
        case noCorpus, disconnected, split, clock, transcript, ownership
        var errorDescription: String? {
            switch self {
            case .noCorpus: "The paired sentence corpus is missing from the app."
            case .disconnected: "Motion stopped. Completed halves were saved; reconnect and re-record this pair."
            case .split: "This passage crosses sentence data splits. Recording stopped."
            case .ownership: "The sentence recording was stopped or replaced. Completed halves keep their original session metadata."
            case .clock: "Audio clock metadata is missing; no word timing was invented. Re-record this pair."
            case .transcript: "Whisper did not match enough of the spoken text. Review the transcript or re-record the pair; missing words were not assigned timings."
            }
        }
    }
}
