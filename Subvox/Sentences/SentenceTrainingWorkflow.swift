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
    var isAcquiring: Bool { [.relax, .speak, .mouth, .recovery].contains(phase) }
    var isBusy: Bool { isAcquiring || phase == .preparing || phase == .processing }
    var coverage: [String: Int] { data.coverage(lexicon: .bundled) }
    var fittingWords: [Template] { data.derivedWords(role: .training, lexicon: .bundled) }
    var repairWords: [String] { data.repairWords(lexicon: .bundled) }
    var plan: [[SentencePrompt]] { data.plannedSentences(limit: sentenceCount, passage: presentation == .passage, lexicon: .bundled) }
    var estimatedSeconds: Double { plan.reduce(0) { $0 + 2 * (pace.baseline + cueDuration($1.map(\.text).joined(separator: " ")) + pace.recovery) } }

    init() {
        do {
            data = try SentenceStore.load()
            if data.prompts.isEmpty {
                var generator = SystemRandomNumberGenerator()
                data.add(SentencePrompt.bundled, using: &generator)
                guard !data.prompts.isEmpty else { throw Failure.noCorpus }
                try SentenceStore.save(data)
            }
        } catch { self.error = error.localizedDescription; storageReady = false }
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
            recorder.start(audio: audio)
            guard recorder.isRecording else {
                if ownedAudio { audio.stop() }
                error = recorder.error ?? "Could not start recording."; phase = .setup; return
            }
            var visit = SentenceVisit(notes: notes, audioDevice: audio.activeDeviceName,
                                      pace: timing.rawValue, presentation: presentation.rawValue)
            data.visits.append(visit)
            var pairIDs: [UUID] = []
            do {
                for (groupIndex, group) in batch.enumerated() {
                    guard canRecord() else { throw Failure.disconnected }
                    guard let role = data.roles[group[0].id],
                          group.allSatisfy({ data.roles[$0.id] == role }), role != .finalTest else { throw Failure.split }
                    var pair = SentencePair(visitID: visit.id, promptIDs: group.map(\.id),
                                            text: group.map(\.text).joined(separator: " "), role: role)
                    data.pairs.append(pair)
                    pairIDs.append(pair.id)
                    try persist()
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
                        if condition == .normal { pair.spoken = recording } else { pair.silent = recording }
                        if let p = data.pairs.firstIndex(where: { $0.id == pair.id }) { data.pairs[p] = pair }
                        recorder.logEvent(recording.samples, kind: condition == .normal ? "sentence-spoken" : "sentence-silent",
                                          label: pair.text, confidence: nil)
                        try persist() // Save the spoken half even if the silent half is interrupted.
                        buffer = []; index += 1; phase = .recovery
                        try await countdown(timing.recovery)
                    }
                }
            } catch is CancellationError {
                // Never turn an interrupted cue into a completed reference or silent take.
            } catch { self.error = error.localizedDescription }
            buffer = []
            recorder.stop()
            if ownedAudio { audio.stop() }
            visit.endedAt = Date(); visit.directory = recorder.lastSessionURL?.path; visit.pairIDs = pairIDs
            let rates = data.pairs.filter { $0.visitID == visit.id }.compactMap { $0.spoken?.quality.sampleRate }
            if !rates.isEmpty { visit.motionRate = rates.reduce(0, +) / Double(rates.count) }
            if let i = data.visits.firstIndex(where: { $0.id == visit.id }) { data.visits[i] = visit }
            for id in pairIDs {
                if let i = data.pairs.firstIndex(where: { $0.id == id }) {
                    data.pairs[i].directory = visit.directory
                    data.pairs[i].sessionStartUptime = recorder.lastStartUptime
                    data.pairs[i].audioOffset = recorder.lastAudioOffset
                }
            }
            guard save() else { phase = .finished; return }
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

    func validate(settings: DecoderSettings) {
        guard !isBusy, storageReady else { return }
        let snapshot = data
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
    private func persist() throws { try SentenceStore.save(data) }
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
        case noCorpus, disconnected, split, clock, transcript
        var errorDescription: String? {
            switch self {
            case .noCorpus: "The paired sentence corpus is missing from the app."
            case .disconnected: "Motion stopped. Completed halves were saved; reconnect and re-record this pair."
            case .split: "This passage crosses sentence data splits. Recording stopped."
            case .clock: "Audio clock metadata is missing; no word timing was invented. Re-record this pair."
            case .transcript: "Whisper did not match enough of the spoken text. Review the transcript or re-record the pair; missing words were not assigned timings."
            }
        }
    }
}
