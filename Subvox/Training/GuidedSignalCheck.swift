import AVFoundation
import Observation

/// Timed paired acquisition. Word cue is hidden during the baseline; no per-trial keypress.
@MainActor @Observable
final class GuidedSignalCheck {
    enum Phase: Equatable {
        case setup, preparing, baseline, cue, recovery, processing, finished, failed(String)
    }
    struct Verification: Codable {
        var takeID: UUID
        var expected: String
        var heard: String
        var matched: Bool
        var timingNote = "Each cue transcribed separately without word hints; Whisper times are estimates, not phoneme alignment."
    }
    struct Summary {
        var count: Int
        var voiced: Int
        var verified: Int?
        var directory: URL
        var transcriptionError: String?
    }
    private(set) var phase: Phase = .setup
    private(set) var position = 0
    private(set) var total = 0
    private(set) var secondsRemaining: Double = 0
    private(set) var trial: SignalCheckPlan.Trial?
    private(set) var summary: Summary?
    var rounds = 3
    var pace: SignalCheckPace = .comfortable
    var notes = ""
    var isBusy: Bool { [.preparing, .baseline, .cue, .recovery, .processing].contains(phase) }
    var isAcquiring: Bool { [.baseline, .cue, .recovery].contains(phase) }
    var progress: Double { total == 0 ? 0 : Double(position) / Double(total) }
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var samples: [MotionSample] = []

    func ingest(_ sample: MotionSample) {
        if phase == .baseline || phase == .cue { samples.append(sample) }
    }

    func reset() {
        guard !isBusy else { return }
        phase = .setup; summary = nil; trial = nil; position = 0
    }

    func stop() {
        guard isAcquiring || phase == .preparing else { return }
        task?.cancel()
    }

    func start(trials: [SignalCheckPlan.Trial], visitID: UUID, audio: AudioService,
               recorder: SessionRecorder, whisper: LocalWhisper,
               canRecord: @escaping () -> Bool,
               onTake: @escaping (Template) -> Void,
               onFinished: @escaping (URL?) -> Void) {
        guard !isBusy, !trials.isEmpty else { return }
        let timing = pace
        phase = .preparing; position = 0; total = trials.count; summary = nil
        task = Task {
            let ownedAudio = !audio.isRunning
            await audio.start()
            guard audio.isRunning, !Task.isCancelled else {
                if ownedAudio { audio.stop() }
                phase = .failed(audio.error ?? "Microphone was not started.")
                onFinished(nil)
                return
            }
            recorder.condition = .mixed
            recorder.prompt = "Paired signal check; voiced, silent and mouth-still cues. Visit \(visitID)"
            recorder.start(audio: audio)
            guard recorder.isRecording else {
                if ownedAudio { audio.stop() }
                phase = .failed(recorder.error ?? "Could not start session recording.")
                onFinished(nil)
                return
            }
            var completed: [Template] = []
            var failure: String?
            do {
                for (index, item) in trials.enumerated() {
                    try Task.checkCancellation()
                    guard canRecord() else { throw AcquisitionError.disconnected }
                    position = index; trial = item; samples = []
                    phase = .baseline
                    try await countdown(timing.baseline)
                    guard canRecord() else { throw AcquisitionError.disconnected }
                    let cueStart = ProcessInfo.processInfo.systemUptime
                    phase = .cue
                    try await countdown(timing.cue)
                    let cueEnd = ProcessInfo.processInfo.systemUptime
                    guard canRecord() else { throw AcquisitionError.disconnected }
                    let quality = TakeQuality.inspect(samples, maximumDuration: 15)
                    let take = Template(label: item.word, samples: samples, sessionID: visitID,
                        condition: item.purpose == .voicedReference ? .normal : .silent,
                        measuredSampleRate: quality.sampleRate,
                        review: quality.invalid ? .pending : .accepted, purpose: item.purpose,
                        cueStartUptime: cueStart, cueEndUptime: cueEnd)
                    completed.append(take)
                    onTake(take)
                    samples = []
                    phase = .recovery
                    position = index + 1
                    try await countdown(timing.recovery)
                }
            } catch is CancellationError {
                // Incomplete current trial is deliberately discarded; completed trials remain.
            } catch { failure = error.localizedDescription }
            samples = []
            recorder.stop()
            if ownedAudio { audio.stop() }
            guard let directory = recorder.lastSessionURL else {
                phase = .failed("The session could not be saved.")
                onFinished(nil); return
            }
            onFinished(directory)
            // Clear cancellation before postprocessing a deliberately stopped partial session.
            phase = .processing
            let audioOffset = recorder.lastAudioOffset
            let startUptime = recorder.lastStartUptime
            let voiced = completed.filter { $0.purpose == .voicedReference }
            var verification: [Verification] = []
            var transcriptionError = failure
            if !voiced.isEmpty, failure == nil {
                do {
                    // Detached from the cancelled acquisition task; the local service owns its
                    // process lifecycle and applies its own bounded request timeout.
                    for take in voiced {
                        guard let range = SignalCueTiming.audioRange(for: take, startUptime: startUptime, audioOffset: audioOffset) else {
                            transcriptionError = "Audio clock offset is missing; per-trial verification is unavailable."
                            continue
                        }
                        // Long-session word spans may bridge silent trials. Verify the actual
                        // cue's audio independently rather than trusting overlapping intervals.
                        let transcript = try await Task { @MainActor in
                            try await whisper.transcribe(audioURL: directory.appending(path: "audio.caf"),
                                                         outputDirectory: directory, timeRange: range)
                        }.value
                        verification.append(Verification(takeID: take.id, expected: take.label, heard: transcript.text,
                                                         matched: Lexicon.words(in: transcript.text).contains(take.label)))
                    }
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    try encoder.encode(verification).write(to: directory.appending(path: "voice-verification.json"), options: .atomic)
                } catch { transcriptionError = error.localizedDescription }
            }
            summary = Summary(count: completed.count, voiced: voiced.count,
                              verified: verification.isEmpty ? nil : verification.filter(\.matched).count,
                              directory: directory, transcriptionError: transcriptionError)
            trial = nil; phase = .finished; task = nil
        }
    }

    private func countdown(_ seconds: Double) async throws {
        let end = ProcessInfo.processInfo.systemUptime + seconds
        while ProcessInfo.processInfo.systemUptime < end {
            try Task.checkCancellation()
            secondsRemaining = max(0, end - ProcessInfo.processInfo.systemUptime)
            try await Task.sleep(for: .milliseconds(50))
        }
        secondsRemaining = 0
    }

    enum AcquisitionError: LocalizedError {
        case disconnected
        var errorDescription: String? { "AirPods motion stopped. Completed trials were kept; reconnect and start a new visit." }
    }
}
