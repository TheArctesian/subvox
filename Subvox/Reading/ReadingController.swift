import AVFoundation
import Observation
import Speech

/// Teleprompter reading of the corpus with automatic labelling.
///
/// While reading, it records a normal session (motion.csv, audio.caf) and the time each line was
/// on screen. Afterwards it writes labels.csv next to them: every line, word, phone and vowel
/// jaw class with start and end times on the motion clock. Voiced and whispered readings are
/// timed from the speech recognizer; silent readings, and words the recognizer missed, are
/// spread over the line's active motion window and marked `estimated`.
@MainActor @Observable
final class ReadingController {
    enum Phase: Equatable {
        case setup
        case reading
        case processing(done: Int, of: Int)
        case finished(Summary)
        case failed(String)
    }

    struct Summary: Equatable {
        var lines: Int
        var words: Int
        var timedFromSpeech: Int
        var phones: Int
        var jawCounts: [String: Int]
        var directory: URL
    }

    var script: ReadingScript = .passage {
        didSet { firstLine = 0 }
    }
    var firstLine = 0
    var lineCount = 40

    private(set) var phase: Phase = .setup
    private(set) var lines: [String] = []
    private(set) var position = 0

    var isReading: Bool { phase == .reading }
    var current: String? { lines.indices.contains(position) ? lines[position] : nil }

    /// The slice of the script the next run will read.
    var plannedLines: ArraySlice<String> {
        let all = script.lines
        guard script == .bank else { return all[...] }
        let start = min(firstLine, max(all.count - 1, 0))
        return all[start..<min(start + lineCount, all.count)]
    }

    @ObservationIgnored private var condition: SessionRecorder.Condition = .silent
    @ObservationIgnored private var spans: [ClosedRange<TimeInterval>?] = []
    @ObservationIgnored private var shownAt: TimeInterval = 0
    @ObservationIgnored private var samples: [MotionSample] = []
    @ObservationIgnored private weak var recorder: SessionRecorder?

    func start(recorder: SessionRecorder, audio: AudioService) async {
        guard !isReading, !recorder.isRecording else { return }
        lines = Array(plannedLines)
        guard !lines.isEmpty else { return }
        condition = recorder.condition
        if condition != .silent, !audio.isRunning { await audio.start() }

        let range = script == .bank ? "lines \(firstLine + 1)–\(firstLine + lines.count)" : "all"
        recorder.prompt = "Read: \(script.rawValue), \(range)"
        recorder.start(audio: audio)
        guard recorder.isRecording else {
            phase = .failed(recorder.error ?? "Couldn't start recording.")
            return
        }
        self.recorder = recorder
        spans = Array(repeating: nil, count: lines.count)
        samples = []
        position = 0
        shownAt = ProcessInfo.processInfo.systemUptime
        phase = .reading
    }

    func ingest(_ sample: MotionSample) {
        if isReading { samples.append(sample) }
    }

    func next() {
        guard isReading else { return }
        let now = ProcessInfo.processInfo.systemUptime
        spans[position] = shownAt...now
        shownAt = now
        position += 1
        if position >= lines.count { finish() }
    }

    /// Goes back a line to re-read it; the new reading replaces the old timing.
    func back() {
        guard isReading, position > 0 else { return }
        position -= 1
        shownAt = ProcessInfo.processInfo.systemUptime
    }

    /// Ends the run early; lines not yet finished are left out of the labels.
    func finish() {
        guard isReading, let recorder else { return }
        recorder.stop()
        guard let directory = recorder.lastSessionURL else {
            phase = .failed("The session wasn't saved.")
            return
        }
        Task { await process(directory: directory, startUptime: recorder.lastStartUptime,
                             audioOffset: recorder.lastAudioOffset) }
    }

    func reset() {
        phase = .setup
        position = 0
        lines = []
    }

    // MARK: Labelling

    private func process(directory: URL, startUptime: TimeInterval, audioOffset: Double?) async {
        let read = lines.indices.compactMap { i in spans[i].map { (i, $0) } }
        phase = .processing(done: 0, of: read.count)

        var audio: AVAudioFile?
        if condition != .silent, audioOffset != nil, await LineRecognizer.authorize() {
            audio = try? AVAudioFile(forReading: directory.appending(path: "audio.caf"))
        }

        let lexicon = Lexicon.bundled
        var labels: [LabelInterval] = []
        for (done, (index, span)) in read.enumerated() {
            let line = lines[index]
            let relative = (span.lowerBound - startUptime)...(span.upperBound - startUptime)
            labels.append(LabelInterval(tier: .line, start: relative.lowerBound, end: relative.upperBound,
                                        label: line, source: .prompt))

            var heard: [RecognizedWord] = []
            if let audio, let audioOffset {
                heard = await LineRecognizer.words(in: audio, from: relative.lowerBound - audioOffset,
                                                   to: relative.upperBound - audioOffset, hint: line)
                    .map { RecognizedWord(text: $0.text, start: $0.start + relative.lowerBound,
                                          end: $0.end + relative.lowerBound) }
            }
            let timed = Alignment.timeWords(expected: Lexicon.words(in: line), recognized: heard,
                                            window: activeWindow(span, startUptime: startUptime) ?? relative,
                                            lexicon: lexicon)
            labels += Alignment.label(timed, lexicon: lexicon)
            phase = .processing(done: done + 1, of: read.count)
        }
        samples = []

        let csv = ([LabelInterval.csvHeader] + labels.map(\.csvRow)).joined(separator: "\n") + "\n"
        do {
            try csv.write(to: directory.appending(path: "labels.csv"), atomically: true, encoding: .utf8)
        } catch {
            phase = .failed("Couldn't write labels: \(error.localizedDescription)")
            return
        }
        let words = labels.filter { $0.tier == .word }
        phase = .finished(Summary(
            lines: read.count,
            words: words.count,
            timedFromSpeech: words.filter { $0.source == .speech }.count,
            phones: labels.filter { $0.tier == .phone }.count,
            jawCounts: Dictionary(grouping: labels.filter { $0.tier == .jaw }, by: \.label).mapValues(\.count),
            directory: directory))
    }

    /// The part of a line's on-screen time where the jaw was actually moving, relative to session start.
    private func activeWindow(_ span: ClosedRange<TimeInterval>, startUptime: TimeInterval) -> ClosedRange<Double>? {
        let inside = samples.filter { span.contains($0.t) }
        let active = MotionFeatures.trim(inside)
        guard let first = active.first, let last = active.last, last.t > first.t else { return nil }
        return (first.t - startUptime)...(last.t - startUptime)
    }
}

/// Offline word timing for one line of recorded audio.
enum LineRecognizer {
    private static let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))

    static func authorize() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0 == .authorized) }
        } && recognizer?.isAvailable == true
    }

    /// Recognized words between `from` and `to` seconds into the file, timed relative to `from`.
    @MainActor
    static func words(in file: AVAudioFile, from: Double, to: Double, hint: String) async -> [RecognizedWord] {
        guard let recognizer else { return [] }
        let rate = file.processingFormat.sampleRate
        let start = AVAudioFramePosition(max(0, from) * rate)
        let count = AVAudioFrameCount(max(0, min(Double(file.length), to * rate) - Double(start)))
        guard count > 0, let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: count) else {
            return []
        }
        do {
            file.framePosition = start
            try file.read(into: buffer, frameCount: count)
        } catch {
            return []
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = false
        request.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
        request.contextualStrings = Lexicon.words(in: hint)
        request.append(buffer)
        request.endAudio()

        let lead = max(0, from) - from  // when the line began before the audio did
        return await withCheckedContinuation { continuation in
            let once = Once()
            recognizer.recognitionTask(with: request) { result, error in
                guard error != nil || result?.isFinal == true else { return }
                let words = (result?.bestTranscription.segments ?? []).flatMap { segment in
                    Lexicon.words(in: segment.substring).map {
                        RecognizedWord(text: $0, start: segment.timestamp + lead,
                                       end: segment.timestamp + segment.duration + lead)
                    }
                }
                once.run { continuation.resume(returning: words) }
            }
        }
    }
}

/// Speech callbacks can arrive more than once; a continuation must resume exactly once.
private final class Once: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func run(_ body: () -> Void) {
        let first = lock.withLock { () -> Bool in
            defer { done = true }
            return !done
        }
        if first { body() }
    }
}
