import AVFoundation
import Observation

/// Immutable identity and clock for one recorder session. Never infer ownership from "last".
struct RecordingSession: Codable, Equatable, Sendable {
    var id: UUID
    var directory: URL
    var startUptime: TimeInterval
    var audioOffset: Double?
}

/// Records synchronized motion CSV + microphone audio + labelled segment events for offline
/// analysis (normal speech vs. whisper vs. silent mouthing).
///
/// Output: ~/Documents/Subvox/Sessions/<timestamp>-<condition>/
///   motion.csv  — every headphone-motion sample, t relative to session start
///   audio.caf   — microphone audio (if the mic is running)
///   events.csv  — captured segments with their label / prediction
///   meta.json   — condition, prompt, clock offsets
@MainActor @Observable
final class SessionRecorder {
    enum Condition: String, CaseIterable, Identifiable, Codable {
        case mixed = "Mixed diagnostic trials"
        case normal = "Normal speech"
        case whisper = "Whisper"
        case silent = "Silent mouthing"
        var id: String { rawValue }
        var slug: String {
            switch self {
            case .mixed: "mixed"
            case .normal: "normal"
            case .whisper: "whisper"
            case .silent: "silent"
            }
        }
    }

    var condition: Condition = .silent
    var prompt = ""
    private(set) var isRecording = false
    private(set) var startedAt: Date?
    private(set) var motionCount = 0
    private(set) var eventCount = 0
    private(set) var lastSession: RecordingSession?
    var lastSessionURL: URL? { lastSession?.directory }
    /// Clock of the last finished session: its start (seconds since boot) and where its audio
    /// begins relative to that start, for anyone post-processing the files.
    var lastStartUptime: TimeInterval { lastSession?.startUptime ?? 0 }
    var lastAudioOffset: Double? { lastSession?.audioOffset }
    private(set) var activeSessionID: UUID?
    var currentSession: RecordingSession? {
        guard isRecording, let id = activeSessionID, let directory else { return nil }
        return RecordingSession(id: id, directory: directory, startUptime: startUptime,
                                audioOffset: audioWriter?.currentInfo?.firstHostSeconds.map { $0 - startUptime })
    }
    private(set) var error: String?

    static var root: URL {
        URL.documentsDirectory.appending(path: "Subvox/Sessions", directoryHint: .isDirectory)
    }

    @ObservationIgnored private let recordingRoot: URL
    @ObservationIgnored private var recordedCondition: Condition = .silent
    @ObservationIgnored private var recordedPrompt = ""
    @ObservationIgnored private var directory: URL?
    @ObservationIgnored private var startUptime: TimeInterval = 0
    @ObservationIgnored private var motionHandle: FileHandle?
    @ObservationIgnored private var eventsHandle: FileHandle?
    @ObservationIgnored private var audioWriter: AudioFileWriter?
    @ObservationIgnored private var sinkID: UUID?
    @ObservationIgnored private weak var sink: AudioSink?
    @ObservationIgnored private var audioDeviceName: String?

    init(root: URL? = nil) { recordingRoot = root ?? Self.root }

    @discardableResult
    func start(audio: AudioService) -> RecordingSession? {
        start(audioSink: audio.isRunning ? audio.sink : nil, audioDevice: audio.activeDeviceName)
    }

    /// Accept a sink separately so capture and recorder lifetime are independent.
    @discardableResult
    func start(audioSink: AudioSink?, audioDevice: String? = nil) -> RecordingSession? {
        guard !isRecording else { return nil }
        error = nil
        let stamp = Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: false))
            .replacingOccurrences(of: ":", with: "")
        let dir = recordingRoot.appending(path: "\(stamp)-\(condition.slug)-\(UUID().uuidString.prefix(8))", directoryHint: .isDirectory)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            motionHandle = try Self.makeFile(dir.appending(path: "motion.csv"), header: MotionSample.csvHeader)
            eventsHandle = try Self.makeFile(dir.appending(path: "events.csv"), header: "start,end,kind,label,confidence")
        } catch {
            self.error = error.localizedDescription
            try? motionHandle?.close(); try? eventsHandle?.close()
            motionHandle = nil; eventsHandle = nil
            return nil
        }

        directory = dir
        startUptime = ProcessInfo.processInfo.systemUptime
        startedAt = Date()
        activeSessionID = UUID()
        recordedCondition = condition
        recordedPrompt = prompt
        audioDeviceName = audioDevice
        motionCount = 0
        eventCount = 0

        if let audioSink {
            let writer = AudioFileWriter(url: dir.appending(path: "audio.caf"))
            audioWriter = writer
            sink = audioSink
            sinkID = audioSink.add { buffer, time in writer.write(buffer, time) }
        }
        isRecording = true
        do { return try checkpoint() }
        catch {
            self.error = error.localizedDescription
            stop()
            return nil
        }
    }

    /// Save the clock while acquisition is still running; completed takes can recover it.
    func checkpoint(sessionID: UUID? = nil) throws -> RecordingSession? {
        guard let session = currentSession, sessionID == nil || sessionID == session.id else { return nil }
        try motionHandle?.synchronize()
        try eventsHandle?.synchronize()
        try writeMetadata(session: session, audioInfo: audioWriter?.currentInfo, complete: false)
        return session
    }

    @discardableResult
    func stop(sessionID: UUID? = nil) -> RecordingSession? {
        guard let session = currentSession, sessionID == nil || sessionID == session.id else { return nil }
        isRecording = false
        if let sinkID { sink?.remove(sinkID) }
        sinkID = nil
        let audioInfo = audioWriter?.finish()
        audioWriter = nil
        try? motionHandle?.close()
        try? eventsHandle?.close()
        motionHandle = nil
        eventsHandle = nil

        let finished = RecordingSession(id: session.id, directory: session.directory, startUptime: session.startUptime,
                                       audioOffset: audioInfo?.firstHostSeconds.map { $0 - session.startUptime })
        do { try writeMetadata(session: finished, audioInfo: audioInfo, complete: true) }
        catch { self.error = error.localizedDescription }
        lastSession = finished
        activeSessionID = nil
        self.directory = nil
        return finished
    }

    private func writeMetadata(session: RecordingSession, audioInfo: AudioFileWriter.Info?, complete: Bool) throws {
        let meta: [String: Any] = [
            "sessionID": session.id.uuidString,
            "complete": complete,
            "condition": recordedCondition.rawValue,
            "prompt": recordedPrompt,
            "startedAt": startedAt?.ISO8601Format() ?? "",
            "startUptime": session.startUptime,
            "durationSeconds": ProcessInfo.processInfo.systemUptime - session.startUptime,
            "motionSamples": motionCount,
            "events": eventCount,
            "audioDevice": audioDeviceName ?? NSNull(),
            "audioSampleRate": audioInfo?.sampleRate ?? NSNull(),
            "audioStartOffsetSeconds": session.audioOffset ?? NSNull(),
            "note": "Motion t and audioStartOffsetSeconds are relative to startUptime (seconds since boot).",
        ]
        let data = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: session.directory.appending(path: "meta.json"), options: .atomic)
    }

    func record(_ sample: MotionSample) {
        guard isRecording else { return }
        motionHandle?.write(Data((sample.csvRow(relativeTo: startUptime) + "\n").utf8))
        motionCount += 1
    }

    func logEvent(_ segment: [MotionSample], kind: String, label: String, confidence: Double?) {
        guard isRecording, let first = segment.first, let last = segment.last else { return }
        let row = [
            String(format: "%.4f", first.t - startUptime),
            String(format: "%.4f", last.t - startUptime),
            kind,
            label.replacingOccurrences(of: ",", with: " "),
            confidence.map { String(format: "%.3f", $0) } ?? "",
        ].joined(separator: ",")
        eventsHandle?.write(Data((row + "\n").utf8))
        eventCount += 1
    }

    private static func makeFile(_ url: URL, header: String) throws -> FileHandle {
        try Data((header + "\n").utf8).write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        return handle
    }
}

/// Writes tap buffers to disk from the audio thread.
private final class AudioFileWriter: @unchecked Sendable {
    struct Info {
        var sampleRate: Double
        var firstHostSeconds: Double?
    }

    private let url: URL
    private let lock = NSLock()
    private var file: AVAudioFile?
    private var info: Info?
    private var failed = false
    private var closed = false

    init(url: URL) {
        self.url = url
    }

    func write(_ buffer: AVAudioPCMBuffer, _ time: AVAudioTime) {
        lock.withLock {
            guard !failed, !closed else { return }
            if file == nil {
                do {
                    let format = buffer.format
                    file = try AVAudioFile(
                        forWriting: url, settings: format.settings,
                        commonFormat: format.commonFormat, interleaved: format.isInterleaved)
                    info = Info(
                        sampleRate: format.sampleRate,
                        firstHostSeconds: time.isHostTimeValid ? AVAudioTime.seconds(forHostTime: time.hostTime) : nil)
                } catch {
                    failed = true
                    return
                }
            }
            try? file?.write(from: buffer)
        }
    }

    var currentInfo: Info? { lock.withLock { info } }

    func finish() -> Info? {
        lock.withLock {
            closed = true
            file = nil
            return info
        }
    }
}
