import AVFoundation
import Observation
import os

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
    enum Condition: String, Codable {
        case paired = "Paired sentence training"
        case signalCheck = "Repeated sentence signal check"
        case observability = "Mouth observability check"
        var slug: String {
            switch self {
            case .paired: "paired"
            case .signalCheck: "signal-check"
            case .observability: "observability"
            }
        }
    }

    var condition: Condition = .paired
    var prompt = ""
    private(set) var isRecording = false
    private(set) var startedAt: Date?
    private(set) var motionCount = 0
    private(set) var eventCount = 0
    private(set) var cameraPointCount = 0
    private(set) var cameraWriteFailures = 0
    private(set) var lastSession: RecordingSession?
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
    @ObservationIgnored private var recordedCondition: Condition = .paired
    @ObservationIgnored private var recordedPrompt = ""
    @ObservationIgnored private var directory: URL?
    @ObservationIgnored private var startUptime: TimeInterval = 0
    @ObservationIgnored private var motionHandle: FileHandle?
    @ObservationIgnored private var eventsHandle: FileHandle?
    @ObservationIgnored private var audioWriter: AudioFileWriter?
    @ObservationIgnored private var cameraHandle: FileHandle?
    @ObservationIgnored private var recordsAudio = false
    @ObservationIgnored private var sinkID: UUID?
    @ObservationIgnored private weak var sink: AudioSink?
    @ObservationIgnored private var audioDeviceName: String?

    init(root: URL? = nil) { recordingRoot = root ?? Self.root }

    @discardableResult
    func start(audio: AudioService, cameraPoints: Bool = false) -> RecordingSession? {
        start(audioSink: audio.isRunning ? audio.sink : nil, audioDevice: audio.activeDeviceName, cameraPoints: cameraPoints)
    }

    /// Accept a sink separately so capture and recorder lifetime are independent.
    @discardableResult
    func start(audioSink: AudioSink?, audioDevice: String? = nil, cameraPoints: Bool = false) -> RecordingSession? {
        guard !isRecording else { return nil }
        error = nil
        let stamp = Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: false))
            .replacingOccurrences(of: ":", with: "")
        let dir = recordingRoot.appending(path: "\(stamp)-\(condition.slug)-\(UUID().uuidString.prefix(8))", directoryHint: .isDirectory)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            motionHandle = try Self.makeFile(dir.appending(path: "motion.csv"), header: MotionSample.csvHeader)
            eventsHandle = try Self.makeFile(dir.appending(path: "events.csv"), header: "start,end,kind,label,confidence")
            if cameraPoints {
                let path = dir.appending(path: "camera-points.ndjson")
                try Data().write(to: path, options: .withoutOverwriting)
                cameraHandle = try FileHandle(forWritingTo: path)
            }
        } catch {
            self.error = error.localizedDescription
            try? motionHandle?.close(); try? eventsHandle?.close(); try? cameraHandle?.close()
            motionHandle = nil; eventsHandle = nil; cameraHandle = nil
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
        eventCount = 0; cameraPointCount = 0; cameraWriteFailures = 0
        recordsAudio = audioSink != nil

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

    private static let signposter = OSSignposter(subsystem: "com.joeblau.subvox", category: "Recording")

    /// Save the clock while acquisition is still running; completed takes can recover it.
    func checkpoint(sessionID: UUID? = nil) throws -> RecordingSession? {
        guard let session = currentSession, sessionID == nil || sessionID == session.id else { return nil }
        let interval = Self.signposter.beginInterval("Recording checkpoint")
        defer { Self.signposter.endInterval("Recording checkpoint", interval) }
        try motionHandle?.synchronize()
        try eventsHandle?.synchronize()
        try cameraHandle?.synchronize()
        try writeMetadata(session: session, audioInfo: audioWriter?.currentInfo, complete: false)
        return session
    }

    /// Flush saved take data on a worker using separate handles. A stopped or replaced
    /// session can finish flushing, but cannot write a new checkpoint after it loses ownership.
    func checkpointCompletedTake(sessionID: UUID) async throws -> RecordingSession? {
        guard let session = currentSession, session.id == sessionID else { return nil }
        let interval = Self.signposter.beginInterval("Flush completed take")
        defer { Self.signposter.endInterval("Flush completed take", interval) }
        let files = cameraHandle == nil ? ["motion.csv", "events.csv"] : ["motion.csv", "events.csv", "camera-points.ndjson"]
        try await Task.detached(priority: .userInitiated) {
            for name in files {
                let handle = try FileHandle(forWritingTo: session.directory.appending(path: name))
                defer { try? handle.close() }
                try handle.synchronize()
            }
        }.value
        guard let current = currentSession, current.id == sessionID else { return nil }
        try writeMetadata(session: current, audioInfo: audioWriter?.currentInfo, complete: false)
        return current
    }

    @discardableResult
    func stop(sessionID: UUID? = nil) -> RecordingSession? {
        guard let session = currentSession, sessionID == nil || sessionID == session.id else { return nil }
        let interval = Self.signposter.beginInterval("Finish recording session")
        defer { Self.signposter.endInterval("Finish recording session", interval) }
        isRecording = false
        if let sinkID { sink?.remove(sinkID) }
        sinkID = nil
        let audioInfo = audioWriter?.finish()
        audioWriter = nil
        try? motionHandle?.close()
        try? eventsHandle?.close()
        try? cameraHandle?.close()
        motionHandle = nil
        eventsHandle = nil
        cameraHandle = nil

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
            "audioRecordingEnabled": recordsAudio,
            "videoRecordingEnabled": false,
            "cameraPointSamples": cameraPointCount,
            "cameraPointWriteFailures": cameraWriteFailures,
            "cameraPointSchemaVersion": 1,
            "cameraPointCoordinates": "Full-image normalized coordinates, lower-left origin. Shape uses roll-aligned inter-eye-distance units; yaw/pitch are not corrected.",
            "audioDevice": audioDeviceName ?? NSNull(),
            "audioSampleRate": audioInfo?.sampleRate ?? NSNull(),
            "audioStartOffsetSeconds": session.audioOffset ?? NSNull(),
            "note": "Motion t, audioStartOffsetSeconds and camera sessionSeconds are relative to startUptime. Camera reading.t is monotonic uptime, with receipt-time fallback flagged per frame.",
        ]
        let data = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: session.directory.appending(path: "meta.json"), options: .atomic)
    }

    func record(_ sample: MotionSample) {
        guard isRecording else { return }
        motionHandle?.write(Data((sample.csvRow(relativeTo: startUptime) + "\n").utf8))
        motionCount += 1
    }

    /// Numeric-only frame sidecar, streamed so an interrupted take still has its points.
    @discardableResult
    func recordCamera(_ reading: CameraPoseReading, sessionID: UUID) -> Bool {
        guard isRecording, activeSessionID == sessionID, let cameraHandle else { return false }
        struct Row: Encodable {
            let schemaVersion = 1
            let sessionID: UUID
            let sessionSeconds: Double
            let reading: CameraPoseReading
        }
        do {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            var bytes = try encoder.encode(Row(sessionID: sessionID, sessionSeconds: reading.t - startUptime, reading: reading))
            bytes.append(0x0A)
            try cameraHandle.write(contentsOf: bytes)
            cameraPointCount += 1
            return true
        } catch {
            cameraWriteFailures += 1
            self.error = "Camera point recording failed: " + error.localizedDescription
            return false
        }
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

    /// Explicit cue boundaries, even if no sensor sample arrived at the key press.
    func logMarker(uptime: Double, kind: String, label: String) {
        guard isRecording, uptime.isFinite else { return }
        let time = String(format: "%.6f", uptime - startUptime)
        let safeLabel = label.replacingOccurrences(of: ",", with: " ").replacingOccurrences(of: "\n", with: " ")
        eventsHandle?.write(Data("\(time),\(time),\(kind),\(safeLabel),\n".utf8))
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
