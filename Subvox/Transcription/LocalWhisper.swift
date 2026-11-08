import AVFoundation
import Darwin
import Foundation
import Observation

struct LocalWhisperWord: Codable, Hashable, Sendable {
    var text: String
    /// Audio-relative seconds, estimated by Whisper or spread within its segment.
    var start: Double
    var end: Double
    var alignmentPeak: Double?
}

struct LocalWhisperTranscript: Codable, Hashable, Sendable {
    var text: String
    var words: [LocalWhisperWord]
    var sourceAudioStartSeconds: Double?
    var timingMethod: String?
    var timingReview: AudioTimingReview?
}

/// Offline Whisper adapter. Each transcription owns a temporary loopback-only server;
/// it never attaches to an existing OpenWhispr process or supplies the expected prompt.
@MainActor @Observable
final class LocalWhisper {
    var modelPath: String { didSet { defaults.set(modelPath, forKey: "LocalWhisper.modelPath") } }
    var executablePath: String { didSet { defaults.set(executablePath, forKey: "LocalWhisper.executablePath") } }
    private(set) var isTranscribing = false
    private(set) var error: String?
    private(set) var status = "Ready for local transcription"
    var isAvailable: Bool {
        Self.validModel(URL(fileURLWithPath: Self.expanded(modelPath)))
            && FileManager.default.isExecutableFile(atPath: Self.expanded(executablePath))
    }
    var isConfigured: Bool { isAvailable }
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var process: Process?

    static let bundledServer = "/Applications/OpenWhispr.app/Contents/Resources/bin/whisper-server-darwin-arm64"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        modelPath = defaults.string(forKey: "LocalWhisper.modelPath") ?? Self.discoverModels().first?.path ?? ""
        executablePath = defaults.string(forKey: "LocalWhisper.executablePath")
            ?? (FileManager.default.isExecutableFile(atPath: Self.bundledServer) ? Self.bundledServer : "")
    }

    func stopOwnedProcess() {
        if let process, process.isRunning { process.terminate() }
    }

    enum Failure: LocalizedError {
        case unavailable, alignmentUnavailable, busy, invalidAudio, conversion(String), server(String), timeout, malformedResponse, nonLoopback
        var errorDescription: String? {
            switch self {
            case .unavailable: "Choose an existing GGML Whisper model and local whisper-server executable. No models are downloaded automatically."
            case .alignmentUnavailable: "This model has no supported audio-attention alignment preset. Choose a standard GGML tiny, base, small, medium or versioned large Whisper model."
            case .busy: "Local Whisper is already transcribing."
            case .invalidAudio: "Audio is empty, invalid, or longer than the 15-minute local transcription limit."
            case .conversion(let reason): "Could not convert audio locally: \(reason)"
            case .server(let reason): "Local Whisper server failed: \(reason)"
            case .timeout: "Local Whisper timed out. Its server has been stopped."
            case .malformedResponse: "Local Whisper returned an unsupported transcript response."
            case .nonLoopback: "Local Whisper only permits HTTP requests to 127.0.0.1."
            }
        }
    }

    nonisolated static func discoverModels() -> [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let roots = [home.appending(path: ".cache/openwhispr/whisper-models"),
                     home.appending(path: "Library/Application Support/Recordly/whisper"),
                     home.appending(path: "Library/Application Support/OpenWhispr/models")]
        var result: [URL] = []
        for root in roots {
            let contents = (try? FileManager.default.contentsOfDirectory(at: root,
                includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])) ?? []
            result += contents.filter(validModel).sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        // Prefer the user's known OpenWhispr base model, then other compatible files.
        let preferred = roots[0].appending(path: "ggml-base.bin")
        if let index = result.firstIndex(of: preferred) { result.insert(result.remove(at: index), at: 0) }
        return result
    }

    nonisolated private static func expanded(_ path: String) -> String { (path as NSString).expandingTildeInPath }
    nonisolated private static func validModel(_ url: URL) -> Bool {
        guard url.lastPathComponent.hasPrefix("ggml-"), url.pathExtension.lowercased() == "bin",
              FileManager.default.isReadableFile(atPath: url.path),
              (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return false }
        // ONNX/Parakeet and CoreML directories are deliberately excluded.
        return true
    }

    func transcribe(audioURL: URL, outputDirectory: URL, timeRange: ClosedRange<Double>? = nil, language: String = "en") async throws -> LocalWhisperTranscript {
        guard !isTranscribing else { throw Failure.busy }
        guard isAvailable else { error = Failure.unavailable.localizedDescription; throw Failure.unavailable }
        isTranscribing = true
        error = nil
        let model = Self.expanded(modelPath)
        guard let preset = Self.alignmentPreset(modelPath: model) else {
            isTranscribing = false
            error = Failure.alignmentUnavailable.localizedDescription
            status = "Audio alignment unavailable"
            throw Failure.alignmentUnavailable
        }
        let executable = Self.expanded(executablePath)
        let id = UUID().uuidString
        let wav = outputDirectory.appending(path: "local-whisper-\(id).wav")
        let logURL = outputDirectory.appending(path: "local-whisper-\(id)-server.log")
        let child = Process()
        let delegate = LoopbackOnlyDelegate()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        configuration.timeoutIntervalForRequest = 180
        configuration.timeoutIntervalForResource = 185
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        var logHandle: FileHandle?
        defer {
            session.invalidateAndCancel()
            if child.isRunning { child.terminate() }
            if process === child { process = nil }
            try? logHandle?.close()
            try? FileManager.default.removeItem(at: wav)
            isTranscribing = false
        }
        do {
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
            status = "Converting audio locally to 16 kHz mono…"
            let conversion = Task.detached(priority: .userInitiated) { try Self.convertToWAV(audioURL, destination: wav, timeRange: timeRange) }
            try await withTaskCancellationHandler(operation: { try await conversion.value }, onCancel: { conversion.cancel() })
            try Task.checkCancellation()
            let port = try Self.unusedPort()
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
            logHandle = try FileHandle(forWritingTo: logURL)
            child.executableURL = URL(fileURLWithPath: executable)
            child.arguments = ["--model", model, "--host", "127.0.0.1", "--port", String(port),
                               "--inference-path", "/inference", "--language", language,
                               "--dtw", preset, "--no-flash-attn"]
            child.currentDirectoryURL = outputDirectory
            child.standardOutput = logHandle
            child.standardError = logHandle
            try child.run()
            process = child
            status = "Loading local Whisper model…"
            return try await withTaskCancellationHandler {
                try await waitForServer(child, port: port, session: session)
                try Task.checkCancellation()
                status = "Transcribing locally (no expected-word hints)…"
                let request = try Self.inferenceRequest(port: port, wav: Data(contentsOf: wav))
                let (data, response) = try await session.data(for: request)
                try Task.checkCancellation()
                guard child.isRunning else { throw Failure.server("Process exited with status \(child.terminationStatus). See \(logURL.lastPathComponent).") }
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw Failure.server("Inference HTTP status \((response as? HTTPURLResponse)?.statusCode ?? 0). See \(logURL.lastPathComponent).")
                }
                try data.write(to: outputDirectory.appending(path: "local-whisper-\(id)-raw.json"), options: .atomic)
                let parsed = try AudioWordTiming.attentionTranscript(data)
                let timing = Task.detached(priority: .userInitiated) { try AudioWordTiming.inspect(wav: wav, transcript: parsed) }
                var transcript = try await withTaskCancellationHandler(operation: { try await timing.value }, onCancel: { timing.cancel() })
                try Task.checkCancellation()
                if let timeRange {
                    transcript.sourceAudioStartSeconds = timeRange.lowerBound
                    transcript.words = transcript.words.map {
                        LocalWhisperWord(text: $0.text, start: $0.start + timeRange.lowerBound,
                                         end: $0.end + timeRange.lowerBound,
                                         alignmentPeak: $0.alignmentPeak.map { $0 + timeRange.lowerBound })
                    }
                }
                try JSONEncoder().encode(transcript).write(to: outputDirectory.appending(path: "local-whisper-\(id).json"), options: .atomic)
                status = transcript.text.isEmpty ? "Local transcription complete: no speech recognized" : "Local transcription complete"
                return transcript
            } onCancel: {
                // Only the child created by this call is ever terminated; never kill by name.
                if child.isRunning { child.terminate() }
                session.invalidateAndCancel()
            }
        } catch {
            if Task.isCancelled || error is CancellationError {
                status = "Local transcription cancelled"
                throw CancellationError()
            }
            let reported: Error = (error as? URLError)?.code == .timedOut ? Failure.timeout : error
            self.error = reported.localizedDescription
            status = "Local transcription failed"
            throw reported
        }
    }

    nonisolated static func alignmentPreset(modelPath: String) -> String? {
        let name = URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent
        guard name.hasPrefix("ggml-") else { return nil }
        let model = String(name.dropFirst(5))
        let presets = ["large-v3-turbo", "large-v3", "large-v2", "large-v1", "medium.en", "medium",
                       "small.en", "small", "base.en", "base", "tiny.en", "tiny"]
        guard let preset = presets.first(where: { model == $0 || model.hasPrefix($0 + "-q") }) else { return nil }
        return preset.replacingOccurrences(of: "large-v", with: "large.v").replacingOccurrences(of: "-turbo", with: ".turbo")
    }

    private func waitForServer(_ child: Process, port: Int, session: URLSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(75))
        while ContinuousClock.now < deadline {
            try Task.checkCancellation()
            guard child.isRunning else { throw Failure.server("Exited during model loading (status \(child.terminationStatus)). Check the session's local server log.") }
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/health")!)
            request.timeoutInterval = 1
            if let (_, response) = try? await session.data(for: request),
               let http = response as? HTTPURLResponse, [200, 404].contains(http.statusCode) { return }
            try await Task.sleep(for: .milliseconds(200))
        }
        throw Failure.timeout
    }

    /// Testable request construction: no endpoint parameter or expected-transcript prompt.
    nonisolated static func inferenceRequest(port: Int, wav: Data, boundary: String = UUID().uuidString) throws -> URLRequest {
        guard (1...65535).contains(port), !boundary.isEmpty,
              boundary.utf8.allSatisfy({ (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || $0 == 45 }) else {
            throw Failure.nonLoopback
        }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/inference")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 180
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func append(_ string: String) { body.append(Data(string.utf8)) }
        // The bundled runtime supports verbose_json. Unknown timestamp-granularity fields
        // are harmless on older whisper.cpp servers; segment estimates remain explicit.
        for (name, value) in [("response_format", "verbose_json"), ("temperature", "0.0"),
                              ("temperature_inc", "0.0"), ("word_timestamps", "true"),
                              ("timestamp_granularities[]", "word")] {
            append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n")
        }
        append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
        body.append(wav)
        append("\r\n--\(boundary)--\r\n")
        request.httpBody = body
        return request
    }

    nonisolated static func parseResponse(_ data: Data) throws -> LocalWhisperTranscript {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              json["error"] == nil else { throw Failure.malformedResponse }
        func number(_ value: Any?) -> Double? {
            let parsed = (value as? NSNumber)?.doubleValue ?? (value as? String).flatMap(Double.init)
            return parsed.flatMap { $0.isFinite ? $0 : nil }
        }
        func interval(_ entry: [String: Any]) -> (Double, Double)? {
            if let start = number(entry["start"]), let end = number(entry["end"]), start >= 0, end >= start { return (start, end) }
            if let offsets = entry["offsets"] as? [String: Any],
               let from = number(offsets["from"]), let to = number(offsets["to"]), from >= 0, to >= from { return (from / 1000, to / 1000) }
            return nil
        }
        func timedWord(_ entry: [String: Any]) -> LocalWhisperWord? {
            guard let text = (entry["word"] ?? entry["text"]) as? String, let (start, end) = interval(entry) else { return nil }
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("[_"), !trimmed.hasPrefix("<|") else { return nil }
            return LocalWhisperWord(text: trimmed, start: start, end: end)
        }
        let segments = (json["segments"] ?? json["transcription"]) as? [[String: Any]] ?? []
        let text = (json["text"] as? String ?? segments.compactMap { $0["text"] as? String }.joined(separator: " "))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard json["text"] is String || json["segments"] is [[String: Any]] || json["transcription"] is [[String: Any]] else {
            throw Failure.malformedResponse
        }
        if ["[BLANK_AUDIO]", "[SILENCE]", "[NO_SPEECH]"].contains(text.uppercased()) {
            return LocalWhisperTranscript(text: "", words: [])
        }
        var words = (json["words"] as? [[String: Any]] ?? []).compactMap(timedWord)
        if words.isEmpty {
            for segment in segments {
                let direct = (segment["words"] as? [[String: Any]] ?? []).compactMap(timedWord)
                if !direct.isEmpty { words += direct; continue }
                guard let (start, end) = interval(segment), let value = segment["text"] as? String else { continue }
                let parts = value.split(whereSeparator: \.isWhitespace).map(String.init)
                    .filter { !$0.hasPrefix("[_") && !$0.hasPrefix("<|") }
                guard !parts.isEmpty else { continue }
                // Segment-only responses are spread estimates, never phone alignment.
                for (index, part) in parts.enumerated() {
                    words.append(LocalWhisperWord(text: part,
                        start: start + (end - start) * Double(index) / Double(parts.count),
                        end: start + (end - start) * Double(index + 1) / Double(parts.count)))
                }
            }
        }
        words.sort { $0.start == $1.start ? $0.end < $1.end : $0.start < $1.start }
        // Some whisper.cpp builds expose token fragments in `words`. When the fragments
        // reproduce the rendered transcript exactly (ignoring whitespace), regroup by its
        // word boundaries and take the enclosing token-time range. No prompt is consulted.
        let rendered = text.split(whereSeparator: \.isWhitespace).map(String.init)
        let compactText = text.filter { !$0.isWhitespace }
        if !rendered.isEmpty, words.map(\.text).joined().filter({ !$0.isWhitespace }) == compactText {
            var ranges: [(start: Int, end: Int, word: LocalWhisperWord)] = []
            var cursor = 0
            for word in words {
                let length = word.text.filter { !$0.isWhitespace }.count
                ranges.append((cursor, cursor + length, word))
                cursor += length
            }
            cursor = 0
            words = rendered.compactMap { part in
                let start = cursor
                cursor += part.count
                let overlapping = ranges.filter { $0.start < cursor && $0.end > start }
                guard let first = overlapping.first, let last = overlapping.last else { return nil }
                return LocalWhisperWord(text: part, start: first.word.start, end: last.word.end)
            }
        }
        return LocalWhisperTranscript(text: text, words: words)
    }

    nonisolated private static func unusedPort() throws -> Int {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw Failure.server("Could not reserve a loopback port.") }
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0 else { throw Failure.server("Could not bind a loopback port.") }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let resolved = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(descriptor, $0, &length) }
        }
        guard resolved == 0 else { throw Failure.server("Could not read the loopback port.") }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    nonisolated static func convertToWAV(_ source: URL, destination: URL, timeRange: ClosedRange<Double>? = nil) throws {
        let input = try AVAudioFile(forReading: source)
        let format = input.processingFormat
        guard format.sampleRate > 0, input.length > 0,
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: format, to: target) else { throw Failure.invalidAudio }
        var endFrame = input.length
        if let timeRange {
            guard timeRange.lowerBound.isFinite, timeRange.upperBound.isFinite,
                  timeRange.lowerBound >= 0, timeRange.upperBound > timeRange.lowerBound else { throw Failure.invalidAudio }
            let start = timeRange.lowerBound * format.sampleRate
            guard start < Double(input.length) else { throw Failure.invalidAudio }
            input.framePosition = AVAudioFramePosition(start)
            endFrame = AVAudioFramePosition(min(Double(input.length), timeRange.upperBound * format.sampleRate))
        }
        // Repeated takes may make a visit long; only the selected cue is sent to Whisper.
        guard Double(endFrame - input.framePosition) / format.sampleRate <= 900 else { throw Failure.invalidAudio }
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
        let output = try AVAudioFile(forWriting: destination, settings: settings,
                                     commonFormat: .pcmFormatFloat32, interleaved: false)
        var readError: Error?
        var produced: AVAudioFramePosition = 0
        while true {
            try Task.checkCancellation()
            guard let buffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 8192) else { throw Failure.invalidAudio }
            var conversionError: NSError?
            let result = converter.convert(to: buffer, error: &conversionError) { requested, status in
                if Task.isCancelled { status.pointee = .endOfStream; return nil }
                guard input.framePosition < endFrame else { status.pointee = .endOfStream; return nil }
                guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(1, min(requested, 32768, AVAudioFrameCount(endFrame - input.framePosition)))) else {
                    status.pointee = .endOfStream; return nil
                }
                do {
                    try input.read(into: chunk)
                    status.pointee = chunk.frameLength == 0 ? .endOfStream : .haveData
                    return chunk.frameLength == 0 ? nil : chunk
                } catch { readError = error; status.pointee = .endOfStream; return nil }
            }
            if let readError { throw readError }
            if let conversionError { throw Failure.conversion(conversionError.localizedDescription) }
            guard result != .error else { throw Failure.conversion("AVAudioConverter failed.") }
            if buffer.frameLength > 0 { try output.write(from: buffer); produced += AVAudioFramePosition(buffer.frameLength) }
            if result == .endOfStream { break }
            guard buffer.frameLength > 0, produced <= 16000 * 901 else { throw Failure.conversion("Conversion made no progress or exceeded its duration bound.") }
        }
        try Task.checkCancellation()
        guard produced > 0 else { throw Failure.invalidAudio }
    }
}

/// Redirects are refused even if a local server asks to send audio elsewhere.
private final class LoopbackOnlyDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
