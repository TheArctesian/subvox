import AVFoundation
import XCTest
@testable import Subvox

final class LocalWhisperTests: XCTestCase {
    func testVerboseWordTimestampsAreAudioRelativeAndRoundTrip() throws {
        let data = Data(#"{"text":" hello world ","words":[{"word":"hello","start":0.2,"end":0.5},{"word":" world","start":0.6,"end":1.1}]}"#.utf8)
        let result = try LocalWhisper.parseResponse(data)
        XCTAssertEqual(result.text, "hello world")
        XCTAssertEqual(result.words.map(\.text), ["hello", "world"])
        XCTAssertEqual(result.words[0].start, 0.2)
        XCTAssertEqual(result.words[1].end, 1.1)
        XCTAssertEqual(try JSONDecoder().decode(LocalWhisperTranscript.self, from: JSONEncoder().encode(result)), result)
    }

    func testSegmentOnlyResponseProducesExplicitlyEstimatedWordSpans() throws {
        let data = Data(#"{"text":"two words","segments":[{"start":2,"end":4,"text":" two words"}]}"#.utf8)
        let result = try LocalWhisper.parseResponse(data)
        XCTAssertEqual(result.words, [.init(text: "two", start: 2, end: 3), .init(text: "words", start: 3, end: 4)])
    }

    func testNestedWordsAndWhisperCppMillisecondOffsets() throws {
        let direct = Data(#"{"segments":[{"text":"fine timing","start":0,"end":3,"words":[{"word":"fine","start":0.3,"end":0.7},{"word":"timing","start":1.8,"end":2.1}]}]}"#.utf8)
        let words = try LocalWhisper.parseResponse(direct).words
        XCTAssertEqual(words[0].start, 0.3)
        XCTAssertEqual(words[1].start, 1.8)
        let cpp = Data(#"{"transcription":[{"offsets":{"from":1000,"to":3000},"text":"hello again"}]}"#.utf8)
        XCTAssertEqual(try LocalWhisper.parseResponse(cpp).words,
                       [.init(text: "hello", start: 1, end: 2), .init(text: "again", start: 2, end: 3)])
    }

    func testMalformedAndInvalidTimingCannotBecomeLabels() throws {
        XCTAssertThrowsError(try LocalWhisper.parseResponse(Data("not JSON".utf8)))
        XCTAssertThrowsError(try LocalWhisper.parseResponse(Data(#"{"error":"bad model"}"#.utf8)))
        XCTAssertThrowsError(try LocalWhisper.parseResponse(Data(#"{"unexpected":42}"#.utf8)))
        let invalid = Data(#"{"text":"untimed","words":[{"word":"a","start":-1,"end":2},{"word":"b","start":4,"end":2},{"word":"c","start":"nan","end":2}]}"#.utf8)
        XCTAssertTrue(try LocalWhisper.parseResponse(invalid).words.isEmpty)
        XCTAssertEqual(try LocalWhisper.parseResponse(Data(#"{"text":"","segments":[]}"#.utf8)), .init(text: "", words: []))
    }

    func testTokenFragmentsAreRegroupedAndBlankAudioIsNotALabel() throws {
        let tokens = Data(#"{"text":"hello world","words":[{"word":"hel","start":0,"end":0.2},{"word":"lo","start":0.2,"end":0.3},{"word":"world","start":0.5,"end":1}]}"#.utf8)
        XCTAssertEqual(try LocalWhisper.parseResponse(tokens).words,
                       [.init(text: "hello", start: 0, end: 0.3), .init(text: "world", start: 0.5, end: 1)])
        let silence = Data(#"{"text":"[BLANK_AUDIO]","words":[{"word":"[","start":0,"end":0.1},{"word":"BLANK","start":0.1,"end":0.2}]}"#.utf8)
        XCTAssertEqual(try LocalWhisper.parseResponse(silence), .init(text: "", words: []))
    }

    func testRequestIsLoopbackMultipartAndContainsNoPrompt() throws {
        let audio = Data([0, 1, 2, 3])
        let request = try LocalWhisper.inferenceRequest(port: 49152, wav: audio, boundary: "test-boundary")
        XCTAssertEqual(request.url?.absoluteString, "http://127.0.0.1:49152/inference")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "multipart/form-data; boundary=test-boundary")
        let body = try XCTUnwrap(request.httpBody)
        XCTAssertNotNil(body.range(of: audio))
        let text = String(decoding: body, as: UTF8.self)
        XCTAssertTrue(text.contains("name=\"response_format\"\r\n\r\nverbose_json"))
        XCTAssertTrue(text.hasSuffix("--test-boundary--\r\n"))
        XCTAssertFalse(text.contains("prompt"))
        XCTAssertThrowsError(try LocalWhisper.inferenceRequest(port: 0, wav: audio))
        XCTAssertThrowsError(try LocalWhisper.inferenceRequest(port: 49152, wav: audio, boundary: "bad\r\nheader"))
    }

    @MainActor func testPathsPersistWithoutChangingGlobalPreferences() throws {
        let name = "Subvox.LocalWhisperTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let empty = FileManager.default.temporaryDirectory.appending(path: "no-whisper-model-\(UUID())")
        let service = LocalWhisper(defaults: defaults, modelDirectory: empty)
        service.modelPath = "/missing/ggml-test.bin"
        service.executablePath = "/missing/whisper-server"
        let restored = LocalWhisper(defaults: defaults, modelDirectory: empty)
        XCTAssertEqual(restored.modelPath, service.modelPath)
        XCTAssertEqual(restored.executablePath, service.executablePath)
        XCTAssertFalse(restored.isConfigured)
        XCTAssertFalse(restored.isTranscribing)
    }

    func testLocalCAFConversionPreservesDurationAndCreates16KMonoWAV() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "local-whisper-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appending(path: "input.caf")
        let destination = directory.appending(path: "converted.wav")
        let format = try XCTUnwrap(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 2, interleaved: false))
        do {
            let file = try AVAudioFile(forWriting: source, settings: format.settings)
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 48000))
            buffer.frameLength = 48000
            for channel in 0..<2 {
                let pointer = try XCTUnwrap(buffer.floatChannelData?[channel])
                for index in 0..<48000 { pointer[index] = 0 }
            }
            try file.write(from: buffer)
        }
        try LocalWhisper.convertToWAV(source, destination: destination)
        let result = try AVAudioFile(forReading: destination)
        XCTAssertEqual(result.processingFormat.sampleRate, 16000)
        XCTAssertEqual(result.processingFormat.channelCount, 1)
        XCTAssertEqual(Double(result.length) / 16000, 1, accuracy: 0.01)
    }
    func testCroppedConversionUsesOnlyTheRequestedAudioWindow() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "local-whisper-crop-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appending(path: "input.caf")
        let destination = directory.appending(path: "cue.wav")
        let format = try XCTUnwrap(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false))
        do {
            let file = try AVAudioFile(forWriting: source, settings: format.settings)
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 96000))
            buffer.frameLength = 96000
            let pointer = try XCTUnwrap(buffer.floatChannelData?[0])
            for i in 0..<96000 { pointer[i] = i < 48000 ? 0 : 0.1 }
            try file.write(from: buffer)
        }
        try LocalWhisper.convertToWAV(source, destination: destination, timeRange: 1...1.5)
        let clip = try AVAudioFile(forReading: destination)
        XCTAssertEqual(Double(clip.length) / 16000, 0.5, accuracy: 0.01)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: clip.processingFormat, frameCapacity: AVAudioFrameCount(clip.length)))
        try clip.read(into: buffer)
        let pointer = try XCTUnwrap(buffer.floatChannelData?[0])
        let mean = (0..<Int(buffer.frameLength)).reduce(0.0) { $0 + Double(pointer[$1]) } / Double(buffer.frameLength)
        XCTAssertGreaterThan(mean, 0.08) // No silent first second may leak into this crop.
        XCTAssertThrowsError(try LocalWhisper.convertToWAV(source, destination: destination, timeRange: 3...4))
    }

    func testShortCueCanBeConvertedAfterRetakesExtendTheVisitPastFifteenMinutes() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "local-whisper-long-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appending(path: "long.wav"), destination = directory.appending(path: "cue.wav")
        // Sparse PCM WAV: a long source without allocating or processing fifteen minutes of audio.
        let byteCount: UInt32 = 960 * 16000 * 2
        var header = Data()
        func append<T: FixedWidthInteger>(_ value: T) {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { header.append(contentsOf: $0) }
        }
        header.append(Data("RIFF".utf8)); append(byteCount + 36); header.append(Data("WAVEfmt ".utf8))
        append(UInt32(16)); append(UInt16(1)); append(UInt16(1)); append(UInt32(16000))
        append(UInt32(32000)); append(UInt16(2)); append(UInt16(16))
        header.append(Data("data".utf8)); append(byteCount)
        try header.write(to: source)
        let file = try FileHandle(forWritingTo: source)
        try file.truncate(atOffset: UInt64(header.count) + UInt64(byteCount)); try file.close()
        try LocalWhisper.convertToWAV(source, destination: destination, timeRange: 930...930.25)
        let clip = try AVAudioFile(forReading: destination)
        XCTAssertEqual(Double(clip.length) / clip.processingFormat.sampleRate, 0.25, accuracy: 0.01)
        XCTAssertThrowsError(try LocalWhisper.convertToWAV(source, destination: destination))
    }

}
