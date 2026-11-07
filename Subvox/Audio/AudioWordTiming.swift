import AVFoundation
import Foundation

struct AudioEnergyFrame: Codable, Hashable, Sendable {
    var start: Double
    var rms: Double
}

struct AudioTimingReview: Codable, Hashable, Sendable {
    var frameSeconds: Double
    var envelope: [AudioEnergyFrame]
    var activityThreshold: Double
    var omittedWords: [String]
    var warnings: [String]
}

/// Acoustic plausibility checks, not a forced phoneme alignment or a speech detector.
/// Attention supplies the word locations; energy only rejects unsupported locations.
enum AudioWordTiming {
    static func inspect(wav: URL, transcript: LocalWhisperTranscript) throws -> LocalWhisperTranscript {
        let file = try AVAudioFile(forReading: wav)
        guard file.processingFormat.sampleRate == 16000, file.processingFormat.channelCount == 1,
              file.length > 0, file.length <= 16000 * 901 else { throw LocalWhisper.Failure.invalidAudio }
        var envelope: [AudioEnergyFrame] = []
        var sum = 0.0, count = 0
        let frameSize = 320
        while file.framePosition < file.length {
            try Task.checkCancellation()
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 8192) else {
                throw LocalWhisper.Failure.invalidAudio
            }
            try file.read(into: buffer)
            guard let values = buffer.floatChannelData?[0], buffer.frameLength > 0 else { break }
            for i in 0..<Int(buffer.frameLength) {
                let value = Double(values[i])
                guard value.isFinite else { throw LocalWhisper.Failure.invalidAudio }
                sum += value * value; count += 1
                if count == frameSize {
                    envelope.append(.init(start: Double(envelope.count) * 0.02, rms: sqrt(sum / Double(count))))
                    sum = 0; count = 0
                }
            }
        }
        if count > 0 { envelope.append(.init(start: Double(envelope.count) * 0.02, rms: sqrt(sum / Double(count)))) }
        return checked(transcript, envelope: envelope, duration: Double(file.length) / 16000)
    }

    static func checked(_ transcript: LocalWhisperTranscript, envelope: [AudioEnergyFrame], duration: Double) -> LocalWhisperTranscript {
        var result = transcript
        let levels = envelope.map(\.rms).sorted()
        let peak = levels.last ?? 0
        let noise = levels.isEmpty ? 0 : levels[levels.count / 5]
        let threshold = max(0.00008, noise * 4, peak * 0.015)
        var omitted: [String] = []
        var clipped: [String] = []
        var previousEnd = 0.0
        result.words = transcript.words.compactMap { original in
            var word = original
            if word.end > duration, word.end <= duration + 0.2, let point = word.alignmentPeak, point <= duration {
                word.end = duration; clipped.append(word.text)
            }
            guard word.start.isFinite, word.end.isFinite, word.start >= 0, word.end > word.start,
                  word.end <= duration + 0.02, word.start >= previousEnd - 1e-8,
                  let point = word.alignmentPeak, point.isFinite, point >= word.start, point <= word.end,
                  envelope.contains(where: { abs($0.start + 0.01 - point) <= 0.12 && $0.rms > threshold }) else {
                omitted.append(word.text); return nil
            }
            previousEnd = word.end
            return word
        }
        var warnings = ["Word boundaries are audio-attention estimates. Energy checks reject quiet locations; they do not verify exact boundaries."]
        if !omitted.isEmpty { warnings.append("Untimed or acoustically unsupported words were omitted: " + omitted.joined(separator: ", ")) }
        if !clipped.isEmpty { warnings.append("Estimated trailing boundaries were clipped to the recorded audio: " + clipped.joined(separator: ", ")) }
        if result.words.isEmpty { warnings.append("No usable audio-aligned words. Check the microphone and local Whisper alignment support.") }
        result.timingReview = AudioTimingReview(frameSeconds: 0.02, envelope: envelope,
            activityThreshold: threshold, omittedWords: omitted, warnings: warnings)
        return result
    }

    /// whisper.cpp's t_dtw is a token attention location in 10 ms units, not an interval.
    /// Group token fragments using rendered text, then estimate boundaries between locations.
    /// Shared locations across adjacent words remain untimed rather than being split evenly.
    static func attentionTranscript(_ data: Data) throws -> LocalWhisperTranscript {
        var result = try LocalWhisper.parseResponse(data)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let segments = json["segments"] as? [[String: Any]] ?? []
        var aligned: [LocalWhisperWord] = []
        for segment in segments {
            guard let text = segment["text"] as? String,
                  let tokens = segment["words"] as? [[String: Any]], !tokens.isEmpty else { continue }
            let rendered = text.split(whereSeparator: \.isWhitespace).map(String.init)
            struct Token { var lo: Int; var hi: Int; var point: Double? }
            var ranges: [Token] = [], cursor = 0, compact = ""
            for token in tokens {
                guard let part = token["word"] as? String else { continue }
                let value = part.filter { !$0.isWhitespace }
                compact += value
                let tick = (token["t_dtw"] as? NSNumber)?.doubleValue
                let point = tick.flatMap { $0.isFinite && $0 >= 0 ? $0 * 0.01 : nil }
                ranges.append(Token(lo: cursor, hi: cursor + value.count, point: point))
                cursor += value.count
            }
            guard compact == text.filter({ !$0.isWhitespace }) else { continue }
            struct Word { var text: String; var first: Double?; var last: Double? }
            var words: [Word] = []; cursor = 0
            for part in rendered {
                let lo = cursor; cursor += part.count
                // Punctuation has its own attention point; it must not extend a word into silence.
                let letters = part.filter { $0.isLetter || $0.isNumber }
                guard !letters.isEmpty else { continue }
                let overlaps = ranges.filter { $0.lo < cursor && $0.hi > lo }
                // Ignore pure punctuation tokens when finding the spoken word's locations.
                let lexical = overlaps.filter { range in
                    let value = String(compact.dropFirst(range.lo).prefix(range.hi - range.lo))
                    return value.contains(where: { $0.isLetter || $0.isNumber })
                }
                guard !lexical.isEmpty else { continue }
                let points = lexical.compactMap(\.point)
                let valid = points.count == lexical.count && zip(points, points.dropFirst()).allSatisfy { $0 <= $1 }
                words.append(Word(text: part, first: valid ? points.first : nil, last: valid ? points.last : nil))
            }
            for i in words.indices {
                let word = words[i]
                guard let first = word.first, let last = word.last, last >= first else {
                    aligned.append(.init(text: word.text, start: 0, end: 0)); continue
                }
                let previous = i > 0 ? words[i - 1].last : nil
                let next = i + 1 < words.count ? words[i + 1].first : nil
                if (i > 0 && (previous == nil || previous! >= first)) ||
                    (i + 1 < words.count && (next == nil || next! <= last)) {
                    aligned.append(.init(text: word.text, start: first, end: first, alignmentPeak: first)); continue
                }
                let start = previous.map { ($0 + first) / 2 } ?? max(0, first - 0.16)
                let end = next.map { (last + $0) / 2 } ?? last + 0.16
                aligned.append(.init(text: word.text, start: start, end: end, alignmentPeak: (first + last) / 2))
            }
        }
        result.words = aligned
        result.timingMethod = "whisper.cpp audio-attention DTW; midpoint boundary estimates"
        return result
    }
}
