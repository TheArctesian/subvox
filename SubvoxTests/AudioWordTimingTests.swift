import AVFoundation
import XCTest
@testable import Subvox

final class AudioWordTimingTests: XCTestCase {
    func testAttentionReplacesBadLeadingTimestampsAndRetainsAmbiguousWordsAsUntimed() throws {
        // Regression from the installed whisper.cpp runtime: ordinary starts at 0 s,
        // acoustic attention at 2.32 s, and adjacent words sharing the 3.92 s location.
        let raw = Data(#"{"text":"Her favorite song played softly","segments":[{"text":" Her favorite song played softly","start":0,"end":5,"words":[{"word":" Her","start":0,"end":0.34,"t_dtw":232},{"word":" favorite","start":0.34,"end":1.25,"t_dtw":278},{"word":" song","start":1.25,"end":1.7,"t_dtw":314},{"word":" played","start":2.24,"end":2.38,"t_dtw":392},{"word":" softly","start":2.48,"end":3.06,"t_dtw":392}]}]}"#.utf8)
        let result = try AudioWordTiming.attentionTranscript(raw)
        XCTAssertEqual(result.words.first?.start ?? 0, 2.16, accuracy: 1e-8)
        XCTAssertEqual(result.words.first?.alignmentPeak, 2.32)
        XCTAssertEqual(result.words.map(\.text), ["Her", "favorite", "song", "played", "softly"])
        XCTAssertEqual(result.words[3].start, result.words[3].end)
        XCTAssertEqual(result.words[4].start, result.words[4].end)
        XCTAssertTrue(result.timingMethod?.contains("estimates") == true)
    }

    func testTokenFragmentsGroupWithoutPunctuationExtendingTheLastWord() throws {
        let raw = Data(#"{"text":"hello world.","segments":[{"text":" hello world.","words":[{"word":" hel","t_dtw":100},{"word":"lo","t_dtw":110},{"word":" world","t_dtw":150},{"word":".","t_dtw":300}]}]}"#.utf8)
        let result = try AudioWordTiming.attentionTranscript(raw)
        XCTAssertEqual(result.words.map(\.text), ["hello", "world."])
        XCTAssertEqual(result.words[0].alignmentPeak ?? 0, 1.05, accuracy: 1e-8)
        XCTAssertEqual(result.words[1].end, 1.66, accuracy: 1e-8)
    }

    func testMissingAttentionNeverFallsBackToEvenlySpacedSegmentLabels() throws {
        let raw = Data(#"{"text":"two words","segments":[{"text":"two words","start":1,"end":3}]}"#.utf8)
        XCTAssertTrue(try AudioWordTiming.attentionTranscript(raw).words.isEmpty)
        let unavailable = Data(#"{"text":"one","segments":[{"text":" one","words":[{"word":" one","start":0,"end":1,"t_dtw":-1}]}]}"#.utf8)
        let parsed = try AudioWordTiming.attentionTranscript(unavailable)
        XCTAssertEqual(parsed.words[0].start, parsed.words[0].end)
    }

    func testEnergyRejectsQuietAndCollapsedLocationsWithoutChangingTheTranscript() {
        let transcript = LocalWhisperTranscript(text: "quiet her played softly", words: [
            .init(text: "quiet", start: 0, end: 0.4, alignmentPeak: 0.2),
            .init(text: "her", start: 2.16, end: 2.55, alignmentPeak: 2.32),
            .init(text: "played", start: 3.92, end: 3.92, alignmentPeak: 3.92),
            .init(text: "softly", start: 4.1, end: 4.4, alignmentPeak: 4.2)])
        let envelope = (0..<250).map { i in AudioEnergyFrame(start: Double(i) * 0.02, rms: (110...125).contains(i) ? 0.05 : 0.00001) }
        let result = AudioWordTiming.checked(transcript, envelope: envelope, duration: 5)
        XCTAssertEqual(result.text, transcript.text)
        XCTAssertEqual(result.words.map(\.text), ["her"])
        XCTAssertEqual(result.timingReview?.omittedWords, ["quiet", "played", "softly"])
    }

    func testOutOfFileAndOverlappingWordsAreOmittedAndSilenceProvidesNoLabels() {
        let words: [LocalWhisperWord] = [.init(text: "a", start: 1, end: 2, alignmentPeak: 1.5),
            .init(text: "b", start: 1.5, end: 2.5, alignmentPeak: 2),
            .init(text: "c", start: 3, end: 20, alignmentPeak: 4)]
        let envelope = (0..<250).map { i in AudioEnergyFrame(start: Double(i) * 0.02, rms: i > 50 ? 0.05 : 0) }
        let result = AudioWordTiming.checked(.init(text: "a b c", words: words), envelope: envelope, duration: 5)
        XCTAssertEqual(result.words.map(\.text), ["a"])
        let silence = envelope.map { AudioEnergyFrame(start: $0.start, rms: 0) }
        XCTAssertTrue(AudioWordTiming.checked(.init(text: "a", words: [words[0]]), envelope: silence, duration: 5).words.isEmpty)
    }

    func testFastManualEndClipsTheEstimatedTailWithoutMovingTheAttentionLocation() {
        let transcript = LocalWhisperTranscript(text: "word", words: [
            .init(text: "word", start: 0.84, end: 1.16, alignmentPeak: 1)])
        let envelope = (0..<53).map { i in AudioEnergyFrame(start: Double(i) * 0.02, rms: i > 40 ? 0.05 : 0) }
        let result = AudioWordTiming.checked(transcript, envelope: envelope, duration: 1.05)
        XCTAssertEqual(result.words.first?.end, 1.05)
        XCTAssertEqual(result.words.first?.alignmentPeak, 1)
        XCTAssertTrue(result.timingReview?.warnings.contains { $0.contains("clipped") } == true)
    }

    func testSupportedModelPresetsAreExplicitAndQuantizationDoesNotChangeThePreset() {
        XCTAssertEqual(LocalWhisper.alignmentPreset(modelPath: "/models/ggml-base.bin"), "base")
        XCTAssertEqual(LocalWhisper.alignmentPreset(modelPath: "ggml-base.en-q5_0.bin"), "base.en")
        XCTAssertEqual(LocalWhisper.alignmentPreset(modelPath: "ggml-large-v3-turbo.bin"), "large.v3.turbo")
        XCTAssertNil(LocalWhisper.alignmentPreset(modelPath: "ggml-custom.bin"))
        XCTAssertNil(LocalWhisper.alignmentPreset(modelPath: "ggml-base-special.bin"))
    }

    func testOldTranscriptsStillDecodeAndInvalidTimingHasADifferentErrorFromTheClock() throws {
        let legacy = Data(#"{"text":"word","words":[{"text":"word","start":0,"end":0}]}"#.utf8)
        let result = try JSONDecoder().decode(LocalWhisperTranscript.self, from: legacy)
        XCTAssertNil(result.timingReview)
        XCTAssertThrowsError(try SentenceReference.anchors(result, sessionStart: 100, audioOffset: 0)) {
            XCTAssertEqual(($0 as? SentenceReference.Failure)?.localizedDescription, SentenceReference.Failure.invalidTiming.localizedDescription)
        }
        XCTAssertThrowsError(try SentenceReference.anchors(result, sessionStart: .nan, audioOffset: 0)) {
            XCTAssertEqual(($0 as? SentenceReference.Failure)?.localizedDescription, SentenceReference.Failure.invalidClock.localizedDescription)
        }
    }
}
