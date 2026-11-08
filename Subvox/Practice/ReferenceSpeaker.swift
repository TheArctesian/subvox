import AVFoundation

/// Speaks a drill phrase with the system voice for its language, before the learner says it.
/// System voices are a stand-in for recorded native audio.
@MainActor final class ReferenceSpeaker: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var finished: CheckedContinuation<Void, Never>?

    override init() { super.init(); synthesizer.delegate = self }

    static func voiceLanguage(_ language: String) -> String {
        ["en": "en-GB", "fr": "fr-FR", "es": "es-ES", "zh": "zh-CN", "es-MX": "es-MX", "zh-Hans": "zh-CN", "zh-Hant": "zh-TW"][language] ?? language
    }

    func speak(_ text: String, language: String) async {
        synthesizer.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: Self.voiceLanguage(language))
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.9
        await withCheckedContinuation { continuation in
            finished = continuation
            synthesizer.speak(utterance)
        }
    }

    /// Speaks an IPA string (one sound, e.g. "uː" or "ʃa") with the language's system voice.
    func speakIPA(_ ipa: String, language: String) async {
        synthesizer.stopSpeaking(at: .immediate)
        let text = NSMutableAttributedString(string: "sound")
        text.addAttribute(NSAttributedString.Key(AVSpeechSynthesisIPANotationAttribute), value: ipa, range: NSRange(location: 0, length: text.length))
        let utterance = AVSpeechUtterance(attributedString: text)
        utterance.voice = AVSpeechSynthesisVoice(language: Self.voiceLanguage(language))
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.75
        await withCheckedContinuation { continuation in
            finished = continuation
            synthesizer.speak(utterance)
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self.finished?.resume(); self.finished = nil }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self.finished?.resume(); self.finished = nil }
    }
}
