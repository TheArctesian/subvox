import AVFoundation
import Observation

/// Plays one reference at a time. Delegate callbacks from a replaced utterance are ignored.
@MainActor @Observable
final class ReferenceSpeaker: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var finished: CheckedContinuation<Void, Never>?
    private var utteranceID: ObjectIdentifier?
    private(set) var playing: String?

    override init() { super.init(); synthesizer.delegate = self }

    static func voiceLanguage(_ language: String) -> String {
        ["en": "en-GB", "fr": "fr-FR", "es": "es-ES", "zh": "zh-CN", "es-MX": "es-MX", "zh-Hans": "zh-CN", "zh-Hant": "zh-TW"][language] ?? language
    }

    func stop() {
        if let utteranceID { complete(utteranceID) }
        synthesizer.stopSpeaking(at: .immediate)
    }

    func speak(_ text: String, language: String) async {
        await play(AVSpeechUtterance(string: text), key: text, language: language, rate: 0.9)
    }

    func speakIPA(_ ipa: String, language: String) async {
        let text = NSMutableAttributedString(string: "sound")
        text.addAttribute(NSAttributedString.Key(AVSpeechSynthesisIPANotationAttribute), value: ipa, range: NSRange(location: 0, length: text.length))
        await play(AVSpeechUtterance(attributedString: text), key: ipa, language: language, rate: 0.75)
    }

    private func play(_ utterance: AVSpeechUtterance, key: String, language: String, rate: Float) async {
        guard !Task.isCancelled else { return }
        stop()
        utterance.voice = AVSpeechSynthesisVoice(language: Self.voiceLanguage(language))
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * rate
        let id = ObjectIdentifier(utterance)
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                finished = continuation
                utteranceID = id
                playing = key
                synthesizer.speak(utterance)
            }
        } onCancel: {
            Task { @MainActor in if self.utteranceID == id { self.stop() } }
        }
    }

    private func complete(_ id: ObjectIdentifier) {
        guard utteranceID == id else { return }
        utteranceID = nil
        playing = nil
        let continuation = finished
        finished = nil
        continuation?.resume()
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in self.complete(id) }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in self.complete(id) }
    }
}
