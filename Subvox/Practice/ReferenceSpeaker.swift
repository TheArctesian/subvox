import AVFoundation
import Observation

/// Plays one reference at a time. Delegate callbacks from a replaced utterance are ignored.
@MainActor @Observable
final class ReferenceSpeaker: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var finished: CheckedContinuation<Bool, Never>?
    private var utteranceID: ObjectIdentifier?
    private var rehearsalID: UUID?
    private(set) var playing: String?

    override init() { super.init(); synthesizer.delegate = self }

    static func voiceLanguage(_ language: String) -> String {
        ["en": "en-GB", "fr": "fr-FR", "es": "es-ES", "zh": "zh-CN", "es-MX": "es-MX", "zh-Hans": "zh-CN", "zh-Hant": "zh-TW"][language] ?? language
    }

    func stop() {
        rehearsalID = nil
        if let utteranceID { complete(utteranceID, playedToEnd: false) }
        synthesizer.stopSpeaking(at: .immediate)
    }

    func speak(_ text: String, language: String) async {
        await rehearse(text, language: language, slower: false, repeats: false)
    }

    func rehearse(_ text: String, language: String, slower: Bool, repeats: Bool, key: String? = nil) async {
        guard !text.isEmpty, !Task.isCancelled else { return }
        stop()
        let rehearsal = UUID()
        rehearsalID = rehearsal
        defer { if rehearsalID == rehearsal { rehearsalID = nil } }
        repeat {
            let utterance = AVSpeechUtterance(string: text)
            if repeats { utterance.postUtteranceDelay = 0.5 }
            let completed = await play(utterance, key: key ?? text, language: language, rate: slower ? 0.68 : 0.9)
            if !completed { return }
        } while repeats && rehearsalID == rehearsal && !Task.isCancelled
    }

    func speakIPA(_ ipa: String, language: String) async {
        guard !Task.isCancelled else { return }
        stop()
        let text = NSMutableAttributedString(string: "sound")
        text.addAttribute(NSAttributedString.Key(AVSpeechSynthesisIPANotationAttribute), value: ipa, range: NSRange(location: 0, length: text.length))
        _ = await play(AVSpeechUtterance(attributedString: text), key: ipa, language: language, rate: 0.75)
    }

    private func play(_ utterance: AVSpeechUtterance, key: String, language: String, rate: Float) async -> Bool {
        guard !Task.isCancelled else { return false }
        utterance.voice = AVSpeechSynthesisVoice(language: Self.voiceLanguage(language))
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * rate
        let id = ObjectIdentifier(utterance)
        return await withTaskCancellationHandler {
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

    private func complete(_ id: ObjectIdentifier, playedToEnd: Bool) {
        guard utteranceID == id else { return }
        utteranceID = nil
        playing = nil
        let continuation = finished
        finished = nil
        continuation?.resume(returning: playedToEnd)
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in self.complete(id, playedToEnd: true) }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        let id = ObjectIdentifier(utterance)
        Task { @MainActor in self.complete(id, playedToEnd: false) }
    }
}
