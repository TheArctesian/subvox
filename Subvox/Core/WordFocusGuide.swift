import Foundation

/// Visual reading targets only. These are never measured word/phoneme boundaries or
/// decoder inputs. The user's Done button still determines the recording's END.
struct WordFocusGuide: Codable, Equatable, Sendable {
    var leadIn: Double
    var wordSeconds: Double
    static let comfortable = WordFocusGuide(leadIn: 0.4, wordSeconds: 0.65)

    static func words(in text: String) -> [String] {
        text.split(whereSeparator: \.isWhitespace).map(String.init)
    }

    /// When the last word has had its full share of time.
    func duration(count: Int) -> Double { leadIn + Double(max(0, count)) * wordSeconds }

    func wordIndex(elapsed: Double, count: Int) -> Int? {
        guard elapsed.isFinite, leadIn.isFinite, leadIn >= 0,
              wordSeconds.isFinite, wordSeconds > 0, count > 0, elapsed >= leadIn else { return nil }
        let position = (elapsed - leadIn) / wordSeconds
        // Hold the last word until Done rather than cycling back through the sentence.
        if position >= Double(count - 1) { return count - 1 }
        return Int(floor(position + 1e-9))
    }
}
