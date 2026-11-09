import Foundation

/// Visual reading targets only. These are never measured word/phoneme boundaries or
/// decoder inputs. The user's Done button still determines the recording's END.
struct WordFocusGuide: Codable, Equatable, Sendable {
    var leadIn: Double
    var wordSeconds: Double
    static let comfortable = WordFocusGuide(leadIn: 0.4, wordSeconds: 0.65)

    static func words(in text: String) -> [String] {
        fragments(in: text).filter { !$0.allSatisfy(\.isWhitespace) }
    }

    /// Keep the original spacing and attach punctuation after a Han character to its beat.
    static func fragments(in text: String) -> [String] {
        var result: [String] = []
        var word = ""
        var afterHan = false
        for character in text {
            let isHan = character.unicodeScalars.first?.properties.isIdeographic == true
            if character.isWhitespace || isHan {
                if isHan, !word.isEmpty, word.allSatisfy(\.isPunctuation) {
                    result.append(word + String(character))
                } else {
                    if !word.isEmpty { result.append(word) }
                    result.append(String(character))
                }
                word = ""
                afterHan = isHan
            } else if afterHan && character.isPunctuation {
                result[result.count - 1].append(character)
            } else {
                word.append(character)
                afterHan = false
            }
        }
        if !word.isEmpty { result.append(word) }
        return result
    }

    static func pinyinByWord(in text: String, pinyin: String?) -> [String?] {
        let words = words(in: text)
        let han = words.map { $0.unicodeScalars.first { $0.properties.isIdeographic } }
        let syllables = pinyin?.split(whereSeparator: \.isWhitespace).map(String.init) ?? []
        var extraCharacters = han.compactMap { $0 }.count - syllables.count
        var result = Array<String?>(repeating: nil, count: words.count)
        guard extraCharacters >= 0, !syllables.isEmpty else { return result }
        var syllableIndex = 0
        var index = 0
        while index < words.count {
            defer { index += 1 }
            guard han[index] != nil, syllableIndex < syllables.count else { continue }
            let syllable = syllables[syllableIndex]
            result[index] = syllable
            syllableIndex += 1
            // Erhua attaches the written 儿/兒 to the preceding syllable.
            if extraCharacters > 0, syllable.hasSuffix("r"), index + 1 < han.count,
               han[index + 1] == "儿" || han[index + 1] == "兒" {
                index += 1
                extraCharacters -= 1
            }
        }
        return extraCharacters == 0 && syllableIndex == syllables.count ? result : Array(repeating: nil, count: words.count)
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
