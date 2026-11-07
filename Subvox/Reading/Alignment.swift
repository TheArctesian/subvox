import Foundation

/// One labelled stretch of the session timeline, in seconds from session start (the same
/// clock as `t` in motion.csv). Tiers nest like a Praat TextGrid: line > word > phone, and
/// every vowel phone also gets a jaw-class interval.
struct LabelInterval: Hashable, Sendable {
    enum Tier: String, Sendable { case line, word, phone, jaw }
    /// `speech`: the word was found by the recognizer and timed from audio.
    /// `estimated`: timed by spreading the line's words over a window, weighted by phone count.
    /// `prompt`: a line, timed by when it was on the teleprompter.
    enum Source: String, Sendable { case speech, estimated, prompt }

    var tier: Tier
    var start: Double
    var end: Double
    var label: String
    var source: Source

    static let csvHeader = "tier,start,end,label,source"
    var csvRow: String {
        let quoted = "\"" + label.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        return "\(tier.rawValue),\(String(format: "%.4f", start)),\(String(format: "%.4f", end)),\(quoted),\(source.rawValue)"
    }
}

struct RecognizedWord: Hashable, Sendable {
    var text: String
    var start: Double
    var end: Double
}

struct TimedWord: Hashable, Sendable {
    var text: String
    var start: Double
    var end: Double
    var source: LabelInterval.Source
}

enum Alignment {
    /// Times every expected word of a line. Words the recognizer heard (matched in order by
    /// longest common subsequence) keep its timing; the rest are spread over the gaps between
    /// those anchors, or over `window` when nothing matched (e.g. silent reading).
    static func timeWords(expected: [String], recognized: [RecognizedWord],
                          window: ClosedRange<Double>, lexicon: Lexicon) -> [TimedWord] {
        let matches = commonSubsequence(expected, recognized.map(\.text))
        var anchors: [Int: RecognizedWord] = [:]
        for (i, j) in matches { anchors[i] = recognized[j] }

        var timed: [TimedWord] = []
        var i = 0
        while i < expected.count {
            if let hit = anchors[i] {
                timed.append(TimedWord(text: expected[i], start: hit.start, end: hit.end, source: .speech))
                i += 1
                continue
            }
            // A run of unheard words between the previous anchor and the next one.
            var j = i
            while j < expected.count, anchors[j] == nil { j += 1 }
            let gapStart = timed.last?.end ?? window.lowerBound
            let gapEnd = max(gapStart, j < expected.count ? anchors[j]!.start : window.upperBound)
            let run = Array(expected[i..<j])
            let spans = spread(weights: run.map { Double(max(lexicon.phones($0)?.count ?? 1, 1)) },
                               over: gapStart...gapEnd)
            for (word, span) in zip(run, spans) {
                timed.append(TimedWord(text: word, start: span.lowerBound, end: span.upperBound, source: .estimated))
            }
            i = j
        }
        return timed
    }

    /// Word, phone and jaw intervals for timed words. Phones split their word's span with vowels
    /// weighted double, a rough stand-in for vowels being the longest segments in a syllable.
    static func label(_ words: [TimedWord], lexicon: Lexicon) -> [LabelInterval] {
        var labels: [LabelInterval] = []
        for word in words {
            labels.append(LabelInterval(tier: .word, start: word.start, end: word.end, label: word.text, source: word.source))
            guard let phones = lexicon.phones(word.text), !phones.isEmpty else { continue }
            let spans = spread(weights: phones.map { Lexicon.vowels.contains($0) ? 2 : 1 }, over: word.start...word.end)
            for (phone, span) in zip(phones, spans) {
                labels.append(LabelInterval(tier: .phone, start: span.lowerBound, end: span.upperBound,
                                            label: phone, source: .estimated))
                if let jaw = Lexicon.jawClass(phone) {
                    labels.append(LabelInterval(tier: .jaw, start: span.lowerBound, end: span.upperBound,
                                                label: jaw, source: .estimated))
                }
            }
        }
        return labels
    }

    static func spread(weights: [Double], over range: ClosedRange<Double>) -> [ClosedRange<Double>] {
        let total = weights.reduce(0, +)
        guard total > 0 else { return weights.map { _ in range } }
        var cursor = range.lowerBound
        let length = range.upperBound - range.lowerBound
        return weights.map { weight in
            let end = cursor + length * weight / total
            defer { cursor = end }
            return cursor...end
        }
    }

    /// Index pairs of a longest common subsequence of two word lists.
    static func commonSubsequence(_ a: [String], _ b: [String]) -> [(Int, Int)] {
        guard !a.isEmpty, !b.isEmpty else { return [] }
        var table = [[Int]](repeating: [Int](repeating: 0, count: b.count + 1), count: a.count + 1)
        for i in stride(from: a.count - 1, through: 0, by: -1) {
            for j in stride(from: b.count - 1, through: 0, by: -1) {
                table[i][j] = a[i] == b[j] ? table[i + 1][j + 1] + 1 : max(table[i + 1][j], table[i][j + 1])
            }
        }
        var pairs: [(Int, Int)] = []
        var i = 0, j = 0
        while i < a.count, j < b.count {
            if a[i] == b[j] {
                pairs.append((i, j))
                i += 1
                j += 1
            } else if table[i + 1][j] >= table[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        return pairs
    }
}
