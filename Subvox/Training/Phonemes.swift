import Foundation

/// The 39 General American phonemes (ARPAbet, as in the CMU dictionary), grouped by where they
/// are made. The grouping is also the first guess at which ones the jaw sensor will confuse:
/// sounds made at the same place move the jaw alike.
struct Phoneme: Identifiable, Hashable, Sendable {
    enum Group: String, CaseIterable, Sendable {
        case vowels = "Vowels"
        case diphthongs = "Diphthongs"
        case lips = "Lips"
        case lipTeeth = "Lip and teeth"
        case tongueTeeth = "Tongue and teeth"
        case ridge = "Tongue ridge"
        case palate = "Palate"
        case velum = "Back of tongue"
        case glides = "Glides and h"
    }

    var symbol: String
    var ipa: String
    var example: String
    var group: Group
    var id: String { symbol }

    static let all: [Phoneme] = [
        Phoneme(symbol: "IY", ipa: "i", example: "see", group: .vowels),
        Phoneme(symbol: "IH", ipa: "ɪ", example: "sit", group: .vowels),
        Phoneme(symbol: "EH", ipa: "ɛ", example: "bed", group: .vowels),
        Phoneme(symbol: "AE", ipa: "æ", example: "cat", group: .vowels),
        Phoneme(symbol: "AA", ipa: "ɑ", example: "father", group: .vowels),
        Phoneme(symbol: "AO", ipa: "ɔ", example: "thought", group: .vowels),
        Phoneme(symbol: "AH", ipa: "ʌ/ə", example: "cut", group: .vowels),
        Phoneme(symbol: "UH", ipa: "ʊ", example: "book", group: .vowels),
        Phoneme(symbol: "UW", ipa: "u", example: "food", group: .vowels),
        Phoneme(symbol: "ER", ipa: "ɝ", example: "bird", group: .vowels),
        Phoneme(symbol: "EY", ipa: "eɪ", example: "day", group: .diphthongs),
        Phoneme(symbol: "AY", ipa: "aɪ", example: "my", group: .diphthongs),
        Phoneme(symbol: "OY", ipa: "ɔɪ", example: "boy", group: .diphthongs),
        Phoneme(symbol: "AW", ipa: "aʊ", example: "now", group: .diphthongs),
        Phoneme(symbol: "OW", ipa: "oʊ", example: "go", group: .diphthongs),
        Phoneme(symbol: "P", ipa: "p", example: "pen", group: .lips),
        Phoneme(symbol: "B", ipa: "b", example: "bat", group: .lips),
        Phoneme(symbol: "M", ipa: "m", example: "man", group: .lips),
        Phoneme(symbol: "F", ipa: "f", example: "fan", group: .lipTeeth),
        Phoneme(symbol: "V", ipa: "v", example: "van", group: .lipTeeth),
        Phoneme(symbol: "TH", ipa: "θ", example: "thin", group: .tongueTeeth),
        Phoneme(symbol: "DH", ipa: "ð", example: "this", group: .tongueTeeth),
        Phoneme(symbol: "T", ipa: "t", example: "top", group: .ridge),
        Phoneme(symbol: "D", ipa: "d", example: "dog", group: .ridge),
        Phoneme(symbol: "S", ipa: "s", example: "sun", group: .ridge),
        Phoneme(symbol: "Z", ipa: "z", example: "zoo", group: .ridge),
        Phoneme(symbol: "N", ipa: "n", example: "no", group: .ridge),
        Phoneme(symbol: "L", ipa: "l", example: "leg", group: .ridge),
        Phoneme(symbol: "SH", ipa: "ʃ", example: "she", group: .palate),
        Phoneme(symbol: "ZH", ipa: "ʒ", example: "measure", group: .palate),
        Phoneme(symbol: "CH", ipa: "tʃ", example: "chin", group: .palate),
        Phoneme(symbol: "JH", ipa: "dʒ", example: "jam", group: .palate),
        Phoneme(symbol: "K", ipa: "k", example: "cat", group: .velum),
        Phoneme(symbol: "G", ipa: "ɡ", example: "go", group: .velum),
        Phoneme(symbol: "NG", ipa: "ŋ", example: "sing", group: .velum),
        Phoneme(symbol: "R", ipa: "ɹ", example: "red", group: .glides),
        Phoneme(symbol: "W", ipa: "w", example: "wet", group: .glides),
        Phoneme(symbol: "Y", ipa: "j", example: "yes", group: .glides),
        Phoneme(symbol: "HH", ipa: "h", example: "hat", group: .glides),
    ]

    static let bySymbol = Dictionary(uniqueKeysWithValues: all.map { ($0.symbol, $0) })
}

/// Picks the fewest prompt words that bring every phoneme up to its target count.
enum CoveragePlanner {
    /// Greedy set cover over `candidates` (most common first, so ties go to everyday words).
    /// Each word is used at most once and words in `used` are skipped, so extra rounds hear
    /// sounds in new contexts rather than repeating the same words.
    static func plan<G: RandomNumberGenerator>(
        needs: [String: Int], candidates: [String], lexicon: Lexicon, avoiding used: Set<String>,
        using generator: inout G
    ) -> [String] {
        var need = needs.filter { $0.value > 0 }
        var pool: [(word: String, phones: [String: Int])] = candidates.compactMap { word in
            guard !used.contains(word), let phones = lexicon.phones(word), isAccentStable(phones) else { return nil }
            return (word, Dictionary(phones.map { ($0, 1) }, uniquingKeysWith: +))
        }
        func gain(_ phones: [String: Int]) -> Int {
            phones.reduce(0) { $0 + min($1.value, need[$1.key] ?? 0) }
        }

        var chosen: [String] = []
        while !need.isEmpty {
            var best: Int?
            var bestGain = 0
            for (i, candidate) in pool.enumerated() {
                let g = gain(candidate.phones)
                // Prefer more coverage, then shorter words (cleaner takes).
                if g > bestGain || (g == bestGain && g > 0 && best.map { candidate.phones.count < pool[$0].phones.count } == true) {
                    best = i
                    bestGain = g
                }
            }
            guard let best else { break }
            let pick = pool.remove(at: best)
            chosen.append(pick.word)
            for (phone, count) in pick.phones where need[phone] != nil {
                need[phone]! -= count
                if need[phone]! <= 0 { need[phone] = nil }
            }
        }
        // Greedy order front-loads long words; shuffle so the session doesn't.
        return chosen.shuffled(using: &generator)
    }

    /// False for words whose General American label may not match how the speaker says them,
    /// because English accents split these word classes differently (Wells' lexical sets):
    /// - a vowel + R not followed by a vowel (car, more): the R is dropped in non-rhotic accents
    /// - BATH words, /æ/ before f, θ, ð, s, nt, ns, nd, nch, mp (after, dance): /ɑː/ in British accents
    /// - CLOTH words, /ɔ/ before f, θ, s, ŋ, g (off, long, dog): /ɒ/ in British accents
    /// Training only on words every accent labels alike keeps a mixed accent's labels consistent;
    /// genuine mergers (cot = caught) still show up in the test as confused sounds.
    static func isAccentStable(_ phones: [String]) -> Bool {
        for (i, phone) in phones.enumerated() {
            let next = i + 1 < phones.count ? phones[i + 1] : nil
            let after = i + 2 < phones.count ? phones[i + 2] : nil
            if Lexicon.vowels.contains(phone), next == "R", after.map({ !Lexicon.vowels.contains($0) }) ?? true {
                return false
            }
            if phone == "AE", ["F", "TH", "DH", "S"].contains(next)
                || (next == "N" && ["T", "S", "D", "CH"].contains(after)) || (next == "M" && after == "P") {
                return false
            }
            if phone == "AO", ["F", "TH", "S", "NG", "G"].contains(next) {
                return false
            }
        }
        return true
    }
}
