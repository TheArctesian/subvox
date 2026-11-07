import CryptoKit
import Foundation

/// Linguistic descriptions are kept separate from empirical sensor distinguishability.
struct ArticulatoryAttributes: Codable, Hashable, Sendable {
    var place: String
    var manner: String
    var voiced: Bool?
    var height: String?
    var backness: String?
    var rounded: Bool?
    var trajectory: [String] = []
}

struct ArticulatoryPhone: Codable, Hashable, Sendable {
    var ipa: String
    var arpabet: String?
    var attributes: ArticulatoryAttributes
    /// Broad visual hypotheses, NOT fitted visemes. Voicing/tongue place are deliberately
    /// absent: the camera cannot measure them reliably. Empirical overlap is learned separately.
    var gesturePrior: [Double]
}

enum ArticulatoryInventory {
    static let gestureNames = ["lip closure", "lip–teeth contact", "rounding", "spreading", "open jaw", "mid jaw", "close jaw", "unobserved articulation"]
    static let all: [ArticulatoryPhone] = {
        var result: [ArticulatoryPhone] = []
        func consonant(_ arpa: String, _ ipa: String, _ place: String, _ manner: String, _ voiced: Bool, _ gesture: Int) {
            var g = Array(repeating: 0.0, count: 8); g[gesture] = 1
            result.append(.init(ipa: ipa, arpabet: arpa, attributes: .init(place: place, manner: manner, voiced: voiced), gesturePrior: g))
        }
        for (a, i, p, m, v, g) in [
            ("P","p","bilabial","stop",false,0),("B","b","bilabial","stop",true,0),("M","m","bilabial","nasal",true,0),
            ("F","f","labiodental","fricative",false,1),("V","v","labiodental","fricative",true,1),
            ("TH","θ","dental","fricative",false,7),("DH","ð","dental","fricative",true,7),
            ("T","t","alveolar","stop",false,7),("D","d","alveolar","stop",true,7),
            ("S","s","alveolar","fricative",false,7),("Z","z","alveolar","fricative",true,7),
            ("N","n","alveolar","nasal",true,7),("L","l","alveolar","lateral approximant",true,7),
            ("SH","ʃ","postalveolar","fricative",false,2),("ZH","ʒ","postalveolar","fricative",true,2),
            ("CH","tʃ","postalveolar","affricate",false,2),("JH","dʒ","postalveolar","affricate",true,2),
            ("K","k","velar","stop",false,7),("G","ɡ","velar","stop",true,7),("NG","ŋ","velar","nasal",true,7),
            ("R","ɹ","alveolar/postalveolar","approximant",true,7),("W","w","labial-velar","approximant",true,2),
            ("Y","j","palatal","approximant",true,7),("HH","h","glottal","fricative",false,7)
        ] { consonant(a,i,p,m,v,g) }
        func vowel(_ arpa: String?, _ ipa: String, _ height: String, _ back: String, _ round: Bool, _ trajectory: [String] = []) {
            var g = Array(repeating: 0.0, count: 8)
            g[round ? 2 : 3] = 0.5
            g[height == "open" || height == "near-open" ? 4 : height.contains("close") && !height.contains("mid") ? 6 : 5] = 0.5
            result.append(.init(ipa: ipa, arpabet: arpa, attributes: .init(place: "vowel", manner: trajectory.isEmpty ? "vowel" : "diphthong", voiced: true, height: height, backness: back, rounded: round, trajectory: trajectory), gesturePrior: g))
        }
        vowel("IY","i","close","front",false); vowel("IH","ɪ","near-close","near-front",false)
        vowel("EH","ɛ","open-mid","front",false); vowel("AE","æ","near-open","front",false)
        vowel("AA","ɑ","open","back",false); vowel("AO","ɔ","open-mid","back",true)
        vowel("AH","ʌ","open-mid","back/central",false); vowel("UH","ʊ","near-close","near-back",true)
        vowel("UW","u","close","back",true); vowel("ER","ɝ","mid","central/rhotic",false)
        vowel("EY","eɪ","close-mid","front",false,["e","ɪ"]); vowel("AY","aɪ","open","central→front",false,["a","ɪ"])
        vowel("OY","ɔɪ","open-mid","back→front",true,["ɔ","ɪ"]); vowel("AW","aʊ","open","central→back",false,["a","ʊ"])
        vowel("OW","oʊ","close-mid","back",true,["o","ʊ"])
        // Optional speaker variants; these are available for editing, never imposed by ancestry.
        vowel(nil,"ə","mid","central",false); vowel(nil,"ɚ","mid","central/rhotic",false)
        vowel(nil,"ɒ","open","back",true); vowel(nil,"ɑː","open","back",false)
        vowel(nil,"ɜː","open-mid","central",false); vowel(nil,"əʊ","mid","central→back",true,["ə","ʊ"])
        vowel(nil,"e","close-mid","front",false); vowel(nil,"o","close-mid","back",true)
        vowel(nil,"a","open","front",false)
        vowel(nil,"iː","close","front",false);vowel(nil,"uː","close","back",true)
        vowel(nil,"ɔː","open-mid","back",true);vowel(nil,"ɛː","open-mid","front",false)
        vowel(nil,"ɜ","open-mid","central",false);vowel(nil,"ɐ","near-open","central",false)
        vowel(nil,"ɪə","near-close","front→central",false,["ɪ","ə"])
        vowel(nil,"eə","close-mid","front→central",false,["e","ə"])
        vowel(nil,"ʊə","near-close","back→central",true,["ʊ","ə"])
        consonant("","ɾ","alveolar","tap",true,7); result[result.count-1].arpabet = nil
        consonant("","ʔ","glottal","stop",false,7); result[result.count-1].arpabet = nil
        consonant("","r","alveolar","trill",true,7); result[result.count-1].arpabet = nil
        return result
    }()
    static let byIPA = Dictionary(uniqueKeysWithValues: all.map { ($0.ipa, $0) })
    static let byARPAbet = Dictionary(uniqueKeysWithValues: all.compactMap { p in p.arpabet.map { ($0,p) } })
    static func gesturePath(_ ipa:String)->[[Double]] {
        guard let phone=byIPA[ipa] else {return []}
        let path=phone.attributes.trajectory.compactMap {byIPA[$0]?.gesturePrior}
        return path.isEmpty ? [phone.gesturePrior]:path
    }
    static func phone(arpabet token: String) -> ArticulatoryPhone? {
        let base = token.uppercased().trimmingCharacters(in: .decimalDigits)
        if base == "AH", token.hasSuffix("0") { return byIPA["ə"] }
        if base == "ER", token.hasSuffix("0") { return byIPA["ɚ"] }
        return byARPAbet[base]
    }
}

struct PronunciationVariant: Codable, Identifiable, Hashable, Sendable {
    var id: String
    var ipa: [String]
    var arpabet: [String]?
    var source: String
    var speakerConfirmed: Bool
}

struct PronunciationProfile: Codable, Sendable {
    var version = 1
    var id = UUID(uuidString: "B51D9354-EE8D-4A40-98FC-333967C61244")!
    var variants: [String: [PronunciationVariant]] = [:]
    var notes = "Mixed accent; dictionary entries are proposals, not ground truth."

    func alternatives(_ word: String, lexicon: Lexicon) -> [PronunciationVariant] {
        let word = Lexicon.wordKey(word)
        if let edited = variants[word] { return edited }
        guard let tokens = lexicon.pronunciationTokens(word) else { return [] }
        let phones = tokens.compactMap(ArticulatoryInventory.phone(arpabet:))
        guard phones.count == tokens.count, !phones.isEmpty else { return [] }
        return [.init(id: "dictionary:\(word)", ipa: phones.map(\.ipa), arpabet: tokens,
                      source: "bundled ARPAbet dictionary proposal", speakerConfirmed: false)]
    }

    mutating func set(word: String, ipaLines: String) throws {
        let key = Lexicon.wordKey(word)
        guard Lexicon.words(in: key) == [key] else { throw LinguisticFailure.pronunciation }
        let lines = ipaLines.split(separator: "\n").map { $0.split(whereSeparator: \.isWhitespace).map(String.init) }
        guard !lines.isEmpty, lines.count <= 8,
              lines.allSatisfy({ !$0.isEmpty && $0.count <= 40 && $0.allSatisfy { ArticulatoryInventory.byIPA[$0] != nil } }) else { throw LinguisticFailure.pronunciation }
        variants[key] = lines.map { phones in
            .init(id: "speaker:" + SHA256.hash(data: Data((key+" "+phones.joined(separator:" ")).utf8)).map {String(format:"%02x",$0)}.joined(), ipa: phones, arpabet: nil,
                  source: "speaker-edited IPA alternatives", speakerConfirmed: true)
        }
    }
    func fingerprint() throws -> String {
        let e = JSONEncoder(); e.outputFormatting = [.sortedKeys]
        return SHA256.hash(data: try e.encode(self)).map { String(format: "%02x", $0) }.joined()
    }
    static var url: URL { SentenceStore.directory.appending(path: "pronunciation-profile.json") }
    static func load(from url: URL = Self.url) throws -> Self {
        FileManager.default.fileExists(atPath: url.path) ? try JSONDecoder().decode(Self.self, from: Data(contentsOf: url)) : Self()
    }
    func save(to url: URL = Self.url) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(self).write(to: url, options: .atomic)
    }
}

enum LinguisticFailure: LocalizedError {
    case pronunciation, alignment, split, insufficient, incompatible, source
    var errorDescription: String? {
        switch self {
        case .pronunciation: "Enter one word and one or more space-separated IPA pronunciations, one variant per line. Unsupported IPA symbols remain unlabelled."
        case .alignment: "The phone alignment is incompatible, incomplete or outside the recording clock. No phone boundaries were invented."
        case .split: "Sentence groups or recording sessions overlap, or a reserved final sentence entered this experiment."
        case .insufficient: "More complete training examples with motion and camera coverage are needed."
        case .incompatible: "The frozen representation model or evaluation manifest is incompatible."
        case .source: "AirPod source changed or motion has a gap. This attempt was saved for review; redo after the source settles."
        }
    }
}
